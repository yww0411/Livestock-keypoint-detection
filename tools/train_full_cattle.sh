#!/usr/bin/env bash
set -eo pipefail

# ---- ROOT & PYTHONPATH ----
if [[ -n "${BASH_SOURCE:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "[ROOT] ${ROOT}"

export TORCH_DETER_WARN_ONLY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${ROOT}:${ROOT}/SimCC-main:${ROOT}/SAR-main:${ROOT}/yolox_pose-main:${PYTHONPATH:-}"
set -u

# ---- Entrypoints ----
TRAIN_PY="${ROOT}/tools/train.py"
TEST_PY="${ROOT}/tools/test.py"
[[ -f "${TRAIN_PY}" && -f "${TEST_PY}" ]] || { echo "[ERR] missing ${TRAIN_PY} or ${TEST_PY}"; exit 1; }

# ---- Resume control（"", "auto", or a ckpt path）----
RESUME="${RESUME:-}"

# ---- Data roots ----
DATA_ROOT="${ROOT}/data_process"
CATTLE_ROOT="${DATA_ROOT}/cattle"

# ---- Configs (Cattle) ----
SIMCC_W48_C="${ROOT}/SimCC-main/configs/cattle/simcc_hrnet-w48_8xb32-280e_cattle-384x288.py"
SIMCC_HRTR_C="${ROOT}/SimCC-main/configs/cattle/simcc_hrtran_8xb32-280e_cattle-384x288.py"
SIMCC_LITE_C="${ROOT}/SimCC-main/configs/cattle/simcc_litehrnet_8xb32-280e_cattle-384x288.py"
SIMCC_R50_C="${ROOT}/SimCC-main/configs/cattle/simcc_res50_8xb32-280e_cattle-384x288.py"
SIMCC_SWIN_C="${ROOT}/SimCC-main/configs/cattle/simcc_swim_8xb32-280e_cattle-384x288.py"

SAR_W48_C="${ROOT}/SAR-main/configs/cattle/SAR_hrnet-w48_8xb32-280e_cattle-384x288.py"
SAR_R50_C="${ROOT}/SAR-main/configs/cattle/SAR_res50_8xb32-280e_cattle-384x288.py"

YOLOX_S_C="${ROOT}/yolox_pose-main/configs/yolox-pose_s_8xb32-300e_cattle_coco.py"
YOLOX_M_C="${ROOT}/yolox_pose-main/configs/yolox-pose_m_4xb16-300e_cattle_coco.py"

# ---- Work & Outputs ----
WORK_ROOT="${ROOT}/work_dirs_train_cattle"
RAW_JSON_ROOT="${ROOT}/analysis/raw_train_cattle"
mkdir -p "${WORK_ROOT}" "${RAW_JSON_ROOT}"

# ---- Helpers ----
pick_ann () {
  local root="$1"; shift
  for rel in "$@"; do
    [[ -f "${root}/${rel%.*}_canon.json" ]] && { echo "${root}/${rel%.*}_canon.json"; return; }
    [[ -f "${root}/${rel}" ]] && { echo "${root}/${rel}"; return; }
  done
  echo ""
}
pick_meta () {
  local n="$1"
  for p in \
    "${ROOT}/configs/_base_/datasets/custom4${n}.py" \
    "${ROOT}/SimCC-main/configs/_base_/datasets/custom4${n}.py" \
    "${ROOT}/SAR-main/configs/_base_/datasets/custom4${n}.py" \
    "${ROOT}/yolox_pose-main/configs/_base_/datasets/custom4${n}.py"
  do [[ -f "$p" ]] && { echo "$p"; return; }; done
  echo ""
}

META_C="$(pick_meta cattle)"; echo "[META] cattle: ${META_C}"
[[ -n "${META_C}" ]] || { echo "[ERR] no metainfo for cattle"; exit 1; }

# ---- Unified opts (正式训练) ----
SEEDS=(${SEEDS:-3407 1337 2029})
BS=32
NUM_WORKERS=${NUM_WORKERS:-8}

BS_OPTS="train_dataloader.batch_size=${BS} val_dataloader.batch_size=${BS} test_dataloader.batch_size=${BS} \
         train_dataloader.num_workers=${NUM_WORKERS} val_dataloader.num_workers=${NUM_WORKERS} test_dataloader.num_workers=${NUM_WORKERS}"

SCOPE_OPTS="val_evaluator.0._scope_=mmpose test_evaluator.0._scope_=mmpose"
TD_EPOCH_OPTS="train_cfg.max_epochs=280"
TD_SIZE_SAR="model.test_cfg.flip_test=True"

BU_SIZE="
img_scale=[320,320] \
data_preprocessor.batch_augments.0.random_size_range=[320,320] \
train_pipeline_stage1.2.img_scale=[320,320] \
train_pipeline_stage1.3.border=[-160,-160] \
train_pipeline_stage1.4.img_scale=[320,320] \
train_pipeline_stage2.2.scale=[320,320] \
test_pipeline.2.scale=[320,320]
"
YOLOX_EPOCH280_OPTS="train_cfg.max_epochs=280 param_scheduler.1.T_max=260 param_scheduler.1.end=260 param_scheduler.2.begin=260 param_scheduler.2.end=280"
YOLOX_META_C="${ROOT}/yolox_pose-main/configs/_base_/datasets/custom4cattle.py"

# 仅 p1A 清空 data_prefix.img
PREFIX_EMPTY="train_dataloader.dataset.data_prefix.img= val_dataloader.dataset.data_prefix.img= test_dataloader.dataset.data_prefix.img="

BEST_OPTS="default_hooks.checkpoint.save_best=coco/AP default_hooks.checkpoint.rule=greater"

# ---- Resume helper（返回需要添加到命令行的 --resume，必要时写入 last_checkpoint）----
make_resume_arg () {
  local wd="$1"
  local arg=""
  if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" == "auto" ]]; then
      # 若无 last_checkpoint，尝试用最新 epoch_*.pth 补上
      if [[ ! -f "${wd}/last_checkpoint" ]]; then
        local latest="$(ls -1t "${wd}"/epoch_*.pth 2>/dev/null | head -n1 || true)"
        [[ -n "${latest}" ]] && echo "${latest}" > "${wd}/last_checkpoint"
      fi
      [[ -f "${wd}/last_checkpoint" ]] && arg="--resume"
    else
      # 指定了 ckpt 路径
      mkdir -p "${wd}"
      echo "${RESUME}" > "${wd}/last_checkpoint"
      arg="--resume"
    fi
  fi
  echo "${arg}"
}

run_one () {
  local cfg="$1" tag="$2" root="$3" tr_ann="$4" va_ann="$5" te_ann="$6" extra_opts="$7" meta="$8" is_yolox="$9" need_prefix_empty="${10:-0}"

  for seed in "${SEEDS[@]}"; do
    local wd="${WORK_ROOT}/${tag}_cattle_s${seed}"
    local out="${RAW_JSON_ROOT}/cattle_${tag}_s${seed}"
    mkdir -p "${wd}"

    echo "[TRAIN CATTLE] ${tag} | seed=${seed} | work=${wd}"

    # 已完成则跳过
    if [[ -f "${wd}/epoch_280.pth" ]]; then
      echo "[SKIP done] ${wd} has epoch_280.pth"
      continue
    fi

    local maybe_prefix=""
    [[ "${need_prefix_empty}" == "1" ]] && maybe_prefix="${PREFIX_EMPTY}"

    local resume_arg
    resume_arg="$(make_resume_arg "${wd}")"

    # 公共 cfg 片段
    local common_opts=(
      ${BEST_OPTS}
      randomness.seed=${seed} randomness.deterministic=True
      ${BS_OPTS} ${SCOPE_OPTS} ${extra_opts} ${maybe_prefix}
      train_dataloader.dataset.data_root="${root}"
      val_dataloader.dataset.data_root="${root}"
      test_dataloader.dataset.data_root="${root}"
      train_dataloader.dataset.ann_file="${tr_ann}"
      val_dataloader.dataset.ann_file="${va_ann}"
      test_dataloader.dataset.ann_file="${te_ann}"
    )

    if [[ "${is_yolox}" == "1" ]]; then
      python "${TRAIN_PY}" "${cfg}" ${resume_arg} \
        --work-dir "${wd}" \
        --cfg-options \
          "${common_opts[@]}" \
          ${YOLOX_EPOCH280_OPTS} \
          bbox_head.loss_pose.metainfo="${meta}" \
          model.train_cfg.assigner.oks_calculator.metainfo="${meta}" \
          val_evaluator.0.type=mmpose.CocoMetric val_evaluator.0.ann_file="${va_ann}"

      local CKPT="${wd}/best.pth"; [[ -f "${CKPT}" ]] || CKPT="$(ls -1t "${wd}"/*.pth | head -n1)"

      python "${TEST_PY}" "${cfg}" "${CKPT}" \
        --cfg-options \
          ${BS_OPTS} ${SCOPE_OPTS} ${extra_opts} ${maybe_prefix} \
          test_dataloader.dataset.data_root="${root}" \
          test_dataloader.dataset.ann_file="${te_ann}" \
          bbox_head.loss_pose.metainfo="${meta}" \
          test_evaluator.0.type=mmpose.CocoMetric test_evaluator.0.ann_file="${te_ann}" \
          test_evaluator.0.outfile_prefix="${out}"
    else
      python "${TRAIN_PY}" "${cfg}" ${resume_arg} \
        --work-dir "${wd}" \
        --cfg-options \
          "${common_opts[@]}" \
          ${TD_EPOCH_OPTS} \
          train_dataloader.dataset.metainfo.from_file="${meta}" \
          val_dataloader.dataset.metainfo.from_file="${meta}" \
          test_dataloader.dataset.metainfo.from_file="${meta}" \
          val_evaluator.0.type=CocoMetric val_evaluator.0.ann_file="${va_ann}"

      local CKPT="${wd}/best.pth"; [[ -f "${CKPT}" ]] || CKPT="$(ls -1t "${wd}"/*.pth | head -n1)"

      python "${TEST_PY}" "${cfg}" "${CKPT}" \
        --cfg-options \
          ${BS_OPTS} ${SCOPE_OPTS} ${extra_opts} ${maybe_prefix} \
          test_dataloader.dataset.data_root="${root}" \
          test_dataloader.dataset.ann_file="${te_ann}" \
          test_dataloader.dataset.metainfo.from_file="${meta}" \
          test_evaluator.0.type=CocoMetric test_evaluator.0.ann_file="${te_ann}" \
          test_evaluator.0.outfile_prefix="${out}"
    fi
  done
}

# ---- Cattle annotations ----
TR=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1A_train.json")
VA=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1A_val.json")
TE=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1A_test.json")
[[ -n "$TR" && -n "$VA" && -n "$TE" ]] || { echo "[ERR] cattle p1A json not found"; exit 1; }

C_P1B_TR=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1B_train.json")
C_P1B_VA=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1B_val.json")
C_P1B_TE=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1B_test.json")
[[ -n "$C_P1B_TR" && -n "$C_P1B_VA" && -n "$C_P1B_TE" ]] || { echo "[ERR] cattle p1B json not found"; exit 1; }

C_P1C_TR=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1C_train.json")
C_P1C_VA=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1C_val.json")
C_P1C_TE=$(pick_ann "${CATTLE_ROOT}" "annotations_p1/cattle_p1C_test.json")
[[ -n "$C_P1C_TR" && -n "$C_P1C_VA" && -n "$C_P1C_TE" ]] || { echo "[ERR] cattle p1C json not found"; exit 1; }

# ---- Run all models (p1A/B/C) ----
# p1A（清空前缀）
run_one "${SIMCC_W48_C}"  "simcc_w48_p1A"       "${CATTLE_ROOT}" "$TR" "$VA" "$TE" ""               "${META_C}" "0" 1
run_one "${SIMCC_HRTR_C}" "simcc_hrtran_p1A"    "${CATTLE_ROOT}" "$TR" "$VA" "$TE" ""               "${META_C}" "0" 1
run_one "${SIMCC_LITE_C}" "simcc_litehrnet_p1A" "${CATTLE_ROOT}" "$TR" "$VA" "$TE" ""               "${META_C}" "0" 1
run_one "${SIMCC_R50_C}"  "simcc_res50_p1A"     "${CATTLE_ROOT}" "$TR" "$VA" "$TE" ""               "${META_C}" "0" 1
run_one "${SIMCC_SWIN_C}" "simcc_swin_p1A"      "${CATTLE_ROOT}" "$TR" "$VA" "$TE" ""               "${META_C}" "0" 1

run_one "${SAR_W48_C}"    "sar_w48_p1A"         "${CATTLE_ROOT}" "$TR" "$VA" "$TE" "${TD_SIZE_SAR}" "${META_C}" "0" 1
run_one "${SAR_R50_C}"    "sar_res50_p1A"       "${CATTLE_ROOT}" "$TR" "$VA" "$TE" "${TD_SIZE_SAR}" "${META_C}" "0" 1

run_one "${YOLOX_S_C}"    "yolox_s_p1A"         "${CATTLE_ROOT}" "$TR" "$VA" "$TE" "${BU_SIZE}"     "${YOLOX_META_C}" "1" 1
run_one "${YOLOX_M_C}"    "yolox_m_p1A"         "${CATTLE_ROOT}" "$TR" "$VA" "$TE" "${BU_SIZE}"     "${YOLOX_META_C}" "1" 1

# p1B（不清前缀）
run_one "${SIMCC_W48_C}"  "simcc_w48_p1B"        "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" ""               "${META_C}"        "0" 0
run_one "${SIMCC_HRTR_C}" "simcc_hrtran_p1B"     "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" ""               "${META_C}"        "0" 0
run_one "${SIMCC_LITE_C}" "simcc_litehrnet_p1B"  "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" ""               "${META_C}"        "0" 0
run_one "${SIMCC_R50_C}"  "simcc_res50_p1B"      "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" ""               "${META_C}"        "0" 0
run_one "${SIMCC_SWIN_C}" "simcc_swin_p1B"       "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" ""               "${META_C}"        "0" 0

run_one "${SAR_W48_C}"    "sar_w48_p1B"          "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" "${TD_SIZE_SAR}" "${META_C}"        "0" 0
run_one "${SAR_R50_C}"    "sar_res50_p1B"        "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" "${TD_SIZE_SAR}" "${META_C}"        "0" 0

run_one "${YOLOX_S_C}"    "yolox_s_p1B"          "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" "${BU_SIZE}"     "${YOLOX_META_C}"  "1" 0
run_one "${YOLOX_M_C}"    "yolox_m_p1B"          "${CATTLE_ROOT}" "${C_P1B_TR}" "${C_P1B_VA}" "${C_P1B_TE}" "${BU_SIZE}"     "${YOLOX_META_C}"  "1" 0

# p1C（不清前缀）
run_one "${SIMCC_W48_C}"  "simcc_w48_p1C"        "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" ""               "${META_C}"        "0" 0
run_one "${SIMCC_HRTR_C}" "simcc_hrtran_p1C"     "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" ""               "${META_C}"        "0" 0
run_one "${SIMCC_LITE_C}" "simcc_litehrnet_p1C"  "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" ""               "${META_C}"        "0" 0
run_one "${SIMCC_R50_C}"  "simcc_res50_p1C"      "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" ""               "${META_C}"        "0" 0
run_one "${SIMCC_SWIN_C}" "simcc_swin_p1C"       "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" ""               "${META_C}"        "0" 0

run_one "${SAR_W48_C}"    "sar_w48_p1C"          "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" "${TD_SIZE_SAR}" "${META_C}"        "0" 0
run_one "${SAR_R50_C}"    "sar_res50_p1C"        "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" "${TD_SIZE_SAR}" "${META_C}"        "0" 0

run_one "${YOLOX_S_C}"    "yolox_s_p1C"          "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" "${BU_SIZE}"     "${YOLOX_META_C}"  "1" 0
run_one "${YOLOX_M_C}"    "yolox_m_p1C"          "${CATTLE_ROOT}" "${C_P1C_TR}" "${C_P1C_VA}" "${C_P1C_TE}" "${BU_SIZE}"     "${YOLOX_META_C}"  "1" 0


echo "[TRAIN CATTLE DONE] logs: ${WORK_ROOT} | json: ${RAW_JSON_ROOT}"
