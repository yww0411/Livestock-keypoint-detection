#!/usr/bin/env bash
set -eo pipefail

# ========== ROOT & PYTHONPATH ==========
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

TRAIN_PY="${ROOT}/tools/train.py"
TEST_PY="${ROOT}/tools/test.py"
[[ -f "${TRAIN_PY}" && -f "${TEST_PY}" ]] || { echo "[ERR] missing ${TRAIN_PY} or ${TEST_PY}"; exit 1; }

# ========== 断点/筛选 ==========
RESUME="${RESUME:-}"         # "", "auto" 或具体 ckpt 路径
MODEL_FILTER="${MODEL_FILTER:-}"
SEED_FILTER="${SEED_FILTER:-}"

DATA_ROOT="${ROOT}/data_process"
HORSE_ROOT="${DATA_ROOT}/horse"
SHEEP_ROOT="${DATA_ROOT}/sheep"

[[ -d "${HORSE_ROOT}" ]] || { echo "[ERR] HORSE_ROOT not found: ${HORSE_ROOT}"; exit 1; }
[[ -d "${SHEEP_ROOT}" ]] || { echo "[ERR] SHEEP_ROOT not found: ${SHEEP_ROOT}"; exit 1; }

# ---- Configs (Horse/Sheep) ----
# SimCC (5)
SIMCC_W48_H="${ROOT}/SimCC-main/configs/horse/simcc_hrnet-w48_8xb32-280e_horse-384x288.py"
SIMCC_HRTR_H="${ROOT}/SimCC-main/configs/horse/simcc_hrtran_8xb32-280e_horse-384x288.py"
SIMCC_LITE_H="${ROOT}/SimCC-main/configs/horse/simcc_litehrnet_8xb32-280e_horse-384x288.py"
SIMCC_R50_H="${ROOT}/SimCC-main/configs/horse/simcc_res50_8xb32-280e_horse-384x288.py"
SIMCC_SWIN_H="${ROOT}/SimCC-main/configs/horse/simcc_swim_8xb32-280e_horse-384x288.py"

SIMCC_W48_S="${ROOT}/SimCC-main/configs/sheep/simcc_hrnet-w48_8xb32-280e_sheep-384x288.py"
SIMCC_HRTR_S="${ROOT}/SimCC-main/configs/sheep/simcc_hrtran_8xb32-280e_sheep-384x288.py"
SIMCC_LITE_S="${ROOT}/SimCC-main/configs/sheep/simcc_litehrnet_8xb32-280e_sheep-384x288.py"
SIMCC_R50_S="${ROOT}/SimCC-main/configs/sheep/simcc_res50_8xb32-280e_sheep-384x288.py"
SIMCC_SWIN_S="${ROOT}/SimCC-main/configs/sheep/simcc_swim_8xb32-280e_sheep-384x288.py"

# SAR (2)
SAR_W48_H="${ROOT}/SAR-main/configs/horse/SAR_hrnet-w48_8xb32-280e_horse-384x288.py"
SAR_R50_H="${ROOT}/SAR-main/configs/horse/SAR_res50_8xb32-280e_horse-384x288.py"
SAR_W48_S="${ROOT}/SAR-main/configs/sheep/SAR_hrnet-w48_8xb32-280e_sheep-384x288.py"
SAR_R50_S="${ROOT}/SAR-main/configs/sheep/SAR_res50_8xb32-280e_sheep-384x288.py"

# YOLOX (2)
YOLOX_S_H="${ROOT}/yolox_pose-main/configs/yolox-pose_s_8xb32-300e_horse_coco.py"
YOLOX_M_H="${ROOT}/yolox_pose-main/configs/yolox-pose_m_4xb16-300e_horse_coco.py"
YOLOX_S_S="${ROOT}/yolox_pose-main/configs/yolox-pose_s_8xb32-300e_sheep_coco.py"
YOLOX_M_S="${ROOT}/yolox_pose-main/configs/yolox-pose_m_4xb16-300e_sheep_coco.py"

WORK_ROOT="${ROOT}/work_dirs_train_hs"
RAW_JSON_ROOT="${ROOT}/analysis/raw_train_hs"
mkdir -p "${WORK_ROOT}" "${RAW_JSON_ROOT}"

pick_ann () {
  local root="$1"; shift
  for rel in "$@"; do
    [[ -f "${root}/${rel%.*}_canon.json" ]] && { echo "${root}/${rel%.*}_canon.json"; return; }
    [[ -f "${root}/${rel}" ]] && { echo "${root}/${rel}"; return; }
  done; echo ""
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
META_H="$(pick_meta horse)"; echo "[META] horse : ${META_H}"
META_S="$(pick_meta sheep)"; echo "[META] sheep : ${META_S}"
[[ -n "${META_H}" && -n "${META_S}" ]] || { echo "[ERR] metainfo missing"; exit 1; }

SEEDS=(1337 2029 3407)
BS=32
BS_OPTS="train_dataloader.batch_size=${BS} val_dataloader.batch_size=${BS} test_dataloader.batch_size=${BS}"
SCOPE_OPTS="val_evaluator.0._scope_=mmpose test_evaluator.0._scope_=mmpose"
TD_SIZE="codec.input_size=[288,384]"
TD_SIZE_SAR="codec.input_size=[288,384] model.test_cfg.flip_test=True"
BU_SIZE="
img_scale=[320,320] \
data_preprocessor.batch_augments.0.random_size_range=[320,320] \
train_pipeline_stage1.2.img_scale=[320,320] \
train_pipeline_stage1.3.border=[-160,-160] \
train_pipeline_stage1.4.img_scale=[320,320] \
train_pipeline_stage2.2.scale=[320,320] \
test_pipeline.2.scale=[320,320]
"
YOLOX_META_H="${ROOT}/yolox_pose-main/configs/_base_/datasets/custom4horse.py"
YOLOX_META_S="${ROOT}/yolox_pose-main/configs/_base_/datasets/custom4sheep.py"

# ---- 统一的执行函数（参数次序固定：cfg tag species root tr va te size meta [is_yolox] [seed] [resume_mode]）
run_one () {
  local cfg="$1" tag="$2" species="$3" root="$4" tr_ann="$5" va_ann="$6" te_ann="$7" size_opts="$8" meta="$9"
  shift 9
  local is_yolox="${1:-0}"           # 可选：是否 YOLOX
  local seed="${2:-1337}"            # 可选：整数 seed（用于 randomness.seed）
  local resume_mode="${3:-${RESUME:-}}"  # 可选：优先使用函数入参，否则读全局 RESUME

  # 按需过滤
  if [[ -n "${MODEL_FILTER}" && "${tag}" != *"${MODEL_FILTER}"* ]]; then
    echo "[SKIP model-filter] ${tag}"; return
  fi
  if [[ -n "${SEED_FILTER:-}" ]]; then
    local sf="${SEED_FILTER#s}"
    [[ "${seed}" == "${sf}" ]] || { echo "[SKIP seed-filter] ${tag} s${seed}"; return; }
  fi

  local wd="${WORK_ROOT}/${tag}_${species}_s${seed}"
  local out="${RAW_JSON_ROOT}/${species}_${tag}_s${seed}"
  mkdir -p "${wd}"

  echo "[RUN] ${tag} | ${species} | seed=${seed} | yolox=${is_yolox}"

  # ===== 统一续训：写 last_checkpoint，仅传 --resume =====
  local resume_cli=()
  if [[ "${resume_mode}" == "auto" ]]; then
    if [[ -f "${wd}/last_checkpoint" ]]; then
      resume_cli=(--resume)
    else
      # 找最近的 epoch_*.pth，写入 last_checkpoint
      last="$(ls -1t "${wd}"/epoch_*.pth 2>/dev/null | head -n1 || true)"
      if [[ -n "${last}" ]]; then
        printf '%s\n' "${last}" > "${wd}/last_checkpoint"
        resume_cli=(--resume)
        echo "[RESUME auto] ${last}"
      fi
    fi
  elif [[ -n "${resume_mode}" ]]; then
    # 显式路径：写入 last_checkpoint
    printf '%s\n' "${resume_mode}" > "${wd}/last_checkpoint"
    resume_cli=(--resume)
    echo "[RESUME path] ${resume_mode}"
  fi

  # 已完成则跳过（你全局 280e；若担心个别 300e，可自行改这里的条件）
  if [[ -f "${wd}/epoch_280.pth" ]]; then
    echo "[SKIP done] ${wd} has epoch_280.pth"
  else
    # 共同的 cfg 片段
    local common_opts=(
      randomness.seed="${seed}" randomness.deterministic=True
      ${BS_OPTS} ${SCOPE_OPTS} ${size_opts}
      train_dataloader.dataset.data_root="${root}"
      val_dataloader.dataset.data_root="${root}"
      test_dataloader.dataset.data_root="${root}"
      train_dataloader.dataset.ann_file="${tr_ann}"
      val_dataloader.dataset.ann_file="${va_ann}"
      test_dataloader.dataset.ann_file="${te_ann}"
    )

    if [[ "${is_yolox}" == "1" ]]; then
      python "${TRAIN_PY}" "${cfg}" "${resume_cli[@]}" \
        --work-dir "${wd}" \
        --cfg-options \
          "${common_opts[@]}" \
          bbox_head.loss_pose.metainfo="${meta}" \
          model.train_cfg.assigner.oks_calculator.metainfo="${meta}" \
          val_evaluator.0.type=mmpose.CocoMetric val_evaluator.0.ann_file="${va_ann}"
    else
      python "${TRAIN_PY}" "${cfg}" "${resume_cli[@]}" \
        --work-dir "${wd}" \
        --cfg-options \
          "${common_opts[@]}" \
          train_dataloader.dataset.metainfo.from_file="${meta}" \
          val_dataloader.dataset.metainfo.from_file="${meta}" \
          test_dataloader.dataset.metainfo.from_file="${meta}" \
          val_evaluator.0.type=CocoMetric val_evaluator.0.ann_file="${va_ann}"
    fi
  fi

  # 选 ckpt 做 test
  local ckpt="${wd}/best.pth"; [[ -f "${ckpt}" ]] || ckpt="$(ls -1t "${wd}"/*.pth 2>/dev/null | head -n1 || true)"
  if [[ -n "${ckpt}" ]]; then
    if [[ "${is_yolox}" == "1" ]]; then
      python "${TEST_PY}" "${cfg}" "${ckpt}" \
        --cfg-options \
          test_dataloader.dataset.data_root="${root}" \
          test_dataloader.dataset.ann_file="${te_ann}" \
          ${SCOPE_OPTS} ${size_opts} \
          bbox_head.loss_pose.metainfo="${meta}" \
          test_evaluator.0.type=mmpose.CocoMetric test_evaluator.0.ann_file="${te_ann}" \
          test_evaluator.0.outfile_prefix="${out}"
    else
      python "${TEST_PY}" "${cfg}" "${ckpt}" \
        --cfg-options \
          test_dataloader.dataset.data_root="${root}" \
          test_dataloader.dataset.ann_file="${te_ann}" \
          ${SCOPE_OPTS} ${size_opts} \
          test_dataloader.dataset.metainfo.from_file="${meta}" \
          test_evaluator.0.type=CocoMetric test_evaluator.0.ann_file="${te_ann}" \
          test_evaluator.0.outfile_prefix="${out}"
    fi
  else
    echo "[WARN] no checkpoint found to test: ${wd}"
  fi
}

# ---- Pick annotations ----
H_TR=$(pick_ann "${HORSE_ROOT}" "annotations/train_annotations.coco.json" "annotations/train.json")
H_VA=$(pick_ann "${HORSE_ROOT}" "annotations/val_annotations.coco.json"   "annotations/val.json")
H_TE=$(pick_ann "${HORSE_ROOT}" "annotations/test_annotations.coco.json"  "annotations/test.json")
S_TR=$(pick_ann "${SHEEP_ROOT}" "annotations/train_annotations.coco.json" "annotations/train.json")
S_VA=$(pick_ann "${SHEEP_ROOT}" "annotations/val_annotations.coco.json"   "annotations/val.json")
S_TE=$(pick_ann "${SHEEP_ROOT}" "annotations/test_annotations.coco.json"  "annotations/test.json")
[[ -n "$H_TR" && -n "$H_VA" && -n "$H_TE" ]] || { echo "[ERR] horse json not found"; exit 1; }
[[ -n "$S_TR" && -n "$S_VA" && -n "$S_TE" ]] || { echo "[ERR] sheep json not found"; exit 1; }

# ---- Run: Horse & Sheep（9 × 三种子）----
for s in "${SEEDS[@]}"; do
  # 可选 seed 过滤：支持 3407 或 s3407
  if [[ -n "${SEED_FILTER}" ]]; then
    sf="${SEED_FILTER#s}"
    [[ "${s}" == "${sf}" ]] || continue
  fi

  # Horse
  run_one "${SIMCC_W48_H}"  "simcc_w48_std"       "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${TD_SIZE}"     "${META_H}"  "0" "${s}"
  run_one "${SIMCC_HRTR_H}" "simcc_hrtran_std"    "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${TD_SIZE}"     "${META_H}"  "0" "${s}"
  run_one "${SIMCC_LITE_H}" "simcc_litehrnet_std" "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${TD_SIZE}"     "${META_H}"  "0" "${s}"
  run_one "${SIMCC_R50_H}"  "simcc_res50_std"     "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${TD_SIZE}"     "${META_H}"  "0" "${s}"
  run_one "${SIMCC_SWIN_H}" "simcc_swin_std"      "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${TD_SIZE}"     "${META_H}"  "0" "${s}"
  run_one "${SAR_W48_H}"    "sar_w48_std"         "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${TD_SIZE_SAR}" "${META_H}"  "0" "${s}"
  run_one "${SAR_R50_H}"    "sar_res50_std"       "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${TD_SIZE_SAR}" "${META_H}"  "0" "${s}"
  run_one "${YOLOX_S_H}"    "yolox_s_std"         "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${BU_SIZE}"     "${YOLOX_META_H}" "1" "${s}"
  run_one "${YOLOX_M_H}"    "yolox_m_std"         "horse" "${HORSE_ROOT}" "$H_TR" "$H_VA" "$H_TE" "${BU_SIZE}"     "${YOLOX_META_H}" "1" "${s}"

  # Sheep
  run_one "${SIMCC_W48_S}"  "simcc_w48_std"       "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${TD_SIZE}"     "${META_S}"  "0" "${s}"
  run_one "${SIMCC_HRTR_S}" "simcc_hrtran_std"    "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${TD_SIZE}"     "${META_S}"  "0" "${s}"
  run_one "${SIMCC_LITE_S}" "simcc_litehrnet_std" "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${TD_SIZE}"     "${META_S}"  "0" "${s}"
  run_one "${SIMCC_R50_S}"  "simcc_res50_std"     "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${TD_SIZE}"     "${META_S}"  "0" "${s}"
  run_one "${SIMCC_SWIN_S}" "simcc_swin_std"      "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${TD_SIZE}"     "${META_S}"  "0" "${s}"
  run_one "${SAR_W48_S}"    "sar_w48_std"         "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${TD_SIZE_SAR}" "${META_S}"  "0" "${s}"
  run_one "${SAR_R50_S}"    "sar_res50_std"       "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${TD_SIZE_SAR}" "${META_S}"  "0" "${s}"
  run_one "${YOLOX_S_S}"    "yolox_s_std"         "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${BU_SIZE}"     "${YOLOX_META_S}" "1" "${s}"
  run_one "${YOLOX_M_S}"    "yolox_m_std"         "sheep" "${SHEEP_ROOT}" "$S_TR" "$S_VA" "$S_TE" "${BU_SIZE}"     "${YOLOX_META_S}" "1" "${s}"
done

echo "[TRAIN HS DONE] logs: ${WORK_ROOT} | json: ${RAW_JSON_ROOT}"
