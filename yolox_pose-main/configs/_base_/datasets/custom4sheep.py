dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Yiwei Wang',
        title='sheep pose estimation Dataset',
        year='2025',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(name='mouth',
             id=0,
             color=[51, 153, 255],
             type='',
             swap=''),
        1:
        dict(
            name='eye',
            id=1,
            color=[51, 153, 255],
            type='',
            swap=''),
        2:
        dict(
            name='ear',
            id=2,
            color=[51, 153, 255],
            type='',
            swap=''),
        3:
        dict(
            name='neck',
            id=3,
            color=[51, 153, 255],
            type='',
            swap=''),
        4:
        dict(
            name='shoulder',
            id=4,
            color=[0, 255, 0],
            type='',
            swap=''),
        5:
        dict(
            name='chest',
            id=5,
            color=[0, 255, 0],
            type='',
            swap=''),
        6:
        dict(
            name='hip',
            id=6,
            color=[0, 255, 0],
            type='',
            swap=''),
        7:
        dict(
            name='tail',
            id=7,
            color=[0, 255, 0],
            type='',
            swap=''),
        8:
        dict(
            name='elbow',
            id=8,
            color=[255, 128, 0],
            type='',
            swap=''),
        9:
        dict(
            name='l_fore_wrist',
            id=9,
            color=[255, 128, 255],
            type='',
            swap='r_fore_wrist'),
        10:
        dict(
            name='l_fore_foot',
            id=10,
            color=[255, 128, 255],
            type='',
            swap='r_fore_foot'),
        11:
        dict(
            name='r_fore_wrist',
            id=11,
            color=[255, 128, 0],
            type='',
            swap='l_fore_wrist'),
        12:
        dict(
            name='r_fore_foot',
            id=12,
            color=[255, 128, 0],
            type='',
            swap='l_fore_foot'),
        13:
        dict(
            name='hind_knee',
            id=13,
            color=[255, 128, 0],
            type='',
            swap=''),
        14:
        dict(
            name='l_hind_hock',
            id=14,
            color=[255, 128, 255],
            type='',
            swap='r_hind_hock'),
        15:
        dict(
            name='l_hind_foot',
            id=15,
            color=[255, 128, 255],
            type='',
            swap='r_hind_foot'),
        16:
        dict(
            name='r_hind_hock',
            id=16,
            color=[255, 128, 0],
            type='',
            swap='l_hind_hock'),
        17:
        dict(
            name='r_hind_foot',
            id=17,
            color=[255, 128, 0],
            type='',
            swap='l_hind_foot')
    },
    skeleton_info={
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1.
    ],
    sigmas=[
           0.015, 0.014, 0.020, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.050,
           0.051, 0.050, 0.051, 0.045, 0.050, 0.051, 0.050, 0.051
    ])