dataset_type = 'ISAIDDataset'

# need to preprocess isaid dataset using isaid_devkit
data_root = '/root/autodl-tmp/OBBDetection-master_2/data/coco/'
#data_root = '/root/autodl-tmp/OBBDetection-master/data/coco_240_last/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True, with_mask=True),
    dict(type='TopNAreaObject', n=1000),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomOBBRotate', rotate_after_flip=True,
         angles=(0, 0), vert_rate=0.5, vert_cls=['Roundabout', 'storage_tank']),     #取消旋转
    dict(type='Pad', size_divisor=32),
    dict(type='DOTASpecialIgnore', ignore_size=4),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='MaskType', mask_type='bitmap'),
    dict(type='OBBDefaultFormatBundle'),
    dict(type='OBBCollect',
         keys=['img', 'gt_bboxes',  'gt_masks', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale= (1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
'''
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=(1024, 1024),
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip',),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomOBBRotate', rotate_after_flip=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img']),
        ])
]
'''
# does evaluation while training
# uncomments it  when you need evaluate every epoch
data = dict(
     samples_per_gpu=2,
     workers_per_gpu=2,
     persistent_workers=False,
     train=dict(
         type=dataset_type,
         #ann_file=data_root + 'train/instancesonly_filtered_train.json',
         ann_file=data_root + 'annotations/instances_train2017.json',
         img_prefix=data_root + 'train2017_rotate/',
         #img_prefix=data_root + 'train2017/',
         pipeline=train_pipeline),
     val=dict(
         type=dataset_type,
         #ann_file=data_root + 'val/instancesonly_filtered_val.json',
         ann_file=data_root + 'annotations/instances_val2017.json',
         img_prefix=data_root + 'val2017_rotate/',
         #img_prefix=data_root + 'val2017/',
         pipeline=test_pipeline),
     test=dict(
         type=dataset_type,
         #ann_file=data_root + 'test/test_info.json',
         ann_file=data_root + 'annotations/instances_test2017.json',
         img_prefix=data_root + 'test2017_rotate/',
         pipeline=test_pipeline))
#evaluation = dict(metric='mAP')
evaluation = dict(metric=['bbox', 'segm', ], save_best='segm_mAP_50')
#evaluation = dict(metric=['bbox', 'segm', ], save_best='bbox_mAP')
# disable evluation, only need train and test
# uncomments it when use trainval as train
'''data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=[
            data_root + 'train/instancesonly_filtered_train.json',
            data_root + 'val/instancesonly_filtered_val.json'],
        img_prefix=[
            data_root + 'train/images/',
            data_root + 'val/images/'],
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/test_info.json',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm', ], save_best='bbox_mAP')'''
'''model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
        checkpoint='torchvision://resnet101'),
        plugins=[
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0010',
                    kv_stride=2),
                stages=(False, False, True, True),
                position='after_conv2')
        ],
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))'''