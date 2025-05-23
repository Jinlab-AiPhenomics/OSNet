dataset_type = 'ISAIDDataset'

# need to preprocess isaid dataset using isaid_devkit
data_root = '/root/autodl-tmp/datasets/osnet_merge/'

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
         keys=['img', 'gt_obboxes',  'gt_masks', 'gt_labels',"gt_bboxes"])
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

# does evaluation while training
# uncomments it  when you need evaluate every epoch
classes = ('plot',)
data = dict(
     samples_per_gpu=16,
     workers_per_gpu=2,
     persistent_workers=False,
     train=dict(
         type=dataset_type,
         classes=classes,
         ann_file=data_root + 'annotations/instances_train2017.json',
         img_prefix=data_root + 'train2017/',
         pipeline=train_pipeline),
     val=dict(
         type=dataset_type,
         classes=classes,
         ann_file=data_root + 'annotations/instances_val2017.json',
         img_prefix=data_root + 'val2017/',
         pipeline=test_pipeline),
     test=dict(
         type=dataset_type,
         classes=classes,
         ann_file=data_root + 'annotations/instances_test2017.json',
         img_prefix=data_root + 'test2017/',
         pipeline=test_pipeline))

evaluation = dict(interval=3, metric=['bbox', 'segm', ], save_best='segm_mAP_50')