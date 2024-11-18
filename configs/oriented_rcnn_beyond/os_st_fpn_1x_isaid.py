_base_ = [
    '../_base_/datasets/isaid.py',
    '../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]
fp16 = dict(loss_scale=1.) #dynamic

'''
model = dict(
    type='OrientedRCNN',
    #type='MaskRCNN',
    pretrained='torchvision://resnet50',
    #pretrained='open-mmlab://mmdet/mobilenet_v2',
    backbone=dict(
        type='ResNet',
        #type='ResNet_atten',
        #type='MobileNetV2',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        #init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
        #init_cfg=None),   #CoordAtt
        plugins=[dict(cfg=dict(type='ContextBlock', ratio=1. / 16), stages=(False, True, True, True),position='after_conv3')]),
        #plugins=[dict(cfg=dict(type='GeneralizedAttention',spatial_range=-1,num_heads=8,attention_type='0010',kv_stride=2),stages=(False, False, True, True),position='after_conv2')]),


'''
model = dict(
    type='OrientedRCNN',
    pretrained = '/root/autodl-tmp/OBBDetection-master_2/swin_tiny_patch4_window7_224.pth',#'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        #_delete_=True,
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=True),

neck=dict(
        #type='PAFPN',
        #type='myPAFPN',
        type='FPN',
        #type='FPN_CARAFE',
        in_channels=[96, 192, 384, 768],
        #in_channels=[24, 32, 96, 1280],
        #in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        #type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            #ratios=[1.0,1.2,0.8],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            target_means=[.0, .0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    #reg_decoded_bbox=True,
         #loss_bbox=dict(type='PolyIoULoss', loss_weight=1.0)),

    roi_head=dict(
        type='OBBStandardRoIHead',
        bbox_roi_extractor=dict(
            type='OBBSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2),
            out_channels=256,
            extend_factor=(1.4, 1.2),
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='OBBShared2FCBBoxHead',
            start_bbox_type='obb',
            end_bbox_type='obb',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='OBB2OBBDeltaXYWHTCoder',
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            #loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        reg_decoded_bbox=True,
            loss_bbox=dict(type='PolyIoULoss', loss_weight=1.0)),
            #loss_bbox=dict(type='CIoULoss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='OBBSingleRoIExtractor',
            #type='My_obb',
            roi_layer=dict(type='RoIAlignRotated', out_size=14, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='OBBFCNMaskHead',
            #num_convs=4,
            #type='UNet',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            bbox_type='obb',
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
            #loss_mask_edge=dict(
                #type='DiceLoss', use_sigmoid=False, loss_weight=1.0))),
            #loss_mask_edge=dict(
                #type='MSELoss', reduction='mean', loss_weight=1.0))),


# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            gpu_assign_thr=200,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.8,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            #match_low_quality=True,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        sampler=dict(
            type='OBBRandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False)),
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.8,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='obb_nms', iou_thr=0.5),
        max_per_img=2000,
        mask_thr_binary=0.5)))
