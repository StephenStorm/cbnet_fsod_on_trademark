while [ ! -f /opt/tiger/haggs/CBNetV2/work_dirs/cascade_mask_rcnn_cbv2_swin_base_cp_mixup0.5_affine_freeze_stage1_resize_1920_15/epoch_12.pth ]; do
    currTime=$(date +"%Y-%m-%d %T")
    echo $currTime "=====>" Wait 100s
    sleep 100
done
echo Wait ok!
currTime=$(date +"%Y-%m-%d %T")
echo $currTime "=====>" Wait 1500s
sleep 1000

./tools/dist_test.sh \
 /opt/tiger/haggs/CBNetV2/work_dirs/cascade_mask_rcnn_cbv2_swin_base_cp_mixup0.5_affine_freeze_stage1_resize_1920_15/cascade_mask_rcnn_swinb_cp_resize_bigger.py \
 /opt/tiger/haggs/CBNetV2/work_dirs/cascade_mask_rcnn_cbv2_swin_base_cp_mixup0.5_affine_freeze_stage1_resize_1920_15/epoch_12.pth \
 8 \
 --format-only \
 --options "jsonfile_prefix=/opt/tiger/haggs/CBNetV2/work_dirs/cascade_mask_rcnn_cbv2_swin_base_cp_mixup0.5_affine_freeze_stage1_resize_1920_15/resize_bigger_0509"