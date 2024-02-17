python main.py \
    --use_color \
    --use_normal \
    --use_pretrained \
    --warm_lr_epochs 0 \
    --pretrained_params_lr 1e-6 \
    --use_beam_search \
    --base_lr 1e-4 \
    --dataset scene_scanrefer \
    --eval_metric caption \
    --vocabulary scanrefer \
    --detector detector_Vote2Cap_DETR \
    --captioner captioner_dcc \
    --checkpoint_dir exp_scanrefer/Vote2Cap_DETR_RGB_NORMAL \
    --max_epoch 720