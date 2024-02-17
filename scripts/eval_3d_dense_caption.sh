python main.py \
    --use_color \
    --use_normal \
    --dataset scene_scanrefer \
    --vocabulary scanrefer \
    --use_beam_search \
    --detector detector_Vote2Cap_DETR \
    # or use v2 with --detector detector_Vote2Cap_DETRv2
    --captioner captioner_dcc \
    # or use v2 with --captioner captioner_dccv2
    --batchsize_per_gpu 8 \
    --test_ckpt [...]/checkpoint_best.pth \
    --test_caption