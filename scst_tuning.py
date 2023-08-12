import os, argparse, importlib
import numpy as np
import torch

from collections import OrderedDict

from engine import do_train, evaluate_caption, evaluate_detection
from models.model_general import CaptionNet

from utils.io import resume_if_possible
from utils.misc import my_worker_init_fn

def make_args_parser():
    parser = argparse.ArgumentParser("3D Dense Captioning Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--pretrained_params_lr", default=None, type=float)
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=0, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, 
        help="Max L2 norm of the gradient"
    )
    
    ##### Model #####
    parser.add_argument(
        '--vocabulary', default="scanrefer", type=str,
        help="should be one of `gpt2` or `scanrefer`"
    )
    parser.add_argument(
        "--detector", type=str, help="folder of the detector"
    )
    parser.add_argument(
        "--captioner", default=None, type=str, 
        help="folder of the captioner"
    )
    parser.add_argument(
        "--freeze_detector", default=False, action='store_true', 
        help="train detector or not"
    )
    
    parser.add_argument(
        "--use_beam_search", default=False, action='store_true',
        help='whether use beam search during evaluation.'
    )
    
    parser.add_argument(
        "--max_des_len", default=32, type=int, 
        help="maximum length of object descriptions."
    )
    
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")
    parser.add_argument("--use_multiview", default=False, action="store_true")
    
    ##### Dataset #####
    parser.add_argument(
        "--dataset", default='scene_scanrefer',
        help="dataset file which stores `dataset` and `dataset_config` class",
    )
    parser.add_argument(
        "--k_sentence_per_scene", default=None, type=int,
        help="k sentences per scene for training caption model",
    )
    
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=1080, type=int)
    parser.add_argument("--eval_every_iteration", default=2000, type=int)
    parser.add_argument(
        "--eval_metric", default='caption', choices=['caption', 'detection'],
        help='evaluate model through `caption` or `detection`.'
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default='0', type=str)

    ##### Testing #####
    parser.add_argument(
        "--test_min_iou", default=0.50, type=float,
        help='minimum iou for evaluating detection and caption performance'
    )

    ##### I/O #####
    parser.add_argument("--pretrained_captioner", default=None, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    
    
    args = parser.parse_args()
    args.use_height = not args.no_height
    
    return args


def build_dataset(args):
    dataset_module = importlib.import_module(f'datasets.{args.dataset}')
    dataset_config = dataset_module.DatasetConfig()

    datasets = {
        "train": dataset_module.Dataset(
            args,
            dataset_config, 
            split_set="train", 
            use_color=args.use_color,
            use_normal=args.use_normal,
            use_multiview=args.use_multiview,
            use_height=args.use_height,
            augment=True
        ),
        "test": dataset_module.Dataset(
            args,
            dataset_config, 
            split_set="val", 
            use_color=args.use_color,
            use_normal=args.use_normal,
            use_multiview=args.use_multiview,
            use_height=args.use_height,
            augment=False
        ),
    }
    
    dataloaders = {}
    for split in ["train", "test"]:
        if split == "train":
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
        
    return dataset_config, datasets, dataloaders    
    

def main(args):
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ### build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    model = CaptionNet(args, dataset_config, datasets['train']).cuda()

    assert (
        args.checkpoint_dir is not None
    ), "Please specify a checkpoint dir using --checkpoint_dir"
    assert (
        args.pretrained_captioner is not None and args.captioner is not None
    ), "Pretrain captioner is required when training scst!"
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ### whether or not use pretrained weights
    optimizer = torch.optim.AdamW(
        filter(lambda params: params.requires_grad, model.parameters()), 
        lr=args.base_lr, 
        weight_decay=args.weight_decay
    )
    
    print('certain parameters are not trained:')
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            print(name)
    
    model.load_state_dict(
        torch.load(args.pretrained_captioner, map_location='cpu')['model']
    )
    
    loaded_epoch, best_val_metrics = resume_if_possible(
        args.checkpoint_dir, model, optimizer
    )
    args.start_epoch = loaded_epoch + 1
    
    do_train(
        args,
        model,
        optimizer,
        dataset_config,
        dataloaders,
        best_val_metrics,
    )
    

if __name__ == "__main__":
    args = make_args_parser()
    
    print(f"Called with args: {args}")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args.use_scst = True
    main(args)
