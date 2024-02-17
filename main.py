import os, argparse, importlib
import numpy as np
import torch

from collections import OrderedDict

from engine import do_train, evaluate_caption, evaluate_detection
from models.model_general import CaptionNet

from utils.io import resume_if_possible
from utils.misc import my_worker_init_fn

def make_args_parser():
    parser = argparse.ArgumentParser(
        "Vote2Cap-DETR: A set-to-set perspective towards 3D Dense Captioning", 
        add_help=False
    )

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, 
        help="Max L2 norm of the gradient"
    )
    # DISABLE warmup learning rate during dense caption training
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    # only ACTIVATE during dense caption training
    parser.add_argument("--pretrained_params_lr", default=None, type=float)
    
    ##### Model #####
    # input based parameters
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")
    parser.add_argument("--use_multiview", default=False, action="store_true")
    
    parser.add_argument(
        "--detector", default="detector_Vote2Cap_DETR", 
        help="folder of the detector"
    )
    
    ## ACTIVATE during dense captioning training
    parser.add_argument("--use_pretrained", default=False, action="store_true")
    parser.add_argument(
        "--captioner", default=None, type=str, help="folder of the captioner"
    )
    parser.add_argument(
        "--freeze_detector", default=False, action='store_true', 
        help="freeze all parameters other than the caption head"
    )
    # caption related hyper parameters
    parser.add_argument(
        "--use_beam_search", default=False, action='store_true',
        help='whether use beam search during caption generation.'
    )
    parser.add_argument(
        "--max_des_len", default=32, type=int, 
        help="maximum length of object descriptions."
    )
    
    
    ##### Dataset #####
    parser.add_argument(
        "--dataset", default='scannet',
        help="dataset file which stores `dataset` and `dataset_config` class",
    )
    parser.add_argument(
        '--vocabulary', default="scanrefer", type=str,
        help="should be one of `gpt2` or `scanrefer`"
    )
    # only activated during k sentence training
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
        "--eval_metric", default='detection', choices=['caption', 'detection'],
        help='evaluate model through `caption` or `detection`.'
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default='0', type=str)

    ##### Testing #####
    parser.add_argument("--test_detection", default=False, action="store_true")
    parser.add_argument("--test_caption", default=False, action="store_true")
    parser.add_argument(
        "--test_min_iou", default=0.50, type=float,
        help='minimum iou for evaluating dense caption performance'
    )
    parser.add_argument("--test_ckpt", default="", type=str)

    ##### I/O #####
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
    
    if args.checkpoint_dir is not None:
        pass
    elif args.test_ckpt is not None:
        args.checkpoint_dir = os.path.dirname(args.test_ckpt)
        print(f'testing directory: {args.checkpoint_dir}')
    else:
        raise AssertionError(
            'Either checkpoint_dir or test_ckpt should be presented!'
        )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ### build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    model = CaptionNet(args, dataset_config, datasets['train']).cuda()
    
    
    # testing phase
    if args.test_detection or args.test_caption:
        
        if args.test_detection:
            if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
                print('Invalid test_ckpt found, test the scratch model.')
            else:
                checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
                model.load_state_dict(checkpoint["model"])
            
            evaluate_detection(
                args,
                -1,
                model,
                dataset_config,
                dataloaders['test']
            )
        
            
        elif args.test_caption:
            if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
                print(
                    f"Please specify a test checkpoint using --test_ckpt. "
                    f"Found invalid value {args.test_ckpt}"
                    )
                assert args.checkpoint_dir is not None, 'checkpoint_dir is required!'
            else:
                checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
                model.load_state_dict(checkpoint["model"])
            
            if args.checkpoint_dir is None:
                args.checkpoint_dir = os.path.dirname(args.test_ckpt)
                os.makedirs(args.checkpoint_dir, exist_ok=True)
            
            print(f'testing directory: {args.checkpoint_dir}')
            
            evaluate_caption(
                args,
                -1,
                model,
                dataset_config,
                dataloaders['test']
            )
            
        else:
            exit('switch to the wrong mode!')
        
    # training phase
    else:
        
        assert (
            args.checkpoint_dir is not None
        ), "Please specify a checkpoint dir using --checkpoint_dir"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        ### whether or not use pretrained weights
        pretrained_named_parameters = OrderedDict({})
        
        if args.use_pretrained is True:
            
            use_color = "_COLOR" if args.use_color else ""
            use_normal = "_NORMAL" if args.use_normal else ""
            use_multiview = "_MULTIVIEW" if args.use_multiview else ""
            
            prefix = '_'.join(args.detector.split('_')[1:])   # 3detr or votenet
            checkpoint_dir = os.path.join(
                ".", "pretrained", prefix + "_XYZ" + use_color + use_multiview + use_normal
            )
            try:
                checkpoint = torch.load(
                    os.path.join(checkpoint_dir, "checkpoint_best.pth"), 
                    map_location="cpu"
                )
            except FileNotFoundError:
                print(
                    f"model {prefix + '_XYZ' + use_color + use_multiview + use_normal}"
                    f" is not pretrained!"
                )
                exit(-1)
            
            # overwrite parameters
            state_dict = model.state_dict()
            pretrained_named_parameters = {
                name: param for name, param in checkpoint['model'].items() \
                                if name in state_dict
            }
            state_dict.update(pretrained_named_parameters)
            model.load_state_dict(state_dict)
        
        ### optimizer, pending to use different lr for pretrained params or not
        if args.pretrained_params_lr is not None and args.use_pretrained is True:
            
            pretrained_params = [
                param for name, param in model.named_parameters() \
                    if (
                        name in pretrained_named_parameters \
                        or name in model.pretrained_parameters()
                    ) and param.requires_grad is True
            ]
            scratch_params = [
                param for name, param in model.named_parameters() \
                    if name not in pretrained_named_parameters \
                    and name not in model.pretrained_parameters() \
                    and param.requires_grad is True
            ]
            param_groups = [
                {"params": pretrained_params, "lr": args.pretrained_params_lr},
                {"params": scratch_params, "lr": args.base_lr},
            ]
            optimizer = torch.optim.AdamW(
                param_groups, weight_decay=args.weight_decay
            )
            
            print('loaded weights:')
            print(
                '\n'.join(
                    list(pretrained_named_parameters.keys()) \
                        + model.pretrained_parameters()
                )
            )
            
        else:
            optimizer = torch.optim.AdamW(
                filter(lambda params: params.requires_grad, model.parameters()), 
                lr=args.base_lr, 
                weight_decay=args.weight_decay
            )
        
        print('certain parameters are not trained:')
        for name, param in model.named_parameters():
            if param.requires_grad is False:
                print(name)
        
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

    main(args)
