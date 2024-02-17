import os, argparse, importlib, time, json
import numpy as np
import torch

from tqdm import tqdm
from collections import OrderedDict
from models.model_general import CaptionNet

from utils.io import resume_if_possible
from utils.misc import my_worker_init_fn
from utils.box_util import box3d_iou_batch_tensor
from utils.proposal_parser import parse_predictions


def make_args_parser():
    parser = argparse.ArgumentParser("3D Dense Captioning Using Transformers", add_help=False)

    ##### Optimizer #####
    
    ##### Model #####
    parser.add_argument(
        '--vocabulary', default="scanrefer", type=str,
        help="should be one of `gpt2` or `scanrefer`"
    )
    parser.add_argument(
        "--detector", default="detector_Vote2Cap_DETR", type=str, 
        help="folder of the detector"
    )
    parser.add_argument(
        "--captioner", default=None, type=str, 
        help="folder of the captioner"
    )
    parser.add_argument(
        "--use_beam_search", default=False, action='store_true',
        help='whether use beam search during evaluation.'
    )
    
    parser.add_argument(
        "--max_des_len", default=32, type=int, 
        help="maximum length of object descriptions."
    )
    
    parser.add_argument(
        "--freeze_detector", default=True, action='store_true'
    )
    
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")
    parser.add_argument("--use_multiview", default=False, action="store_true")
    
    ##### Dataset #####
    parser.add_argument(
        "--dataset", default='test_scanrefer',
        help="dataset file which stores `dataset` and `dataset_config` class",
    )
    
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default='0', type=str)

    ##### Testing #####
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)

    ##### I/O #####
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
            augment=False
        ),
        "test": dataset_module.Dataset(
            args,
            dataset_config, 
            split_set="test", 
            use_color=args.use_color,
            use_normal=args.use_normal,
            use_multiview=args.use_multiview,
            use_height=args.use_height,
            augment=False
        ),
    }
    
    dataloaders = {}
    
    split = 'test'
    sampler = torch.utils.data.SequentialSampler(datasets[split])

    dataloaders[split] = torch.utils.data.DataLoader(
        datasets[split],
        sampler=sampler,
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,
    )
        
    return dataset_config, datasets, dataloaders    
 

def flip_bounding_boxes_to_scene(bbox_corner):
    bbox_corner[..., [1, 2]] = bbox_corner[..., [2, 1]]
    bbox_corner[..., [2]] = -bbox_corner[..., [2]]
    return bbox_corner


@torch.no_grad()
def run_dense_caption(args, model, dataset_loader):
    
    model.eval()
    dataset = importlib.import_module(f'datasets.{args.dataset}')
    SCANREFER = dataset.SCANREFER
    net_device = next(model.parameters()).device
    prediction_test_set = {}
    
    for curr_iter, batch_data_label in enumerate(tqdm(dataset_loader)):
        
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        
        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
        }
        outputs = model(model_input, is_eval=True)
        
        # ---- add nms to get accurate predictions
        nms_bbox_masks = parse_predictions( # batch x nqueries
            outputs["box_corners"], 
            outputs['sem_cls_prob'], 
            outputs['objectness_prob'], 
            batch_data_label['point_clouds']
        )
        nms_bbox_masks = torch.from_numpy(nms_bbox_masks).long() == 1
        
        ### match objects
        batch_size, nqueries, _, _ = outputs["box_corners"].shape
        
        # ---- Checkout bounding box ious and semantic logits
        good_bbox_masks = outputs["sem_cls_logits"].argmax(-1) != (
            outputs["sem_cls_logits"].shape[-1] - 1
        )
        good_bbox_masks &= nms_bbox_masks.to(good_bbox_masks.device)
        
        captions = outputs["lang_cap"]  # batch, nqueries, [sentence]
        
        sem_prob = outputs["sem_cls_prob"].cpu().tolist()
        objectness_prob = outputs["objectness_prob"].cpu().tolist()
        
        good_bbox_masks = good_bbox_masks.cpu().tolist()
        
        ### calculate measurable indicators on captions
        
        for idx, scene_id in enumerate(batch_data_label["scan_idx"].cpu().tolist()):
            
            scene_name = SCANREFER['scene_list']['test'][scene_id]
            print('evaluating on scene:', scene_name)
            # output_file = os.path.join(args.visualize_dir, scene_name + '.json')
            
            scene_results = []
            for prop_id in range(nqueries):

                if good_bbox_masks[idx][prop_id] is False:
                    continue
                
                scene_results.append({
                    'caption': captions[idx][prop_id],
                    'box': flip_bounding_boxes_to_scene(
                        outputs["box_corners"][idx][prop_id]
                    ).cpu().tolist(),
                    'sem_prob': sem_prob[idx][prop_id],
                    'obj_prob': [
                        1-objectness_prob[idx][prop_id], 
                        objectness_prob[idx][prop_id]
                    ]
                })
            
            prediction_test_set[scene_name] = scene_results
            
    with open(os.path.join(args.visualize_dir, 'test-set-pred.json'), 'w') as file:
        json.dump(prediction_test_set, file, indent=4)
                
    return


def main(args):
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.dirname(args.test_ckpt)
    args.visualize_dir = os.path.join(args.checkpoint_dir, 'prediction-test-set')
    
    os.makedirs(args.visualize_dir, exist_ok=True)
    
    ### build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    model = CaptionNet(args, dataset_config, datasets['train']).cuda()
    
    checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    
    model.eval()
    print(f'testing directory: {args.checkpoint_dir}')
    
    run_dense_caption(
        args,
        model,
        dataloaders['test']
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
