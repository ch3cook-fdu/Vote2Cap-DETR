import os, sys, time, math, json, importlib
import torch
import datetime
from collections import defaultdict, OrderedDict

import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor

from utils.box_util import box3d_iou_batch_tensor
from utils.ap_calculator import APCalculator
from utils.io import save_checkpoint
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions


class Logger:
    def __init__(self, args):
        self.logger = open(os.path.join(args.checkpoint_dir, 'logger.out'), 'a')
    def __call__(self, info_str):
        self.logger.write(info_str + "\n")
        self.logger.flush()
        print(info_str)


def score_captions(corpus: dict, candidates: dict):
    
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    
    score_per_caption = {
        "bleu-1": [float(s) for s in bleu[1][0]],
        "bleu-2": [float(s) for s in bleu[1][1]],
        "bleu-3": [float(s) for s in bleu[1][2]],
        "bleu-4": [float(s) for s in bleu[1][3]],
        "cider": [float(s) for s in cider[1]],
        "rouge": [float(s) for s in rouge[1]],
        "meteor": [float(s) for s in meteor[1]],
    }
    
    message = '\n'.join([
        "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
        ),
        "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
        ),
        "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
        ),
        "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
        ),
        "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            cider[0], max(cider[1]), min(cider[1])
        ),
        "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            rouge[0], max(rouge[1]), min(rouge[1])
        ),
        "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            meteor[0], max(meteor[1]), min(meteor[1])
        )
    ])
    
    eval_metric = {
        "BLEU-4": bleu[0][3],
        "CiDEr": cider[0],
        "Rouge": rouge[0],
        "METEOR": meteor[0],
    }
    return score_per_caption, message, eval_metric


def prepare_corpus(raw_data, max_len: int=30) -> dict:
    # helper function to prepare ground truth captions
    corpus = defaultdict(list)
    object_id_to_name = defaultdict(lambda:'unknown')
    
    for data in raw_data:
        
        (         scene_id,         object_id,         object_name
        ) = data["scene_id"], data["object_id"], data["object_name"]
        
        # parse language tokens
        token = data["token"][:max_len]
        description = " ".join(["sos"] + token + ["eos"])
        key = f"{scene_id}|{object_id}|{object_name}"
        object_id_to_name[f"{scene_id}|{object_id}"] = object_name
        
        corpus[key].append(description)
        
    return corpus, object_id_to_name


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr

def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        if args.pretrained_params_lr is not None and \
            param_group["lr"] == args.pretrained_params_lr:
            continue
        param_group["lr"] = curr_lr
    return curr_lr


def do_train(
    args,
    model,
    optimizer,
    dataset_config,
    dataloaders,
    best_val_metrics=dict()
):
    if args.eval_metric == 'detection':
        args.criterion = f'mAP@{args.test_min_iou}'
        do_eval = evaluate_detection
    elif args.eval_metric == 'caption':
        args.criterion = f'CiDEr@{args.test_min_iou}'
        do_eval = evaluate_caption
    else:
        raise NotImplementedError
        
    logout = Logger(args)
    logout(f"call with args: {args}")
    logout(f"{model}")
    
    curr_iter = args.start_epoch * len(dataloaders['train'])
    max_iters = args.max_epoch * len(dataloaders['train'])
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    for curr_epoch in range(args.start_epoch, args.max_epoch):
        
        for batch_idx, batch_data_label in enumerate(dataloaders['train']):
            
            curr_time = time.time()
            
            curr_iter = curr_epoch * len(dataloaders['train']) + batch_idx
            curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(net_device)
    
            # Forward pass
            optimizer.zero_grad()
    
            outputs = model(batch_data_label, is_eval=False)
            loss = outputs['loss']
    
            if not math.isfinite(loss.item()):
                logout("Loss in not finite. Training will be stopped.")
                sys.exit(1)
    
            loss.backward()
            if args.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
    
            time_delta.update(time.time() - curr_time)
            loss_avg.update(loss.item())
    
            # logging
            if curr_iter % args.log_every == 0:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logout(
                    f"Epoch [{curr_epoch}/{args.max_epoch}]; "
                    f"Iter [{curr_iter}/{max_iters}]; "
                    f"Loss {loss_avg.avg:0.2f}; "
                    f"LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; "
                    f"ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                )
            
            # eval
            if (curr_iter + 1) % args.eval_every_iteration == 0:
                eval_metrics = do_eval(
                    args,
                    curr_epoch,
                    model,
                    dataset_config,
                    dataloaders['test'],
                    logout,
                    curr_train_iter=curr_iter
                )
                model.train()
                if not best_val_metrics or (
                    best_val_metrics[args.criterion] < eval_metrics[args.criterion]
                ):
                    best_val_metrics = eval_metrics
                    filename = "checkpoint_best.pth"
                    save_checkpoint(
                        args.checkpoint_dir,
                        model,
                        optimizer,
                        curr_epoch,
                        args,
                        best_val_metrics,
                        filename="checkpoint_best.pth",
                    )
                    logout(
                        f"Epoch [{curr_epoch}/{args.max_epoch}] "
                        f"saved current best val checkpoint at {filename}; "
                        f"{args.criterion} {eval_metrics[args.criterion]}"
                    )
            # end of an iteration
            
        # end of an epoch
        save_checkpoint(
            args.checkpoint_dir,
            model,
            optimizer,
            curr_epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )
    # end of training
    do_eval(
        args,
        curr_epoch,
        model,
        dataset_config,
        dataloaders['test'],
        logout,
        curr_train_iter=-1
    )
    return 


@torch.no_grad()
def evaluate_detection(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):

    # ap calculator is exact for evaluation. 
    #   This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )
    
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    
    model.eval()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for curr_iter, batch_data_label in enumerate(dataset_loader):
        
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
            
        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
        }
        outputs = model(model_input, is_eval=True)

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        ap_calculator.step_meter(
            {'outputs': outputs}, 
            batch_data_label
        )
        time_delta.update(time.time() - curr_time)
        
        if curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logout(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; "
                f"Evaluating on iter: {curr_train_iter}; "
                f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )
    
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)

    logout("==" * 10)
    logout(f"Evaluate Epoch [{curr_epoch}/{args.max_epoch}]")
    logout(f"{metric_str}")
    logout("==" * 10)
    
    eval_metrics = {
        metric + f'@{args.test_min_iou}': score \
            for metric, score in metrics[args.test_min_iou].items()
    }
    return eval_metrics


@torch.no_grad()
def evaluate_caption(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):
    dataset = importlib.import_module(f'datasets.{args.dataset}')
    SCANREFER = dataset.SCANREFER
    
    # prepare ground truth caption labels
    print("preparing corpus...")
    corpus, object_id_to_name = prepare_corpus(
        SCANREFER['language']['val']
    )
    
    ### initialize and prepare for evaluation
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    
    model.eval()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    candidates = {'caption': OrderedDict({}), 'iou': defaultdict(float)}
    
    for curr_iter, batch_data_label in enumerate(dataset_loader):
        
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        
        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
        }
        outputs = model(model_input, is_eval=True)
        
        ### match objects
        batch_size, MAX_NUM_OBJ, _, _ = batch_data_label["gt_box_corners"].shape
        _, nqueries, _, _ = outputs["box_corners"].shape
        
        match_box_ious = box3d_iou_batch_tensor(    # batch, nqueries, MAX_NUM_OBJ
            (outputs["box_corners"]
             .unsqueeze(2)
             .repeat(1, 1, MAX_NUM_OBJ, 1, 1)
             .view(-1, 8, 3)
             ),
            (batch_data_label["gt_box_corners"]
             .unsqueeze(1)
             .repeat(1, nqueries, 1, 1, 1)
             .view(-1, 8, 3)
             )
        ).view(batch_size, nqueries, MAX_NUM_OBJ)
        match_box_ious, match_box_idxs = match_box_ious.max(-1) # batch, nqueries
        match_box_idxs = torch.gather(
            batch_data_label['gt_box_object_ids'], 1, 
            match_box_idxs
        ) # batch, nqueries
        
        # ---- Checkout bounding box ious and semantic logits
        good_bbox_masks = match_box_ious > args.test_min_iou     # batch, nqueries
        good_bbox_masks &= outputs["sem_cls_logits"].argmax(-1) != (
            outputs["sem_cls_logits"].shape[-1] - 1
        )
        
        # ---- add nms to get accurate predictions
        nms_bbox_masks = parse_predictions( # batch x nqueries
            outputs["box_corners"], 
            outputs['sem_cls_prob'], 
            outputs['objectness_prob'], 
            batch_data_label['point_clouds']
        )
        nms_bbox_masks = torch.from_numpy(nms_bbox_masks).long() == 1
        good_bbox_masks &= nms_bbox_masks.to(good_bbox_masks.device)
        
        good_bbox_masks = good_bbox_masks.cpu().tolist()
        
        captions = outputs["lang_cap"]  # batch, nqueries, [sentence]
        
        match_box_idxs = match_box_idxs.cpu().tolist()
        match_box_ious = match_box_ious.cpu().tolist()
        ### calculate measurable indicators on captions
        for idx, scene_id in enumerate(batch_data_label["scan_idx"].cpu().tolist()):
            scene_name = SCANREFER['scene_list']['val'][scene_id]
            for prop_id in range(nqueries):

                if good_bbox_masks[idx][prop_id] is False:
                    continue
                
                match_obj_id = match_box_idxs[idx][prop_id]
                match_obj_iou = match_box_ious[idx][prop_id]
                
                object_name = object_id_to_name[f"{scene_name}|{match_obj_id}"]
                key = f"{scene_name}|{match_obj_id}|{object_name}"
                
                if match_obj_iou > candidates['iou'][key]:
                    candidates['iou'][key] = match_obj_iou
                    candidates['caption'][key] = [
                        captions[idx][prop_id]
                    ]
                    # DEBUG: checkout how many matched bounding boxes
                    # candidates[key] = ["this is a valid match!"]
                    
        # Memory intensive as it gathers point cloud GT tensor across all ranks
        time_delta.update(time.time() - curr_time)
        
        if curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logout(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; "
                f"Evaluating on iter: {curr_train_iter}; "
                f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )
    
    # end of forward pass traversion
    
    ### message out
    missing_proposals = len(corpus.keys() - candidates['caption'].keys())
    total_captions = len(corpus.keys())
    logout(
        f"\n----------------------Evaluation-----------------------\n"
        f"INFO: iou@{args.test_min_iou} matched proposals: "
        f"[{total_captions - missing_proposals} / {total_captions}], "
    )
    
    ### make up placeholders for undetected bounding boxes
    for missing_key in (corpus.keys() - candidates['caption'].keys()):
        candidates['caption'][missing_key] = ["sos eos"]
    
    # find annotated objects in scanrefer
    candidates = OrderedDict([
        (key, value) for key, value in sorted(candidates['caption'].items()) \
            if not key.endswith("unknown")
    ])
    score_per_caption, message, eval_metric = score_captions(
        OrderedDict([(key, corpus[key]) for key in candidates]), candidates
    )
    
    logout(message)
    
    with open(os.path.join(args.checkpoint_dir, "corpus_val.json"), "w") as f: 
        json.dump(corpus, f, indent=4)
    
    with open(os.path.join(args.checkpoint_dir, "pred_val.json"), "w") as f:
        json.dump(candidates, f, indent=4)
    
    with open(os.path.join(args.checkpoint_dir, "pred_gt_val.json"), "w") as f:
        pred_gt_val = {}
        for scene_object_id, scene_object_id_key in enumerate(candidates):
            pred_gt_val[scene_object_id_key] = {
                'pred': candidates[scene_object_id_key],
                'gt': corpus[scene_object_id_key],
                'score': {
                    'bleu-1': score_per_caption['bleu-1'][scene_object_id],
                    'bleu-2': score_per_caption['bleu-2'][scene_object_id],
                    'bleu-3': score_per_caption['bleu-3'][scene_object_id],
                    'bleu-4': score_per_caption['bleu-4'][scene_object_id],
                    'CiDEr': score_per_caption['cider'][scene_object_id],
                    'rouge': score_per_caption['rouge'][scene_object_id],
                    'meteor': score_per_caption['meteor'][scene_object_id]
                }
            }
        json.dump(pred_gt_val, f, indent=4)
    
    eval_metrics = {
        metric + f'@{args.test_min_iou}': score \
            for metric, score in eval_metric.items()
    }
    return eval_metrics