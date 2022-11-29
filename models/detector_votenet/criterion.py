# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from typing import Dict

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False, return_distance=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
    if return_distance:
        return dist1, idx1, dist2, idx2, pc_dist
    else:
        return dist1, idx1, dist2, idx2


class SetCriterion(nn.Module):
    def __init__(self, cfgs):
        super(SetCriterion, self).__init__()
        mean_size_arr = torch.from_numpy(cfgs.mean_size_arr).float()
        self.register_buffer('mean_size_arr', mean_size_arr)
        
        self.num_size_cluster = cfgs.num_size_cluster
        
        box_loss_weight = 1
        self.loss_dict = {
            'loss_vote': (self.loss_vote, 1),
            'loss_objectness': (self.loss_objectness, 0.5),
            'loss_sem_cls': (self.loss_sem_cls, 0.1),
            'loss_center': (self.loss_center, box_loss_weight * 1),
            'loss_size': (self.loss_size, box_loss_weight * 1),
        }
        
        
    def compute_label_assignment(self, outputs: Dict, targets: Dict) -> Dict:
        # Associate proposal and GT objects by point-to-point distances
        aggregated_vote_xyz = outputs['aggregated_vote_xyz']
        gt_center = targets['gt_box_centers']
        
        batch, max_num_objs, _ = gt_center.shape
        batch, nproposals, _ = aggregated_vote_xyz.shape
        device = aggregated_vote_xyz.device
        
        dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)
    
        # Generate objectness label and mask
        # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
        # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
        euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
        objectness_label = torch.zeros((batch, nproposals), device=device).long()
        objectness_mask = torch.zeros((batch, nproposals), device=device).float()
        
        objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1
        
        return {
            'objectness_label': objectness_label,   # batch x nproposals
            'objectness_mask': objectness_mask,     # batch x nproposals
            'object_assignment': ind1               # batch x nproposals
        }
        
    
    def loss_objectness(
        self, outputs: Dict, targets: Dict, assignments: Dict
    ) -> Dict:
        # Compute objectness loss
        objectness_scores = outputs['objectness_scores']    # batch x nproposal x 2
        objectness_loss = nnf.cross_entropy(                # batch x nproposal
            objectness_scores.reshape(-1, 2),               # (batch x nproposal) x 2
            assignments['objectness_label'].reshape(-1),    # (batch x nproposal)
            weight=torch.Tensor(OBJECTNESS_CLS_WEIGHTS).to(objectness_scores.device), 
            reduction='none'
        ).reshape(assignments['objectness_label'].shape)
        loss = torch.sum(
            objectness_loss * assignments['objectness_mask']
        ) / (torch.sum(assignments['objectness_mask']) + 1e-6)
    
        return {'loss_objectness': loss}
    
    
    def loss_vote(self, outputs: Dict, targets: Dict, assignments: Dict) -> Dict:
        """ Compute vote loss: Match predicted votes to GT votes.
    
        Args:
            end_points: dict (read-only)
        
        Returns:
            vote_loss: scalar Tensor
                
        Overall idea:
            If the seed point belongs to an object (votes_label_mask == 1),
            then we require it to vote for the object center.
    
            Each seed point may vote for multiple translations v1,v2,v3
            A seed point may also be in the boxes of multiple objects:
            o1,o2,o3 with corresponding GT votes c1,c2,c3
    
            Then the loss for this seed point is:
                min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
        """
    
        # Load ground truth votes and assign them to seed points
        batch, num_seed, _ = outputs['seed_xyz'].shape
        vote_xyz = outputs['vote_xyz'] # B,num_seed*vote_factor,3
        seed_inds = outputs['seed_inds'].long() # B,num_seed in [0,num_points-1]
    
        # Get groundtruth votes for the seed points
        # vote_label_mask: Use gather to select B,num_seed from B,num_point
        #   non-object point has no GT vote mask = 0, object point has mask = 1
        # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
        #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
        seed_gt_votes_mask = torch.gather(targets['vote_label_mask'], 1, seed_inds)
        seed_inds_expand = seed_inds.view(batch,num_seed,1).repeat(1, 1, 3*GT_VOTE_FACTOR)
        seed_gt_votes = torch.gather(targets['vote_label'], 1, seed_inds_expand)
        seed_gt_votes += outputs['seed_xyz'].repeat(1,1,3)
    
        # Compute the min of min of distance
        vote_xyz_reshape = vote_xyz.view(batch * num_seed, -1, 3)
        seed_gt_votes_reshape = seed_gt_votes.view(batch*num_seed, GT_VOTE_FACTOR, 3)
        # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
        dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
        votes_dist, _ = torch.min(dist2, dim=1)
        votes_dist = votes_dist.view(batch, num_seed)
        
        loss = torch.sum(
            votes_dist * seed_gt_votes_mask.float()
        ) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
        
        return {'loss_vote': loss}
    
    
    def loss_center(self, outputs: Dict, targets: Dict, assignments: Dict) -> Dict:
    
        # Compute center loss
        pred_center = outputs["center"]
        gt_center = targets["gt_box_centers"]
        box_label_mask = targets["gt_box_present"]
        objectness_label = assignments["objectness_label"].float()
        
        dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)
        
        centroid_reg_loss1 = \
            torch.sum(dist1 * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        centroid_reg_loss2 = \
            torch.sum(dist2 * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
        
        loss = centroid_reg_loss1 + centroid_reg_loss2
        
        return {'loss_center': loss}
    
    
    def loss_size(self, outputs: Dict, targets: Dict, assignments: Dict) -> Dict:
        
        objectness_label = assignments['objectness_label']
        object_assignment = assignments['object_assignment']
        # re-implement to gather labels for votenet
        targets['size_residual_label'] = \
            targets['gt_box_sizes'] - self.mean_size_arr[targets["gt_box_sem_cls_label"]]
        targets['size_residual_label'][targets['gt_box_present'] == 0] = 0
        
        ### Compute size cls loss (sem-classification)
        size_class_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, object_assignment
        )  # select (B,K) from (B,K2)
        
        size_class_loss = nnf.cross_entropy(
            outputs["size_scores"].transpose(2, 1), 
            size_class_label,
            reduction="none"
        )  # (B,K)
        size_class_loss = \
            torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        
        ### Compute size reg loss
        size_residual_label = torch.gather(
            targets["size_residual_label"], 1,
            object_assignment.unsqueeze(-1).repeat(1, 1, 3)
        )  # select (B,K,3) from (B,K2,3)
        
        size_label_one_hot = torch.cuda.FloatTensor(
            objectness_label.shape[0],  # batch
            size_class_label.shape[1], 
            self.num_size_cluster
        ).zero_()
        
        # src==1 so it"s *one-hot* (B, K, num_size_cluster) -> +(,3)
        size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1)
        size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)
        predicted_size_residual_normalized = torch.sum(
            outputs["size_residuals_normalized"] * size_label_one_hot_tiled, 2
        )  # (B, K, 3)
        # 1, 1, num_size_cluster
        mean_size_arr_expanded = self.mean_size_arr.unsqueeze(0).unsqueeze(0)
        
        mean_size_label = \
            torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
        size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
        size_residual_normalized_loss = torch.mean(
            huber_loss(
                predicted_size_residual_normalized - size_residual_label_normalized, 
                delta=1.0
            ),
            dim = -1
        )  # (B,K,3) -> (B,K)
        
        size_residual_normalized_loss = \
            torch.sum(size_residual_normalized_loss * objectness_label) \
                / (torch.sum(objectness_label) + 1e-6)
        
        loss = 0.1 * size_class_loss + size_residual_normalized_loss
        return {'loss_size': loss}
    
    
    def loss_sem_cls(self, outputs: Dict, targets: Dict, assignments: Dict) -> Dict:
        
        objectness_label = assignments['objectness_label']
        object_assignment = assignments['object_assignment']
        
        sem_cls_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, 
            object_assignment
        )  # select (B,K) from (B,K2)
        
        sem_cls_loss = nnf.cross_entropy(
            outputs["sem_cls_scores"].transpose(2, 1), 
            sem_cls_label,
            reduction="none"
        )  # (B,K)
        loss = torch.sum(sem_cls_loss * objectness_label) \
                / (torch.sum(objectness_label) + 1e-6)
        
        return {'loss_sem_cls': loss}
    
    
    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        assignments = self.compute_label_assignment(outputs, targets)
        
        loss = torch.zeros(1)[0].to(targets['point_clouds'].device)
        loss_dict = {}
        
        for loss_name, (loss_fn, loss_weight) in self.loss_dict.items():
            
            loss_intermidiate = loss_fn(outputs, targets, assignments)
            loss_dict.update(loss_intermidiate)
            
            loss += loss_weight * loss_intermidiate[loss_name]
            
        loss *= 10
        
        return assignments, loss, loss_intermidiate


def build_criterion(cfgs):
    
    criterion = SetCriterion(cfgs)
    
    return criterion