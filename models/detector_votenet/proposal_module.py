# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from utils.box_util import flip_axis_to_camera_tensor, get_3d_box_batch_tensor

from third_party.pointnet2.pointnet2_modules import (
    PointnetSAModuleVotes
)
import third_party.pointnet2.pointnet2_utils as pointnet2_utils



def get_3d_box_batch(box_center_unnorm, box_size, box_angle):
    box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
    boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
    return boxes
    
    
    
class ProposalModule(nn.Module):
    def __init__(
        self, 
        num_class: int, 
        num_heading_bin: int, 
        num_size_cluster: int,
        mean_size_arr: np.ndarray,
        num_proposal: int,
        sampling: str, 
        seed_feat_dim: int=256
    ):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.size_decoded = True
        
        self.prop_feat_size = 256
        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            # mlp=[self.seed_feat_dim, 128, 128, 128],
            mlp=[
                self.seed_feat_dim, 
                self.prop_feat_size,
                self.prop_feat_size,
                self.prop_feat_size
            ],
            use_xyz=True,
            normalize_xyz=True
        )
        
        # Object proposal/detection
        #   Objectness scores (2), center residual (3),
        #   heading class + residual (num_heading_bin*2), 
        #   size class + residual(num_size_cluster*4)
        self.proposal = nn.Sequential(
            nn.Conv1d(self.prop_feat_size, self.prop_feat_size, 1, bias=False),
            nn.BatchNorm1d(self.prop_feat_size), nn.ReLU(),
            nn.Conv1d(self.prop_feat_size, self.prop_feat_size, 1, bias=False),
            nn.BatchNorm1d(self.prop_feat_size), nn.ReLU(),
            nn.Conv1d(
                self.prop_feat_size,
                2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,
                1
            )
        )
        

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            raise AssertionError('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            
        end_points['aggregated_vote_xyz'] = xyz             # batch x num_proposal x 3
        end_points['aggregated_vote_inds'] = sample_inds    # batch x num_proposal
        end_points['aggregated_vote_features'] = (
            features.permute(0, 2, 1).contiguous().unsqueeze(0)
        )
        
        box_prediction = self.decode_box_predictions(
            self.proposal(features), 
            end_points, 
            self.num_heading_bin,
            self.num_size_cluster, 
            self.mean_size_arr
        )
        
        return box_prediction
    

    def decode_box_predictions(
        self, net, data_dict, num_heading_bin, num_size_cluster, mean_size_arr
    ):
        """
        decode the predicted parameters for the bounding boxes

        """
        net_transposed = net.transpose(2, 1).contiguous() # batch x nproposal x ...
        batch, nproposal = net_transposed.shape[:2]

        # 1.contain object or not
        objectness_scores = net_transposed[:,:,0:2]
        # 2.object center
        base_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        box_center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
        # 3.object size class 18-d; object size offset 18*3
        size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net_transposed[
            :,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4
        ].view(batch, nproposal, num_size_cluster, 3) # batch x nproposal x num_size_clusterx3
        # 4.semantic
        sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:]

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = box_center

        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(
            mean_size_arr.astype(np.float32)
        ).cuda().unsqueeze(0).unsqueeze(0)
        
        # recover size to get 3d box predictions
        data_dict['pred_size'] = torch.gather(
            # batch, nproposal, num_size_cluster, 3
            data_dict['size_residuals'] + torch.from_numpy(
                mean_size_arr
            ).float().cuda().unsqueeze(0).unsqueeze(0), 2, 
            torch.argmax(size_scores, -1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        ).squeeze(2)  # batch, nproposal, 3
        data_dict['sem_cls_scores'] = sem_cls_scores
        
        # processed box info: batch x nproposal x 8 x 3
        data_dict["box_corners"] = get_3d_box_batch(
            box_center, 
            data_dict['pred_size'], 
            torch.zeros_like(data_dict['center'][..., 0])
        )

        data_dict["prop_features"] = data_dict["aggregated_vote_features"]
        data_dict["bbox_mask"] = objectness_scores.argmax(-1) #0: invalid 1: valid
        data_dict['bbox_sems'] = sem_cls_scores.argmax(-1)
        data_dict['sem_cls'] = sem_cls_scores.argmax(-1)

        # re-gather information for evaluation
        sem_cls_logits = torch.cat(
            (sem_cls_scores, objectness_scores.argmin(-1, keepdim=True) * 1e10), dim=-1
        )
        
        gathered_outputs = {
            "sem_cls_logits": sem_cls_logits,
            "objectness_prob": torch.softmax(objectness_scores, dim=-1)[..., 1],
            "sem_cls_prob": torch.softmax(sem_cls_scores, dim=-1),
        }
        data_dict.update(gathered_outputs)
        
        return data_dict
