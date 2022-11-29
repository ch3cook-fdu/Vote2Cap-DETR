import os
import torch
import numpy as np
from torch import nn, Tensor
from functools import partial
from typing import Dict, Callable, List
from utils.pc_util import scale_points, shift_scale_points

# votenet configuration & submodules
from models.detector_votenet.config import model_config
from models.detector_votenet.backbone_module import Pointnet2Backbone
from models.detector_votenet.proposal_module import ProposalModule
from models.detector_votenet.voting_module import VotingModule

from models.detector_votenet.criterion import build_criterion


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(
        self, 
        dataset_config: object,
        num_class: int, 
        num_heading_bin: int, 
        num_size_cluster: int, 
        mean_size_arr: np.ndarray,
        input_feature_dim: int=0, 
        num_proposal: int=256, 
        vote_factor: int=1, 
        sampling: int='vote_fps',
        criterion: Callable=None,
    ):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.proposal = ProposalModule(
            num_class, 
            num_heading_bin, 
            num_size_cluster,
            mean_size_arr, 
            num_proposal, 
            sampling
        )
        
        self.criterion = criterion
    
    def forward(self, inputs: Dict, is_eval: bool=False) -> Dict:
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.proposal(xyz, features, end_points)
        
        if self.criterion is not None and is_eval is False:
            assignments, end_points['loss'], _ = self.criterion(
                end_points, inputs
            )
            end_points['assignments'] = {
                'proposal_matched_mask': assignments['objectness_label'],   # batch x nproposals
                'per_prop_gt_inds': assignments['object_assignment']        # batch x nproposals
            }
        else:
            try:
                assignments, _, _ = self.criterion(
                    end_points, inputs
                )
                end_points['assignments'] = {
                    'proposal_matched_mask': assignments['objectness_label'],   # batch x nproposals
                    'per_prop_gt_inds': assignments['object_assignment']        # batch x nproposals
                }
            except:
                pass
        
        end_points.update({
            # nlyr x npoint x batch x channel -> nlyr x batch x npoint x channel
            'prop_features': end_points['aggregated_vote_features'],
            # batch x channel x npoints -> batch x npoints x channel
            'enc_features': end_points['fp2_features'].permute(0, 2, 1),
            'enc_xyz': end_points['fp2_xyz'],               # batch x npoints x 3
            'query_xyz': end_points['aggregated_vote_xyz'], # batch x nqueries x 3
            'center_normalized': shift_scale_points(
                end_points['center'], src_range=[
                    inputs["point_cloud_dims_min"],
                    inputs["point_cloud_dims_max"],
                ]
            )
        })
        
        return end_points


def detector(args, dataset_config):
    cfg = model_config(args, dataset_config)
    
    criterion = build_criterion(cfg)
    
    model = VoteNet(
        dataset_config=dataset_config,
        num_class=cfg.num_class, 
        num_heading_bin=cfg.num_heading_bin, 
        num_size_cluster=cfg.num_size_cluster, 
        mean_size_arr=cfg.mean_size_arr,
        input_feature_dim=cfg.input_feature_dim, 
        num_proposal=cfg.num_proposal, 
        vote_factor=cfg.vote_factor, 
        sampling=cfg.sampling,
        criterion=criterion
    )
    return model