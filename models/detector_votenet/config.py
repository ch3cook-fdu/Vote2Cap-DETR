import os
import numpy as np
from datasets.scannet import BASE

class model_config:
    
    def __init__(self, args, dataset_config):
        
        self.dataset_config = dataset_config
        
        self.preenc_npoints = 2048
        self.num_class = dataset_config.num_semcls
        self.num_heading_bin: int=1
        self.num_size_cluster: int=18
        self.mean_size_arr: np.ndarray=np.load(
            os.path.join(
                BASE, 'data', 'scannet', 'meta_data', 'scannet_means.npz'
            )
        )['arr_0']
        
        self.input_feature_dim: int=(
            3   * (int(args.use_color) + int(args.use_normal)) + \
            1   * int(args.use_height) + \
            128 * int(args.use_multiview)
        )
        self.num_proposal: int=256
        self.vote_factor: int=1
        # self.sampling: str='vote_fps'
        self.sampling: str='seed_fps'
        self.out_dim: int=256
        
        ### Matcher
        self.matcher_giou_cost = 2.
        self.matcher_cls_cost = 1.
        self.matcher_center_cost = 0.
        self.matcher_objectness_cost = 0.
        
        ### Loss Weights
        self.loss_giou_weight = 1.
        self.loss_sem_cls_weight = 1.
        self.loss_no_object_weight = 0.25
        self.loss_angle_cls_weight = 0.1
        self.loss_angle_reg_weight = 0.5
        self.loss_center_weight = 5.
        self.loss_size_weight = 1.