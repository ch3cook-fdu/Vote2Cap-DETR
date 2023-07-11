import copy, math, importlib
import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from typing import Dict

from collections import OrderedDict
from transformers import GPT2Config, GPT2LMHeadModel

from models.captioner_dcc.helper import Matcher
from models.captioner_dcc.generation_utils import generation
from models.captioner_dcc.scst import SCST_Training
from utils.box_util import generalized_box3d_iou


@torch.no_grad()
def hungarian_matching(matcher: Matcher, predicts: dict, targets: dict) -> dict:
    
    outputs = predicts.copy()
    nactual_gt = targets["gt_box_present"].sum(axis=1).long()
    num_boxes = torch.clamp(nactual_gt.sum(), min=1).item()
    
    targets["nactual_gt"] = nactual_gt
    targets["num_boxes"] = num_boxes
    targets["num_boxes_replica"] = nactual_gt.sum().item()
    
    # for match only here
    outputs["gious"] = generalized_box3d_iou(
        outputs["box_corners"], 
        targets["gt_box_corners"], 
        targets["nactual_gt"],
        rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
        needs_grad=False,
    )
    center_dist = torch.cdist(
        outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
    )
    outputs["center_dist"] = center_dist
    
    return matcher(outputs, targets)


def proposal_dimension_select(features: Tensor, indices: Tensor) -> Tensor:
    '''
    
    Parameters
    ----------
    features : Tensor, with size [batch x nsrc x ...]
        Data bank, from which to gather information.
    indices : Tensor, with size [batch x ntgt]
        Indices for gathering information from data bank.

    Returns
    -------
    Tensor, with size [batch x ntgt x ...]
        Gathers features in proposal dimension.
    
    '''
    return torch.gather(
        features, 1, 
        indices.reshape(
            *(indices.shape + tuple(1 for _ in features.shape[2:]))
        ).repeat(
            *((1, 1) + features.shape[2:])
        )
    )


def decode_box_corners(box_corners: Tensor) -> Tensor:
    box_corners = copy.deepcopy(box_corners.detach())
    box_corners[..., [1, 2]] = box_corners[..., [2, 1]]
    box_corners[..., -1] *= -1
    return box_corners


def position_embedding(max_len: int, d_model: int) -> Tensor:
    position_embedding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))
    position_embedding[:, 0::2] = torch.sin(position * div_term)
    position_embedding[:, 1::2] = torch.cos(position * div_term)
    return position_embedding


class captioner(nn.Module):

    def __init__(self, args, train_dataset):
        super(captioner, self).__init__()
        
        self.embedding_size = 256
        self.max_positions = 64
        self.max_des_len = args.max_des_len
        
        ## initialize tokenizer for batch decoding
        self.tokenizer = train_dataset.tokenizer
        self.nvocabs = len(self.tokenizer)
        
        ## for label assignment
        self.matcher = Matcher(
            cost_class=1, cost_objectness=0, cost_giou=2, cost_center=0
        )
        
        ## caption generation cores
        gpt2_config = GPT2Config(
            vocab_size=self.nvocabs,
            n_positions=self.max_positions,
            n_embd=self.embedding_size,
            n_layer=2,
            n_head=4,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            add_cross_attention=True,
        )
        self.transformer = GPT2LMHeadModel(config=gpt2_config)
        self.transformer.transformer.wpe = nn.Embedding.from_pretrained(
            position_embedding(self.max_positions, self.embedding_size)
        )
        
        ## for proposal feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(256, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        
        self.context_projector = nn.Sequential(
            nn.Linear(256, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
        )
        
        ## ---- super parameters for evaluation
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 5 if args.use_beam_search is True else None,
        }
        
        self.scst = SCST_Training(args)
        self.use_scst = hasattr(args, 'use_scst') and args.use_scst is True
        self.scst_max_des_per_iter = 32
    
    
    def prepare_object_representations(self, detector_output: dict) -> dict:
        
        ## extract proposal feature: batch x nprop x channel
        last_layer_output = detector_output['prop_features'][-1]
        object_feature = self.feature_projector(last_layer_output)
        prefix_feature = object_feature.unsqueeze(2)
        # batch x nprop x 1 x channel, as RNN-like guidance
        detector_output['object_features'] = prefix_feature
        
        ## batch x nprop x ntgt x channel, as cross attention guidance
        query_xyz = detector_output['query_xyz']
        batch, nprop, _ = query_xyz.shape
        
        # batch x nprop x npoints
        center_distance = torch.cdist(query_xyz, detector_output['enc_xyz'])
        
        # batch x nprop x k
        k_near_indice = center_distance.topk(k=128, largest=False, dim=-1).indices
        k_near_context_feature = proposal_dimension_select(
            self.context_projector(detector_output['enc_features']), 
            k_near_indice.reshape(batch, -1)
        )   # batch x (nprop x k) x channel
        k_near_context_feature = k_near_context_feature.reshape(
            batch, nprop, -1, self.embedding_size
        )   # batch x nprop x k x channel
        
        detector_output['k_near_context'] = k_near_indice
        detector_output['encoder_hidden_states'] = k_near_context_feature
        
        return detector_output
        
    
    def forward(self, detector_output: dict, inputs: dict, is_eval: bool=False) -> dict:
        
        # nlayers x batch x nprop x channel -> batch x nprop x 1 x channel
        detector_output = self.prepare_object_representations(detector_output)
        
        if is_eval is False:
            if self.use_scst is True:
                return self.forward_scst(detector_output, inputs)
            return self.forward_training(detector_output, inputs)
        else:
            return self.forward_evaluation(detector_output, inputs)
    
    
    def forward_training(self, detector_output: Dict, inputs: Dict) -> Dict:
        
        # get word embeddings, NOTE: captioner does not predict <bos> token
        caption_ids = inputs['reference_tokens']    # batch x MAX_NUM_OBJ x ntokens
        embedding_mask = inputs['reference_masks']  # batch x MAX_NUM_OBJ x ntokens
        
        # ---- match proposal bounding boxes with ground truth inds
        assignments = hungarian_matching(
            self.matcher, detector_output, inputs
        )
        
        # ---- generate caption labels for rnn model
        gt_box_cap_label = proposal_dimension_select(
            caption_ids, assignments['per_prop_gt_inds'].long()
        )   # batch x nproposals x max_des_len
        gt_box_cap_masks = proposal_dimension_select(
            embedding_mask, assignments['per_prop_gt_inds'].long()
        )   # batch x nproposals x max_des_len
        
        # no loss for objects with background and non annotated objects
        unvalid_proposal = assignments['proposal_matched_mask']
        unannotated_proposal = (gt_box_cap_label[..., 0] != 0).long()
        annotated_proposal = unvalid_proposal * unannotated_proposal
        assignments['annotated_proposal'] = annotated_proposal
        
        # ---- generate caption embeddings for rnn model
        prefix_tokens = detector_output['object_features']
        inputs_embeds = torch.cat([
            prefix_tokens, self.transformer.transformer.wte(gt_box_cap_label)
        ], dim=2)   # batch x nproposals x (nprefix + max_des_len) x channel
        inputs_masks = torch.cat([
            torch.ones_like(prefix_tokens[..., 0]), gt_box_cap_masks
        ], dim=2)   # batch x nproposals x (nprefix + max_des_len)
        
        outputs = self.transformer( # num_annotated x (1 + max_des_len)
            inputs_embeds=inputs_embeds[annotated_proposal == 1],
            attention_mask=inputs_masks[annotated_proposal == 1],
            encoder_hidden_states=\
                None if detector_output.get('encoder_hidden_states', None) is None else \
                    detector_output['encoder_hidden_states'][annotated_proposal == 1]
        )
        
        detector_output['loss'] += 5 * self.loss_caption(
            logits = outputs.logits[:, prefix_tokens.shape[2] - 1: -1],
            target = gt_box_cap_label[annotated_proposal == 1].long()
        )
        
        return detector_output
    
    
    def forward_scst(self, detector_output: Dict, inputs: Dict) -> Dict:
        
        # get word embeddings, NOTE: captioner does not predict <bos> token
        caption_ids = inputs['reference_tokens']    # batch x MAX_NUM_OBJ x ntokens
        
        # ---- match proposal bounding boxes with ground truth inds
        assignments = hungarian_matching(
            self.matcher, detector_output, inputs
        )
        
        # ---- generate caption labels for rnn model
        gt_box_cap_label = proposal_dimension_select(
            caption_ids, assignments['per_prop_gt_inds'].long()
        )   # batch x nproposals x max_des_len
        
        # no loss for objects with background and non annotated objects
        unvalid_proposal = assignments['proposal_matched_mask']
        unannotated_proposal = (gt_box_cap_label[..., 0] != 0).long()
        annotated_proposal = unvalid_proposal * unannotated_proposal
        
        if torch.sum(annotated_proposal == 1).cpu().tolist() > self.scst_max_des_per_iter:
            random_value = torch.randn(annotated_proposal.shape, device=annotated_proposal.device)
            random_value[annotated_proposal == 0] = 1e8
            
            random_threshold = torch.kthvalue(
                random_value.view(-1), 
                self.scst_max_des_per_iter
            ).values
            
            annotated_proposal *= (random_value <= random_threshold).long()
        
        assignments['annotated_proposal'] = annotated_proposal
        
        # generation with greedy search
        prefix_tokens = detector_output['object_features']
        
        greedy_caption = generation(
            self.transformer, 
            inputs_embeds=prefix_tokens[annotated_proposal == 1],
            encoder_hidden_states=\
                None if detector_output.get('encoder_hidden_states', None) is None else \
                    detector_output['encoder_hidden_states'][annotated_proposal == 1],
            early_stopping = True,
            eos_token_id = self.tokenizer.eos_token_id,
            num_beams = None,
        )
        
        beam_caption = generation(
            self.transformer, 
            inputs_embeds=prefix_tokens[annotated_proposal == 1],
            encoder_hidden_states=\
                None if detector_output.get('encoder_hidden_states', None) is None else \
                    detector_output['encoder_hidden_states'][annotated_proposal == 1],
            **self.caption_config
        )
        scst_loss = self.scst(greedy_caption, beam_caption, inputs, assignments)
        detector_output['loss'] += 5 * scst_loss
        
        return detector_output
        
        
    def loss_caption(self, logits: Tensor, target: Tensor) -> Tensor:
        loss_config = {'reduction': 'none', 'ignore_index': 0}
        
        loss_per_word = nnf.cross_entropy(
            logits.reshape(-1, self.nvocabs),
            target.reshape(-1), 
            **loss_config
        )
        loss_per_word = loss_per_word.reshape(target.shape)
        final_loss = torch.sum(loss_per_word * (target != 0).float()) / torch.sum(
            torch.sum(target != 0).float() + 1e-6
        )
        return final_loss
    
    
    def forward_evaluation(self, detector_output: Dict, inputs: Dict) -> Dict:
        
        # proposal_tokens: batch x nprop x nprefix x channel
        prefix_tokens = detector_output['object_features']
        
        batch, nproposals, nprefix, channel = prefix_tokens.shape
        
        caption_output = OrderedDict()
        
        for batch_id in range(batch):
            scene_cap_output = generation(
                self.transformer, 
                inputs_embeds=prefix_tokens[batch_id],
                encoder_hidden_states=\
                    None if detector_output.get('encoder_hidden_states', None) is None else \
                        detector_output['encoder_hidden_states'][batch_id],
                **self.caption_config
            )
            # update scene output to batch output
            for key, tensor in scene_cap_output.items():
                caption_output[key] = caption_output.get(key, []) + [tensor]
        
        for key, tensor in caption_output.items():
            caption_output[key] = torch.cat(caption_output[key], dim=0)
        
        
        captions = self.tokenizer.batch_decode(
            caption_output['output_ids'].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        detector_output['lang_cap'] = [
            [
                'sos ' + captions[batch_id * nproposals + prop_id] + ' eos' \
                    for prop_id in range(nproposals)
            ] \
            for batch_id in range(batch)
        ]
        
        return detector_output
    
