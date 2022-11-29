import os, importlib, json, tqdm
import torch
from torch import nn, Tensor
from models.captioner_dcc.cider_scorer import Cider
from collections import defaultdict, OrderedDict


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

            
class SCST_Training(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        dataset = importlib.import_module(f'datasets.{args.dataset}')
        self.scan_list = dataset.SCANREFER['scene_list']['train']
        self.scanrefer = dataset.SCANREFER['language']['train']
        
        print('preparing N-Grams in Cider Scorer')
        self.gathered_scanrefer = self.preprocess_and_gather_language()
        
        gathered_corpus = OrderedDict({
            f"{scene_id}|{instance_id}": instance_annotation \
            for scene_id, scene_annotation in self.gathered_scanrefer.items() \
                for instance_id, instance_annotation in scene_annotation.items()
        })
        with open(os.path.join(args.checkpoint_dir, 'train_corpus.json'), 'w') as f:
            json.dump(gathered_corpus, f, indent=4)
        
        self.rewarder = Cider(gathered_corpus)
        
        ScanReferTokenizer = dataset.ScanReferTokenizer
        self.tokenizer = ScanReferTokenizer(dataset.vocabulary['word2idx'])
    
    
    def preprocess_and_gather_language(self):
        
        gathered_language = defaultdict(lambda : defaultdict(list))
        
        for lang_dict in tqdm.tqdm(self.scanrefer):
            scene_id  = lang_dict['scene_id']
            object_id = int(lang_dict['object_id'])
            
            sentence  = ' '.join(lang_dict['token'] + ['eos'])
            gathered_language[scene_id][object_id].append(sentence)
        
        return gathered_language
    
    
    def forward(
        self, 
        greedy_output, 
        beam_search_output, 
        inputs: dict,
        assignments: dict
    ):
        
        unvalid_proposal = assignments['proposal_matched_mask']
        per_prop_gt_inds = assignments['per_prop_gt_inds'].long()
        annotated_proposal = assignments['annotated_proposal']
        
        device = unvalid_proposal.device
        
        # scan_id for each propsal and trained samples
        scan_idx = inputs['scan_idx'].unsqueeze(1).repeat(1, unvalid_proposal.shape[1])
        scan_idx = scan_idx[annotated_proposal == 1]
        # instance index for each proposal and trained samples
        per_prop_instance_id = torch.gather(
            inputs['gt_box_object_ids'], 1, per_prop_gt_inds
        )
        per_prop_instance_id = per_prop_instance_id[annotated_proposal == 1]
        
        
        beam_output_scores = beam_search_output['output_scores']
        
        caption = []
        information = []
        reference = []
        
        batch_size, beam_size = beam_search_output['beam_output_ids'].shape[:2]
        
        for sample_id, (scan_id, instance_id) in enumerate(
            zip(
                scan_idx.cpu().tolist(),
                per_prop_instance_id.cpu().tolist()
            )
        ):
            sentence_reference = \
                self.gathered_scanrefer[self.scan_list[scan_id]][instance_id]
            
            beam_output_ids = beam_search_output['beam_output_ids'][sample_id]
            
            greedy_output_ids = greedy_output['output_ids'][sample_id]
            
            # greedy search baseline
            information.extend([f'{sample_id}|{self.scan_list[scan_id]}|{instance_id}|greedy'])
            caption.append(greedy_output_ids.unsqueeze(0))
            reference.append(sentence_reference)
            
            # beam search for training
            information.extend([
                f'{sample_id}|{self.scan_list[scan_id]}|{instance_id}|beam_{beam_id}' \
                    for beam_id in range(beam_output_ids.shape[0])
            ])
            caption.append(beam_output_ids)
            reference.extend([
                sentence_reference for beam_id in range(beam_output_ids.shape[0])
            ])
        
        caption = torch.cat(caption, dim=0) # (nsample x (1 + beam_size)) x ntokens
        
        # decode caption for scoring
        caption = self.tokenizer.batch_decode(
            caption.cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        for sample_id in range(len(caption)):
            if len(caption[sample_id]) > 0 and caption[sample_id][-1] != ' ': 
                caption[sample_id] += ' '
            caption[sample_id] = [caption[sample_id] + 'eos']
        
        # scoring
        _, cider_score = self.rewarder.compute_score(
            OrderedDict(zip(information, reference)),
            OrderedDict(zip(information, caption))
        )
        
        cider_score = torch.from_numpy(cider_score).to(device).float()
        cider_score = cider_score.reshape(batch_size, 1 + beam_size)
        
        # batch x nbeam
        reward = cider_score[:, 1:] - cider_score[:, [0]]
        
        scst_loss = - torch.mean(reward * beam_output_scores)
        
        
        return scst_loss
    