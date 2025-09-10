import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from typing import List
from comm_loss import CoMMLoss 
from backbone import LateFusion, MMGCN, MultiDialogueGCN, MM_DFN, MultiBiModel
from mmfusion import MMFusion
from smurf_decomp import ThreeModalityModel, compute_corr_loss

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        if args.backbone == "late_fusion":
            self.net = LateFusion(args)
        if args.backbone == "mmgcn":
            self.net = MMGCN(args)
        if args.backbone == "dialogue_gcn":
            self.net = MultiDialogueGCN(args)
        if args.backbone == "mm_dfn":
            self.net = MM_DFN(args)
        if args.backbone == "biddin":
            self.net = MultiBiModel(args)

        self.args = args
        self.modalities = args.modalities
        self.threshold = args.cl_threshold
        self.growing_factor = args.cl_growth
        
        self.use_comm = args.use_comm
        self.use_smurf = args.use_smurf
        # CoMM
        n_modalities = 3
        input_dims = [1024, 1024, 1024]
        input_adapters: List[nn.Module] = [
            nn.Linear(in_dim, 1024) for in_dim in input_dims
        ]
        from comm import CoMM
        comm_fuse = MMFusion(
        input_adapters=input_adapters,
        embed_dim=1024,
        fusion="concat",   # or "x-attn"
        pool="cls",        # or "mean"
        n_heads=4,
        n_layers=1,
        add_bias_kv=False,
        dropout=0.1
    )
        
        self.comm_module = CoMM(
            comm_enc=comm_fuse, 
            hidden_dim=1024, 
            n_classes=6, 
            augmentation_style="linear", 
        )
        hidden_dim = 1024
        # SMURF
        #hardcoded
        D_text, D_audio, D_visual = 768, 512, 1024 #TODO change to args.embedding_dim
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.smurf_model = ThreeModalityModel(t_dim=D_text, a_dim=D_audio, v_dim=D_visual, out_dim=hidden_dim, final_dim=6)
        
        # Fusion layer 
        self.fusion_layer = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, data):
        # legacy pipeline
        joint, logit = self.net(data)
        prob = F.log_softmax(joint, dim=-1)
        prob_m = {
            m: F.log_softmax(logit[m], dim=-1) for m in self.modalities
        }
        with torch.no_grad():
            scores = {
                m: sum([F.softmax(logit[m], dim=1)[i][data["label_tensor"][i]]
                       for i in range(prob.size(0))])
                for m in self.modalities
            }

            min_score = min(scores.values())
            ratio = {
                m: scores[m] / min_score
                for m in self.modalities
            }
            
        return prob, prob_m, ratio

    def get_loss(self, data):
        # Get CoMM loss:
        # CoMM added
        if self.use_comm:
            textf = data["tensor"]['t']
            audiof = data["tensor"]['a']
            visualf = data["tensor"]['v']
            z1, z2, all_transformer_out = self.comm_module(textf, audiof, visualf, None)
            comm_loss = CoMMLoss()
            comm_loss_values = comm_loss({
                    "aug1_embed": z1,
                    "aug2_embed": z2,
                    "prototype": -1  # You need to define/select this somewhere
                })
        else:
            comm_loss_values = 0
        # Legacy loss
        joint, logit  = self.net(data)

        # if not self.use_smurf:
        prob = F.log_softmax(joint, dim=-1)
        # else:
        #     prob = self.forward(data)[3]
        prob_m = {
            m: F.log_softmax(logit[m], dim=-1) for m in self.modalities
        }

        loss = F.nll_loss(prob, data["label_tensor"], reduction='none')

        loss_m = {
            m: F.nll_loss(prob_m[m], data["label_tensor"])
            for m in self.modalities
        }

        with torch.no_grad():
            sum_len = [0] + torch.cumsum(data["length"], dim=0).tolist()
            score_dialogs = [
                torch.stack([torch.sum(torch.stack([F.softmax(logit[m], dim=1)[i][data["label_tensor"][i]]
                                                    for i in range(sum_len[j-1], sum_len[j])]))
                            for j in range(1, len(sum_len))])
                for m in self.modalities]
            score_dialogs = torch.stack(score_dialogs, dim=0).std(dim=0)
            dialogue_score = torch.zeros(prob.size(0)).to(self.args.device)

            for j in range(1, len(sum_len)):
                dialogue_score[sum_len[j-1]:sum_len[j]
                               ].fill_(score_dialogs[j-1])

            v = self.hard_regularization(dialogue_score, loss)

            batch_score = {
                m: sum([F.softmax(logit[m], dim=1)[
                       i][data["label_tensor"][i]] * v[i] for i in range(prob.size(0))])
                for m in self.modalities
            }

            min_score = min(batch_score.values())
            ratio = {
                m: batch_score[m] / min_score
                for m in self.modalities
            }
            
        loss = loss * v

        take_sample = torch.sum(v)

        return loss.mean(), ratio, take_sample, loss_m, comm_loss_values # hardcoded comm loss for now

    def hard_regularization(self, scores, loss):
        if self.args.use_cl:
            diff = 2 / (1 / scores + 1 / loss)
            v = diff <= self.threshold
        else:
            v = torch.ones_like(scores)
        return v.int()

    def increase_threshold(self):
        self.threshold *= self.growing_factor
        if self.threshold > 60:
            self.threshold = 60
