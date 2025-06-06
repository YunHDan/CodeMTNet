import torch.nn.functional as F
import torch.nn as nn
from .mambablock_arch import MambaBlock
from typing import Optional
from torch import Tensor

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        pos = pos.unsqueeze(2)      # [HW, B, C] -> [HW, B, 1, C]
        pos = pos.permute(1, 3, 2, 0)  # [HW, B, 1, C] -> [B, C, 1, HW]

        HW = tensor.shape[0]
        pos = F.interpolate(pos, size=(1, HW), mode='bilinear', align_corners=False)
        pos = pos.squeeze(2).permute(2, 0, 1)

        return pos + tensor


    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class MACR(nn.Module):
    def __init__(self, dim, norm_layer, attn_drop_rate, d_state, mlp_ratio, nhead,
                           dim_mlp, dropout, mamba_layers, transformer_layers, codebook_size):
        super(MACR, self).__init__()
        blocks1 = []
        blocks2 = []
        # blocks3 = []

        for _ in range(mamba_layers):
            blocks1.append(MambaBlock(hidden_dim=dim, norm_layer=norm_layer, attn_drop_rate=attn_drop_rate, d_state=d_state, expand=mlp_ratio))
        self.mam1 = nn.ModuleList(blocks1)

        self.feat_emb = nn.Linear(256, dim)

        for _ in range(transformer_layers):
            blocks2.append(TransformerSALayer(embed_dim=dim, nhead=nhead, dim_mlp=dim_mlp, dropout=dropout))
        self.attn = nn.ModuleList(blocks2)
        # for _ in range(mamba_layers):
        #     blocks3.append(MambaBlock(hidden_dim=dim, norm_layer=norm_layer, attn_drop_rate=attn_drop_rate, d_state=d_state, expand=mlp_ratio))
        # self.mam2 = nn.ModuleList(blocks3)

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, codebook_size, bias=False))

    def forward(self, lq_feat, pos):
        # B, C, H, W = lq_feat.shape
        #   Mamba Block
        for block in self.mam1:
            out1 = block(lq_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        feat_emb = self.feat_emb(out1.flatten(2).permute(2, 0, 1))
        query_emb = feat_emb

        # Attention Block
        for block in self.attn:
            out = block(query_emb, query_pos=pos)

        # # (HW)BC -> BCHW
        # feat_out = self.feat_out(out2.permute(1, 0, 2).reshape(B*H*W, C)).view(B, C, H, W)

        # # Mamba Block
        # for block in self.mam2:
        #     out = block(feat_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        logits = self.idx_pred_layer(out)
        return logits

