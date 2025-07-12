
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional
from archs.vqgan_arch import *
from utils.registry import ARCH_REGISTRY
from .DAE_arch import DAE, Downsample
from .mcrb_arch import mcrb
from .vqgan_arch import Upsample


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Fuse_sft_block(nn.Module):  # CFT
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)

        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


@ARCH_REGISTRY.register()
class codemtnet(VQAutoEncoder):
    def __init__(self, dim_embd=256, n_head=8, mamba_layers=4,
                 codebook_size=1024, latent_size=256,
                 connect_list=['32', '64', '128', '256'],
                 fix_modules=['hq_encoder', 'quantize', 'generator'], vqgan_path=None,
                 img_size=256, nf=64, ch_mult=[1, 2, 2, 4, 4], res_blocks=2, attn_resolutions=[16],
                 d_state=16, mlp_ratio=2, transformer_layers=2, fuse_encoder_block=[12, 8, 4], fuse_generator_block=[10, 13, 16],
                 L=8, hidden_list=[256, 256, 256]):
        super(codemtnet, self).__init__(256, 64, [1, 2, 2, 4, 4], 'nearest', 2, [16], codebook_size)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'])

        self.connect_list = connect_list
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2
        self.latent_size = latent_size

        self.position_emb = nn.Parameter(torch.zeros(self.latent_size, self.dim_embd))

        self.dae = DAE(
            in_channels=3,
            nf=nf,
            emb_dim=self.dim_embd,
            ch_mult=ch_mult,
            num_res_blocks=res_blocks,
            resolution=img_size,
            attn_resolutions=attn_resolutions,
            L=L,
            hidden_list=hidden_list
        )

        # macb
        self.macr = mcrb(dim=dim_embd, norm_layer=nn.LayerNorm, attn_drop_rate=0, d_state=d_state,
                         mlp_ratio=mlp_ratio, nhead=n_head,
                         dim_mlp=self.dim_mlp, dropout=0.0, mamba_layers=mamba_layers,
                         transformer_layers=transformer_layers,
                         codebook_size=codebook_size)

        self.channels = {
            '4': 128,
            '8': 128,
            '10': 256,
            '12': 256,
            '13': 128,
            '16': 128,
        }

        self.fuse_encoder_block = fuse_encoder_block
        self.fuse_generator_block = fuse_generator_block

        self.link_list = {
            '12': 10,
            '8': 13,
            '4': 16,
        }

        self.set_link = {
            '10': 32,
            '13': 64,
            '16': 128,
        }

        self.fuse_convs_dict = nn.ModuleDict()
        for number in self.fuse_generator_block:
            in_ch = self.channels[str(number)]
            self.fuse_convs_dict[str(self.set_link[str(number)])] = Fuse_sft_block(in_ch, in_ch)        # 10, 13, 16

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, mode='train', test_shape=None):
        # ################### Encoder #####################
        enc_feat_dict = {}
        out_list = self.fuse_encoder_block        # 12, 8, 4
        fea_size = []
        for i, block in enumerate(self.dae.blocks):
            if isinstance(block, Downsample):
                fea_size.append((x.shape[2], x.shape[3]))
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(self.link_list[str(i)])] = x.clone()       # 10, 13, 16

        # ################# mcrb ###################
        lq_feat = x  # (BCHW)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)

        # Mamba-Assisted Codebook Retrival
        logits = self.macr(lq_feat, pos_emb)
        logits = logits.permute(1, 0, 2)  # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        if mode == 'train':
            quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])
        else:
            quant_feat = self.quantize.get_codebook_feat(top_idx, shape=test_shape)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = self.fuse_generator_block   # 10, 13, 16

        for i, block in enumerate(self.generator.blocks):
            if isinstance(block, Upsample):
                fea = fea_size.pop()
                x = block(x, fea)
            else:
                x = block(x)
            if i in fuse_list:  # fuse after i-th block
                if w > 0:
                    x = self.fuse_convs_dict[str(self.set_link[str(i)])](enc_feat_dict[str(i)].detach(), x, w)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        return out, logits, lq_feat
