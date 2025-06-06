import torch
import torch.nn as nn
from .NRDA_arch import NRDA
import torch.nn.functional as F


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, original_size):
        # Apply transpose convolution to upsample
        x = self.conv_transpose(x)
        # Apply final convolution to refine output
        x = self.conv_final(x)
        # Crop or pad to match the original size
        # Calculate padding sizes
        pad_h = max(0, original_size[0] - x.size(2))
        pad_w = max(0, original_size[1] - x.size(3))
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Calculate cropping sizes
        crop_h = max(0, x.size(2) - original_size[0])
        crop_w = max(0, x.size(3) - original_size[1])

        # Crop if necessary
        if crop_h > 0 or crop_w > 0:
            x = x[:, :, :x.size(2) - crop_h, :x.size(3) - crop_w]
        return x


class Emb_proj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv(x)
        return out


class DAE(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions, L=8, hidden_list=[256, 256, 256]):
        super(DAE, self).__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)  # 5
        self.num_res_blocks = num_res_blocks  # 2
        self.resolution = resolution  # img_size
        self.attn_resolutions = attn_resolutions  # 16

        curr_res = self.resolution  # 256, 128, 64, 32, 16
        in_ch_mult = (1,) + tuple(ch_mult)  # (1, 1, 2, 2, 4, 4)
        # L = [35, 18, 112, 112]

        self.emb = Emb_proj(in_channels, nf, 3, 1, 1)

        blocks = []
        # initial convultion
        blocks.append(self.emb)

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]  # 64, 64, 128, 128, 256
            block_out_ch = nf * ch_mult[i]  # 64, 128, 128, 256, 256
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != 0:
                blocks.append(NRDA(n_fea_in=block_in_ch * 2, n_fea_out=block_in_ch, n_fea_middle=block_in_ch * 4, emb_ch=block_in_ch, L=L, hidden_list=hidden_list))
            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))  # 512
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(NRDA(n_fea_in=block_in_ch * 2, n_fea_out=block_in_ch, n_fea_middle=block_in_ch * 4, emb_ch=block_in_ch, L=L, hidden_list=hidden_list))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        fea_size = []
        for block in self.blocks:
            print(block)
            if isinstance(block, Downsample):
                fea_size.append((x.shape[2], x.shape[3]))
            x = block(x)
        return x, fea_size

if __name__ == '__main__':
    model = DAE(3, 64, 256, [1, 2, 2, 4, 4], 2, 256, [16]).to('cuda')
    x = torch.randn(1, 3, 256, 256).to('cuda')
    y, fea = model(x)