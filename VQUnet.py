import torch
import einops
import numpy as np
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops.layers.torch import Rearrange
from timm.layers import DropPath

from quant import VectorQuantizer3


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels, freq_inv=100):
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq.to(tensor.device))[..., None]
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).flatten(-2)
        return emb_x


def get_lattice_meshgrid_np(shape_size, no_shift=False):
    linspace = [np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    ) for shape in shape_size]
    mesh = np.stack(
        np.meshgrid(linspace[0], linspace[1], linspace[2], indexing="ij"),
        axis=-1,
    )
    return mesh


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, drop_path_r=0., stride=1, groups=1,
                 activation_class=nn.ReLU, conv_class=nn.Conv3d, affine=False,
                 checkpoint=False, **kwargs):
        super().__init__()
        self.activation_fn = activation_class()
        self.conv1 = conv_class(in_planes, planes, kernel_size=1, bias=False, groups=groups)
        self.norm1 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv2 = conv_class(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.norm2 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv3 = conv_class(planes, self.expansion * planes, kernel_size=1, bias=False, groups=groups)
        self.norm3 = nn.InstanceNorm3d(self.expansion * planes, affine=affine)
        self.drop_path = DropPath(drop_path_r) if drop_path_r > 0. else nn.Identity()
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, groups=groups)
        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(self, x):
        out = self.activation_fn(self.norm1(self.conv1(x)))
        out = self.activation_fn(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out = self.drop_path(out)
        out += self.shortcut_conv(x)
        out = self.activation_fn(out)
        return out

    def forward_checkpoint(self, x):
        return torch_checkpoint(self.forward_normal, x, preserve_rng_state=False)


class ConvBuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activate_class=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            self.activate_function,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.shortcut_conv = nn.Identity()
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        return self.activate_function(self.conv1(x) + self.shortcut_conv(x))


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path_r=0., stride=1, scale=4, activate_class=nn.ReLU):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, out_channels * self.scale, 1, 1, 0, bias=False), nn.InstanceNorm3d(out_channels * self.scale, affine=True))
        self.norm1 = nn.InstanceNorm3d(out_channels * self.scale, affine=True)
        self.conv_list = nn.ModuleList([nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) for _ in range(self.scale - 1)])
        self.activate_class = activate_class()
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels * self.scale, out_channels, 1, 1, 0, bias=False), nn.InstanceNorm3d(out_channels, affine=True))
        self.drop_path = DropPath(drop_path_r) if drop_path_r > 0. else nn.Identity()
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        x_list = self.activate_class(self.conv1(x)).chunk(self.scale, dim=1)
        y_list = []
        for ii, xi in enumerate(x_list):
            if ii == 0:
                y_list.append(xi)
            elif ii == 1:
                y_list.append(self.conv_list[ii - 1](xi))
            else:
                y_list.append(self.conv_list[ii - 1](xi + y_list[-1]))
        residual = self.conv2(self.activate_class(self.norm1(torch.cat(y_list, dim=1))))
        residual = self.drop_path(residual)
        return self.activate_class(residual + self.shortcut_conv(x))


class AttentionGate(nn.Module):
    def __init__(self, down_features, up_features, out_features, attention_features=64, attention_heads=8):
        super().__init__()
        self.dfz = down_features
        self.ufz = up_features
        self.ofz = out_features
        self.afz = attention_features
        self.ahz = attention_heads
        self.conv_q = nn.Sequential(nn.Conv3d(self.ufz, self.afz, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(self.afz, affine=True))
        self.conv_k = nn.Sequential(nn.Conv3d(self.dfz, self.afz, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(self.afz, affine=True))
        self.conv_v = nn.Sequential(nn.Conv3d(self.dfz, self.ufz, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(self.ufz, affine=True))
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(self.afz, self.ahz, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.conv_back = ConvBuildingBlock(self.ufz, self.ofz)

    def forward(self, us, ds):
        ds_shape = ds.shape
        D, H, W = ds_shape[2:]
        upsampled = nn.functional.interpolate(input=us, size=(D, H, W), mode='trilinear', align_corners=True)
        query = self.conv_q(upsampled)
        key = self.conv_k(ds)
        value = self.conv_v(ds)
        value = einops.rearrange(value, "N (afz ahz) d h w -> N afz ahz d h w", ahz=self.ahz)
        gate = self.gate(query + key)
        out = value * gate[:, None]
        out = einops.rearrange(out, "N afz ahz d h w -> N (afz ahz) d h w", ahz=self.ahz)
        return self.conv_back(self.relu(out + upsampled))


def ThreeD_Rope(q, k, pos_emb, edge_index=None):
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
    if edge_index is None:
        q_new = q * cos_pos[..., None, :] + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[..., None, :]
        k_new = k * cos_pos[..., None, :] + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[..., None, :]
    else:
        q_new = q * cos_pos[..., None, :] + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[..., None, :]
        k_new = k * cos_pos[edge_index][..., None, :] + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[edge_index][..., None, :]
    return q_new, k_new


class Transition(nn.Module):
    def __init__(self, in_features, norm, n=3, drop_path_r=0.):
        super().__init__()
        self.norm = norm(in_features)
        self.w1 = nn.Linear(in_features, in_features * n, bias=False)
        self.w2 = nn.Linear(in_features, in_features * n, bias=False)
        self.w3 = nn.Linear(in_features * n, in_features, bias=False)
        self.short = nn.Identity()
        self.drop_path = DropPath(drop_path_r) if drop_path_r > 0. else nn.Identity()

    def forward(self, x):
        y = self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))
        y = self.norm(self.drop_path(y) + self.short(x))
        return y


class AttentionWith3DRoPE(nn.Module):
    def __init__(self, in_features, attention_heads, attention_features, if_cross=False, drop_path_r=0., **kwargs):
        super().__init__()
        self.if_cross = if_cross
        self.y_shape = kwargs.get("y_shape", None)
        self.ifz = in_features
        self.ahz = attention_heads
        self.afz = attention_features
        self.attention_scale = np.sqrt(self.afz)
        self.q = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.k = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.v = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.back = nn.Sequential(
            Rearrange("B L ahz afz -> B L (ahz afz)", ahz=self.ahz, afz=self.afz),
            nn.Linear(self.ahz * self.afz, self.ifz, bias=False)
        )
        self.pos_encoding = SinusoidalPositionalEncoding(channels=attention_features // 3)
        self.norm1 = nn.LayerNorm(in_features)
        self.drop_path = DropPath(drop_path_r) if drop_path_r > 0. else nn.Identity()
        self.transition1 = Transition(in_features, nn.LayerNorm, drop_path_r=drop_path_r)

    def forward(self, x, y=None):
        B, C, H, D, W = x.shape
        pos_emb = torch.from_numpy(get_lattice_meshgrid_np((H, D, W), no_shift=True)).float().to(x.device)[None].repeat(B, 1, 1, 1, 1)
        pos_emb = einops.rearrange(pos_emb, "B H D W C -> B (H D W) C") * 1.5
        pos_emb = self.pos_encoding(pos_emb).flatten(-2)
        x_vec = einops.rearrange(x, "B C H D W -> B (H D W) C")
        query = self.q(x_vec)
        if self.if_cross:
            y_vec = einops.rearrange(y, "B C H D W -> B (H D W) C")
            key = self.k(y_vec)
            value = self.v(y_vec)
        else:
            key = self.k(x_vec)
            value = self.v(x_vec)
        query, key = ThreeD_Rope(query, key, pos_emb)
        attention_weights = (torch.einsum('blai,bkai->blka', query, key) / self.attention_scale).softmax(dim=-2)
        out = torch.einsum('blka,bkai->blai', attention_weights, value)
        out = self.norm1(x_vec + self.drop_path(self.back(out)))
        out = self.transition1(out)
        return einops.rearrange(out, "B (H D W) C -> B C H D W", B=B, C=C, H=H, D=D, W=W) + x


class FusionGateV2(nn.Module):
    def __init__(self, C, num_groups=8):
        super().__init__()
        self.gate = nn.Conv3d(C, C, 1, groups=num_groups)
        self.conv = nn.Conv3d(C, C, 3, padding=1)
        nn.init.constant_(self.gate.bias, 0.)

    def forward(self, ds_3, h):
        fused = ds_3
        g = torch.sigmoid(self.gate(fused))
        fused = fused + g * self.conv(h)
        return fused


class Inhead(nn.Module):
    def __init__(self, out_channels=256, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv3d(1, base_channels, 3, padding=1, bias=False)
        self.act1 = nn.ReLU()
        self.conv_out = nn.Conv3d(base_channels, out_channels, 1, bias=False)
        self.act_out = nn.ReLU()

    def forward(self, x):
        density = x[:, :1]
        feat = self.act1(self.conv1(density))
        feat = self.act_out(self.conv_out(feat))
        return feat


class MMVQUnet(nn.Module):
    def __init__(self, Cvae=256, vs=4096, n_heads=8, drop_path_rate=0.2, config=[4, 4, 4, 4, 4]):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0
        self.inhead = Inhead(out_channels=256, base_channels=64)
        self.downsample1 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample2 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample3 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.quantize = VectorQuantizer3(
            vocab_size=vs, Cvae=Cvae, using_znorm=False, beta=0.25,
            default_qresi_counts=0, v_patch_nums=(1, 2, 3, 4, 6), quant_resi=0.5,
            n_heads=n_heads, patience=128
        )
        self.main0 = nn.Sequential(*[AttentionWith3DRoPE(256, 8, 48, drop_path_r=dpr[i]) for i in range(config[0])])
        begin += config[0]
        self.fusion = FusionGateV2(Cvae)
        self.main1 = self.main_layer(256, 3, config[1], dpr[begin:begin + config[1]])
        begin += config[1]
        self.attn2 = AttentionGate(256, 256, 128)
        self.main2 = self.main_layer(128, 4, config[2], dpr[begin:begin + config[2]])
        begin += config[2]
        self.attn3 = AttentionGate(256, 128, 64)
        self.main3 = self.main_layer(64, 4, config[3], dpr[begin:begin + config[3]])
        begin += config[3]
        self.attn4 = AttentionGate(256, 64, 64)
        self.main4 = self.main_layer(64, 4, config[4], dpr[begin:begin + config[4]])
        self.input_info_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 2, kernel_size=1)
        )
        self.conv14 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv3d(64, 32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv3d(64, 32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv13 = nn.Conv3d(in_channels=32 * 3, out_channels=2, padding=1, kernel_size=3)

    def main_layer(self, input_channels, expansion, num_layers, dpr_lst):
        layer = []
        for i in range(num_layers):
            layer.append(Res2NetBlock(input_channels, input_channels, drop_path_r=dpr_lst[i], scale=expansion))
        return nn.Sequential(*layer)

    def forward(self, V0):
        ds_0 = self.inhead(V0)
        ds_1 = self.downsample1(ds_0)
        ds_2 = self.downsample2(ds_1)
        ds_3 = self.downsample3(ds_2)
        f_hat, avg_usage, vqloss = self.quantize(ds_3, ret_usages=True)
        ds_f = self.fusion(ds_3, f_hat)
        c3 = self.main1(self.main0(ds_f) + ds_f)
        c2 = self.main2(self.attn2(c3, ds_2))
        c1 = self.main3(self.attn3(c2, ds_1))
        c0 = self.main4(self.attn4(c1, ds_0))
        input_props = self.input_info_head(c0)
        input_mask = torch.sigmoid(input_props[:, 0:1])
        input_quality = torch.tanh(input_props[:, 1:2])
        c0_gated = c0 * input_mask
        f3 = self.conv14(c0_gated)
        f5 = self.conv11(c0_gated)
        f7 = self.conv12(c0_gated)
        f = torch.cat((f3, f5, f7), dim=1)
        f = self.relu1(f)
        f = self.conv13(f)
        return f[:, 0:1], torch.tanh(f[:, 1:2]), input_mask, input_quality, avg_usage, vqloss
