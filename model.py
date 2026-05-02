from typing import Self, Sequence

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        pre_norm=None,
        post_norm=None,
        pre_permute=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SquaredReLU(nn.Module):
    """
    Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(
        self,
        scale_value=1.0,
        bias_value=0.0,
        scale_learnable=True,
        bias_learnable=True,
        mode=None,
        inplace=False,
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(
            scale_value * torch.ones(1), requires_grad=scale_learnable
        )
        self.bias = nn.Parameter(
            bias_value * torch.ones(1), requires_grad=bias_learnable
        )

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(
        self,
        dim,
        head_dim=32,
        num_heads=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        proj_bias=False,
        **kwargs,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerNormGeneral(nn.Module):
    def __init__(
        self, affine_shape=None, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5
    ):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(
            x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps
        )


class SepConv(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act1_layer=StarReLU,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class Mlp(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=StarReLU,
        drop=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        # drop_probs =

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    def __init__(
        self,
        dim,
        num_classes=1000,
        mlp_ratio=4,
        act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm,
        head_dropout=0.0,
        bias=True,
    ):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        token_mixer=nn.Identity,
        mlp=Mlp,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        layer_scale_init_value=None,
        res_scale_init_value=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)

        self.layer_scale1 = (
            Scale(dim=dim, init_value=layer_scale_init_value)
            if layer_scale_init_value
            else nn.Identity()
        )
        self.res_scale1 = (
            Scale(dim=dim, init_value=res_scale_init_value)
            if res_scale_init_value
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.layer_scale2 = (
            Scale(dim=dim, init_value=layer_scale_init_value)
            if layer_scale_init_value
            else nn.Identity()
        )
        self.res_scale2 = (
            Scale(dim=dim, init_value=res_scale_init_value)
            if res_scale_init_value
            else nn.Identity()
        )

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(self.token_mixer(self.norm1(x)))
        x = self.res_scale2(x) + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


DOWNSAMPLE_LAYERS_FOUR_STAGES = [
    partial(
        Downsampling,
        kernel_size=7,
        stride=4,
        padding=2,
        post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
    )
] + [
    partial(
        Downsampling,
        kernel_size=3,
        stride=2,
        padding=1,
        pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
        pre_permute=True,
    )
] * 3


class MetaFormer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        depths=(3, 12, 18, 3),
        dims=(96, 192, 384, 576),
        token_mixers=(SepConv, SepConv, Attention, Attention),
        dulbrn=16,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        res_scale_init_values=(None, None, 1.0, 1.0),
    ):
        super().__init__()
        self.dulbrn = dulbrn
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.dulbrn, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dulbrn, self.dulbrn * 2, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_channels = [self.dulbrn, self.dulbrn * 2]
        self.out_channels.extend(dims[:-1])
        num_stage = len(depths) - 1
        self.num_stage = num_stage
        down_dims = [in_chans] + list(dims)
        self.downsample_layers = nn.ModuleList(
            [
                DOWNSAMPLE_LAYERS_FOUR_STAGES[i](down_dims[i], down_dims[i + 1])
                for i in range(num_stage)
            ]
        )
        mlps = [mlps] * num_stage
        norm_layers = [norm_layers] * num_stage
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers[i],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i],
                        res_scale_init_value=res_scale_init_values[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        features = [f1, f2]
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x.permute(0, 3, 1, 2))

        return features


def fuse_conv_bn(conv, bn):
    W = conv.weight
    if conv.bias is None:
        b = torch.zeros(W.size(0), device=W.device)
    else:
        b = conv.bias

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)

    W_fused = W * (gamma / std).reshape(-1, 1, 1, 1)
    b_fused = (b - mean) * (gamma / std) + beta

    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=True,
    )

    fused_conv.weight.data = W_fused
    fused_conv.bias.data = b_fused

    return fused_conv


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)
        self.eval_conv = None
        super(Conv2dReLU, self).__init__(conv, bn, relu)

    def train(self, mode: bool = True) -> Self:
        super(Conv2dReLU, self).train(mode)
        if not mode:
            self.eval_conv = fuse_conv_bn(self[0], self[1])

    def forward(self, input):
        if self.training:
            return super().forward(input)
        else:
            return F.relu_(self.eval_conv(input))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.size()[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)

        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
    ):
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        self.conv41 = DecoderBlock(
            encoder_channels[0], encoder_channels[1], decoder_channels[1]
        )

        self.conv31 = DecoderBlock(
            encoder_channels[1], encoder_channels[2], decoder_channels[2]
        )
        self.conv32 = DecoderBlock(
            decoder_channels[1], decoder_channels[2], decoder_channels[2]
        )

        self.conv21 = DecoderBlock(
            encoder_channels[2], encoder_channels[3], decoder_channels[3]
        )
        self.conv22 = DecoderBlock(
            decoder_channels[2], decoder_channels[3], decoder_channels[3]
        )
        self.conv23 = DecoderBlock(
            decoder_channels[2], decoder_channels[3], decoder_channels[3]
        )

        self.conv11 = DecoderBlock(
            encoder_channels[3], encoder_channels[4], decoder_channels[4]
        )
        self.conv12 = DecoderBlock(
            decoder_channels[3], decoder_channels[4], decoder_channels[4]
        )
        self.conv13 = DecoderBlock(
            decoder_channels[3], decoder_channels[4], decoder_channels[4]
        )
        self.conv14 = DecoderBlock(
            decoder_channels[3], decoder_channels[4], decoder_channels[4]
        )

    def forward(self, *features):
        features = list(features)
        features[0] = self.conv11(features[1], features[0])
        features[1] = self.conv21(features[2], features[1])
        features[2] = self.conv31(features[3], features[2])
        features[3] = self.conv41(features[4], features[3])

        features[0] = self.conv12(features[1], features[0])
        features[1] = self.conv22(features[2], features[1])
        features[2] = self.conv32(features[3], features[2])

        features[0] = self.conv13(features[1], features[0])
        features[1] = self.conv23(features[2], features[1])

        features[0] = self.conv14(features[1], features[0])
        return features[0]


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class DDN(nn.Module):
    def __init__(self, granu=-5, scales: Sequence[float] = (1.0,)):
        super(DDN, self).__init__()
        self.encoder = MetaFormer()
        self.granu = granu
        encoder_channels = self.encoder.out_channels
        self.decoder_channels = (256, 128, 64, 32, 16)
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels, decoder_channels=self.decoder_channels
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )
        self.decoder_1 = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=self.decoder_channels,
        )
        self.segmentation_head_1 = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )
        self.scales = scales
        self.forward_mode = (
            self.multi_scale
            if len(scales) > 1
            else self.one_scale
            if scales[0] != 1.0
            else self.real_forward
        )

    def real_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        pad_top = 2
        pad_bottom = 2 + pad_h
        pad_left = 2
        pad_right = 2 + pad_w
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        features = self.encoder(x)
        results = self.segmentation_head(self.decoder(*features))
        std = self.segmentation_head_1(self.decoder_1(*features))
        std = F.softplus(std)
        results = F.sigmoid(results + std * self.granu)
        results = results[:, :, pad_top : pad_top + H, pad_left : pad_left + W]
        return results

    def multi_scale(self, inp: torch.Tensor) -> torch.Tensor:
        b, _, h, w = inp.shape
        result = torch.zeros([b, 1, h, w]).to(inp)
        for scale in self.scales:

            x = F.interpolate(
                inp,
                scale_factor=scale,
                mode="bilinear",
                antialias=True,
                align_corners=True,
            )
            x = self.real_forward(x)

            x = F.interpolate(
                x, size=(h, w), mode="bilinear", antialias=True, align_corners=True
            )
            result = result + x
        return result/len(self.scales)

    def one_scale(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = F.interpolate(
            x,
            scale_factor=self.scales[0],
            mode="bilinear",
            antialias=True,
            align_corners=True,
        )
        x = self.real_forward(x)
        return F.interpolate(
            x, size=(h, w), mode="bilinear", antialias=True, align_corners=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_mode(x)


def ddn_init(granu=-5, scales: Sequence[float] = (1.0,)):
    model = DDN(granu, scales)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://github.com/umzi2/DDN_clean/releases/download/v1.0.0/BSDS-best-checkpoint.pth"
    )
    model.load_state_dict(state_dict["state_dict"])
    return model
