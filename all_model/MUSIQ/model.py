import torch
from torch import nn as nn
from torch.nn import functional as F

from builtins import ValueError
from collections import OrderedDict
import math
import collections.abc
from itertools import repeat
import numpy as np
from typing import Tuple

def extract_image_patches(x, kernel, stride=1, dilation=1):
    """
    Ref: https://stackoverflow.com/a/65886666
    """
    # Do TF 'SAME' Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2))

    # Extract patches
    patches = F.unfold(x, kernel, dilation, stride=stride)
    return patches


def _ceil_divide_int(x, y):
    """Returns ceil(x / y) as int"""
    return int(math.ceil(x / y))


def resize_preserve_aspect_ratio(image, h, w, longer_side_length):
    """Aspect-ratio-preserving resizing with tf.image.ResizeMethod.GAUSSIAN.
    Args:
      image: The image tensor (n_crops, c, h, w).
      h: Height of the input image.
      w: Width of the input image.
      longer_side_length: The length of the longer side after resizing.
    Returns:
      A tuple of [Image after resizing, Resized height, Resized width].
    """
    # Computes the height and width after aspect-ratio-preserving resizing.
    ratio = longer_side_length / max(h, w)
    rh = round(h * ratio)
    rw = round(w * ratio)

    resized = F.interpolate(image, (rh, rw), mode='bicubic')
    return resized, rh, rw


def _pad_or_cut_to_max_seq_len(x, max_seq_len):
    """Pads (or cuts) patch tensor `max_seq_len`.
    Args:
        x: input tensor of shape (n_crops, c, num_patches).
        max_seq_len: max sequence length.
    Returns:
        The padded or cropped tensor of shape (n_crops, c, max_seq_len).
    """
    # Shape of x (n_crops, c, num_patches)
    # Padding makes sure that # patches > max_seq_length. Note that it also
    # makes the input mask zero for shorter input.
    n_crops, c, num_patches = x.shape
    paddings = torch.zeros((n_crops, c, max_seq_len)).to(x)
    x = torch.cat([x, paddings], dim=-1)
    x = x[:, :, :max_seq_len]
    return x


def get_hashed_spatial_pos_emb_index(grid_size, count_h, count_w):
    """Get hased spatial pos embedding index for each patch.
    The size H x W is hashed to grid_size x grid_size.
    Args:
      grid_size: grid size G for the hashed-based spatial positional embedding.
      count_h: number of patches in each row for the image.
      count_w: number of patches in each column for the image.
    Returns:
      hashed position of shape (1, HxW). Each value corresponded to the hashed
      position index in [0, grid_size x grid_size).
    """
    pos_emb_grid = torch.arange(grid_size).float()

    pos_emb_hash_w = pos_emb_grid.reshape(1, 1, grid_size)
    pos_emb_hash_w = F.interpolate(pos_emb_hash_w, (count_w), mode='nearest')
    pos_emb_hash_w = pos_emb_hash_w.repeat(1, count_h, 1)

    pos_emb_hash_h = pos_emb_grid.reshape(1, 1, grid_size)
    pos_emb_hash_h = F.interpolate(pos_emb_hash_h, (count_h), mode='nearest')
    pos_emb_hash_h = pos_emb_hash_h.transpose(1, 2)
    pos_emb_hash_h = pos_emb_hash_h.repeat(1, 1, count_w)

    pos_emb_hash = pos_emb_hash_h * grid_size + pos_emb_hash_w

    pos_emb_hash = pos_emb_hash.reshape(1, -1)
    return pos_emb_hash


def _extract_patches_and_positions_from_image(image, patch_size, patch_stride, hse_grid_size, n_crops, h, w, c,
                                              scale_id, max_seq_len):
    """Extracts patches and positional embedding lookup indexes for a given image.
    Args:
      image: the input image of shape [n_crops, c, h, w]
      patch_size: the extracted patch size.
      patch_stride: stride for extracting patches.
      hse_grid_size: grid size for hash-based spatial positional embedding.
      n_crops: number of crops from the input image.
      h: height of the image.
      w: width of the image.
      c: number of channels for the image.
      scale_id: the scale id for the image in the multi-scale representation.
      max_seq_len: maximum sequence length for the number of patches. If
        max_seq_len = 0, no patch is returned. If max_seq_len < 0 then we return
        all the patches.
    Returns:
      A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
      is (n_crops, num_patches, patch_size * patch_size * c + 3).
    """
    n_crops, c, h, w = image.shape
    p = extract_image_patches(image, patch_size, patch_stride)
    assert p.shape[1] == c * patch_size**2

    count_h = _ceil_divide_int(h, patch_stride)
    count_w = _ceil_divide_int(w, patch_stride)

    # Shape (1, num_patches)
    spatial_p = get_hashed_spatial_pos_emb_index(hse_grid_size, count_h, count_w)
    # Shape (n_crops, 1, num_patches)
    spatial_p = spatial_p.unsqueeze(1).repeat(n_crops, 1, 1)
    scale_p = torch.ones_like(spatial_p) * scale_id
    mask_p = torch.ones_like(spatial_p)

    # Concatenating is a hacky way to pass both patches, positions and input
    # mask to the model.
    # Shape (n_crops, c * patch_size * patch_size + 3, num_patches)
    out = torch.cat([p, spatial_p.to(p), scale_p.to(p), mask_p.to(p)], dim=1)
    if max_seq_len >= 0:
        out = _pad_or_cut_to_max_seq_len(out, max_seq_len)
    return out


def get_multiscale_patches(image,
                           patch_size=32,
                           patch_stride=32,
                           hse_grid_size=10,
                           longer_side_lengths=[224, 384],
                           max_seq_len_from_original_res=None):
    """Extracts image patches from multi-scale representation.
    Args:
      image: input image tensor with shape [n_crops, 3, h, w]
      patch_size: patch size.
      patch_stride: patch stride.
      hse_grid_size: Hash-based positional embedding grid size.
      longer_side_lengths: List of longer-side lengths for each scale in the
        multi-scale representation.
      max_seq_len_from_original_res: Maximum number of patches extracted from
        original resolution. <0 means use all the patches from the original
        resolution. None means we don't use original resolution input.
    Returns:
      A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
      is (n_crops, num_patches, patch_size * patch_size * c + 3).
    """
    # Sorting the list to ensure a deterministic encoding of the scale position.
    longer_side_lengths = sorted(longer_side_lengths)

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    n_crops, c, h, w = image.shape

    outputs = []
    for scale_id, longer_size in enumerate(longer_side_lengths):
        resized_image, rh, rw = resize_preserve_aspect_ratio(image, h, w, longer_size)

        max_seq_len = int(np.ceil(longer_size / patch_stride)**2)
        out = _extract_patches_and_positions_from_image(resized_image, patch_size, patch_stride, hse_grid_size, n_crops,
                                                        rh, rw, c, scale_id, max_seq_len)
        outputs.append(out)

    if max_seq_len_from_original_res is not None:
        out = _extract_patches_and_positions_from_image(image, patch_size, patch_stride, hse_grid_size, n_crops, h, w,
                                                        c, len(longer_side_lengths), max_seq_len_from_original_res)
        outputs.append(out)

    outputs = torch.cat(outputs, dim=-1)
    return outputs.transpose(1, 2)

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)

def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """Convert distribution prediction to mos score.
    For datasets with detailed score labels, such as AVA
    Args:
        dist_score (tensor): (*, C), C is the class number
    Output:
        mos_score (tensor): (*, 1)
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score

def symm_pad(im: torch.Tensor, padding: Tuple[int, int, int, int]):
    """Symmetric padding same as tensorflow.
    Ref: https://discuss.pytorch.org/t/symmetric-padding/19866/3
    """
    h, w = im.shape[-2:]
    left, right, top, bottom = padding
 
    x_idx = np.arange(-left, w+right)
    y_idx = np.arange(-top, h+bottom)
 
    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2*rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod+double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w-0.5)
    y_pad = reflect(y_idx, -0.5, h-0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]

def excact_padding_2d(x, kernel, stride=1, dilation=1, mode='same'):
    assert len(x.shape) == 4, f'Only support 4D tensor input, but got {x.shape}'
    kernel = to_2tuple(kernel)
    stride = to_2tuple(stride)
    dilation = to_2tuple(dilation)
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    pad_l, pad_r, pad_t, pad_b = (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2)

    mode = mode if mode != 'same' else 'constant'
    if mode != 'symmetric':
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    elif mode == 'symmetric':
        x = symm_pad(x, (pad_l, pad_r, pad_t, pad_b)) 

    return x


class ExactPadding2d(nn.Module):
    r"""This function calculate exact padding values for 4D tensor inputs,
    and support the same padding mode as tensorflow.
    Args:
        kernel (int or tuple): kernel size.
        stride (int or tuple): stride size.
        dilation (int or tuple): dilation size, default with 1.
        mode (srt): padding mode can be ('same', 'symmetric', 'replicate', 'circular')
    """

    def __init__(self, kernel, stride=1, dilation=1, mode='same'):
        super().__init__()
        self.kernel = to_2tuple(kernel)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.mode = mode

    def forward(self, x):
        return excact_padding_2d(x, self.kernel, self.stride, self.dilation, self.mode)

class StdConv(nn.Conv2d):
    """
    Reference: https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def forward(self, x):
        # implement same padding
        x = excact_padding_2d(x, self.kernel_size, self.stride, mode='same')
        weight = self.weight
        weight = weight - weight.mean((1, 2, 3), keepdim=True)
        weight = weight / (weight.std((1, 2, 3), keepdim=True) + 1e-5)
        return F.conv2d(x, weight, self.bias, self.stride)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()

        width = inplanes

        self.conv1 = StdConv(inplanes, width, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, width, eps=1e-4)
        self.conv2 = StdConv(width, width, 3, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, width, eps=1e-4)
        self.conv3 = StdConv(width, outplanes, 1, 1, bias=False)
        self.gn3 = nn.GroupNorm(32, outplanes, eps=1e-4)

        self.relu = nn.ReLU(True)

        self.needs_projection = inplanes != outplanes or stride != 1
        if self.needs_projection:
            self.conv_proj = StdConv(inplanes, outplanes, 1, stride, bias=False)
            self.gn_proj = nn.GroupNorm(32, outplanes, eps=1e-4)

    def forward(self, x):
        identity = x
        if self.needs_projection:
            identity = self.gn_proj(self.conv_proj(identity))

        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        out = self.relu(x + identity)

        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads=6, bias=False, attn_drop=0., out_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask_h = mask.reshape(B, 1, N, 1)
            mask_w = mask.reshape(B, 1, 1, N)
            mask2d = mask_h * mask_w
            attn = attn.masked_fill(mask2d == 0, -1e3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.out_drop(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_dim,
                 num_heads,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attention = MultiHeadAttention(dim, num_heads, bias=True, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, inputs_masks):
        y = self.norm1(x)
        y = self.attention(y, inputs_masks)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AddHashSpatialPositionEmbs(nn.Module):
    """Adds learnable hash-based spatial embeddings to the inputs."""

    def __init__(self, spatial_pos_grid_size, dim):
        super().__init__()
        self.position_emb = nn.parameter.Parameter(torch.randn(1, spatial_pos_grid_size * spatial_pos_grid_size, dim))
        nn.init.normal_(self.position_emb, std=0.02)

    def forward(self, inputs, inputs_positions):
        return inputs + self.position_emb.squeeze(0)[inputs_positions.long()]


class AddScaleEmbs(nn.Module):
    """Adds learnable scale embeddings to the inputs."""

    def __init__(self, num_scales, dim):
        super().__init__()
        self.scale_emb = nn.parameter.Parameter(torch.randn(num_scales, dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    def forward(self, inputs, inputs_scale_positions):
        return inputs + self.scale_emb[inputs_scale_positions.long()]


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        mlp_dim=1152,
        attention_dropout_rate=0.,
        dropout_rate=0,
        num_heads=6,
        num_layers=14,
        num_scales=3,
        spatial_pos_grid_size=10,
        use_scale_emb=True,
        use_sinusoid_pos_emb=False,
    ):
        super().__init__()
        self.use_scale_emb = use_scale_emb
        self.posembed_input = AddHashSpatialPositionEmbs(spatial_pos_grid_size, input_dim)
        self.scaleembed_input = AddScaleEmbs(num_scales, input_dim)

        self.cls = nn.parameter.Parameter(torch.zeros(1, 1, input_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_norm = nn.LayerNorm(input_dim, eps=1e-6)

        self.transformer = nn.ModuleDict()
        for i in range(num_layers):
            self.transformer[f'encoderblock_{i}'] = TransformerBlock(input_dim, mlp_dim, num_heads, dropout_rate,
                                                                     attention_dropout_rate)

    def forward(self, x, inputs_spatial_positions, inputs_scale_positions, inputs_masks):
        n, _, c = x.shape

        x = self.posembed_input(x, inputs_spatial_positions)
        if self.use_scale_emb:
            x = self.scaleembed_input(x, inputs_scale_positions)

        cls_token = self.cls.repeat(n, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        cls_mask = torch.ones((n, 1)).to(inputs_masks)
        inputs_mask = torch.cat([cls_mask, inputs_masks], dim=1)
        x = self.dropout(x)

        for k, m in self.transformer.items():
            x = m(x, inputs_mask)
        x = self.encoder_norm(x)

        return x


class MUSIQ(nn.Module):
    r"""
    Evaluation:
        - n_crops: currently only test with 1 crop evaluation
    Reference:
        Ke, Junjie, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang.
        "Musiq: Multi-scale image quality transformer." In Proceedings of the
        IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5148-5157. 2021.
    """

    def __init__(
        self,
        patch_size=32,
        num_class=1,
        hidden_size=384,
        mlp_dim=1152,
        attention_dropout_rate=0.,
        dropout_rate=0,
        num_heads=6,
        num_layers=14,
        num_scales=3,
        spatial_pos_grid_size=10,
        use_scale_emb=True,
        use_sinusoid_pos_emb=False,
        # data opts
        longer_side_lengths=[224, 384],
        max_seq_len_from_original_res=-1,
    ):
        super(MUSIQ, self).__init__()

        resnet_token_dim = 64
        self.patch_size = patch_size

        self.data_preprocess_opts = {
            'patch_size': patch_size,
            'patch_stride': patch_size,
            'hse_grid_size': spatial_pos_grid_size,
            'longer_side_lengths': longer_side_lengths,
            'max_seq_len_from_original_res': max_seq_len_from_original_res
        }

        self.conv_root = StdConv(3, resnet_token_dim, 7, 2, bias=False)
        self.gn_root = nn.GroupNorm(32, resnet_token_dim, eps=1e-6)
        self.root_pool = nn.Sequential(
            nn.ReLU(True),
            ExactPadding2d(3, 2, mode='same'),
            nn.MaxPool2d(3, 2),
        )

        token_patch_size = patch_size // 4
        self.block1 = Bottleneck(resnet_token_dim, resnet_token_dim * 4)

        self.embedding = nn.Linear(resnet_token_dim * 4 * token_patch_size**2, hidden_size)
        self.transformer_encoder = TransformerEncoder(hidden_size, mlp_dim, attention_dropout_rate, dropout_rate,
                                                      num_heads, num_layers, num_scales, spatial_pos_grid_size,
                                                      use_scale_emb, use_sinusoid_pos_emb)

        if num_class > 1:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, num_class),
                nn.Softmax(dim=-1),
            )
        else:
            self.head = nn.Linear(hidden_size, num_class)

    def forward(self, x, return_mos=True, return_dist=False):
        x = (x - 0.5) * 2
        x = get_multiscale_patches(x, **self.data_preprocess_opts)

        assert len(x.shape) in [3, 4]
        if len(x.shape) == 4:
            b, num_crops, seq_len, dim = x.shape
            x = x.reshape(b * num_crops, seq_len, dim)
        else:
            b, seq_len, dim = x.shape
            num_crops = 1

        inputs_spatial_positions = x[:, :, -3]
        inputs_scale_positions = x[:, :, -2]
        inputs_masks = x[:, :, -1].bool()
        x = x[:, :, :-3]

        x = x.reshape(-1, 3, self.patch_size, self.patch_size)
        x = self.conv_root(x)
        x = self.gn_root(x)
        x = self.root_pool(x)
        x = self.block1(x)
        # to match tensorflow channel order
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, seq_len, -1)
        x = self.embedding(x)
        x = self.transformer_encoder(x, inputs_spatial_positions, inputs_scale_positions, inputs_masks)
        q = self.head(x[:, 0])

        q = q.reshape(b, num_crops, -1)
        q = q.mean(dim=1)  # for multiple crops evaluation
        mos = dist_to_mos(q)

        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(q)

        if len(return_list) > 1:
            return return_list
        else:
            return {
                'final_result': return_list[0]
            }


if __name__ == "__main__":
    model = MUSIQ().cuda()
    x = torch.rand((1,3,500,856)).cuda()
    y = model(x)
    print(y)