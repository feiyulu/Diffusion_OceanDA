# --- unet_model.py ---
# This file defines the U-Net architecture, including the custom PartialConv2d layer.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Custom Partial Convolution Layer for handling masked regions
class PartialConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 3D kernel for updating the 3D mask
        self.register_buffer('sum_kernel', torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        self.sum_kernel.requires_grad = False
        self.window_size = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]

    def forward(self, input_tensor, mask_in):
        # Apply 3D convolution
        output = super().forward(input_tensor)
        
        # Update the 3D mask
        with torch.no_grad():
            mask_for_kernel = mask_in[:, :1, :, :, :]
            mask_padded = F.pad(mask_for_kernel, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode='constant', value=1)
            mask_out = F.conv3d(mask_padded, self.sum_kernel, bias=None, stride=self.stride, padding=0, dilation=self.dilation, groups=1)
            mask_out = mask_out / self.window_size
            mask_out = (mask_out > 0).float()
            
        output = output * mask_out
        return output, mask_out

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)))
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ResidualBlock(nn.Module):
    """A residual block with two 3D partial convolutions and time conditioning."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout_prob):
        super().__init__()
        self.conv1 = PartialConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels * 2) if time_emb_dim > 0 else None
        self.conv2 = PartialConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.residual_conv = PartialConv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, time_emb, mask):
        h, h_mask = self.conv1(x, mask)
        h = self.norm1(h); h = self.act1(h); h = self.dropout(h)
        if self.time_mlp is not None:
            # Reshape time embedding for 3D broadcasting (add D, H, W dimensions)
            time_emb_reshaped = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            scale, shift = torch.chunk(time_emb_reshaped, 2, dim=1)
            h = h * (1 + scale) + shift
        h, h_mask = self.conv2(h, h_mask); h = self.norm2(h); h = self.act2(h); h = self.dropout(h)
        residual, residual_mask = self.residual_conv(x, mask) if isinstance(self.residual_conv, PartialConv3d) else (self.residual_conv(x), mask)
        combined_mask = h_mask * residual_mask
        return (h + residual) * combined_mask, combined_mask

class SelfAttentionBlock(nn.Module):
    """A self-attention block for 3D data."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        # Use Conv3d for projections
        self.to_qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x, mask):
        N, C, D, H, W = x.shape
        x_norm = self.norm(x)
        q, k, v = self.to_qkv(x_norm).chunk(3, dim=1)
        # Rearrange for 3D multi-head attention, flattening the D, H, W dimensions
        q = rearrange(q, 'n (h c) d y x -> n h (d y x) c', h=self.num_heads)
        k = rearrange(k, 'n (h c) d y x -> n h (d y x) c', h=self.num_heads)
        v = rearrange(v, 'n (h c) d y x -> n h (d y x) c', h=self.num_heads)
        attn_scores = torch.einsum('nhic,nhjc->nhij', q, k) * (C // self.num_heads)**-0.5
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('nhij,nhjc->nhic', attn_probs, v)
        # Rearrange back to 3D spatial dimensions
        out = rearrange(out, 'n h (d y x) c -> n (h c) d y x', d=D, y=H, x=W)
        out = self.proj_out(out)
        return (x + out) * mask[:, :1, :, :, :].repeat(1, C, 1, 1, 1), mask

class CrossAttentionBlock(nn.Module):
    """A cross-attention block for 3D data."""
    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv3d(channels, channels, 1)
        self.to_kv = nn.Linear(context_dim, channels * 2)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x, context, mask):
        N, C, D, H, W = x.shape
        x_norm = self.norm(x)
        q = self.to_q(x_norm)
        q = rearrange(q, 'n (h c) d y x -> n h (d y x) c', h=self.num_heads)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = rearrange(k, 'n (h c) -> n h c', h=self.num_heads)
        v = rearrange(v, 'n (h c) -> n h c', h=self.num_heads)
        
        attn_scores = torch.einsum('nhic,nhjc->nhij', q, k.unsqueeze(2)) * (C // self.num_heads)**-0.5
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('nhij,nhjc->nhic', attn_probs, v.unsqueeze(2))
        out = rearrange(out, 'n h (d y x) c -> n (h c) d y x', d=D, y=H, x=W)
        out = self.proj_out(out)
        return (x + out) * mask[:, :1, :, :, :].repeat(1, C, 1, 1, 1), mask

class DownBlock(nn.Module):
    """A downsampling block for the 3D U-Net."""
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim, dropout_prob, has_attn=False, num_res_blocks=2):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            block_in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(block_in_ch, out_channels, time_emb_dim, dropout_prob))
            if has_attn:
                self.attn_blocks.append(nn.ModuleList([SelfAttentionBlock(out_channels), CrossAttentionBlock(out_channels, context_dim)]))
            else:
                self.attn_blocks.append(nn.ModuleList([nn.Identity(), nn.Identity()]))
        # Downsample only in horizontal dimensions (H, W) to preserve vertical resolution
        self.downsample = PartialConv3d(out_channels, out_channels, kernel_size=3, stride=(1, 2, 2), padding=1)

    def forward(self, x, time_emb, mask, context=None):
        for res_block, (self_attn, cross_attn) in zip(self.res_blocks, self.attn_blocks):
            x, mask = res_block(x, time_emb, mask)
            if isinstance(self_attn, SelfAttentionBlock):
                x, mask = self_attn(x, mask)
            if isinstance(cross_attn, CrossAttentionBlock) and context is not None:
                x, mask = cross_attn(x, context, mask)
        skip_x, skip_mask = x, mask
        x_down, mask_down = self.downsample(x, mask)
        return x_down, mask_down, skip_x, skip_mask

class UpBlock(nn.Module):
    """An upsampling block for the 3D U-Net."""
    def __init__(self, in_channels_from_prev_up, skip_channels, out_channels, time_emb_dim, context_dim, dropout_prob, has_attn=False, num_res_blocks=2):
        super().__init__()
        # Upsample only in horizontal dimensions (H, W)
        self.upsample_conv = nn.ConvTranspose3d(in_channels_from_prev_up, out_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            block_in_ch = (out_channels + skip_channels) if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(block_in_ch, out_channels, time_emb_dim, dropout_prob))
            if has_attn:
                self.attn_blocks.append(nn.ModuleList([SelfAttentionBlock(out_channels), CrossAttentionBlock(out_channels, context_dim)]))
            else:
                self.attn_blocks.append(nn.ModuleList([nn.Identity(), nn.Identity()]))

    def forward(self, x, skip_x, time_emb, mask_from_prev_up, skip_mask, context=None):
        x_upsampled = self.upsample_conv(x)
        # Handle potential size mismatches from non-power-of-2 dimensions
        if x_upsampled.shape[-3:] != skip_x.shape[-3:]:
            x_upsampled = F.interpolate(x_upsampled, size=skip_x.shape[-3:], mode='trilinear', align_corners=False)
        
        mask_upsampled = F.interpolate(mask_from_prev_up[:,:1,:,:,:], size=x_upsampled.shape[-3:], mode='nearest')
        mask_upsampled = (mask_upsampled > 0.5).float().repeat(1, x_upsampled.shape[1], 1, 1, 1)
        
        x_combined = torch.cat((x_upsampled, skip_x), dim=1)
        combined_mask = mask_upsampled * skip_mask
        current_x, current_mask = x_combined, combined_mask
        for res_block, (self_attn, cross_attn) in zip(self.res_blocks, self.attn_blocks):
            current_x, current_mask = res_block(current_x, time_emb, current_mask)
            if isinstance(self_attn, SelfAttentionBlock):
                current_x, current_mask = self_attn(current_x, current_mask)
            if isinstance(cross_attn, CrossAttentionBlock) and context is not None:
                current_x, current_mask = cross_attn(current_x, context, current_mask)
        return current_x, current_mask

class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=32,
                 time_embedding_dim=256,
                 conditioning_configs={},
                 location_embedding_channels=2,
                 channel_multipliers=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attn_resolutions=(8,),
                 dropout_prob=0.1,
                 verbose_init=False):
        super().__init__()

        self.location_embedding_channels = location_embedding_channels
        self.conditioning_configs = conditioning_configs

        # 1. Handle time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4), nn.GELU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )

        # 2. Setup MLPs for each scalar conditioner dynamically
        self.condition_mlps = nn.ModuleDict()
        total_condition_dim = 0
        for name, config in self.conditioning_configs.items():
            emb_dim = config['dim']
            self.condition_mlps[name] = nn.Sequential(
                nn.Linear(1, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)
            )
            total_condition_dim += emb_dim
        
        # This context_dim is what the CrossAttentionBlocks will use
        self.context_dim = total_condition_dim

        # 3. Initial convolution (handles concatenated location embedding)
        initial_conv_in_channels = in_channels + location_embedding_channels
        self.initial_conv = PartialConv3d(initial_conv_in_channels, base_channels, kernel_size=3, padding=1)

        # 4. Downsampling path
        self.down_stages = nn.ModuleList()
        current_channels = base_channels
        for i, multiplier in enumerate(channel_multipliers):
            out_channels_stage = base_channels * multiplier
            self.down_stages.append(DownBlock(
                current_channels, out_channels_stage, time_embedding_dim, self.context_dim,
                dropout_prob, multiplier in attn_resolutions, num_res_blocks
            ))
            current_channels = out_channels_stage
        
        # 5. Bottleneck
        self.bottleneck_res1 = ResidualBlock(current_channels, current_channels, time_embedding_dim, dropout_prob)
        if channel_multipliers[-1] in attn_resolutions:
            self.bottleneck_self_attn = SelfAttentionBlock(current_channels)
            self.bottleneck_cross_attn = CrossAttentionBlock(current_channels, self.context_dim)
        else:
            self.bottleneck_self_attn = self.bottleneck_cross_attn = nn.Identity()
        self.bottleneck_res2 = ResidualBlock(current_channels, current_channels, time_embedding_dim, dropout_prob)

        # 6. Upsampling path
        self.up_stages = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            multiplier = channel_multipliers[i]
            skip_channels = base_channels * multiplier
            out_channels_stage = base_channels * multiplier
            self.up_stages.append(UpBlock(
                current_channels, skip_channels, out_channels_stage, time_embedding_dim, self.context_dim,
                dropout_prob, multiplier in attn_resolutions, num_res_blocks + 1
            ))
            current_channels = out_channels_stage

        # 7. Final layers
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()
        self.output_conv = PartialConv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, mask, conditions=None, location_field=None, verbose_forward=False):
        # x: (N, C, H, W), t: (N,), mask: (N, 1, H, W)
        # conditions: dict, e.g., {'dayofyear': (N,), 'co2': (N,)}
        # location_field: (N, L, H, W)
        
        current_x, current_mask = x, mask

        # Concatenate location embedding if used
        if self.location_embedding_channels > 0 and location_field is not None:
            location_field_3d = location_field.unsqueeze(2).repeat(1, 1, x.shape[2], 1, 1)
            current_x = torch.cat((current_x, location_field_3d), dim=1)
            current_mask = mask.repeat(1, current_x.shape[1], 1, 1, 1)
        
        # 1. Process time embedding
        time_emb = self.time_mlp(t)
        
        # 2. Process all scalar conditions to create a single context vector for cross-attention
        context_embeddings = []
        if conditions is not None and self.context_dim > 0:
            for name, value in conditions.items():
                if name in self.condition_mlps:
                    value = value.unsqueeze(1).float() if value.dim() == 1 else value.float()
                    context_embeddings.append(self.condition_mlps[name](value))
            context = torch.cat(context_embeddings, dim=1) if context_embeddings else None
        else:
            context = None

        # --- U-Net Path ---
        current_x, current_mask = self.initial_conv(current_x, current_mask)
        skip_connections, masks_for_skip = [], []

        # Downsampling
        for stage in self.down_stages:
            current_x, current_mask, skip_x, skip_mask = stage(current_x, time_emb, current_mask, context)
            skip_connections.append(skip_x); masks_for_skip.append(skip_mask)
        
        # Bottleneck
        current_x, current_mask = self.bottleneck_res1(current_x, time_emb, current_mask)
        if isinstance(self.bottleneck_self_attn, SelfAttentionBlock):
            current_x, current_mask = self.bottleneck_self_attn(current_x, current_mask)
        if isinstance(self.bottleneck_cross_attn, CrossAttentionBlock) and context is not None:
            current_x, current_mask = self.bottleneck_cross_attn(current_x, context, current_mask)
        current_x, current_mask = self.bottleneck_res2(current_x, time_emb, current_mask)

        # Upsampling
        for stage in self.up_stages:
            skip_x, skip_mask = skip_connections.pop(), masks_for_skip.pop()
            current_x, current_mask = stage(current_x, skip_x, time_emb, current_mask, skip_mask, context)

        # Final output
        current_x = self.final_norm(self.final_act(current_x))
        output_prediction, _ = self.output_conv(current_x, current_mask)
        
        # Ensure final output respects the original mask
        final_mask = mask.repeat(1, output_prediction.shape[1], 1, 1)
        return output_prediction * final_mask

def count_parameters(model):
    """ Function to count trainable parameters in a model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
