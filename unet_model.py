# --- unet_model.py ---
# This file defines the U-Net architecture, including the custom PartialConv3d layer.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint # Import checkpointing utility

# Custom Partial Convolution Layer for handling masked regions
class PartialConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 3D kernel for updating the 3D mask
        self.register_buffer('sum_kernel', torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        self.sum_kernel.requires_grad = False
        self.window_size = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]

    def forward(self, input_tensor, mask_in):
        # A more robust implementation of partial convolution to prevent shape mismatches.
        masked_input = input_tensor * mask_in
        output = super().forward(masked_input)

        with torch.no_grad():
            update_mask = F.conv3d(
                mask_in[:, :1, :, :, :], self.sum_kernel, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
            )
            mask_ratio = self.window_size / (update_mask + 1e-8)
            mask_out = (update_mask > 0).float()
            mask_ratio = torch.clamp(mask_ratio, 0.0, 1e4)

        corrected_output = output * mask_ratio
        final_output = corrected_output * mask_out
        final_mask = mask_out.repeat(1, self.out_channels, 1, 1, 1)
        return final_output, final_mask

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
        if self.time_mlp is not None and time_emb is not None:
            time_emb_reshaped = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            scale, shift = torch.chunk(time_emb_reshaped, 2, dim=1)
            h = h * (1 + scale) + shift
        h, h_mask = self.conv2(h, h_mask); h = self.norm2(h); h = self.act2(h); h = self.dropout(h)
        residual, residual_mask = self.residual_conv(x, mask) if isinstance(self.residual_conv, PartialConv3d) else (self.residual_conv(x), mask)
        
        if h.shape != residual.shape:
             residual = F.interpolate(residual, size=h.shape[-3:], mode='trilinear', align_corners=False)
             residual_mask = F.interpolate(residual_mask, size=h_mask.shape[-3:], mode='nearest')

        combined_mask = h_mask * residual_mask
        return (h + residual) * combined_mask, combined_mask

class SelfAttentionBlock(nn.Module):
    """A self-attention block for 3D data."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x, mask):
        N, C, D, H, W = x.shape
        x_norm = self.norm(x)
        q, k, v = self.to_qkv(x_norm).chunk(3, dim=1)
        q = rearrange(q, 'n (h c) d y x -> n h (d y x) c', h=self.num_heads)
        k = rearrange(k, 'n (h c) d y x -> n h (d y x) c', h=self.num_heads)
        v = rearrange(v, 'n (h c) d y x -> n h (d y x) c', h=self.num_heads)
        attn_scores = torch.einsum('nhic,nhjc->nhij', q, k) * (C // self.num_heads)**-0.5
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('nhij,nhjc->nhic', attn_probs, v)
        out = rearrange(out, 'n h (d y x) c -> n (h c) d y x', d=D, y=H, x=W)
        out = self.proj_out(out)
        return (x + out) * mask, mask

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
        k = rearrange(k, 'n (h c) -> n h () c', h=self.num_heads)
        v = rearrange(v, 'n (h c) -> n h () c', h=self.num_heads)
        
        attn_scores = torch.einsum('nhic,nhjc->nhij', q, k) * (q.shape[-1])**-0.5
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.einsum('nhij,nhjc->nhic', attn_probs, v)
        out = rearrange(out, 'n h (d y x) c -> n (h c) d y x', d=D, y=H, x=W)
        out = self.proj_out(out)
        return (x + out) * mask, mask

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
        # Using a larger horizontal kernel to capture larger spatial patterns,
        # while maintaining a balanced stride for gradual downsampling.
        self.downsample = PartialConv3d(out_channels, out_channels, kernel_size=(3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3))

    def forward(self, x, time_emb, mask, context=None, use_checkpointing=False):
        skip_outputs = []
        for res_block, (self_attn, cross_attn) in zip(self.res_blocks, self.attn_blocks):
            # Apply gradient checkpointing here 
            if use_checkpointing:
                x, mask = checkpoint(res_block, x, time_emb, mask)
            else:
                x, mask = res_block(x, time_emb, mask)
            
            if isinstance(self_attn, SelfAttentionBlock):
                x, mask = self_attn(x, mask)
            if isinstance(cross_attn, CrossAttentionBlock) and context is not None:
                x, mask = cross_attn(x, context, mask)
            skip_outputs.append(x)
        
        x_down, mask_down = self.downsample(x, mask)
        return x_down, mask_down, skip_outputs

class UpBlock(nn.Module):
    """An upsampling block for the 3D U-Net."""
    def __init__(self, in_channels, skip_channels_in, out_channels, time_emb_dim, context_dim, dropout_prob, has_attn=False, num_res_blocks=2):
        super().__init__()
        # Instead of ConvTranspose3d, we use a standard convolution.
        # The upsampling will be handled by F.interpolate in the forward pass.
        self.conv_after_upsample = PartialConv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        total_in_channels = out_channels + skip_channels_in
        for i in range(num_res_blocks):
            block_in_ch = total_in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(block_in_ch, out_channels, time_emb_dim, dropout_prob))
            if has_attn:
                self.attn_blocks.append(nn.ModuleList([SelfAttentionBlock(out_channels), CrossAttentionBlock(out_channels, context_dim)]))
            else:
                self.attn_blocks.append(nn.ModuleList([nn.Identity(), nn.Identity()]))

    def forward(self, x, skip_xs, time_emb, mask, context=None, use_checkpointing=False):
        # Get the spatial shape (D, H, W) from the corresponding skip connection tensor.
        # This is the target shape we need to upsample our current tensor 'x' to.
        skip_ref = skip_xs[0]
        target_shape = skip_ref.shape[-3:]

        # Upsample the input tensor 'x' and its mask to the target shape.
        # F.interpolate is used instead of ConvTranspose3d to avoid checkerboard artifacts
        # and to guarantee the output size matches the skip connection's size perfectly.
        x = F.interpolate(x, size=target_shape, mode='trilinear', align_corners=False)
        mask = F.interpolate(mask, size=target_shape, mode='nearest')
        
        # Now that 'x' has the same spatial dimensions as the skip connections,
        # we can apply a convolution.
        x, mask = self.conv_after_upsample(x, mask)
        
        # Concatenate the upsampled tensor with all the skip connections from the corresponding DownBlock.
        # This is guaranteed to work because we explicitly matched the sizes above.
        x = torch.cat([x] + skip_xs, dim=1)

        mask = mask[:, :1, :, :, :].repeat(1, x.shape[1], 1, 1, 1)

        for res_block, (self_attn, cross_attn) in zip(self.res_blocks, self.attn_blocks):
            if use_checkpointing:
                x, mask = checkpoint(res_block, x, time_emb, mask)
            else:
                x, mask = res_block(x, time_emb, mask)

            if isinstance(self_attn, SelfAttentionBlock):
                x, mask = self_attn(x, mask)
            if isinstance(cross_attn, CrossAttentionBlock) and context is not None:
                x, mask = cross_attn(x, context, mask)
        return x, mask

class UNet(nn.Module):
    def __init__(self, config, verbose_init=False):
        super().__init__()
        self.in_channels = config.channels
        self.out_channels = config.channels
        self.location_embedding_channels = config.location_embedding_channels
        self.conditioning_configs = config.conditioning_configs
        self.verbose_init = verbose_init
        self.has_logged_forward = False
        #  Add flag for checkpointing
        self.use_checkpointing = getattr(config, 'use_checkpointing', False)
        
        base_channels = config.base_unet_channels
        time_embedding_dim = base_channels * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, time_embedding_dim), nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        self.condition_mlps = nn.ModuleDict()
        total_condition_dim = 0
        if self.conditioning_configs:
            for name, cfg in self.conditioning_configs.items():
                emb_dim = cfg['dim']
                self.condition_mlps[name] = nn.Sequential(nn.Linear(1, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
                total_condition_dim += emb_dim
        self.context_dim = total_condition_dim

        initial_conv_in_channels = self.in_channels + self.location_embedding_channels
        self.initial_conv = PartialConv3d(initial_conv_in_channels, base_channels, kernel_size=3, padding=1)

        self.down_stages = nn.ModuleList()
        current_channels = base_channels
        channel_multipliers = getattr(config, 'channel_multipliers', (1, 2, 4, 8))
        num_res_blocks = getattr(config, 'num_res_blocks', 2)
        attn_resolutions = getattr(config, 'attn_resolutions', (16,))
        
        d, h, w = config.data_shape

        for i, multiplier in enumerate(channel_multipliers):
            out_ch = base_channels * multiplier
            has_attn = (h // (2**i)) in attn_resolutions
            self.down_stages.append(DownBlock(
                current_channels, out_ch, time_embedding_dim, self.context_dim,
                config.dropout_prob, has_attn, num_res_blocks
            ))
            current_channels = out_ch
        
        self.bottleneck = nn.ModuleList([
            ResidualBlock(current_channels, current_channels, time_embedding_dim, config.dropout_prob),
            SelfAttentionBlock(current_channels),
            ResidualBlock(current_channels, current_channels, time_embedding_dim, config.dropout_prob)
        ])

        self.up_stages = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            multiplier = channel_multipliers[i]
            in_ch = channel_multipliers[i+1] * base_channels if i + 1 < len(channel_multipliers) else current_channels
            out_ch = base_channels * multiplier
            skip_ch_in = out_ch * num_res_blocks
            has_attn = (h // (2**i)) in attn_resolutions
            
            self.up_stages.append(UpBlock(
                in_ch, skip_ch_in, out_ch, time_embedding_dim, self.context_dim,
                config.dropout_prob, has_attn, num_res_blocks
            ))
            current_channels = out_ch

        self.final_conv = nn.Conv3d(base_channels, self.out_channels, kernel_size=1)

    def forward(self, x, t, mask, conditions=None, location_field=None, verbose_forward=True):
        is_main_process = not x.device.type == 'cuda' or x.device.index == 0
        should_log = verbose_forward and not self.has_logged_forward

        if should_log and is_main_process:
            print("\n--- UNet Forward Pass (Actual Shapes) ---")

        if self.location_embedding_channels > 0 and location_field is not None:
            location_field_3d = location_field.unsqueeze(2).repeat(1, 1, x.shape[2], 1, 1)
            x = torch.cat((x, location_field_3d), dim=1)
            if should_log and is_main_process: print(f"After Location Concat: {x.shape}")
        
        time_emb = self.time_mlp(t)
        
        context = None
        if conditions is not None and self.context_dim > 0:
            context_embeddings = [self.condition_mlps[name](value.unsqueeze(-1).float() if value.dim() == 1 else value.float()) for name, value in conditions.items() if name in self.condition_mlps]
            if context_embeddings:
                context = torch.cat(context_embeddings, dim=1)

        initial_mask = mask.repeat(1, x.shape[1], 1, 1, 1)
        x, current_mask = self.initial_conv(x, initial_mask)
        if should_log and is_main_process: print(f"After Initial Conv:    {x.shape}")
        
        skip_connections = []
        for i, stage in enumerate(self.down_stages):
            if should_log and is_main_process: print(f"  [Down Stage {i+1}] Input:  {x.shape}")
            x, current_mask, skips = stage(x, time_emb, current_mask, context, use_checkpointing=self.use_checkpointing)
            if should_log and is_main_process:
                print(f"  [Down Stage {i+1}] Skips:  {[s.shape for s in skips]}")
                print(f"  [Down Stage {i+1}] Output: {x.shape}")
            skip_connections.append(skips)
        
        if should_log and is_main_process: print(f"Bottleneck Input:        {x.shape}")
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock):
                if self.use_checkpointing:
                    x, current_mask = checkpoint(layer, x, time_emb, current_mask)
                else:
                    x, current_mask = layer(x, time_emb, current_mask)
            elif isinstance(layer, SelfAttentionBlock):
                x, current_mask = layer(x, current_mask)
        if should_log and is_main_process: print(f"Bottleneck Output:       {x.shape}")

        for i, stage in enumerate(self.up_stages):
            if should_log and is_main_process: print(f"  [Up Stage {i+1}] Input:    {x.shape}")
            skips_for_stage = skip_connections.pop()
            if should_log and is_main_process: print(f"  [Up Stage {i+1}] Using Skips: {[s.shape for s in skips_for_stage]}")
            x, current_mask = stage(x, skips_for_stage, time_emb, current_mask, context, use_checkpointing=self.use_checkpointing)
            if should_log and is_main_process: print(f"  [Up Stage {i+1}] Output:   {x.shape}")

        output_prediction = self.final_conv(x)
        
        if should_log:
            if is_main_process:
                print(f"After Final Conv:      {output_prediction.shape}")
                print("-----------------------------------------")
            # Set the flag on ALL replicas after the first pass.
            # On the next forward pass, should_log will be False for everyone.
            self.has_logged_forward = True
        
        return output_prediction * mask

def count_parameters(model):
    """ Function to count trainable parameters in a model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
