# --- unet_model.py ---
# This file defines the U-Net architecture, including the custom PartialConv3d layer.
# REFACTORED: Implemented a factorized pseudo-3D architecture with a configurable
# multi-encoder/multi-decoder design. Input channels can be grouped to share
# vertical encoders, allowing the model to learn both specialized and joint representations.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint 

# --- 2D Convolutional Layers (for the main U-Net) ---

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('sum_kernel', torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]))
        self.sum_kernel.requires_grad = False
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

    def forward(self, input_tensor, mask_in):
        masked_input = input_tensor * mask_in
        output = super().forward(masked_input)
        with torch.no_grad():
            update_mask = F.conv2d(
                mask_in[:, :1, :, :], self.sum_kernel, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
            )
            mask_ratio = self.window_size / (update_mask + 1e-8)
            mask_out = (update_mask > 0).float()
            mask_ratio = torch.clamp(mask_ratio, 0.0, 1e4)
        corrected_output = output * mask_ratio
        final_output = corrected_output * mask_out
        final_mask = mask_out.repeat(1, self.out_channels, 1, 1)
        return final_output, final_mask

# --- 1D Convolutional Layers (for Vertical Encoder/Decoder) ---

class PartialConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('sum_kernel', torch.ones(1, 1, self.kernel_size[0]))
        self.sum_kernel.requires_grad = False
        self.window_size = self.kernel_size[0]
    def forward(self, input_tensor, mask_in):
        masked_input = input_tensor * mask_in
        output = super().forward(masked_input)
        with torch.no_grad():
            update_mask = F.conv1d(
                mask_in[:, :1, :], self.sum_kernel, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
            )
            mask_ratio = self.window_size / (update_mask + 1e-8)
            mask_out = (update_mask > 0).float()
            mask_ratio = torch.clamp(mask_ratio, 0.0, 1e4)
        corrected_output = output * mask_ratio
        final_output = corrected_output * mask_out
        final_mask = mask_out.repeat(1, self.out_channels, 1)
        return final_output, final_mask

# --- Positional Embedding ---

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

# --- NEW: Vertical Encoder and Decoder Modules ---

class VerticalEncoder(nn.Module):
    """Encodes the vertical dimension of a GROUP of channels into a latent feature space."""
    def __init__(self, in_channels, latent_dim, depth_size):
        super().__init__()
        self.conv_in = PartialConv1d(in_channels, latent_dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.fc = nn.Linear(depth_size * latent_dim, latent_dim)
        
    def forward(self, x, mask):
        # x shape: (n, num_channels_in_group, d, h, w)
        n, c, d, h, w = x.shape
        x_reshaped = rearrange(x, 'n c d h w -> (n h w) c d')
        # Use the mask of the first channel in the group as representative
        mask_reshaped = rearrange(mask, 'n c d h w -> (n h w) c d')[:, :1, :]
        
        x_conv, mask_conv = self.conv_in(x_reshaped, mask_reshaped)
        x_act = self.act(x_conv) * mask_conv
        
        x_flat = rearrange(x_act, 'b c d -> b (c d)')
        x_encoded = self.fc(x_flat)
        
        output = rearrange(x_encoded, '(n h w) ld -> n ld h w', n=n, h=h, w=w)
        # The horizontal mask is the max projection of the multi-channel 3D mask
        horizontal_mask = (torch.sum(mask, dim=(1,2)) > 0).float().unsqueeze(1)
        return output, horizontal_mask

class VerticalDecoder(nn.Module):
    """Decodes from a latent space back to the physical vertical dimension for a GROUP of channels."""
    def __init__(self, latent_dim, out_channels, depth_size):
        super().__init__()
        self.fc = nn.Linear(latent_dim, latent_dim * depth_size)
        self.act = nn.SiLU()
        self.conv_out = PartialConv1d(latent_dim, out_channels, kernel_size=3, padding=1)
        self.depth_size = depth_size
        self.latent_dim = latent_dim

    def forward(self, x, original_mask_3d_group):
        n, ld, h, w = x.shape
        x_reshaped = rearrange(x, 'n ld h w -> (n h w) ld')
        
        x_fc = self.fc(x_reshaped)
        x_unflat = rearrange(x_fc, 'b (ld d) -> b ld d', d=self.depth_size, ld=self.latent_dim)
        x_act = self.act(x_unflat)

        # Use the representative mask for the group, reshaped for the 1D conv
        mask_reshaped = rearrange(original_mask_3d_group, 'n c d h w -> (n h w) c d')[:, :1, :]
        
        x_decoded, _ = self.conv_out(x_act, mask_reshaped)
        output = rearrange(x_decoded, '(n h w) c d -> n c d h w', n=n, h=h, w=w)
        return output

# --- Core 2D U-Net Components (Unchanged) ---

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, dropout_prob):
        super().__init__()
        self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_channels % 8 == 0 else out_channels, out_channels)
        self.act1 = nn.SiLU()
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels * 2) if time_embedding_dim > 0 else None
        self.conv2 = PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8 if out_channels % 8 == 0 else out_channels, out_channels)
        self.act2 = nn.SiLU()
        self.residual_conv = PartialConv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, time_emb, mask):
        h, h_mask = self.conv1(x, mask)
        h = self.norm1(h); h = self.act1(h); h = self.dropout(h)
        if self.time_mlp is not None and time_emb is not None:
            time_emb_reshaped = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            scale, shift = torch.chunk(time_emb_reshaped, 2, dim=1)
            h = h * (1 + scale) + shift
        h, h_mask = self.conv2(h, h_mask); h = self.norm2(h); h = self.act2(h); h = self.dropout(h)
        residual, residual_mask = self.residual_conv(x, mask) if isinstance(self.residual_conv, PartialConv2d) else (self.residual_conv(x), mask)
        combined_mask = h_mask * residual_mask
        return (h + residual) * combined_mask, combined_mask

class SelfAttentionBlock2D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else channels, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, mask):
        N, C, H, W = x.shape
        x_norm = self.norm(x)
        q, k, v = self.to_qkv(x_norm).chunk(3, dim=1)
        q = rearrange(q, 'n (h c) y x -> n h (y x) c', h=self.num_heads)
        k = rearrange(k, 'n (h c) y x -> n h (y x) c', h=self.num_heads)
        v = rearrange(v, 'n (h c) y x -> n h (y x) c', h=self.num_heads)
        attn_scores = torch.einsum('nhic,nhjc->nhij', q, k) * (C // self.num_heads)**-0.5
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('nhij,nhjc->nhic', attn_probs, v)
        out = rearrange(out, 'n h (y x) c -> n (h c) y x', y=H, x=W)
        out = self.proj_out(out)
        return (x + out) * mask, mask

class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, dropout_prob, has_attn=False, num_res_blocks=2):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlock2D(in_channels if i == 0 else out_channels, out_channels, time_embedding_dim, dropout_prob)
            for i in range(num_res_blocks)
        ])
        self.attn_block = SelfAttentionBlock2D(out_channels) if has_attn else nn.Identity()
        self.downsample = PartialConv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_emb, mask, use_checkpointing=False, should_log=False, stage_name=""):
        skip_outputs = []
        for i, res_block in enumerate(self.res_blocks):
            x, mask = checkpoint(res_block, x, time_emb, mask, use_reentrant=False) if use_checkpointing else res_block(x, time_emb, mask)
            if should_log: print(f"    {stage_name} ResBlock {i+1} Out: {x.shape}")
            skip_outputs.append(x)
        if isinstance(self.attn_block, SelfAttentionBlock2D):
            x, mask = self.attn_block(x, mask)
            if should_log: print(f"    {stage_name} SelfAttn Out: {x.shape}")
        x_down, mask_down = self.downsample(x, mask)
        return x_down, mask_down, skip_outputs

class UpBlock2D(nn.Module):
    def __init__(self, in_channels, skip_channels_in, out_channels, time_embedding_dim, dropout_prob, has_attn=False, num_res_blocks=2):
        super().__init__()
        self.conv_after_upsample = PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
            ResidualBlock2D((out_channels + skip_channels_in) if i == 0 else out_channels, out_channels, time_embedding_dim, dropout_prob)
            for i in range(num_res_blocks)
        ])
        self.attn_block = SelfAttentionBlock2D(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip_xs, time_emb, mask, use_checkpointing=False, should_log=False, stage_name=""):
        target_shape = skip_xs[0].shape[-2:] # Get target H, W
        x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, size=target_shape, mode='nearest')
        
        x, mask = self.conv_after_upsample(x, mask)
        
        x = torch.cat([x] + skip_xs, dim=1)
        if should_log: print(f"    {stage_name} After Skip Concat: {x.shape}")
        mask = mask[:, :1, :, :].repeat(1, x.shape[1], 1, 1)

        for i, res_block in enumerate(self.res_blocks):
            x, mask = checkpoint(res_block, x, time_emb, mask, use_reentrant=False) if use_checkpointing else res_block(x, time_emb, mask)
            if should_log: print(f"    {stage_name} ResBlock {i+1} Out: {x.shape}")
        if isinstance(self.attn_block, SelfAttentionBlock2D):
            x, mask = self.attn_block(x, mask)
            if should_log: print(f"    {stage_name} SelfAttn Out: {x.shape}")
        return x, mask

class UNet2D(nn.Module):
    _has_logged_forward = False
    
    def __init__(self, config, in_channels):
        super().__init__()
        self.use_checkpointing = config.use_checkpointing
        base_channels = config.base_unet_channels
        time_embedding_dim = base_channels * 4
        
        self.initial_conv = PartialConv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.down_stages = nn.ModuleList()
        current_channels = base_channels
        channel_multipliers = config.channel_multipliers
        num_res_blocks = config.num_res_blocks
        attn_resolutions = config.attn_resolutions
        h, w = config.data_shape[1], config.data_shape[2]

        for i, multiplier in enumerate(channel_multipliers):
            out_ch = base_channels * multiplier
            # BUG FIX: The resolution check was off by one power of 2.
            # It should check the resolution *after* the downsampling of the current stage.
            current_res = h // (2**(i+1))
            has_attn = current_res in attn_resolutions
            self.down_stages.append(DownBlock2D(current_channels, out_ch, time_embedding_dim, config.dropout_prob, has_attn, num_res_blocks))
            current_channels = out_ch
        
        self.bottleneck = nn.ModuleList([
            ResidualBlock2D(current_channels, current_channels, time_embedding_dim, config.dropout_prob),
            SelfAttentionBlock2D(current_channels),
            ResidualBlock2D(current_channels, current_channels, time_embedding_dim, config.dropout_prob)
        ])

        self.up_stages = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            multiplier = channel_multipliers[i]
            in_ch = channel_multipliers[i+1] * base_channels if i + 1 < len(channel_multipliers) else current_channels
            out_ch = base_channels * multiplier
            skip_ch_in = out_ch * num_res_blocks
            # Apply the same fix here for consistency
            current_res = h // (2**(i+1))
            has_attn = current_res in attn_resolutions
            self.up_stages.append(UpBlock2D(in_ch, skip_ch_in, out_ch, time_embedding_dim, config.dropout_prob, has_attn, num_res_blocks))
        
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, time_emb, mask):
        is_main_process = not x.device.type == 'cuda' or x.device.index == 0
        should_log = is_main_process and not UNet2D._has_logged_forward

        if should_log: print("\n--- 2D U-Net Core Forward Pass ---")
        
        h, current_mask = self.initial_conv(x, mask)
        if should_log: print(f"After Initial 2D Conv: {h.shape}")
        
        skip_connections = []
        for i, stage in enumerate(self.down_stages):
            stage_name = f"[2D Down Stage {i+1}]"
            if should_log: print(f"  {stage_name} Input: {h.shape}")
            h, current_mask, skips = stage(h, time_emb, current_mask, self.use_checkpointing, should_log, stage_name)
            if should_log: print(f"  {stage_name} Output: {h.shape}")
            skip_connections.append(skips)
        
        if should_log: print(f"2D Bottleneck Input: {h.shape}")
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock2D):
                h, current_mask = checkpoint(layer, h, time_emb, current_mask, use_reentrant=False) if self.use_checkpointing else layer(h, time_emb, current_mask)
            else:
                h, current_mask = layer(h, current_mask)
        if should_log: print(f"2D Bottleneck Output: {h.shape}")

        for i, stage in enumerate(self.up_stages):
            stage_name = f"[2D Up Stage {i+1}]"
            if should_log: print(f"  {stage_name} Input: {h.shape}")
            skips_for_stage = skip_connections.pop()
            h, current_mask = stage(h, skips_for_stage, time_emb, current_mask, self.use_checkpointing, should_log, stage_name)
            if should_log: print(f"  {stage_name} Output: {h.shape}")
            
        final_output = self.final_conv(h)
        if should_log:
            print(f"After Final 2D Conv: {final_output.shape}")
            print("------------------------------------")
            UNet2D._has_logged_forward = True
            
        return final_output

# --- Main UNet Wrapper ---

class UNet(nn.Module):
    _has_logged_forward = False

    def __init__(self, config, verbose_init=False):
        super().__init__()
        self.config = config
        base_channels = config.base_unet_channels
        time_embedding_dim = base_channels * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, time_embedding_dim), nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # NEW: Configurable multi-encoder/decoder architecture
        self.encoder_groups = config.vertical_encoder_groups
        self.latent_dims = config.vertical_latent_dims
        self.depth_size = config.data_shape[0]
        
        self.vertical_encoders = nn.ModuleList()
        self.vertical_decoders = nn.ModuleList()

        for i, group in enumerate(self.encoder_groups):
            num_channels_in_group = len(group)
            latent_dim_for_group = self.latent_dims[i]
            
            self.vertical_encoders.append(
                VerticalEncoder(in_channels=num_channels_in_group, latent_dim=latent_dim_for_group, depth_size=self.depth_size)
            )
            self.vertical_decoders.append(
                VerticalDecoder(latent_dim=latent_dim_for_group, out_channels=num_channels_in_group, depth_size=self.depth_size)
            )
        
        total_latent_dim = sum(self.latent_dims)
        unet2d_in_channels = total_latent_dim + config.location_embedding_channels
        self.unet_2d = UNet2D(config, unet2d_in_channels)
        

    def forward(self, x, t, mask, conditions=None, location_field=None):
        is_main_process = not x.device.type == 'cuda' or x.device.index == 0
        should_log = is_main_process and not UNet._has_logged_forward

        if should_log:
            print("\n--- UNet Wrapper Forward Pass ---")
            print(f"Initial Input 'x': {x.shape}")

        # 1. Encode each channel group's vertical dimension
        latent_repr_list = []
        horizontal_mask_list = []
        for i, group in enumerate(self.encoder_groups):
            x_group = x[:, group, :, :, :]
            mask_group = mask[:, group, :, :, :]
            
            latent_repr, horizontal_mask = self.vertical_encoders[i](x_group, mask_group)
            if should_log: print(f"Encoder Group {i} Output: {latent_repr.shape}")
            latent_repr_list.append(latent_repr)
            horizontal_mask_list.append(horizontal_mask)

        # 2. Concatenate all latent representations and masks
        concatenated_latent_repr = torch.cat(latent_repr_list, dim=1)
        representative_horizontal_mask = horizontal_mask_list[0]
        if should_log: print(f"Concatenated Latent Repr: {concatenated_latent_repr.shape}")

        # 3. Prepare inputs for the 2D U-Net by adding location embeddings
        if self.config.location_embedding_channels > 0 and location_field is not None:
            concatenated_latent_repr = torch.cat([concatenated_latent_repr, location_field], dim=1)
            location_mask = torch.ones_like(location_field)
            unet_input_mask = torch.cat([representative_horizontal_mask.repeat(1, sum(self.latent_dims), 1, 1), location_mask], dim=1)
            if should_log: print(f"After Location Concat: {concatenated_latent_repr.shape}")
        else:
            unet_input_mask = representative_horizontal_mask.repeat(1, sum(self.latent_dims), 1, 1)

        time_emb = self.time_mlp(t)

        # 4. Run the 2D U-Net
        unet_output = self.unet_2d(concatenated_latent_repr, time_emb, unet_input_mask)
        
        # 5. Decode each latent group back to the physical 3D space
        output_list = torch.zeros_like(x)
        
        latent_start_idx = 0
        for i, group in enumerate(self.encoder_groups):
            latent_end_idx = latent_start_idx + self.latent_dims[i]
            
            latent_output_group = unet_output[:, latent_start_idx:latent_end_idx, :, :]
            original_mask_group = mask[:, group, :, :, :]
            
            decoded_group = self.vertical_decoders[i](latent_output_group, original_mask_group)
            if should_log: print(f"Decoder Group {i} Output: {decoded_group.shape}")
            
            output_list[:, group, :, :, :] = decoded_group
            latent_start_idx = latent_end_idx
        
        if should_log:
            print(f"Final Reconstructed Output: {output_list.shape}")
            print("---------------------------------")
            UNet._has_logged_forward = True

        return output_list * mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
