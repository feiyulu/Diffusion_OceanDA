# --- unet_model.py ---
# This file defines the U-Net architecture, including the custom PartialConv2d layer.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Partial Convolution Layer for handling masked regions
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a convolutional layer that calculates the sum of valid input pixels in the receptive field. 
        # This sum is used for normalization.
        # The sum_kernel should always operate on a single-channel mask.
        # Its input and output channels should be 1, regardless of the feature channels.

        self.register_buffer('sum_kernel', torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]))
        self.sum_kernel.requires_grad = False # This kernel is fixed

        # Store window_size for normalization, as in the reference implementation
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        # Print out partial conv information during initialization
        print(f"  PartialConv2d: kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}")
        print(f"    Weight shape: {self.weight.shape}")
        if self.bias is not None:
            print(f"    Bias shape: {self.bias.shape}")

    def forward(self, input_tensor, mask_in):
        # input_tensor: (N, C_in, H, W)
        # mask_in: (N, C_in, H, W) -- mask for valid data (1) vs land (0)

        # Apply convolution to the input tensor
        output = super().forward(input_tensor) # (N, C_out, H', W')

        # Compute mask_out by convolving the mask_in
        with torch.no_grad(): # Mask operations do not require gradients
            # Use only the first channel of mask_in for sum_kernel convolution
            # The sum_kernel is designed for single-channel input (in_channels=1).
            # Assuming that if mask_in is multi-channel, all channels are identical for the mask logic.
            mask_for_kernel = mask_in[:, :1, :, :] 
            
            # Pad mask_in to match input_tensor padding for convolution
            mask_padded = F.pad(
                mask_for_kernel,
                (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), 
                mode='constant', value=1)

            # Perform convolution on the mask to determine valid output regions
            # Use sum_kernel which is all ones, essentially counting valid input pixels in the receptive field
            mask_out = F.conv2d(
                mask_padded, self.sum_kernel, bias=None, stride=self.stride, 
                padding=0, dilation=self.dilation, groups=1)
            
            # Normalize mask_out by window_size to get a ratio of valid pixels (0 to 1)
            mask_out = mask_out / self.window_size # This assumes all values are > 0 if there's any valid input

            # Threshold mask_out to be binary: 1 for valid, 0 for invalid
            mask_out = torch.clamp(mask_out, 0, 1) # Ensure values are between 0 and 1
            mask_out = (mask_out > 0).float() # Binarize: 1 if any valid input, 0 otherwise

        # Zero out results where mask_out is 0
        output = output * mask_out # Ensure land areas in output are zeroed

        return output, mask_out

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * 
            -(torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1))
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout_prob):
        super().__init__()
        self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels * 2)

        self.conv2 = PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        # Residual connection: 1x1 conv if channel sizes mismatch
        if in_channels != out_channels:
            self.residual_conv = PartialConv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity() # No change needed

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, time_emb, mask):
        # time_emb: (N, time_emb_dim)
        # x: (N, C, H, W)
        # mask: (N, C, H, W)

        # First convolution block
        h, h_mask = self.conv1(x, mask)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.dropout(h)

        # Time embedding conditioning: scale and shift
        # time_emb_reshaped: (N, out_channels * 2, 1, 1)
        time_emb_reshaped = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        # Split into (N, out_channels, 1, 1) each
        scale, shift = torch.chunk(time_emb_reshaped, 2, dim=1)
        h = h * (1 + scale) + shift # Apply conditioning

        # Second convolution block
        h, h_mask = self.conv2(h, h_mask) # Pass the updated mask
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)

        # Residual connection
        if isinstance(self.residual_conv, PartialConv2d):
            residual, residual_mask = self.residual_conv(x, mask)
        else: # It's an nn.Identity()
            residual = self.residual_conv(x) # nn.Identity takes only one input
            residual_mask = mask # Mask is unchanged if Identity is used
        
        # Combine residual and main path (element-wise addition)
        combined_mask = h_mask * residual_mask # Element-wise multiplication for combined mask
        
        return (h + residual) * combined_mask, combined_mask

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, mask):
        # x: (N, C, H, W)
        # mask: (N, C, H, W)
        
        N, C, H, W = x.shape
        x_norm = self.norm(x)

        q = self.query(x_norm).view(N, self.num_heads, C // self.num_heads, H * W)
        k = self.key(x_norm).view(N, self.num_heads, C // self.num_heads, H * W)
        v = self.value(x_norm).view(N, self.num_heads, C // self.num_heads, H * W)

        # Calculate attention scores
        attn_scores = torch.einsum("nhcl,nhck->nhlk", q, k) * (C // self.num_heads)**-0.5
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to value
        out = torch.einsum("nhlk,nhck->nhcl", attn_probs, v)
        out = out.reshape(N, C, H, W) # Reshape back to image format

        out = self.proj_out(out)

        # Residual connection
        return x + out, mask # Attention doesn't change mask, just features within valid regions

# DownBlock class encapsulating residual blocks, attention, and downsampling
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout_prob, has_attn=False, num_res_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        # First residual block might have different in_channels
        self.blocks.append(ResidualBlock(in_channels, out_channels, time_emb_dim, dropout_prob))
        self.blocks.append(AttentionBlock(out_channels) if has_attn else nn.Identity())
        
        # Subsequent residual blocks in this down stage
        for _ in range(num_res_blocks - 1): # num_res_blocks is total, so (total - 1) more
            self.blocks.append(ResidualBlock(out_channels, out_channels, time_emb_dim, dropout_prob))
            self.blocks.append(AttentionBlock(out_channels) if has_attn else nn.Identity())

        # Downsampling layer after all residual blocks
        self.downsample = PartialConv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_emb, mask):
        # Pass through residual blocks and attention for this down stage
        for i, block in enumerate(self.blocks):
            if isinstance(block, ResidualBlock):
                x, mask = block(x, time_emb, mask)
            else: # Must be AttentionBlock or Identity
                if isinstance(block, AttentionBlock):
                    x, mask = block(x, mask)
                else: # nn.Identity
                    x = block(x) # nn.Identity takes only one input, mask remains unchanged
        
        # The skip connection is the output *before* final downsampling
        skip_x = x
        skip_mask = mask

        # Apply downsampling
        x_downsampled, mask_downsampled = self.downsample(x, mask)
        
        return x_downsampled, mask_downsampled, skip_x, skip_mask # Return downsampled features and skip for UNet


# UpBlock class encapsulating upsampling, residual blocks, and attention
class UpBlock(nn.Module):
    def __init__(self, in_channels_from_prev_up, skip_channels, out_channels, time_emb_dim, dropout_prob, has_attn=False, num_res_blocks=2):
        super().__init__()
        # Takes `in_channels_from_prev_up` and outputs `out_channels`
        # This output will then be concatenated with `skip_channels`.
        self.upsample_conv = nn.ConvTranspose2d(in_channels_from_prev_up, out_channels, kernel_size=4, stride=2, padding=1)
        
        self.blocks = nn.ModuleList()
        # The first residual block in an UpBlock takes concatenated features
        self.blocks.append(ResidualBlock(out_channels + skip_channels, out_channels, time_emb_dim, dropout_prob))
        self.blocks.append(AttentionBlock(out_channels) if has_attn else nn.Identity())

        # Subsequent residual blocks operate on `out_channels`
        for _ in range(num_res_blocks - 1):
            self.blocks.append(ResidualBlock(out_channels, out_channels, time_emb_dim, dropout_prob))
            self.blocks.append(AttentionBlock(out_channels) if has_attn else nn.Identity())

    def forward(self, x, skip_x, time_emb, mask_from_prev_up, skip_mask):
        # Upsample input from previous level
        x_upsampled = self.upsample_conv(x)
        
        # Upsample mask from previous level
        mask_upsampled = F.interpolate(mask_from_prev_up[:, :1, :, :], size=x_upsampled.shape[-2:], mode='nearest')
        mask_upsampled = (mask_upsampled > 0.5).float()
        mask_upsampled_repeated = mask_upsampled.repeat(1, x_upsampled.shape[1], 1, 1)

        # Handle potential size mismatch for skip connection (interpolate skip_x to match x_upsampled)
        # This can occur if ConvTranspose2d's output size slightly differs from target.
        if x_upsampled.shape[-2:] != skip_x.shape[-2:]:
            skip_x = F.interpolate(skip_x, size=x_upsampled.shape[-2:], mode='nearest')
            skip_mask = F.interpolate(skip_mask[:, :1, :, :], size=x_upsampled.shape[-2:], mode='nearest')
            skip_mask = (skip_mask > 0.5).float()
            skip_mask = skip_mask.repeat(1, skip_x.shape[1], 1, 1) # Repeat for skip_x channels

        # Concatenate upsampled features with skip connection
        x_combined = torch.cat((x_upsampled, skip_x), dim=1)
        combined_mask = mask_upsampled_repeated * skip_mask # Element-wise multiplication for combined mask

        # Pass through the sequence of residual blocks and attention
        current_x = x_combined
        current_mask = combined_mask

        for i, block in enumerate(self.blocks):
            if isinstance(block, ResidualBlock):
                # For the first ResidualBlock (j=0), input is x_combined, combined_mask
                # For subsequent RBs, input is output from previous RB (current_x, current_mask)
                current_x, current_mask = block(current_x, time_emb, current_mask)
            else: # Must be AttentionBlock or Identity
                if isinstance(block, AttentionBlock):
                    current_x, current_mask = block(current_x, current_mask)
                else: # nn.Identity
                    current_x = block(current_x) # nn.Identity takes one input, mask is unchanged
        
        return current_x, current_mask

class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=32,
                 time_embedding_dim=256,
                 dayofyear_embedding_dim=64,
                 use_dayofyear_embedding=False,
                 use_2d_location_embedding=False,
                 use_co2_embedding=False,
                 co2_embedding_dim=64,
                 location_embedding_channels=2,
                 channel_multipliers=(1, 2, 4, 8),
                 num_res_blocks=2, # Number of residual blocks per stage
                 attn_resolutions=(8,), # Multipliers at which to add attention (e.g., (4,8) for 128, 256 channels)
                 dropout_prob=0.1,
                 verbose_init=False):
        super().__init__()

        self.use_dayofyear_embedding = use_dayofyear_embedding
        self.use_2d_location_embedding = use_2d_location_embedding
        self.dayofyear_embedding_dim = dayofyear_embedding_dim
        self.location_embedding_channels = location_embedding_channels
        self.use_co2_embedding = use_co2_embedding
        self.co2_embedding_dim = co2_embedding_dim
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions

        if verbose_init: print("--- Initializing UNet ---")
        if verbose_init: print(f"  UNet: in_channels={in_channels}, out_channels={out_channels}, base_channels={base_channels}")
        if verbose_init: print(f"  Time embedding dim: {time_embedding_dim}")
        if verbose_init: print(f"  Day of year embedding enabled: {use_dayofyear_embedding}, dim: {dayofyear_embedding_dim}")
        if verbose_init: print(f"  2D location embedding enabled: {use_2d_location_embedding}, channels: {location_embedding_channels}")
        if verbose_init: print(f"  CO2 embedding enabled: {use_co2_embedding}, dim: {co2_embedding_dim}") # NEW

        # 1. Handle time embedding setup
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.GELU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )

        # 2. Day of year embedding setup
        if self.use_dayofyear_embedding:
            self.dayofyear_mlp = nn.Sequential(
                nn.Linear(1, self.dayofyear_embedding_dim),
                nn.GELU(),
                nn.Linear(self.dayofyear_embedding_dim, self.dayofyear_embedding_dim)
            )
        
        # CO2 embedding setup
        if self.use_co2_embedding:
            self.co2_mlp = nn.Sequential(
                nn.Linear(1, self.co2_embedding_dim), # CO2 is a scalar value
                nn.GELU(),
                nn.Linear(self.co2_embedding_dim, self.co2_embedding_dim)
            )

        # Calculate the total dimension for the condition embedding passed to blocks
        total_condition_embedding_dim = time_embedding_dim
        if self.use_dayofyear_embedding:
            total_condition_embedding_dim += self.dayofyear_embedding_dim
        if self.use_co2_embedding:
            total_condition_embedding_dim += self.co2_embedding_dim
        if verbose_init: print(f"  Total condition embedding dim for blocks: {total_condition_embedding_dim}")

        # 3. Initial convolution layer
        initial_conv_in_channels = in_channels
        if self.use_2d_location_embedding:
            initial_conv_in_channels += self.location_embedding_channels
        if verbose_init: print(f"  Initial Conv: input {initial_conv_in_channels} -> output {base_channels}")
        self.initial_conv = PartialConv2d(initial_conv_in_channels, base_channels, kernel_size=3, padding=1)

        # 4. Downsampling stages
        self.down_stages = nn.ModuleList()
        current_channels = base_channels # Start from initial_conv output
        for i, multiplier in enumerate(channel_multipliers):
            out_channels_stage = base_channels * multiplier
            self.down_stages.append(DownBlock(
                in_channels=current_channels,
                out_channels=out_channels_stage,
                time_emb_dim=total_condition_embedding_dim,
                dropout_prob=dropout_prob,
                has_attn=(multiplier in attn_resolutions),
                num_res_blocks=num_res_blocks
            ))
            current_channels = out_channels_stage # The output of this DownBlock (after its internal downsample) becomes input for next
            if verbose_init: print(f"  Down Stage {i+1}: In {self.down_stages[-1].blocks[0].conv1.in_channels} -> Out {current_channels}. Has Attn: {multiplier in attn_resolutions}")

        # 5. Bottleneck stage
        if verbose_init: print(f"  Bottleneck: channels {current_channels}")
        self.bottleneck_res1 = ResidualBlock(current_channels, current_channels, total_condition_embedding_dim, dropout_prob)
        self.bottleneck_attn = AttentionBlock(current_channels) if channel_multipliers[-1] in attn_resolutions else nn.Identity()
        self.bottleneck_res2 = ResidualBlock(current_channels, current_channels, total_condition_embedding_dim, dropout_prob)

        # 6. Upsampling stages
        self.up_stages = nn.ModuleList()

        # `current_channels` is currently the bottleneck output channels.
        # `channels_from_prev_up_stage` tracks the number of channels flowing *into* the upsample_conv of the current UpBlock.
        channels_from_prev_up_stage = current_channels

        # Iterate from the largest multiplier down to the smallest
        for i in range(len(channel_multipliers) - 1, -1, -1):
            current_multiplier = channel_multipliers[i]
            # Channels for the skip connection coming from the corresponding down-stage.
            skip_channels_for_this_up_stage = base_channels * current_multiplier
            
            # The target output channels for the residual blocks within this UpBlock.
            out_channels_up_stage = base_channels * current_multiplier

            self.up_stages.append(UpBlock(
                in_channels_from_prev_up=channels_from_prev_up_stage,
                skip_channels=skip_channels_for_this_up_stage,
                out_channels=out_channels_up_stage,
                time_emb_dim=total_condition_embedding_dim,
                dropout_prob=dropout_prob,
                has_attn=(current_multiplier in attn_resolutions),
                num_res_blocks=num_res_blocks
            ))
            # After adding this UpBlock, its output (`out_channels_up_stage`) becomes the input for the next UpBlock.
            channels_from_prev_up_stage = out_channels_up_stage
            if verbose_init: print(f"  Up Stage {len(channel_multipliers)-i}: In_prev {self.up_stages[-1].upsample_conv.in_channels} -> Out_current {self.up_stages[-1].upsample_conv.out_channels}. SkipCh {skip_channels_for_this_up_stage}. Has Attn: {current_multiplier in attn_resolutions}")

        # 7. Final output layers
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()
        self.output_conv = PartialConv2d(base_channels, out_channels, kernel_size=3, padding=1)
        if verbose_init: print(f"  Final Output Conv: input {base_channels} -> output {out_channels}")
        if verbose_init: print("--- UNet Initialization Complete ---")

    def forward(
        self, x, t, mask, 
        dayofyear_batch=None, location_field=None, co2_batch=None,
        verbose_forward=False):
        # x: (N, C, H, W) - noisy input data (original channels only)
        # t: (N,) - Diffusion timestep
        # mask: (N, 1, H, W) - Land mask (1 for ocean, 0 for land)
        # dayofyear_batch: (N,) - Day of year for embedding
        # location_field: (N, location_embedding_channels, H, W) - 2D lat/lon embedding
        # co2_batch: (N,) - CO2 concentration for embedding

        if verbose_forward: print("--- Starting UNet Forward Pass ---")
        if verbose_forward: print(f"Input X shape: {x.shape}, T shape: {t.shape}, Mask shape: {mask.shape}")
        if self.use_dayofyear_embedding and verbose_forward: print(f"Dayofyear batch shape: {dayofyear_batch.shape}")
        if self.use_2d_location_embedding and verbose_forward: print(f"Location field shape: {location_field.shape}")
        if self.use_co2_embedding and verbose_forward: print(f"CO2 batch shape: {co2_batch.shape}")

        # Apply 2D location embedding if enabled, by concatenating channels
        current_x = x # Rename to avoid confusion with `x` used in internal blocks
        current_mask = mask # Rename for consistency

        if self.use_2d_location_embedding and location_field is not None:
            if verbose_forward: print(f"Before location concat: Current X shape {current_x.shape}")
            current_x = torch.cat((current_x, location_field), dim=1)
            # The mask needs to be repeated to match the new number of channels for `x`
            current_mask = current_mask.repeat(1, current_x.shape[1], 1, 1) # Expand mask channels
            if verbose_forward: print(f"After location concat: Current X shape {current_x.shape}, Current Mask shape {current_mask.shape}")

        # Process time embedding
        time_emb = self.time_mlp(t)
        if verbose_forward: print(f"Time embedding (time_emb) shape: {time_emb.shape}")
        
        # Combine time embedding with day of year embedding if enabled
        condition_emb = time_emb
        if self.use_dayofyear_embedding and dayofyear_batch is not None:
            if dayofyear_batch.dim() == 1:
                dayofyear_batch = dayofyear_batch.unsqueeze(1).float()
            else:
                dayofyear_batch = dayofyear_batch.float()
            dayofyear_emb = self.dayofyear_mlp(dayofyear_batch)
            if verbose_forward: print(f"Day of year embedding (dayofyear_emb) shape: {dayofyear_emb.shape}")
            condition_emb = torch.cat((condition_emb, dayofyear_emb), dim=1)
            if verbose_forward: print(f"After dayofyear concat: Condition embedding (condition_emb) shape: {condition_emb.shape}")
        
        # Combine with CO2 embedding if enabled
        if self.use_co2_embedding and co2_batch is not None:
            if co2_batch.dim() == 1:
                co2_batch = co2_batch.unsqueeze(1).float()
            else:
                co2_batch = co2_batch.float()
            co2_emb = self.co2_mlp(co2_batch)
            if verbose_forward: print(f"CO2 embedding (co2_emb) shape: {co2_emb.shape}")
            condition_emb = torch.cat((condition_emb, co2_emb), dim=1)
            if verbose_forward: print(f"After CO2 concat: Condition embedding (condition_emb) shape: {condition_emb.shape}")

        # Initial convolution
        current_x, current_mask = self.initial_conv(current_x, current_mask)
        if verbose_forward: print(f"After initial_conv: Current X shape: {current_x.shape}, Current Mask shape: {current_mask.shape}")

        skip_connections = []
        masks_for_skip = []

        # Downsampling path stages
        for i, down_stage in enumerate(self.down_stages):
            # Each DownBlock returns (downsampled_x, downsampled_mask, skip_x, skip_mask)
            current_x, current_mask, skip_x, skip_mask = down_stage(current_x, condition_emb, current_mask)
            skip_connections.append(skip_x)
            masks_for_skip.append(skip_mask)
            if verbose_forward: print(f"  After Down Stage {i+1}: Current X shape {current_x.shape}, Current Mask shape {current_mask.shape}. Skip X shape {skip_x.shape}")

        # Bottleneck
        current_x, current_mask = self.bottleneck_res1(current_x, condition_emb, current_mask)
        if isinstance(self.bottleneck_attn, AttentionBlock):
            current_x, current_mask = self.bottleneck_attn(current_x, current_mask)
        else:
            current_x = self.bottleneck_attn(current_x)
        current_x, current_mask = self.bottleneck_res2(current_x, condition_emb, current_mask)
        if verbose_forward: print(f"After Bottleneck: Current X shape: {current_x.shape}, Current Mask shape: {current_mask.shape}")

        # Upsampling path stages
        # Skips were appended in order from initial conv (highest resolution) down to bottleneck
        # We need to pop them in reverse order to get the lowest resolution skip first
        skip_connections.reverse()
        masks_for_skip.reverse()

        for i, up_stage in enumerate(self.up_stages):
            # Get the skip connection and mask for this specific up-stage
            skip_x_for_stage = skip_connections.pop(0) 
            skip_mask_for_stage = masks_for_skip.pop(0)

            # UpBlock.forward now takes:
            # - x (current feature map from previous up-stage or bottleneck)
            # - skip_x (from corresponding down-stage)
            # - time_emb
            # - mask_from_prev_up (mask for x)
            # - skip_mask (mask for skip_x)
            current_x, current_mask = up_stage(
                current_x, 
                skip_x_for_stage, 
                condition_emb, 
                current_mask, 
                skip_mask_for_stage
            )
            if verbose_forward: print(f"  After Up Stage {i+1}: Current X shape {current_x.shape}, Current Mask shape {current_mask.shape}")

        # Final output layers
        current_x = self.final_norm(self.final_act(current_x))
        if verbose_forward: print(f"After final_norm and activation: Current X shape: {current_x.shape}")

        output_prediction, _ = self.output_conv(current_x, current_mask) # Output `out_channels`, H, W
        if verbose_forward: print(f"After output_conv (output prediction): {output_prediction.shape}")

        # Ensure land points are zeroed out in the final output prediction.
        final_output_mask_broadcast = current_mask[:, :1, :, :].float().repeat(1, output_prediction.shape[1], 1, 1) 
        output_prediction = output_prediction * final_output_mask_broadcast # Zero out land regions
        if verbose_forward: print(f"Final output after applying mask: {output_prediction.shape}")
        if verbose_forward: print("--- End UNet Forward Pass ---")

        return output_prediction

# Function to count trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
