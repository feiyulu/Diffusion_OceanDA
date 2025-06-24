# --- unet_model.py ---
# This file defines the U-Net architecture, including the custom PartialConv2d layer.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Partial Convolution Layer for handling masked regions (e.g., land points)
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a convolutional layer that calculates the sum of valid input pixels
        # in the receptive field. This sum is used for normalization.
        # The sum_kernel should always operate on a single-channel mask.
        # Its input and output channels should be 1, regardless of the feature channels.
        self.register_buffer('sum_kernel', torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]))
        self.sum_kernel.requires_grad = False # This kernel is fixed

        # Store window_size for normalization, as in the reference implementation
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        # These prints are fine as they happen once at initialization
        print(f"  PartialConv2d: kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}")
        print(f"    Weight shape: {self.weight.shape}")
        if self.bias is not None:
            print(f"    Bias shape: {self.bias.shape}")

    def forward(self, input_tensor, mask_in):
        # input_tensor: (N, C_in, H, W) - the feature map
        # mask_in: (N, 1, H, W) - a binary mask (1 for valid/ocean, 0 for invalid/land)
        
        # 1. Apply convolution directly to the input_tensor (unmasked)
        conv_data = F.conv2d(
            input_tensor, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

        # 2. Convolve the input mask with a kernel of ones to get the sum of valid pixels
        with torch.no_grad():
            update_mask = F.conv2d(
                mask_in, self.sum_kernel, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
            )
            # Avoid division by zero and handle cases where all pixels are masked
            update_mask = torch.max(update_mask, torch.ones_like(update_mask) * 1e-6)

        # 3. Generate the output mask based on where valid pixels contributed
        output_mask = torch.where(update_mask > 0, 1.0, 0.0)

        # 4. Calculate the mask_ratio (normalization factor)
        mask_ratio = self.window_size / update_mask

        # 5. Apply the mask_ratio and the output_mask to the convolved data
        output_feature = conv_data * mask_ratio * output_mask.float().repeat(1, conv_data.shape[1], 1, 1)

        return output_feature, output_mask

# --- U-Net Components ---
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Generate sinusoidal embeddings for the timestep
        # This helps the model incorporate time information effectively.
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = PartialConv2d(in_channels, out_channels, 3, padding=1)
        
        self.up = up # Store the up/down flag for use in forward

        if up:
            # For upsampling path, use a PartialConv2d (stride 1) after interpolation
            self.transform_conv = PartialConv2d(out_channels, out_channels, 3, padding=1)
            print(f"  Block (up): Interpolate + PartialConv2d: in_channels={out_channels}, out_channels={out_channels}, kernel_size=3, padding=1, stride=1")
        else:
            # For downsampling path, use Conv2d with stride 2
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            print(f"  Block (down): Conv2d: in_channels={out_channels}, out_channels={out_channels}, kernel_size=4, stride=2, padding=1")

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() # SiLU (Sigmoid Linear Unit) is a common activation function

        # Print weights of the transform layer (only for Conv2d, PartialConv2d weights printed in its init)
        if not up: # Only for downsampling Conv2d
            if hasattr(self.transform, 'weight'):
                print(f"    Transform Weight shape: {self.transform.weight.shape}")
            if hasattr(self.transform, 'bias') and self.transform.bias is not None:
                print(f"    Transform Bias shape: {self.transform.bias.shape}")


    def forward(self, x, time_emb, mask_in):
        # x: input feature map, time_emb: time embedding, mask_in: corresponding input mask
        
        # Apply the convolutional layer and get the updated feature map and mask
        h, mask_out_conv = self.conv1(x, mask_in)
        h = self.norm(self.act(h)) # Apply batch normalization and activation

        # Incorporate time embedding into the feature map
        # time_emb is (N, time_dim), expand to (N, time_dim, 1, 1) to be added to (N, C, H, W)
        time_emb_reshaped = self.act(self.time_mlp(time_emb)).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb_reshaped

        # Propagate mask through the transform layer (downsampling/upsampling)
        if self.up: # Upsampling block
            # Interpolate both feature map and mask to double the size
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            mask_out_transform = F.interpolate(mask_out_conv, scale_factor=2, mode='nearest')
            
            # Apply the PartialConv2d after interpolation. This changes channels, not spatial size.
            h, mask_out_transform = self.transform_conv(h, mask_out_transform)
        else: # Downsampling block
            # Calculate the output mask by convolving the mask with a kernel of ones
            mask_transform_kernel = torch.ones(1, 1, self.transform.kernel_size[0], self.transform.kernel_size[1]).to(mask_out_conv.device)
            mask_out_transform = F.conv2d(
                mask_out_conv, mask_transform_kernel, bias=None,
                stride=self.transform.stride, padding=self.transform.padding,
                dilation=self.transform.dilation, groups=1
            )
            mask_out_transform = torch.where(mask_out_transform > 0, 1.0, 0.0)
            h = self.transform(h) # Apply the Conv2d transform to the feature map

        return h, mask_out_transform # Return both feature map and its propagated mask

class UNet(nn.Module):
    # Added base_channels argument
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_dim: int = 256, 
        image_size: tuple = (128, 128), 
        base_channels: int = 32):
        
        super().__init__()
        self.time_dim = time_dim
        self.image_size = image_size # Store image size as a tuple (H, W)
        self.base_channels = base_channels # Store base channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        print("--- UNet Layer Initialization and Weight Shapes ---")
        # Use base_channels for the initial convolution output
        self.initial_conv = PartialConv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path blocks - scale channels based on base_channels
        self.down1 = Block(base_channels, base_channels * 2, time_dim, up=False)
        self.down2 = Block(base_channels * 2, base_channels * 4, time_dim, up=False)
        self.down3 = Block(base_channels * 4, base_channels * 8, time_dim, up=False)

        # Bottleneck block - scale channels based on base_channels
        self.bottleneck_conv1 = PartialConv2d(base_channels * 8, base_channels * 16, 3, padding=1)
        self.bottleneck_norm1 = nn.BatchNorm2d(base_channels * 16)
        self.bottleneck_act1 = nn.SiLU()
        
        # Calculate target size for bottleneck ConvTranspose2d to match x3's size
        # x3's size is original_size // 4
        bottleneck_target_h = image_size[0] // 4
        bottleneck_target_w = image_size[1] // 4

        # Calculate actual input sizes to bottleneck_conv_transpose after down3
        input_h_bottleneck = image_size[0] // 8
        input_w_bottleneck = image_size[1] // 8

        # Initialize output_padding_h and output_padding_w to 0 to prevent UnboundLocalError
        # In case the calculation on the right-hand side fails for an unforeseen reason.
        output_padding_h = 0 
        output_padding_w = 0

        # Calculate required output padding to perfectly match target H/4, W/4
        # H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        # For stride=2, padding=1, kernel_size=4, dilation=1: H_out = 2*H_in + output_padding
        # So, output_padding = H_out - 2*H_in
        output_padding_h = bottleneck_target_h - (2 * input_h_bottleneck)
        output_padding_w = bottleneck_target_w - (2 * input_w_bottleneck)

        # Clamp output_padding to 0 or 1, as values greater than 1 are unusual and imply larger architectural issues
        output_padding_h = max(0, min(1, output_padding_h))
        output_padding_w = max(0, min(1, output_padding_w))

        if output_padding_h not in [0, 1] or output_padding_w not in [0, 1]:
            print(f"Warning: Unusual output padding for bottleneck: ({output_padding_h}, {output_padding_w}).")
            print(f"Input to bottleneck_conv_transpose: ({input_h_bottleneck}, {input_w_bottleneck})")
            print(f"Target from bottleneck_conv_transpose: ({bottleneck_target_h}, {bottleneck_target_w})")


        self.bottleneck_conv_transpose = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 4, 2, 1, output_padding=(output_padding_h, output_padding_w)) 
        print(f"  Bottleneck ConvTranspose2d: in_channels={base_channels * 16}, out_channels={base_channels * 8}, kernel_size=4, stride=2, padding=1, output_padding=({output_padding_h}, {output_padding_w})")
        if hasattr(self.bottleneck_conv_transpose, 'weight'):
            print(f"    Bottleneck Transform Weight shape: {self.bottleneck_conv_transpose.weight.shape}")
        if hasattr(self.bottleneck_conv_transpose, 'bias') and self.bottleneck_conv_transpose.bias is not None:
            print(f"    Bottleneck Transform Bias shape: {self.bottleneck_conv_transpose.bias.shape}")


        # Upsampling path blocks (in_channels is (previous layer output channels + skip connection channels))
        # Ensure skip connection channels match the corresponding downsampling block's output
        self.up1 = Block(base_channels * 8 + base_channels * 4, base_channels * 4, time_dim, up=True)
        self.up2 = Block(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim, up=True)

        # Final stage: concatenate output of up2 with initial_conv_output
        self.final_upsample_conv = PartialConv2d(base_channels * 2 + base_channels, base_channels, 3, padding=1)
        self.final_norm = nn.BatchNorm2d(base_channels)
        self.final_act = nn.SiLU()
        print(f"  Final Upsample Conv: PartialConv2d: in_channels={base_channels * 2 + base_channels}, out_channels={base_channels}, kernel_size=3, padding=1, stride=1")
        if hasattr(self.final_upsample_conv, 'weight'):
            print(f"    Final Upsample Conv Weight shape: {self.final_upsample_conv.weight.shape}")
        if hasattr(self.final_upsample_conv, 'bias') and self.final_upsample_conv.bias is not None:
            print(f"    Final Upsample Conv Bias shape: {self.final_upsample_conv.bias.shape}")


        # The very last convolution to map to output channels
        self.output_conv = PartialConv2d(base_channels, out_channels, 3, padding=1)
        print(f"  Output Conv: PartialConv2d: in_channels={base_channels}, out_channels={out_channels}, kernel_size=3, padding=1, stride=1")
        if hasattr(self.output_conv, 'weight'):
            print(f"    Output Conv Weight shape: {self.output_conv.weight.shape}")
        if hasattr(self.output_conv, 'bias') and self.output_conv.bias is not None:
            print(f"    Output Conv Bias shape: {self.output_conv.bias.shape}")

        print("--- End UNet Layer Initialization ---")

    def forward(self, x, time, mask, verbose_forward=False):
        # x: input noisy image, time: current timestep, mask: land/ocean mask (1=ocean, 0=land)
        
        if verbose_forward: # Only print verbose info if verbose_forward is True
            print("\n--- UNet Forward Pass - Tensor Shapes ---")
            print(f"Input X shape: {x.shape}, Input Mask shape: {mask.shape}")

        # Time embeddings
        time_emb = self.time_mlp(time)

        # Downsampling path (Encoder)
        # Each block returns (feature_map, mask_for_feature_map)
        x1, mask1 = self.initial_conv(x, mask) # x1 (base_channels channels, H, W)
        if verbose_forward: print(f"After initial_conv: X1 shape: {x1.shape}, Mask1 shape: {mask1.shape}")

        x2, mask2 = self.down1(x1, time_emb, mask1) # x2 (base_channels * 2 channels, H/2, W/2)
        if verbose_forward: print(f"After down1: X2 shape: {x2.shape}, Mask2 shape: {mask2.shape}")

        x3, mask3 = self.down2(x2, time_emb, mask2) # x3 (base_channels * 4 channels, H/4, W/4)
        if verbose_forward: print(f"After down2: X3 shape: {x3.shape}, Mask3 shape: {mask3.shape}")

        x4, mask4 = self.down3(x3, time_emb, mask3) # x4 (base_channels * 8 channels, H/8, W/8)
        if verbose_forward: print(f"After down3: X4 shape: {x4.shape}, Mask4 shape: {mask4.shape}")

        # Bottleneck
        x4_pre_transform, mask4_bottleneck_conv = self.bottleneck_conv1(x4, mask4) # x4 (base_channels * 16 channels, H/8, W/8)
        x4_pre_transform = self.bottleneck_norm1(self.bottleneck_act1(x4_pre_transform))
        if verbose_forward: print(f"Before bottleneck_conv_transpose: X4_pre_transform shape: {x4_pre_transform.shape}")
        
        # x4 will be upsampled to H/4, W/4 to match x3
        x4 = self.bottleneck_conv_transpose(x4_pre_transform) # x4 (base_channels * 8 channels, H/4, W/4)
        # Ensure correct size after upsampling if image dimensions aren't perfectly divisible
        target_h_bottleneck = self.image_size[0] // 4
        target_w_bottleneck = self.image_size[1] // 4
        if x4.shape[2] != target_h_bottleneck or x4.shape[3] != target_w_bottleneck:
            x4 = F.interpolate(x4, size=(target_h_bottleneck, target_w_bottleneck), mode='nearest')
        
        if verbose_forward: print(f"After bottleneck_conv_transpose: {x4.shape}")
        
        # Upsample the mask corresponding to the bottleneck output
        mask4_up = F.interpolate(mask4_bottleneck_conv, size=(target_h_bottleneck, target_w_bottleneck), mode='nearest')
        if verbose_forward: print(f"Mask4_up shape (interpolated to X3 size): {mask4_up.shape}")
        
        # Upsampling path (Decoder) with skip connections
        # Combine masks using logical OR and cast to float for single-channel mask input to next block
        
        # Up1: Concatenate bottleneck output (x4) with skip connection from down2 (x3)
        x_concat_up1 = torch.cat((x4, x3), dim=1) # (base_channels * 8 + base_channels * 4) channels, H/4, W/4
        mask_concat_up1 = (mask4_up.bool() | mask3.bool()).float() # Single channel mask
        if verbose_forward: print(f"Before up1: X_concat_up1 shape: {x_concat_up1.shape}, Mask_concat_up1 shape: {mask_concat_up1.shape}")
        x, mask_up1 = self.up1(x_concat_up1, time_emb, mask_concat_up1) # x (base_channels * 4 channels, H/2, W/2)
        if verbose_forward: print(f"After up1: X shape: {x.shape}, Mask_up1 shape: {mask_up1.shape}")

        # Up2: Concatenate current output (x) with skip connection from down1 (x2)
        x_concat_up2 = torch.cat((x, x2), dim=1) # (base_channels * 4 + base_channels * 2) channels, H/2, W/2
        mask_concat_up2 = (mask_up1.bool() | mask2.bool()).float() # Single channel mask
        if verbose_forward: print(f"Before up2: X_concat_up2 shape: {x_concat_up2.shape}, Mask_concat_up2 shape: {mask_concat_up2.shape}")
        x, mask_up2 = self.up2(x_concat_up2, time_emb, mask_concat_up2) # x (base_channels * 2 channels, H, W)
        if verbose_forward: print(f"After up2: X shape: {x.shape}, Mask_up2 shape: {mask_up2.shape}")

        # Final Stage: Concatenate output of Up2 with Initial Conv (x1)
        # Both x and x1 are now at original H, W dimensions.
        x_final_stage_concat = torch.cat((x, x1), dim=1) # (base_channels * 2 + base_channels) channels, H, W
        mask_final_stage_concat = (mask_up2.bool() | mask1.bool()).float()
        if verbose_forward: print(f"Before final_upsample_conv: X_final_stage_concat shape: {x_final_stage_concat.shape}, Mask_final_stage_concat shape: {mask_final_stage_concat.shape}")

        # Apply final upsample conv, norm, act
        h, final_mask_processed = self.final_upsample_conv(x_final_stage_concat, mask_final_stage_concat)
        h = self.final_norm(self.final_act(h))
        if verbose_forward: print(f"After final_upsample_conv and activation: H shape: {h.shape}, Final_mask_processed shape: {final_mask_processed.shape}")

        # Final output convolution
        output_prediction, _ = self.output_conv(h, final_mask_processed) # Output `out_channels`, H, W
        if verbose_forward: print(f"After output_conv (output prediction): {output_prediction.shape}")

        # Crucially, the model should only predict noise for the *ocean* regions.
        # Ensure land points are zeroed out in the final output prediction.
        # The input `mask` is the ground truth for land/ocean.
        final_output_mask_broadcast = mask.float().repeat(1, output_prediction.shape[1], 1, 1)
        output_prediction = output_prediction * final_output_mask_broadcast # Zero out land regions
        if verbose_forward: print(f"Final output after applying original mask: {output_prediction.shape}")
        if verbose_forward: print("--- End UNet Forward Pass ---")

        # Return only the output prediction
        return output_prediction

# Function to count trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
