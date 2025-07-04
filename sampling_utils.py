import torch
from tqdm import tqdm
import numpy as np

# Assuming config.py, unet_model.py, and diffusion_process.py are in the same directory
from config import Config # To access configuration parameters like image_size, channels, etc.
from unet_model import UNet # For the model type passed to sample_conditional
from diffusion_process import Diffusion # For the diffusion process logic

# --- Conditional Sampling Function ---
@torch.no_grad() # Disable gradient calculations for inference (sampling)
def sample_conditional(
    model, 
    diffusion, 
    num_samples, 
    observations, 
    observed_mask, 
    land_mask, 
    channels, 
    image_size, 
    device, 
    sampling_method='ddpm', 
    observation_fidelity_weight=1.0,
    target_dayofyear=None, 
    target_location_field=None, 
    config=None):
    """
    Generates new ocean states conditionally guided by sparse observations and a land mask.

    Args:
        model (nn.Module): The trained U-Net model.
        diffusion (Diffusion): Diffusion process utility object.
        num_samples (int): Number of samples to generate.
        observations (torch.Tensor): Tensor containing sparse observations (1, channels, H, W).
        observed_mask (torch.Tensor): Binary mask indicating observation locations (1, channels, H, W).
        land_mask (torch.Tensor): Global land/ocean mask (1, 1, H, W).
        channels (int): Number of data channels.
        image_size (tuple): Image spatial size (height, width).
        device (torch.device): Device to perform computations on.
        sampling_method (str): 'ddpm' for DDPM sampling, 'ddim' for DDIM sampling.
        observation_fidelity_weight (float): Weight for observations (0.0=ignore, 1.0=hard).

    Returns:
        torch.Tensor: Generated ocean states (num_samples, channels, H, W).
    """
    model.eval() # Set model to evaluation mode
    print("Starting conditional sampling...")
    print(f"Using {sampling_method.upper()} sampling method with observation fidelity weight {observation_fidelity_weight:.2f}")
    
    img_h, img_w = image_size
    # Start with pure random noise, but ensure land regions are zeroed out from the beginning.
    initial_noise = torch.randn((num_samples, channels, img_h, img_w), device=device)
    land_mask_expanded_channels = land_mask.float().repeat(num_samples, channels, 1, 1)
    x_t = initial_noise * land_mask_expanded_channels # Noise only exists on ocean regions

    # Ensure the land_mask for the U-Net input has batch dimension
    current_land_mask_batch = land_mask.float().repeat(num_samples, 1, 1, 1).to(device) # (N, 1, H, W)
    
    # Move observation tensors to the correct device
    observations_tensor = observations.to(device)
    observed_mask_tensor = observed_mask.to(device)

    # Prepare optional embedding inputs for the model
    doy_input_for_sampling = None
    if config.use_dayofyear_embedding:
        if target_dayofyear is None:
            raise ValueError("target_dayofyear must be provided if use_dayofyear_embedding is True.")
        doy_input_for_sampling = torch.full((num_samples,), target_dayofyear, device=device, dtype=torch.long)

    # Prepare 2D location field for sampling
    loc_field_input_for_sampling = None
    if config.use_2d_location_embedding:
        if target_location_field is None:
            raise ValueError("target_location_field must be provided if use_2d_location_embedding is True.")
        # target_location_field is (1, 2, H, W), repeat it for num_samples
        loc_field_input_for_sampling = target_location_field.repeat(num_samples, 1, 1, 1).to(device) # NEW

    # Prepare timesteps for DDIM, if used. For DDPM, iterate `reversed(range(0, config.timesteps))`.
    if sampling_method == 'ddim':
        timesteps_to_sample = list(reversed(range(0, diffusion.timesteps)))
    else: # 'ddpm'
        timesteps_to_sample = list(reversed(range(0, diffusion.timesteps)))

    # Flag to control verbose printing in UNet.forward for the first sampling iteration
    print_unet_forward_shapes_sample = True 

    # Iterate backward through the diffusion timesteps (denoising process)
    for i_idx, i in enumerate(tqdm(timesteps_to_sample, desc="Sampling")):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long) # Current timestep

        # Predict the noise for the current noisy image x_t, passing all conditioning inputs
        predicted_noise = model(x_t, t, current_land_mask_batch, 
                                dayofyear_batch=doy_input_for_sampling, 
                                location_field=loc_field_input_for_sampling, # MODIFIED: Pass location_field
                                verbose_forward=print_unet_forward_shapes_sample)

        # After the first forward pass, set the flag to False to suppress further verbose printing
        if print_unet_forward_shapes_sample:
            print_unet_forward_shapes_sample = False

        # Estimate the clean image (x_0_pred) from x_t and the predicted noise
        x_0_pred = diffusion.predict_x0_from_noise(x_t, t, predicted_noise, current_land_mask_batch)

        # Enforce observations for conditional sampling (softer enforcement) 
        effective_observed_mask = observed_mask_tensor.bool() * land_mask_expanded_channels.bool()
        
        # Potential to introduc localization schemes for the observations
        x_0_pred[effective_observed_mask] = (
            observation_fidelity_weight * observations_tensor[effective_observed_mask] +
            (1.0 - observation_fidelity_weight) * x_0_pred[effective_observed_mask]
        )
        
        x_0_pred = torch.clamp(x_0_pred, 0., 1.)
        x_0_pred = x_0_pred * land_mask_expanded_channels

        # If it's the very last step (t=0), x_0_pred is our final generated sample.
        if i == 0:
            x_t = x_0_pred
        else:
            if sampling_method == 'ddpm':
                t_prev_index = i - 1 
                sqrt_alpha_cumprod_prev_t = diffusion.sqrt_alphas_cumprod[t_prev_index] if t_prev_index >= 0 else torch.tensor(1.0).to(device)
                sqrt_one_minus_alpha_cumprod_prev_t = diffusion.sqrt_one_minus_alphas_cumprod[t_prev_index] if t_prev_index >= 0 else torch.tensor(0.0).to(device)

                noise_for_next_step = torch.randn_like(x_t) * land_mask_expanded_channels
                
                x_t = sqrt_alpha_cumprod_prev_t * x_0_pred + sqrt_one_minus_alpha_cumprod_prev_t * noise_for_next_step
            
            elif sampling_method == 'ddim':
                if i_idx + 1 < len(timesteps_to_sample):
                    t_prev = timesteps_to_sample[i_idx + 1]
                else:
                    t_prev = -1
                x_t = diffusion.ddim_sample(model, x_t, t, t_prev, i, current_land_mask_batch)
            
            x_t = x_t * land_mask_expanded_channels

    return x_t