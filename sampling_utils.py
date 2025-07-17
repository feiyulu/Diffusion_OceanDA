import torch
from tqdm import tqdm
import numpy as np

# Assuming config.py, unet_model.py, and diffusion_process.py are in the same directory
from config import Config # To access configuration parameters like image_size, channels, etc.
from unet_model import UNet # For the model type passed to sample_conditional
from diffusion_process import Diffusion, DPMSolver # Import the new DPMSolver

# --- Conditional Sampling Function ---
@torch.no_grad() # Disable gradient calculations for inference (sampling)
def sample_conditional(
    model, diffusion, num_samples, observations, observed_mask, land_mask, 
    channels, image_size, device, sampling_method='ddpm', 
    observation_fidelity_weight=1.0,
    target_conditions=None, 
    target_location_field=None, 
    config=None):
    """
    Generates new ocean states conditionally guided by sparse observations and a land mask.
    """
    model.eval() # Set model to evaluation mode
    print("Starting conditional sampling...")
    print(f"Using {sampling_method.upper()} sampling method.")
    if sampling_method.lower() == 'ddim':
        print(f"DDIM eta (stochasticity): {config.ddim_eta}")
    elif sampling_method.lower() == 'dpm-solver++':
        print(f"DPM-Solver++ will use {config.sampling_steps} steps.")
    
    img_h, img_w = image_size
    initial_noise = torch.randn((num_samples, channels, img_h, img_w), device=device)
    land_mask_expanded_channels = land_mask.float().repeat(num_samples, channels, 1, 1)
    x_t = initial_noise * land_mask_expanded_channels

    current_land_mask_batch = land_mask.float().repeat(num_samples, 1, 1, 1).to(device)
    observations_tensor = observations.to(device)
    observed_mask_tensor = observed_mask.to(device)

    conditions_input = None
    if target_conditions and config.conditioning_configs:
        conditions_input = {key: torch.full((num_samples,), val.item(), device=device) for key, val in target_conditions.items()}
    
    loc_field_input = None
    if config.use_2d_location_embedding and target_location_field is not None:
        loc_field_input = target_location_field.repeat(num_samples, 1, 1, 1).to(device)

    # --- Set up sampler-specific timestep schedule ---
    sampler_name = sampling_method.lower()
    if sampler_name in ['ddpm', 'ddim']:
        timesteps_to_sample = list(reversed(range(0, diffusion.timesteps)))
    elif sampler_name == 'dpm-solver++':
        steps = config.sampling_steps
        timesteps_to_sample = torch.linspace(diffusion.timesteps - 1, 0, steps + 1, device=device).long()
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    # --- Initialize DPM-Solver if needed ---
    dpm_solver = None
    model_s_list = [] # History of model outputs for multi-step solvers
    if sampler_name == 'dpm-solver++':
        dpm_solver = DPMSolver(diffusion.alphas_cumprod)

    print_unet_forward_shapes_sample = True 

    # --- Main Sampling Loop (Refactored) ---
    loop_iterator = timesteps_to_sample if sampler_name in ['ddpm', 'ddim'] else timesteps_to_sample[:-1]
    
    for i_idx, i in enumerate(tqdm(loop_iterator, desc="Sampling")):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        
        predicted_noise = model(x_t, t, current_land_mask_batch, 
                                conditions=conditions_input, 
                                location_field=loc_field_input,
                                verbose_forward=print_unet_forward_shapes_sample)
        if print_unet_forward_shapes_sample: print_unet_forward_shapes_sample = False

        # --- Guidance Step (common to all samplers) ---
        x_0_pred = diffusion.predict_x0_from_noise(x_t, t, predicted_noise, current_land_mask_batch)
        
        effective_observed_mask = observed_mask_tensor.bool() * land_mask_expanded_channels.bool()
        x_0_pred[effective_observed_mask] = (
            observation_fidelity_weight * observations_tensor[effective_observed_mask] +
            (1.0 - observation_fidelity_weight) * x_0_pred[effective_observed_mask]
        )
        x_0_pred = torch.clamp(x_0_pred, 0., 1.) * land_mask_expanded_channels

        # --- Sampler-specific update step ---
        if i == 0 and sampler_name in ['ddpm', 'ddim']:
            x_t = x_0_pred
            continue

        if sampler_name == 'ddpm':
            t_prev_index = i - 1 
            sqrt_alpha_cumprod_prev_t = diffusion.sqrt_alphas_cumprod[t_prev_index]
            sqrt_one_minus_alpha_cumprod_prev_t = diffusion.sqrt_one_minus_alphas_cumprod[t_prev_index]
            noise_for_next_step = torch.randn_like(x_t) * land_mask_expanded_channels
            x_t = sqrt_alpha_cumprod_prev_t * x_0_pred + sqrt_one_minus_alpha_cumprod_prev_t * noise_for_next_step
        
        elif sampler_name == 'ddim':
            # --- FIX: Use the refactored ddim_sample function ---
            t_prev = timesteps_to_sample[i_idx + 1] if i_idx + 1 < len(timesteps_to_sample) else -1
            # Call the refactored ddim_sample with the guided x_0_pred
            x_t = diffusion.ddim_sample(
                x_t, t, t_prev, x_0_pred, 
                current_land_mask_batch, 
                eta=config.ddim_eta
            )

        elif sampler_name == 'dpm-solver++':
            t_prev = timesteps_to_sample[i_idx + 1]
            sqrt_alpha_cumprod_t = diffusion.sqrt_alphas_cumprod[t, None, None, None]
            sqrt_one_minus_alpha_cumprod_t = diffusion.sqrt_one_minus_alphas_cumprod[t, None, None, None]
            guided_noise = (x_t - sqrt_alpha_cumprod_t * x_0_pred) / sqrt_one_minus_alpha_cumprod_t
            guided_noise *= land_mask_expanded_channels

            if not model_s_list: # First step of the solver
                x_t = dpm_solver.dpm_solver_first_order_update(guided_noise, i, t_prev.item(), x_t)
            else: # Second-order (or higher) step
                x_t = dpm_solver.multistep_dpm_solver_second_order_update(model_s_list, guided_noise, i, t_prev.item(), x_t)
            
            model_s_list.append({'s': i, 'output': guided_noise})
            if len(model_s_list) > 1:
                model_s_list.pop(0)

        x_t = x_t * land_mask_expanded_channels

    return x_t
