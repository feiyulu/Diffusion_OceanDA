# --- sampling_utils.py ---
# This file contains the core logic for generating new samples from the trained model.
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import xarray as xr
from collections import defaultdict
from scipy.spatial import cKDTree
 
from diffusion_process import Diffusion, DPMSolver

def _create_gaussian_kernel(radius, sigma, device):
    """Creates a 2D Gaussian kernel."""
    kernel_size = 2 * radius + 1
    x_cord = torch.arange(kernel_size, device=device, dtype=torch.float32)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    
    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
    
    # Make sure the sum of the kernel is 1
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

@torch.no_grad()
def apply_observation_guidance(x_0_pred, observations, observed_mask, guidance_strength_mask, land_mask_batch, config, operators_to_apply, model, diffusion, conditions_input, loc_field_input, prev_state_input):
    """
    Applies observation-based guidance to the model's prediction of the clean state (x0).
    This is a key step in data assimilation, nudging the model towards reality.
    It iterates through different observation sources and applies their specified operator.
    """
    guided_x_0 = x_0_pred.clone()
    B, C, D, H, W = x_0_pred.shape
    
    # Find all unique guidance configurations from the active sources
    active_sources = [s for s in config.observation_sources if s.get("enabled", False)]

    for source in active_sources:
        source_strength = source.get("guidance_strength", 1.0)
        source_mask = (observed_mask.bool()) & (guidance_strength_mask == source_strength)

        if not torch.any(source_mask):
            continue

        # Subsampling is handled during data creation, so we don't need to do it here.
        # The mask passed in is already the final one.
            
        operator = source.get("operator", "point_replacement").lower()
        
        # Skip this source if its operator is not in the list of operators to apply for this call.
        if operator not in operators_to_apply:
            continue
        
        # 'point_replacement': The simplest form of guidance. Directly nudges the model state
        # at observation locations towards the observed value.
        if operator == "point_replacement":
            # Nudge the prediction towards the observation
            # x_guided = w * y_obs + (1 - w) * x_pred
            nudge = source_strength * (observations - x_0_pred)
            guided_x_0[source_mask] += nudge[source_mask]

        # 'localized_innovation': A more sophisticated method that spreads the influence of an
        # observation to nearby grid cells using a Gaussian kernel.
        elif operator == "localized_innovation":
            radius = source.get("localization_radius", 3)
            sigma = radius / 2.0 # A reasonable default for sigma
            
            # 1. Create the Gaussian kernel for spreading
            kernel = _create_gaussian_kernel(radius, sigma, x_0_pred.device)
            padding = radius

            # 2. Calculate the innovation (observation - prediction) only at observation points
            innovation = (observations - x_0_pred) * source_mask

            # 3. Reshape for 2D convolution (apply to each channel and depth level independently)
            # (B, C, D, H, W) -> (B*C*D, 1, H, W)
            innovation_flat = innovation.view(B * C * D, 1, H, W)
            mask_flat = source_mask.float().view(B * C * D, 1, H, W)

            # 4. Convolve both the innovation and the mask
            smoothed_innovation = F.conv2d(innovation_flat, kernel, padding=padding)
            smoothed_mask_weights = F.conv2d(mask_flat, kernel, padding=padding)

            # 5. Normalize the smoothed innovation to prevent artifacts from overlapping kernels
            normalized_smoothed_innovation = smoothed_innovation / (smoothed_mask_weights + 1e-9)
            
            # 6. Reshape back and apply the update
            update_field = normalized_smoothed_innovation.view(B, C, D, H, W)
            guided_x_0 += source_strength * update_field

        # 'resampling_guidance': A powerful method that uses a short diffusion-denoising
        # loop to let the model itself spread the innovation in a physically realistic way.
        elif operator == "resampling_guidance":
            resampling_steps = source.get("resampling_steps", 50)
            if resampling_steps <= 0:
                continue
            
            # 1. Create a sparse innovation field
            innovation = (observations - x_0_pred) * source_mask
            
            # 2. Create the initial state for resampling by adding the sparse correction
            x_0_hybrid = x_0_pred + source_strength * innovation
            x_0_hybrid = torch.clamp(x_0_hybrid, 0., 1.) * land_mask_batch

            # 3. Diffuse this hybrid state forward for a few steps
            t_resample = torch.tensor([resampling_steps - 1], device=x_0_pred.device)
            x_t_hybrid, _ = diffusion.noise_images(x_0_hybrid, t_resample, land_mask_batch)

            # 4. Denoise the hybrid state back to step 0 using the chosen inner sampler
            x_resampled = x_t_hybrid
            resampling_method = source.get("resampling_method", "dpm-solver++").lower()

            if resampling_method == 'ddpm':
                # Slower, more robust DDPM sampler
                for i in tqdm(reversed(range(resampling_steps)), desc=f"DDPM Resampling ({source['name']})", leave=False):
                    t = torch.full((x_0_pred.shape[0],), i, device=x_0_pred.device, dtype=torch.long)
                    pred_noise = model(x_resampled, t, land_mask_batch, conditions=conditions_input, location_field=loc_field_input, prev_state_surface=prev_state_input)
                    x_0_from_noise = diffusion.predict_x0_from_noise(x_resampled, t, pred_noise, land_mask_batch)
                    x_resampled = diffusion.p_sample_from_x0(x_resampled, t, x_0_from_noise, land_mask_batch)
            
            elif resampling_method == 'dpm-solver++':
                # Faster DPM-Solver++
                dpm_solver = DPMSolver(diffusion.alphas_cumprod)
                model_s_list = []
                inner_timesteps = torch.linspace(resampling_steps - 1, 0, resampling_steps + 1, device=x_0_pred.device).long().tolist()

                for i, step in enumerate(tqdm(inner_timesteps[:-1], desc=f"Fast Resampling ({source['name']})", leave=False)):
                    t = torch.full((x_0_pred.shape[0],), step, device=x_0_pred.device, dtype=torch.long)
                    t_prev_step = inner_timesteps[i + 1]
                    pred_noise = model(x_resampled, t, land_mask_batch, conditions=conditions_input, location_field=loc_field_input, prev_state_surface=prev_state_input)

                    if len(model_s_list) == 0:
                        x_resampled = dpm_solver.dpm_solver_first_order_update(pred_noise, step, t_prev_step, x_resampled)
                    else:
                        x_resampled = dpm_solver.multistep_dpm_solver_second_order_update(model_s_list, pred_noise, step, t_prev_step, x_resampled)
                    
                    # Correctly manage the history for the next step
                    model_s_list.append({'s': step, 'output': pred_noise})
                    if len(model_s_list) > 1: # DPM-Solver++ 2nd order needs a history of size 1
                        model_s_list.pop(0) # Keep the list at size 1
            else:
                raise ValueError(f"Unknown resampling_method: {resampling_method}")

            guided_x_0 = x_resampled

        else:
            raise ValueError(f"Unknown observation operator: {operator}")

    return guided_x_0


@torch.no_grad()
def sample_conditional(model, diffusion, config, observations, observed_mask, 
                       guidance_strength_mask, land_mask, 
                       target_conditions, target_location_field,
                       prev_state_surface=None, num_samples=None):
    """
    Generates new 3D ocean states conditionally guided by multi-source observations.
    """
    model.eval()
    if num_samples is None:
        num_samples = config.ensemble_size

    C, D, H, W = (config.channels, *config.data_shape)
    device = config.device

    initial_noise = torch.randn((num_samples, C, D, H, W), device=device)
    land_mask_batch = land_mask.repeat(num_samples, 1, 1, 1, 1).to(device)
    x_t = initial_noise * land_mask_batch

    observations_tensor = observations.repeat(num_samples, 1, 1, 1, 1).to(device)
    observed_mask_tensor = observed_mask.repeat(num_samples, 1, 1, 1, 1).to(device)
    guidance_strength_tensor = guidance_strength_mask.repeat(num_samples, 1, 1, 1, 1).to(device)

    conditions_input = {key: torch.full((num_samples,), val.item(), device=device) for key, val in target_conditions.items()} if target_conditions else None
    loc_field_input = target_location_field.repeat(num_samples, 1, 1, 1).to(device) if config.location_embedding_channels > 0 and target_location_field is not None else None
    prev_state_input = prev_state_surface.repeat(num_samples, 1, 1, 1).to(device) if config.previous_states and prev_state_surface is not None else None

    sampler_name = config.sampling_method.lower()
    if sampler_name in ['ddpm', 'ddim']:
        timesteps_to_sample = list(reversed(range(0, diffusion.timesteps)))
    elif sampler_name == 'dpm-solver++':
        sampling_steps = config.sampling_steps
        timesteps_to_sample = torch.linspace(diffusion.timesteps - 1, 0, sampling_steps + 1, device=device).long().tolist()
        dpm_solver = DPMSolver(diffusion.alphas_cumprod)
        model_s_list = []
    else:
        raise ValueError(f"Unknown sampling method: {sampler_name}")

    for i, step in enumerate(tqdm(timesteps_to_sample[:-1], desc="Sampling Timesteps")):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)

        # --- Step 1: Get a noise prediction from the model ---
        predicted_noise = model(x_t, t, land_mask_batch, conditions=conditions_input, 
                                location_field=loc_field_input, prev_state_surface=prev_state_input)

        # --- Step 2: Predict x0 from the (potentially guided) noise ---
        x_0_pred = diffusion.predict_x0_from_noise(x_t, t, predicted_noise, land_mask_batch)

        # --- Step 3: Apply guidance in the data space (e.g., localized_innovation) ---
        # This function nudges the predicted clean state (x0) towards the observations.
        # We apply all data-space operators here. Latent blending is handled separately above.
        data_space_ops = ["point_replacement", "localized_innovation", "resampling_guidance"]
        guided_x_0 = apply_observation_guidance(x_0_pred, observations_tensor, observed_mask_tensor, guidance_strength_tensor, land_mask_batch, config, operators_to_apply=data_space_ops, model=model, diffusion=diffusion, conditions_input=conditions_input, loc_field_input=loc_field_input, prev_state_input=prev_state_input)
        
        # Ensure the guided state is physically plausible and respects the land mask.
        guided_x_0 = torch.clamp(guided_x_0, 0., 1.) * land_mask_batch

        # --- Step 4: Use the guided x0 to take one step in the reverse diffusion process ---
        # This computes the state at the next (less noisy) timestep.
        t_prev_step = timesteps_to_sample[i + 1]
        t_prev = torch.full((num_samples,), t_prev_step, device=device, dtype=torch.long)

        if sampler_name == 'ddpm':
            x_t = diffusion.p_sample_from_x0(x_t, t, guided_x_0, land_mask_batch)
        elif sampler_name == 'ddim':
            x_t = diffusion.ddim_sample(x_t, t, t_prev, guided_x_0, land_mask_batch, eta=config.ddim_eta)
        elif sampler_name == 'dpm-solver++':
            guided_noise = (x_t - diffusion.sqrt_alphas_cumprod[t[0]] * guided_x_0) / diffusion.sqrt_one_minus_alphas_cumprod[t[0]]
            if len(model_s_list) == 0:
                x_t = dpm_solver.dpm_solver_first_order_update(guided_noise, step, t_prev_step, x_t)
            else:
                x_t = dpm_solver.multistep_dpm_solver_second_order_update(model_s_list, guided_noise, step, t_prev_step, x_t)
            model_s_list.append({'s': step, 'output': guided_noise})
            if len(model_s_list) > 1:
                model_s_list.pop(0)

        x_t = x_t * land_mask_batch

    return x_t
