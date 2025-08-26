# --- sampling_utils.py ---
# This file contains the core logic for generating new samples from the trained model.
# REFINED: Integrated DPM-Solver++ for accelerated sampling and added num_samples argument for serial generation.
import torch
from tqdm import tqdm

from diffusion_process import Diffusion, DPMSolver

@torch.no_grad()
def sample_conditional(model, diffusion, config, observations, observed_mask, land_mask, 
                       target_conditions, target_location_field, num_samples=None):
    """
    Generates new 3D ocean states conditionally guided by sparse observations.
    This function is completely refactored to handle 5D tensors (N, C, D, H, W)
    and supports multiple sampling methods (DDPM, DDIM, DPM-Solver++).
    """
    model.eval()
    
    # If num_samples is not specified, use the value from the config file
    if num_samples is None:
        num_samples = config.ensemble_size

    # Get data shape from config
    C, D, H, W = (config.channels, *config.data_shape)
    device = config.device

    # Generate initial noise with the correct 5D shape
    initial_noise = torch.randn((num_samples, C, D, H, W), device=device)
    land_mask_batch = land_mask.repeat(num_samples, 1, 1, 1, 1).to(device)
    x_t = initial_noise * land_mask_batch

    # Prepare observation and conditional tensors
    observations_tensor = observations.repeat(num_samples, 1, 1, 1, 1).to(device)
    observed_mask_tensor = observed_mask.repeat(num_samples, 1, 1, 1, 1).to(device)

    conditions_input = None
    if target_conditions and config.conditioning_configs:
        conditions_input = {
            key: torch.full((num_samples,), val.item(), device=device) 
            for key, val in target_conditions.items()
        }
    
    loc_field_input = None
    if config.location_embedding_channels > 0 and target_location_field is not None:
        loc_field_input = target_location_field.repeat(num_samples, 1, 1, 1).to(device)

    # --- Set up sampler-specific timestep schedule ---
    sampler_name = config.sampling_method.lower()
    if sampler_name in ['ddpm', 'ddim']:
        timesteps_to_sample = list(reversed(range(0, diffusion.timesteps)))
        sampling_steps = diffusion.timesteps
    elif sampler_name == 'dpm-solver++':
        sampling_steps = config.sampling_steps
        # Create the specific timestep schedule for DPM-Solver
        timesteps_to_sample = torch.linspace(diffusion.timesteps - 1, 0, sampling_steps + 1, device=device).long().tolist()
        dpm_solver = DPMSolver(diffusion.alphas_cumprod)
        model_s_list = [] # Stores previous model outputs for higher-order steps
    else:
        raise ValueError(f"Unknown sampling method: {sampler_name}")

    # --- Main Sampling Loop ---
    # The progress bar is now handled in the main ensemble_sampling script
    for i, step in enumerate(timesteps_to_sample[:-1]):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        
        # 1. Predict noise using the U-Net model
        predicted_noise = model(x_t, t, land_mask_batch, 
                                conditions=conditions_input, 
                                location_field=loc_field_input)

        # 2. Guidance Step: Predict x0 and nudge it towards the observations
        x_0_pred = diffusion.predict_x0_from_noise(x_t, t, predicted_noise, land_mask_batch)
        
        # Apply observation fidelity weight only on observed points
        effective_observed_mask = observed_mask_tensor.bool() & land_mask_batch.bool()
        guided_x_0 = torch.where(
            effective_observed_mask,
            config.observation_fidelity_weight * observations_tensor + (1.0 - config.observation_fidelity_weight) * x_0_pred,
            x_0_pred
        )
        guided_x_0 = torch.clamp(guided_x_0, 0., 1.) * land_mask_batch

        # 3. Sampler-specific update step to get x_{t-1}
        t_prev_step = timesteps_to_sample[i + 1]
        t_prev = torch.full((num_samples,), t_prev_step, device=device, dtype=torch.long)

        if sampler_name == 'ddpm':
            x_t = diffusion.p_sample_from_x0(x_t, t, guided_x_0, land_mask_batch)
        
        elif sampler_name == 'ddim':
            x_t = diffusion.ddim_sample(x_t, t, t_prev, guided_x_0, land_mask_batch, eta=config.ddim_eta)

        elif sampler_name == 'dpm-solver++':
            # Use the guided x0 to get the guided noise prediction
            guided_noise = (x_t - diffusion.sqrt_alphas_cumprod[t[0]] * guided_x_0) / diffusion.sqrt_one_minus_alphas_cumprod[t[0]]
            
            if dpm_solver.order == 1 or len(model_s_list) == 0:
                # First-order update
                x_t = dpm_solver.dpm_solver_first_order_update(guided_noise, step, t_prev_step, x_t)
            else:
                # Second-order update
                x_t = dpm_solver.multistep_dpm_solver_second_order_update(model_s_list, guided_noise, step, t_prev_step, x_t)
            
            # Update the history for the next step
            model_s_list.append({'s': step, 'output': guided_noise})
            if len(model_s_list) > dpm_solver.order:
                model_s_list.pop(0)

        # Ensure the result stays within the ocean mask
        x_t = x_t * land_mask_batch

    # The final sample is the result of the last step
    final_sample = x_t

    # Return the final denoised and guided sample
    return final_sample
