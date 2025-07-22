# --- sampling_utils.py ---
# This file contains the core logic for generating new samples from the trained model.
import torch
from tqdm import tqdm

from diffusion_process import Diffusion, DPMSolver

@torch.no_grad()
def sample_conditional(model, diffusion, config, observations, observed_mask, land_mask, 
                       target_conditions, target_location_field):
    """
    Generates new 3D ocean states conditionally guided by sparse observations.
    This function is completely refactored to handle 5D tensors (N, C, D, H, W).
    """
    model.eval()
    print("Starting conditional sampling...")
    print(f"Using {config.sampling_method.upper()} sampling method.")
    
    # Get data shape from config
    _, C, D, H, W = (config.ensemble_size, config.channels, *config.data_shape)
    device = config.device
    num_samples = config.ensemble_size

    # FIX: Generate initial noise with the correct 5D shape
    initial_noise = torch.randn((num_samples, C, D, H, W), device=device)
    # The land mask is already 5D from the data loader, just repeat for batch
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
        timesteps_to_sample = torch.linspace(diffusion.timesteps - 1, 0, sampling_steps + 1, device=device).long().tolist()
    else:
        raise ValueError(f"Unknown sampling method: {sampler_name}")

    # --- Main Sampling Loop ---
    for i, step in enumerate(tqdm(timesteps_to_sample, desc="Sampling", total=len(timesteps_to_sample))):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        
        # 1. Predict noise using the U-Net model
        predicted_noise = model(x_t, t, land_mask_batch, 
                                conditions=conditions_input, 
                                location_field=loc_field_input)

        # 2. Guidance Step: Predict x0 and nudge it towards the observations
        x_0_pred = diffusion.predict_x0_from_noise(x_t, t, predicted_noise, land_mask_batch)
        
        # Apply observation fidelity weight only on observed points
        # The mask multiplication ensures this happens only on ocean grid cells with observations
        effective_observed_mask = observed_mask_tensor.bool() & land_mask_batch.bool()
        guided_x_0 = torch.where(
            effective_observed_mask,
            config.observation_fidelity_weight * observations_tensor + (1.0 - config.observation_fidelity_weight) * x_0_pred,
            x_0_pred
        )
        guided_x_0 = torch.clamp(guided_x_0, 0., 1.) * land_mask_batch

        # 3. Sampler-specific update step to get x_{t-1}
        if step == 0:
            x_t = guided_x_0
            continue

        t_prev_step = timesteps_to_sample[i + 1] if i + 1 < len(timesteps_to_sample) else -1
        t_prev = torch.full((num_samples,), t_prev_step, device=device, dtype=torch.long)

        if sampler_name == 'ddpm':
            # For DDPM, the reverse step is derived from the guided x0
            x_t = diffusion.p_sample_from_x0(x_t, t, guided_x_0, land_mask_batch)
        
        elif sampler_name == 'ddim':
            # For DDIM, we use the specific DDIM sampling equation
            x_t = diffusion.ddim_sample(x_t, t, t_prev, guided_x_0, land_mask_batch, eta=config.ddim_eta)
        
        # (DPM-Solver logic would go here if implemented)

        # Ensure the result stays within the ocean mask
        x_t = x_t * land_mask_batch

    # Return the final denoised and guided sample
    return x_t
