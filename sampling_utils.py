# --- sampling_utils.py ---
# This file contains the core logic for generating new samples from the trained model.
import torch
from tqdm import tqdm

from diffusion_process import Diffusion, DPMSolver

@torch.no_grad()
def apply_observation_guidance(x_0_pred, observations, observed_mask, guidance_strength_mask, config):
    """
    Applies guidance to the predicted x0, nudging it towards observations.
    Handles different guidance strengths and stochastic masking for dense data.
    """
    guided_x_0 = x_0_pred.clone()
    
    # Find all unique guidance configurations from the active sources
    active_sources = [s for s in config.observation_sources if s.get("enabled", False)]

    for source in active_sources:
        # Get the mask for the current observation type
        # This is a bit of a placeholder; a real implementation might need a more robust
        # way to link source configs to the mask segments. For now, we assume the masks
        # from different sources are mutually exclusive, which holds for our synthetic data.
        
        # We need to create a mask specific to this source's strength
        source_strength = source.get("guidance_strength", 1.0)
        source_mask = (observed_mask.bool()) & (guidance_strength_mask == source_strength)

        if not torch.any(source_mask):
            continue

        # --- Handle stochastic masking for dense data ---
        subsample_frac = source.get("subsample_fraction")
        if subsample_frac is not None and subsample_frac < 1.0:
            # Flatten the mask to easily subsample
            source_mask_flat = source_mask.flatten()
            true_indices = torch.where(source_mask_flat)[0]
            
            # Choose a random subset of points
            num_to_sample = int(len(true_indices) * subsample_frac)
            sampled_indices = true_indices[torch.randperm(len(true_indices))[:num_to_sample]]
            
            # Create the new subsampled mask
            final_guidance_mask_flat = torch.zeros_like(source_mask_flat)
            final_guidance_mask_flat[sampled_indices] = True
            source_mask = final_guidance_mask_flat.reshape(source_mask.shape)
            
        operator = source.get("operator", "point_replacement").lower()
        
        if operator == "point_replacement":
            # Nudge the prediction towards the observation
            # x_guided = w * y_obs + (1 - w) * x_pred
            guided_x_0[source_mask] = (source_strength * observations[source_mask] +
                                       (1.0 - source_strength) * x_0_pred[source_mask])
        else:
            raise ValueError(f"Unknown observation operator: {operator}")

    return guided_x_0


@torch.no_grad()
def sample_conditional(model, diffusion, config, observations, observed_mask, 
                       guidance_strength_mask, land_mask, 
                       target_conditions, target_location_field, num_samples=None):
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

    for i, step in enumerate(timesteps_to_sample[:-1]):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        
        predicted_noise = model(x_t, t, land_mask_batch, conditions=conditions_input, location_field=loc_field_input)

        # --- Guidance Step ---
        x_0_pred = diffusion.predict_x0_from_noise(x_t, t, predicted_noise, land_mask_batch)
        
        # NEW: Call the dedicated guidance function
        guided_x_0 = apply_observation_guidance(
            x_0_pred, observations_tensor, observed_mask_tensor, guidance_strength_tensor, config
        )
        
        guided_x_0 = torch.clamp(guided_x_0, 0., 1.) * land_mask_batch

        # --- Sampler-specific update step ---
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
            if len(model_s_list) > dpm_solver.order:
                model_s_list.pop(0)

        x_t = x_t * land_mask_batch

    return x_t
