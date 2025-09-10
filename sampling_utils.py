# --- sampling_utils.py ---
# This file contains the core logic for generating new samples from the trained model.
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

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
def apply_observation_guidance(x_0_pred, observations, observed_mask, guidance_strength_mask, config):
    """
    Applies guidance to the predicted x0, nudging it towards observations.
    Handles different guidance strengths and stochastic masking for dense data.
    """
    guided_x_0 = x_0_pred.clone()
    B, C, D, H, W = x_0_pred.shape
    
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
            nudge = source_strength * (observations - x_0_pred)
            guided_x_0[source_mask] += nudge[source_mask]

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

        elif operator == "latent_blending":
            # This operator is handled inside the main sampling loop, as it needs
            # to interact with the model's internal states. We just use this
            # block to acknowledge the operator exists. The actual guidance happens
            # by recalculating the predicted_noise in the main loop.
            # The `guided_x_0` from the unguided pass is returned, and the magic
            # happens in the next step of the `sample_conditional` function.
            pass

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
        
        # --- Step 1: Get a noise prediction. This will be guided if latent_blending is active. ---
        is_latent_blending = any(s.get("operator", "").lower() == "latent_blending" for s in config.observation_sources if s.get("enabled"))

        if is_latent_blending:
            # --- Latent Blending Guidance ---
            # 1a. Get the original unguided prediction
            time_emb = model.time_mlp(t)
            latent_original_2d, h_mask_orig = model.encode_vertical(x_t, land_mask_batch)
            if loc_field_input is not None:
                latent_original_2d = torch.cat([latent_original_2d, loc_field_input], dim=1)
                loc_mask = torch.ones_like(loc_field_input)
                unet_mask_orig = torch.cat([h_mask_orig.repeat(1, sum(model.latent_dims), 1, 1), loc_mask * h_mask_orig], dim=1)
            
            bottleneck_original, _, skips_original = model.unet_2d.encode(latent_original_2d, time_emb, unet_mask_orig)
            
            # 1b. Create the "known" state and encode it
            x_0_pred_orig = diffusion.predict_x0_from_noise(x_t, t, model(x_t, t, land_mask_batch, conditions=conditions_input, location_field=loc_field_input), land_mask_batch)
            x_0_known = torch.where(observed_mask_tensor, observations_tensor, x_0_pred_orig)
            x_t_known, _ = diffusion.noise_images(x_0_known, t, land_mask_batch)

            latent_known_2d, h_mask_known = model.encode_vertical(x_t_known, land_mask_batch)
            if loc_field_input is not None:
                latent_known_2d = torch.cat([latent_known_2d, loc_field_input], dim=1)
                unet_mask_known = torch.cat([h_mask_known.repeat(1, sum(model.latent_dims), 1, 1), loc_mask * h_mask_known], dim=1)

            bottleneck_known, _, _ = model.unet_2d.encode(latent_known_2d, time_emb, unet_mask_known)

            # 1c. Blend the latent space and decode to get a guided noise prediction
            source_cfg = next(s for s in config.observation_sources if s.get("operator", "").lower() == "latent_blending")
            w = source_cfg.get("guidance_strength", 0.5)
            bottleneck_guided = (1 - w) * bottleneck_original + w * bottleneck_known
            
            # Create a correct mask for the bottleneck tensor's shape
            bottleneck_channels = bottleneck_guided.shape[1]
            bottleneck_mask = h_mask_orig.repeat(1, bottleneck_channels, 1, 1)
            
            guided_latent_output = model.unet_2d.decode(bottleneck_guided, bottleneck_mask, skips_original, time_emb)
            predicted_noise = model.decode_vertical(x_t, guided_latent_output, land_mask_batch)
        else:
            # --- Standard (Unguided) Prediction ---
            predicted_noise = model(x_t, t, land_mask_batch, conditions=conditions_input, location_field=loc_field_input)

        # --- Step 2: Predict x0 from the (potentially guided) noise ---
        x_0_pred = diffusion.predict_x0_from_noise(x_t, t, predicted_noise, land_mask_batch)

        # --- Step 3: Apply any output-space guidance methods (e.g., localized_innovation) ---
        # This function will skip any sources that use 'latent_blending'.
        guided_x_0 = apply_observation_guidance(x_0_pred, observations_tensor, observed_mask_tensor, guidance_strength_tensor, config)
        
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
