# --- ensemble_sampling.py ---
# This script is dedicated to generating an ensemble of samples from a trained model
# to evaluate its performance and quantify uncertainty.
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import xarray as xr
import os
import argparse

# Import components from other project files
from config import Config
from data_utils import load_ocean_data
from unet_model import UNet, count_parameters
from diffusion_process import Diffusion
from sampling_utils import sample_conditional

def plot_ensemble_results(
    ensemble_mean, ensemble_spread, true_sample, clim_pred, 
    obs_points_actual, land_mask_np, config, sample_day, num_obs_points, 
    select_size=None):
    """
    Visualizes the results of the ensemble sampling, including mean, spread, and comparisons.
    """
    print("\nVisualizing and saving ensemble results...")
    
    ensemble_size = config.ensemble_size if select_size is None else select_size
    ensemble_mean_np = ensemble_mean.cpu().numpy()
    ensemble_spread_np = ensemble_spread.cpu().numpy()
    true_sample_np = true_sample.cpu().numpy()

    num_channels = config.channels
    fig, axes = plt.subplots(6, num_channels, figsize=(12*num_channels, 24), squeeze=False)

    for c in range(num_channels):
        var_name = "Temperature" if c == 0 else "Salinity"
        cmap = 'viridis' if c == 0 else 'plasma'

        # Mask out land areas for plotting
        masked_true = np.ma.masked_where(land_mask_np == 0, true_sample_np[c])
        masked_mean = np.ma.masked_where(land_mask_np == 0, ensemble_mean_np[c])
        masked_spread = np.ma.masked_where(land_mask_np == 0, ensemble_spread_np[c])
        masked_error = np.ma.masked_where(land_mask_np == 0, ensemble_mean_np[c] - true_sample_np[c])
        clim_error = np.ma.masked_where(land_mask_np == 0, clim_pred[c] - true_sample_np[c])

        # 1. Ground Truth
        ax = axes[0,c]
        im = ax.imshow(masked_true, cmap=cmap, origin='lower', vmin=0., vmax=1.)
        for c, y, x, val in obs_points_actual:
                if c == 0 and land_mask_np[y, x] == 1: # Only plot if it was on ocean
                    ax.scatter(x, y, c=val, cmap=cmap, vmin=0., vmax=1., marker='o', s=5)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Ground Truth {var_name}')
        ax.set_xlabel('X-coordinate'); ax.set_ylabel('Y-coordinate')

        # 2. Climatology
        ax = axes[1,c]
        im = ax.imshow(clim_pred[c], cmap=cmap, origin='lower', vmin=0., vmax=1.)
        std = np.nanstd(clim_pred[c])
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        std = np.nanstd(masked_mean)
        ax.set_title(f'Climatology (Std: {std:.4f})')
        ax.set_xlabel('X-coordinate'); ax.set_ylabel('Y-coordinate')

        # 3. Ensemble mean
        ax = axes[2,c]
        im = ax.imshow(masked_mean, cmap=cmap, origin='lower', vmin=0., vmax=1.)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        std = np.nanstd(masked_mean)
        ax.set_title(f'Ensemble Mean (Std: {std:.4f})')
        ax.set_xlabel('X-coordinate'); ax.set_ylabel('Y-coordinate')

        # 4. Ensemble Spread (Uncertainty)
        ax = axes[3,c]
        im = ax.imshow(masked_spread, cmap='inferno', origin='lower')
        plt.colorbar(im, ax=ax, label='Std. Dev.')
        ax.set_title(f'Ensemble Spread')
        ax.set_xlabel('X-coordinate'); ax.set_ylabel('Y-coordinate')

        # 5. Climatological error
        ax = axes[4,c]
        im = ax.imshow(clim_error, cmap='bwr', origin='lower', vmin=-0.1, vmax=0.1)
        plt.colorbar(im, ax=ax, label=f'Error {var_name}')
        rmse = np.sqrt(np.mean(clim_error**2))
        ax.set_title(f'Climatology Error (RMSE: {rmse:.4f})')
        ax.set_xlabel('X-coordinate'); ax.set_ylabel('Y-coordinate')

        # 6. Mean Error (Bias)
        ax = axes[5,c]
        im = ax.imshow(masked_error, cmap='bwr', origin='lower', vmin=-0.1, vmax=0.1)
        rmse = np.sqrt(np.mean(masked_error**2))
        plt.colorbar(im, ax=ax, label=f'Error {var_name}')
        ax.set_title(f'Ensemble Mean Error (RMSE: {rmse:.4f})')
        ax.set_xlabel('X-coordinate'); ax.set_ylabel('Y-coordinate')

    plt.tight_layout()
    plot_save_path = (
        f"{config.sample_plot_dir}/"
        f"ensemble_obs{num_obs_points}_day{sample_day[0]}_"
        f"ens{ensemble_size}_{config.sampling_method}.png")
    plt.savefig(plot_save_path)
    print(f"Ensemble plot saved to {plot_save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble sampling for Ocean Diffusion Model.")
    parser.add_argument("--work_path", "-w", type=str, required=True, help="Working directory containing the config and model checkpoints.")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to the JSON configuration file relative to work_path.")
    args = parser.parse_args()

    # --- 1. Initialization ---
    config_path = os.path.join(args.work_path, args.config)
    try:
        config = Config.from_json_file(config_path)
    except FileNotFoundError as e:
        print(e); raise

    print(f"Using device: {config.device}")
    print(f"Ensemble size: {config.ensemble_size}")

    # Initialize model and diffusion process
    model = UNet(
        in_channels=config.channels, 
        out_channels=config.channels, 
        base_channels=config.base_unet_channels,
        conditioning_configs=config.conditioning_configs,
        use_2d_location_embedding=config.use_2d_location_embedding,
        location_embedding_channels=config.location_embedding_channels
    ).to(config.device)
    
    diffusion = Diffusion(
        timesteps=config.timesteps, 
        beta_start=config.beta_start, 
        beta_end=config.beta_end, 
        img_size=config.image_size, 
        device=config.device)

    # --- 2. Load Pre-trained Model ---
    latest_checkpoint_path = None
    if os.path.exists(config.model_checkpoint_dir):
        checkpoints = [f for f in os.listdir(config.model_checkpoint_dir) if f.startswith(f"ODA_ch{config.channels}_{config.test_id}_epoch_") and f.endswith(".pth")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
            latest_checkpoint_path = os.path.join(config.model_checkpoint_dir, checkpoints[-1])
    
    if not latest_checkpoint_path:
        raise FileNotFoundError(f"No model checkpoint found in {config.model_checkpoint_dir}. Cannot perform sampling.")
    
    print(f"Loading model from {latest_checkpoint_path} for sampling...")
    checkpoint = torch.load(latest_checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model from epoch {checkpoint['epoch']} loaded successfully.")

    # --- 3. Prepare Data and Observations ---
    land_mask_global = None
    for sample_day in config.sample_days:
        true_sample, land_mask, true_conditions_dict, true_location_field_single, _, _ = load_ocean_data(
            config,
            time_range=sample_day,
            is_test_data=True
        )
        true_sample = true_sample.to(config.device)
        land_mask_global = land_mask.to(config.device)
        target_location_field = true_location_field_single.to(config.device)

        # Get climatological prediction as baseline skills
        clim_pred=[]
        ds_t_training = xr.open_mfdataset(config.filepath_t)
        da_t_training = ds_t_training[config.varname_t]
        clim_t = da_t_training.groupby('time.dayofyear').mean('time')
        clim_t_pred = (clim_t.isel(
            lat=slice(config.lat_range[0],config.lat_range[1]),
            lon=slice(config.lon_range[0],config.lon_range[1]),
            dayofyear=sample_day).values-config.T_range[0])/\
                (config.T_range[1]-config.T_range[0])
        print(f'Climatological Temperature Shape: {clim_t_pred.shape}')
        clim_pred.append(clim_t_pred.squeeze())

        if config.use_salinity: 
                ds_s_training = xr.open_mfdataset(config.filepath_s)
                da_s_training = ds_s_training[config.varname_s]
                clim_s = da_s_training.groupby('time.dayofyear').mean('time')
                clim_s_pred = (clim_s.isel(
                    lat=slice(config.lat_range[0],config.lat_range[1]),
                    lon=slice(config.lon_range[0],config.lon_range[1]),
                    dayofyear=sample_day).values-config.S_range[0])/\
                        (config.S_range[1]-config.S_range[0])
                clim_pred.append(clim_s_pred.squeeze())

        for num_obs_points in config.observation_samples:
            observations = torch.zeros_like(true_sample)
            observed_mask = torch.zeros_like(true_sample, dtype=torch.bool)
            ocean_coords_h, ocean_coords_w = np.where(land_mask_global[0, 0].cpu().numpy() == 1)
            all_ocean_points = list(zip(ocean_coords_h, ocean_coords_w))

            if len(all_ocean_points) < num_obs_points:
                sampled_indices = np.arange(len(all_ocean_points))
            else:
                sampled_indices = np.random.choice(len(all_ocean_points), num_obs_points, replace=False)
            
            obs_points_actual = []
            for idx in sampled_indices:
                y, x = all_ocean_points[idx]
                for c in range(config.channels):
                    val = true_sample[0, c, y, x].item()
                    observations[0, c, y, x] = val
                    observed_mask[0, c, y, x] = True
                    obs_points_actual.append((c, y, x, val))


            # --- 4. Generate Ensemble ---
            ensemble_members = []
            print(f"Generating ensemble of size {config.ensemble_size} for day {sample_day} with {num_obs_points} observations...")
            for _ in tqdm(range(config.ensemble_size), desc="Ensemble Sampling"):
                generated_sample = sample_conditional(
                    model, diffusion, num_samples=1,
                    observations=observations, observed_mask=observed_mask,
                    land_mask=land_mask_global, channels=config.channels, image_size=config.image_size,
                    device=config.device, sampling_method=config.sampling_method,
                    observation_fidelity_weight=config.observation_fidelity_weight,
                    target_conditions=true_conditions_dict,
                    target_location_field=target_location_field,
                    config=config
                )
                ensemble_members.append(generated_sample.squeeze(0))

            # --- 5. Analyze and Visualize Ensemble ---
            ensemble_options = [
                num for num in [1,2,3,4,5,10,20,40,80] 
                if num < config.ensemble_size]
            ensemble_options.append(config.ensemble_size)
            for ensemble_select in ensemble_options:
                ensemble_tensor = torch.stack(ensemble_members[0:ensemble_select])
                ensemble_mean = torch.mean(ensemble_tensor, dim=0)
                ensemble_spread = torch.std(ensemble_tensor, dim=0)

                plot_ensemble_results(
                    ensemble_mean, ensemble_spread, true_sample.squeeze(0), clim_pred, 
                    obs_points_actual, land_mask_global.squeeze(0).squeeze(0).cpu().numpy(), 
                    config, sample_day, num_obs_points, select_size=ensemble_select
                )

    print("\nEnsemble sampling and analysis complete.")
