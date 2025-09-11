# --- plotting_utils.py ---
# This file contains utility functions for creating visualizations.

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

def plot_losses(train_losses, val_losses, save_path):
    """
    Plots the training and validation losses over epochs and saves the plot.
    """
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'o-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'o-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Weighted MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")

def plot_loaded_observations(observed_mask_np, land_mask_np, config, sample_day_datetime, num_obs_points):
    """
    Visualizes the locations of loaded observations on a map for verification.
    """
    print("\nVisualizing loaded observation locations...")
    
    # Find which depth levels have observations
    depths_with_obs = np.where(np.any(observed_mask_np, axis=(0, 2, 3)))[0]
    if len(depths_with_obs) == 0:
        print("No observations found to plot.")
        return

    # Plot only a subset of depth levels if there are too many
    plot_depths = depths_with_obs
    if len(depths_with_obs) > 5:
        plot_depths = depths_with_obs[::len(depths_with_obs)//5]
        print(f"Plotting a subset of depth levels with observations: {plot_depths}")

    for depth_idx in plot_depths:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot land as a background
        ax.imshow(land_mask_np == 0, cmap='Greys', origin='lower', alpha=0.5)
        
        # Plot ocean
        ocean_mask = np.ma.masked_where(land_mask_np == 0, np.ones_like(land_mask_np))
        ax.imshow(ocean_mask, cmap='Blues', origin='lower', alpha=0.3)

        # Overlay observation points for this depth level
        obs_y, obs_x = np.where(np.any(observed_mask_np[:, depth_idx, :, :], axis=0))
        ax.scatter(obs_x, obs_y, c='red', marker='x', s=5, label=f'Obs Points ({len(obs_y)})')

        ax.set_title(f'Loaded Observation Locations for {sample_day_datetime.strftime("%Y-%m-%d")} at Depth Index {depth_idx}')
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])

        plot_save_path = os.path.join(config.sample_plot_dir, f"obs_locations_day{sample_day_datetime.dayofyear}_depth{depth_idx}.png")
        plt.savefig(plot_save_path, dpi=150)
        print(f"Observation location plot saved to {plot_save_path}")
        plt.close(fig)

def plot_ensemble_results_3d(
    ensemble_mean, ensemble_spread, true_sample, clim_pred,
    observed_mask_np,
    land_mask_np, area_weights_np, config, sample_day_datetime, 
    num_obs_points, obs_count_str, depth_level, depth, select_size=None):
    """
    Visualizes a specific depth level of the 3D ensemble sampling results,
    plotting observations directly from the provided mask.
    """
    print(f"\nVisualizing and saving results for depth level {depth_level}...")

    ensemble_mean_level = ensemble_mean[:, depth_level, :, :].cpu().numpy()
    ensemble_spread_level = ensemble_spread[:, depth_level, :, :].cpu().numpy()
    true_sample_level = true_sample[:, depth_level, :, :].cpu().numpy()
    clim_pred_level = clim_pred[:, depth_level, :, :]

    num_channels = config.channels
    
    for c in range(num_channels):
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), squeeze=False)

        vmin=0.
        vmax=np.exp(-depth/2000)
        verror=0.2*np.exp(-depth/2000)
        vstd=0.1*np.exp(-depth/2000)

        var_name = "Temperature" if c == 0 else "Salinity"
        cmap = 'viridis' if c == 0 else 'plasma'
        error_cmap = 'bwr'

        masked_true = np.ma.masked_where(land_mask_np == 0, true_sample_level[c])
        masked_mean = np.ma.masked_where(land_mask_np == 0, ensemble_mean_level[c])
        masked_spread = np.ma.masked_where(land_mask_np == 0, ensemble_spread_level[c])
        masked_error = np.ma.masked_where(land_mask_np == 0, ensemble_mean_level[c] - true_sample_level[c])
        clim_error = np.ma.masked_where(land_mask_np == 0, clim_pred_level[c] - true_sample_level[c])

        # --- Row 1: Ground Truth ---
        ax = axes[0, 0]
        im = ax.imshow(masked_true, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        
        channel_mask_for_level = observed_mask_np[c, depth_level, :, :]
        obs_y, obs_x = np.where(channel_mask_for_level)
        if len(obs_y) > 0:
            title_obs_str = f'Obs {len(obs_y)}'
            if depth_level > 0:
                ax.scatter(obs_x, obs_y, c='red', marker='x', s=1, label='Observations')
        else:
            title_obs_str = "No Obs"
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Ground Truth (Depth {depth_level}: {depth}, {title_obs_str})')

        # --- Row 2: Climatology ---
        ax = axes[1, 0]
        im = ax.imshow(clim_pred_level[c], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Climatology (Depth {depth_level}: {depth})')

        # --- Row 3: Ensemble Mean ---
        ax = axes[2, 0]
        im = ax.imshow(masked_mean, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Ensemble Mean (Depth {depth_level}: {depth})')

        # --- Row 4: Ensemble Spread (Uncertainty) ---
        ax = axes[0, 1]
        im = ax.imshow(masked_spread, cmap='inferno', origin='lower', vmin=0, vmax=vstd)
        plt.colorbar(im, ax=ax, label='Std. Dev.')
        ax.set_title(f'Ensemble Spread (Uncertainty)')

        # --- Helper for Weighted RMSE ---
        def weighted_rmse(error, weights, mask):
            valid_mask = (mask == 1) & ~np.isnan(error)
            if np.sum(valid_mask) == 0: return np.nan
            weighted_sq_error = (error[valid_mask]**2) * weights[valid_mask]
            sum_of_weights = np.sum(weights[valid_mask])
            return np.sqrt(np.sum(weighted_sq_error) / sum_of_weights)

        # --- Row 5: Climatological Error ---
        ax = axes[1, 1]
        rmse = weighted_rmse(clim_error, area_weights_np, land_mask_np)
        bias = np.nanmean(clim_error)
        im = ax.imshow(clim_error, cmap=error_cmap, origin='lower', vmin=-verror, vmax=verror)
        plt.colorbar(im, ax=ax, label='Error')
        ax.set_title(f'Climatology Error (wRMSE: {rmse:.4f}, Bias: {bias:.4f})')

        # --- Row 6: Mean Error (Bias) ---
        ax = axes[2, 1]
        rmse = weighted_rmse(masked_error, area_weights_np, land_mask_np)
        bias = np.nanmean(masked_error)
        im = ax.imshow(masked_error, cmap=error_cmap, origin='lower', vmin=-verror, vmax=verror)
        plt.colorbar(im, ax=ax, label='Error')
        ax.set_title(f'Ensemble Mean Error (wRMSE: {rmse:.4f}, Bias: {bias:.4f})')

        for ax_row in axes:
            for ax_ in ax_row:
                ax_.set_xticks([])
                ax_.set_yticks([])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'Ch{c} Ensemble Analysis for {sample_day_datetime.strftime("%Y-%m-%d")} at Depth {depth_level} ({depth:.2f}m)', fontsize=16)

        plot_save_path = os.path.join(
            config.sample_plot_dir,
            f"ensemble_ch{c}_{obs_count_str}_day{sample_day_datetime.dayofyear}_depth{depth_level}_{config.sampling_method}.png"
        )
        plt.savefig(plot_save_path, dpi=150)
        print(f"Ensemble plot for depth {depth_level} saved to {plot_save_path}")
        plt.close(fig)