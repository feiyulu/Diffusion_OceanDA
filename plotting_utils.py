# --- plotting_utils.py ---
# This file contains utility functions for creating visualizations,
# such as loss plots and comparisons of model outputs.

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
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def plot_ensemble_results_3d(
    ensemble_mean, ensemble_spread, true_sample, clim_pred,
    obs_points_actual, land_mask_np, config, sample_day_datetime,
    num_obs_points, depth_level, select_size=None):
    """
    Visualizes a specific depth level of the 3D ensemble sampling results.
    """
    print(f"\nVisualizing and saving results for depth level {depth_level}...")

    # --- Prepare data for plotting by selecting the specified depth level ---
    ensemble_mean_level = ensemble_mean[:, depth_level, :, :].cpu().numpy()
    ensemble_spread_level = ensemble_spread[:, depth_level, :, :].cpu().numpy()
    true_sample_level = true_sample[:, depth_level, :, :].cpu().numpy()
    clim_pred_level = clim_pred[:, depth_level, :, :].cpu().numpy()

    num_channels = config.channels
    fig, axes = plt.subplots(6, num_channels, figsize=(6 * num_channels, 24), squeeze=False)

    for c in range(num_channels):
        var_name = "Temperature" if c == 0 else "Salinity"
        cmap = 'viridis' if c == 0 else 'plasma'
        error_cmap = 'bwr'

        # Mask out land areas for all plots in this channel
        masked_true = np.ma.masked_where(land_mask_np == 0, true_sample_level[c])
        masked_mean = np.ma.masked_where(land_mask_np == 0, ensemble_mean_level[c])
        masked_spread = np.ma.masked_where(land_mask_np == 0, ensemble_spread_level[c])
        masked_error = np.ma.masked_where(land_mask_np == 0, ensemble_mean_level[c] - true_sample_level[c])
        clim_error = np.ma.masked_where(land_mask_np == 0, clim_pred_level[c] - true_sample_level[c])

        # --- Row 1: Ground Truth ---
        ax = axes[0, c]
        im = ax.imshow(masked_true, cmap=cmap, origin='lower', vmin=0., vmax=1.)
        # Plot observation points for this channel and depth level
        for obs_c, obs_z, obs_y, obs_x, _ in obs_points_actual:
            if obs_c == c and obs_z == depth_level:
                ax.scatter(obs_x, obs_y, c='red', marker='x', s=15)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Ground Truth (Depth Idx: {depth_level})')

        # --- Row 2: Climatology ---
        ax = axes[1, c]
        im = ax.imshow(clim_pred_level[c], cmap=cmap, origin='lower', vmin=0., vmax=1.)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Climatology (Depth Idx: {depth_level})')

        # --- Row 3: Ensemble Mean ---
        ax = axes[2, c]
        im = ax.imshow(masked_mean, cmap=cmap, origin='lower', vmin=0., vmax=1.)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Ensemble Mean (Depth Idx: {depth_level})')

        # --- Row 4: Ensemble Spread (Uncertainty) ---
        ax = axes[3, c]
        im = ax.imshow(masked_spread, cmap='inferno', origin='lower')
        plt.colorbar(im, ax=ax, label='Std. Dev.')
        ax.set_title(f'Ensemble Spread (Uncertainty)')

        # --- Row 5: Climatological Error ---
        ax = axes[4, c]
        rmse = np.sqrt(np.mean(clim_error**2))
        im = ax.imshow(clim_error, cmap=error_cmap, origin='lower', vmin=-0.2, vmax=0.2)
        plt.colorbar(im, ax=ax, label='Error')
        ax.set_title(f'Climatology Error (RMSE: {rmse:.4f})')

        # --- Row 6: Mean Error (Bias) ---
        ax = axes[5, c]
        rmse = np.sqrt(np.mean(masked_error**2))
        im = ax.imshow(masked_error, cmap=error_cmap, origin='lower', vmin=-0.2, vmax=0.2)
        plt.colorbar(im, ax=ax, label='Error')
        ax.set_title(f'Ensemble Mean Error (RMSE: {rmse:.4f})')

        for row in range(6):
            axes[row, c].set_xlabel('Longitude Index')
            axes[row, c].set_ylabel('Latitude Index')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'Ensemble Analysis for {sample_day_datetime.strftime("%Y-%m-%d")} at Depth Index {depth_level}', fontsize=16)

    # Construct a descriptive filename that includes the depth level
    plot_save_path = os.path.join(
        config.sample_plot_dir,
        f"ensemble_obs{num_obs_points}_day{sample_day_datetime.dayofyear}_"
        f"depth{depth_level}_ens{select_size or config.ensemble_size}_{config.sampling_method}.png"
    )
    plt.savefig(plot_save_path, dpi=150)
    print(f"Ensemble plot for depth {depth_level} saved to {plot_save_path}")
    plt.close(fig)