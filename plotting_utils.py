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

    Args:
        train_losses (list): List of training loss values per epoch.
        val_losses (list): List of validation loss values per epoch.
        save_path (str): Full path to save the loss plot image.
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
    plt.close() # Close the figure to free up memory
    print(f"Loss plot saved to {save_path}")


def plot_ensemble_results_3d_surface(
    ensemble_mean, ensemble_spread, true_sample, clim_pred,
    obs_points_actual, land_mask_np, config, sample_day_datetime,
    num_obs_points, select_size=None):
    """
    Visualizes the surface layer (z=0) of the 3D ensemble sampling results.

    Args:
        ensemble_mean (torch.Tensor): The mean of the ensemble members (C, D, H, W).
        ensemble_spread (torch.Tensor): The std dev of the ensemble members (C, D, H, W).
        true_sample (torch.Tensor): The ground truth data (C, D, H, W).
        clim_pred (torch.Tensor): The climatology prediction (C, D, H, W).
        obs_points_actual (list): List of tuples (c, z, y, x, value) for observed points.
        land_mask_np (np.array): The 2D surface land mask (H, W).
        config (Config): The configuration object.
        sample_day_datetime (pd.Timestamp): The date of the sample.
        num_obs_points (int): The number of observations used.
        select_size (int, optional): The size of the ensemble subset being plotted.
    """
    print(f"\nVisualizing and saving surface results for ensemble size {select_size or config.ensemble_size}...")

    # --- Prepare data for plotting by selecting the surface layer (z=0) ---
    ensemble_mean_surface = ensemble_mean[:, 0, :, :].cpu().numpy()
    ensemble_spread_surface = ensemble_spread[:, 0, :, :].cpu().numpy()
    true_sample_surface = true_sample[:, 0, :, :].cpu().numpy()
    clim_pred_surface = clim_pred[:, 0, :, :].cpu().numpy()

    num_channels = config.channels
    # Create a 6-row plot for each channel
    fig, axes = plt.subplots(6, num_channels, figsize=(6 * num_channels, 24), squeeze=False)

    for c in range(num_channels):
        var_name = "Temperature" if c == 0 else "Salinity"
        cmap = 'viridis' if c == 0 else 'plasma'
        error_cmap = 'bwr' # Blue-White-Red for errors

        # Mask out land areas for all plots in this channel
        masked_true = np.ma.masked_where(land_mask_np == 0, true_sample_surface[c])
        masked_mean = np.ma.masked_where(land_mask_np == 0, ensemble_mean_surface[c])
        masked_spread = np.ma.masked_where(land_mask_np == 0, ensemble_spread_surface[c])
        masked_error = np.ma.masked_where(land_mask_np == 0, ensemble_mean_surface[c] - true_sample_surface[c])
        clim_error = np.ma.masked_where(land_mask_np == 0, clim_pred_surface[c] - true_sample_surface[c])

        # --- Row 1: Ground Truth ---
        ax = axes[0, c]
        im = ax.imshow(masked_true, cmap=cmap, origin='lower', vmin=0., vmax=1.)
        # Plot surface observation points for this channel
        for obs_c, obs_z, obs_y, obs_x, _ in obs_points_actual:
            if obs_c == c and obs_z == 0: # Only plot points for this channel and on the surface
                ax.scatter(obs_x, obs_y, c='red', marker='x', s=15)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Ground Truth Surface {var_name}')

        # --- Row 2: Climatology ---
        ax = axes[1, c]
        im = ax.imshow(clim_pred_surface[c], cmap=cmap, origin='lower', vmin=0., vmax=1.)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Climatology Surface (Baseline)')

        # --- Row 3: Ensemble Mean ---
        ax = axes[2, c]
        im = ax.imshow(masked_mean, cmap=cmap, origin='lower', vmin=0., vmax=1.)
        plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')
        ax.set_title(f'Ensemble Mean Surface')

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

        # Set common labels for all subplots in the column
        for row in range(6):
            axes[row, c].set_xlabel('Longitude Index')
            axes[row, c].set_ylabel('Latitude Index')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    fig.suptitle(f'Ensemble Analysis for {sample_day_datetime.strftime("%Y-%m-%d")} ({num_obs_points} obs, {select_size or config.ensemble_size} members)', fontsize=16)

    # Construct a descriptive filename and save the plot
    plot_save_path = os.path.join(
        config.sample_plot_dir,
        f"ensemble_surface_obs{num_obs_points}_day{sample_day_datetime.dayofyear}_"
        f"ens{select_size or config.ensemble_size}_{config.sampling_method}.png"
    )
    plt.savefig(plot_save_path, dpi=150)
    print(f"Ensemble plot saved to {plot_save_path}")
    plt.close(fig)
