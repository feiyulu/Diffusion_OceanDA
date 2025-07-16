# --- train_and_sample.py ---
# This is the main script for training the model and performing conditional sampling.
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os # Import the os module for path operations
import imageio # For creating GIF animations
from torch.utils.data import DataLoader, random_split # Import random_split

# Import components from other files
from unet_model import UNet, count_parameters # count_parameters imported for model size
from diffusion_process import Diffusion # Already imported

# --- Training Function ---
def train_diffusion_model(
    model, train_loader, val_loader, diffusion, optimizer, 
    epochs, device, land_mask, gradient_accumulation_steps=1,
    start_epoch=0,
    # NEW: use_... flags are no longer needed here, the model knows from its config
    save_interval=10, checkpoint_dir="checkpoints",
    test_id="default_test", channels=1, loss_plot_dir="loss_plots"
    ):
    """ Trains the diffusion U-Net model. """
    model.train()
    print("Starting training...")
    train_losses, val_losses = [], []
    print_unet_forward_shapes_train = True 
    land_mask = land_mask.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)")
        optimizer.zero_grad() 

        # NEW: The data loader now yields a dictionary of conditions
        for batch_idx, (x_0, conditions_batch, location_field_batch) in enumerate(pbar):
            x_0 = x_0.to(device)
            current_land_mask = land_mask.repeat(x_0.shape[0], 1, 1, 1)
            t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device).long()
            x_t, true_epsilon = diffusion.noise_images(x_0, t, current_land_mask)
            
            # NEW: Move each tensor in the conditions dictionary to the device
            conditions_input = {k: v.to(device) for k, v in conditions_batch.items()}
            loc_field_input = location_field_batch.to(device)

            # Pass the dictionaries directly to the model
            predicted_epsilon = model(x_t, t, current_land_mask, 
                                      conditions=conditions_input, 
                                      location_field=loc_field_input,
                                      verbose_forward=print_unet_forward_shapes_train) 
            if print_unet_forward_shapes_train: print_unet_forward_shapes_train = False

            loss = F.mse_loss(predicted_epsilon * current_land_mask, true_epsilon * current_land_mask) / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix(loss=loss.item() * gradient_accumulation_steps)

        if (len(train_loader)) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} finished, Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop (also updated) ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)")
            for x_0_val, conditions_batch_val, location_field_batch_val in val_pbar:
                x_0_val = x_0_val.to(device)
                current_land_mask_val = land_mask.repeat(x_0_val.shape[0], 1, 1, 1)
                t_val = torch.randint(0, diffusion.timesteps, (x_0_val.shape[0],), device=device).long()
                x_t_val, true_epsilon_val = diffusion.noise_images(x_0_val, t_val, current_land_mask_val)
                
                conditions_input_val = {k: v.to(device) for k, v in conditions_batch_val.items()}
                loc_field_input_val = location_field_batch_val.to(device)

                predicted_epsilon_val = model(x_t_val, t_val, current_land_mask_val, 
                                              conditions=conditions_input_val, 
                                              location_field=loc_field_input_val)
                val_loss = F.mse_loss(predicted_epsilon_val * current_land_mask_val, true_epsilon_val * current_land_mask_val)
                total_val_loss += val_loss.item()
                val_pbar.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} finished, Average Validation Loss: {avg_val_loss:.4f}")
        model.train()

        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 
                                           f"ODA_ch{channels}_{test_id}_epoch_{epoch+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'epoch': epoch, # Save the last completed epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print("Checkpoint saved.")

            # Save loss plot
            plot_save_dir = os.path.dirname(loss_plot_dir)
            plot_filename = f"loss_plot_{test_id}_epoch_{epoch+1}.png"
            specific_plot_path = os.path.join(plot_save_dir, plot_filename)
            plot_losses(train_losses, val_losses, specific_plot_path)
            print(f"Loss plot for epoch {epoch+1} saved to {specific_plot_path}")

    return train_losses, val_losses

# --- Animation Function for Training Data ---
def create_training_animation(data_tensor, land_mask, channels, use_salinity, save_path, interval=200):
    """
    Creates an animation (GIF) of the generated synthetic training data samples.

    Args:
        data_tensor (torch.Tensor): The tensor of training data samples (N, C, H, W).
        land_mask (torch.Tensor): The global land/ocean mask (1, 1, H, W).
        channels (int): Number of channels (1 for Temp, 2 for Temp+Salinity).
        use_salinity (bool): If True, also plots salinity data.
        save_path (str): File path to save the GIF animation.
        interval (int): Delay between frames in milliseconds.
    """
    print(f"Creating training data animation and saving to {save_path}...")
    frames = []
    land_mask_np = land_mask[0, 0].cpu().numpy()
    
    num_rows = channels if use_salinity else 1
    
    # Limit number of frames to avoid excessively large GIFs
    max_frames = 1000 
    data_to_animate = data_tensor[:min(len(data_tensor), max_frames):10]

    for i, sample in enumerate(tqdm(data_to_animate, desc="Generating animation frames")):
        sample_np = sample.cpu().numpy()
        
        fig, axes = plt.subplots(num_rows, 1, figsize=(12, 6*num_rows))
        if num_rows == 1: # If only one subplot, axes is not an array
            axes = [axes]

        # Plot Temperature Map (always channel 0)
        masked_temp = np.ma.masked_where(land_mask_np == 0, sample_np[0])
        im_temp = axes[0].imshow(masked_temp, cmap='viridis', origin='lower', vmin=0., vmax=1.)
        axes[0].set_title(f'Sample {i+1} - Temperature')
        axes[0].set_xlabel('X-coordinate')
        axes[0].set_ylabel('Y-coordinate')
        plt.colorbar(im_temp, ax=axes[0], label='Normalized Temperature')

        # Plot Salinity Map (only if use_salinity is True)
        if use_salinity:
            masked_sal = np.ma.masked_where(land_mask_np == 0, sample_np[1])
            im_sal = axes[1].imshow(masked_sal, cmap='plasma', origin='lower', vmin=0., vmax=1.)
            axes[1].set_title(f'Sample {i+1} - Salinity (Ocean Only)')
            axes[1].set_xlabel('X-coordinate')
            axes[1].set_ylabel('Y-coordinate')
            plt.colorbar(im_sal, ax=axes[1], label='Normalized Salinity')

        plt.tight_layout()
        
        # Capture the plot as an image and append to frames
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig) # Close the figure to free up memory

    if frames:
        imageio.mimsave(save_path, frames, fps=10) # fps controls speed
        print(f"Animation saved successfully to {save_path}")
    else:
        print("No frames were generated for the animation.")

# --- Loss Plotting Function ---
def plot_losses(train_losses, val_losses, save_path):
    """
    Plots the training and validation losses over epochs.

    Args:
        train_losses (list): List of training loss values per epoch.
        val_losses (list): List of validation loss values per epoch.
        save_path (str): File path to save the loss plot.
    """
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"Loss plot saved to {save_path}")