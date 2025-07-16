# --- diffusion_process.py ---
# This file defines the forward and reverse diffusion processes.
import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion:
    def __init__(self, timesteps, beta_start, beta_end, img_size, device):
        self.timesteps = timesteps
        self.img_h, self.img_w = img_size # Store height and width separately
        self.device = device

        # Define the noise schedule (betas, alphas, cumulative products)
        self.betas = self.prepare_noise_schedule(beta_start, beta_end).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)

        # Calculations for diffusion process q(x_t | x_0) (forward process)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device) # This should be sqrt(alpha_bar)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)

        # Calculations for posterior q(x_{t-1} | x_t, x_0) (used in reverse process)
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(device)

        # --- DDIM specific parameters ---
        # These are initialized but will not be used if sampling_method in config is 'ddpm'
        self.ddim_alphas = self.alphas_cumprod.to(device)
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas).to(device)

        self.eta = 0.0 # Control stochasticity of DDIM: 0.0 for deterministic, 1.0 for DDPM-like.

    def prepare_noise_schedule(self, _beta_start, _beta_end):
        # Linear schedule for beta values
        return torch.linspace(_beta_start, _beta_end, self.timesteps)

    def noise_images(self, x_0, t, mask_in):
        """
        Takes an original image x_0 and a timestep t, and returns a noisy version x_t and the noise added.
        Noise is only added to valid (ocean) regions as defined by mask_in.

        Args:
            x_0 (torch.Tensor): Original image (batch_size, channels, H, W).
            t (torch.Tensor): Timestep (batch_size,).
            mask_in (torch.Tensor): Binary mask (batch_size, 1, H, W).

        Returns:
            tuple: (x_t, epsilon) - noisy image and the true noise added.
        """
        # Ensure correct alpha_cumprod and sqrt_one_minus_alpha_cumprod are used
        sqrt_alpha_cumprod_t = self.alphas_cumprod[t, None, None, None]**0.5 # Corrected sqrt
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t, None, None, None]
        
        epsilon = torch.randn_like(x_0)
        
        # mask_in is (N, 1, H, W), repeat it across channels for multiplication with x_0 and epsilon
        mask_expanded = mask_in.float().repeat(1, x_0.shape[1], 1, 1)
        
        masked_epsilon = epsilon * mask_expanded
        masked_x_0 = x_0 * mask_expanded

        x_t = sqrt_alpha_cumprod_t * masked_x_0 + sqrt_one_minus_alpha_cumprod_t * masked_epsilon
        
        x_t = x_t * mask_expanded # Ensure land points remain 0 in x_t after adding noise

        return x_t, masked_epsilon

    def predict_x0_from_noise(self, x_t, t, epsilon_pred, mask_in):
        """
        Estimates the original clean image (x_0) from a noisy image (x_t)
        and the predicted noise (epsilon_pred).

        Args:
            x_t (torch.Tensor): Noisy image.
            t (torch.Tensor): Timestep.
            epsilon_pred (torch.Tensor): Predicted noise by the U-Net.
            mask_in (torch.Tensor): Land mask (N, 1, H, W).

        Returns:
            torch.Tensor: Estimated x_0.
        """
        # The equation for predicting x_0 from x_t and epsilon_pred remains the same for both DDPM and DDIM
        sqrt_alpha_cumprod_t = self.alphas_cumprod[t, None, None, None]**0.5 # Corrected sqrt
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t, None, None, None]
        
        mask_expanded = mask_in.float().repeat(1, x_t.shape[1], 1, 1)
        masked_epsilon_pred = epsilon_pred * mask_expanded

        predicted_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * masked_epsilon_pred) / sqrt_alpha_cumprod_t
        
        predicted_x0 = predicted_x0 * mask_expanded # Ensure predicted x_0 remains 0 at land points
        return predicted_x0

    @torch.no_grad() # Disable gradient calculations for sampling
    def p_sample(self, model, x, t, t_index, mask_in):
        """
        Performs one step of the reverse diffusion (denoising) process using DDPM.

        Args:
            model (nn.Module): The trained U-Net model.
            x (torch.Tensor): Noisy image at current step t.
            t (torch.Tensor): Timestep tensor for model prediction.
            t_index (int): Scalar index for accessing noise schedule parameters.
            mask_in (torch.Tensor): Land mask (N, 1, H, W).

        Returns:
            torch.Tensor: Denoised image for the next step (x_{t-1}).
        """
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index] # This is 1/sqrt(alpha_t)
        
        model_output_noise = model(x, t, mask_in) # Model predicts epsilon noise
        
        mask_expanded = mask_in.float().repeat(1, x.shape[1], 1, 1)
        model_output_noise = model_output_noise * mask_expanded # Ensure predicted noise is zeroed out on land

        # Calculate mean of the posterior distribution q(x_{t-1} | x_t, x_0)
        # Using the predicted noise to estimate x_0 internally
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output_noise / sqrt_one_minus_alphas_cumprod_t)
        
        model_mean = model_mean * mask_expanded # Zero out land points in the mean

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x) * mask_expanded # Generate noise, zeroed out on land
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def ddim_sample(self, model, x, t, t_prev, current_t_idx, mask_in, eta=0.0):
        """
        Performs one step of the reverse diffusion (denoising) process using DDIM.
        
        Args:
            ...
            eta (float): Controls the stochasticity of the sampling. 0.0 for deterministic DDIM.
        """
        # Predict noise (epsilon_theta)
        model_output_noise = model(x, t, mask_in)
        mask_expanded = mask_in.float().repeat(1, x.shape[1], 1, 1)
        model_output_noise = model_output_noise * mask_expanded

        # Get parameters for current and previous timesteps
        alpha_bar_t = self.ddim_alphas[current_t_idx]
        alpha_bar_prev_t = self.ddim_alphas[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)

        # Estimate x_0 (predicted clean image)
        x_0_pred = (x - self.ddim_sqrt_one_minus_alphas[current_t_idx] * model_output_noise) / (alpha_bar_t**0.5)
        x_0_pred = x_0_pred * mask_expanded

        # --- NEW: Stochastic DDIM Implementation ---
        # Calculate sigma_t, the standard deviation of the noise term
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev_t)
        )

        # The direction pointing to x_t
        direction_pointing_to_xt = torch.sqrt(1 - alpha_bar_prev_t - sigma_t**2) * model_output_noise

        # The random noise term
        noise = torch.randn_like(x) * mask_expanded if t_prev > 0 else torch.zeros_like(x)

        # The full DDIM equation for x_{t-1}
        x_prev = (alpha_bar_prev_t**0.5) * x_0_pred + direction_pointing_to_xt + sigma_t * noise
        x_prev = x_prev * mask_expanded

        return x_prev
