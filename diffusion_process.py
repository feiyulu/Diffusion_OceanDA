# --- diffusion_process.py ---
# This file defines the forward and reverse diffusion processes.
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DPM-Solver++ Implementation ---
class DPMSolver:
    """A self-contained DPM-Solver++ implementation."""
    def __init__(self, alphas_cumprod, order=2):
        self.alphas_cumprod = alphas_cumprod
        self.order = order

    def to_lambda(self, alpha_bar):
        """Convert alpha_bar to log-signal-to-noise ratio (lambda)."""
        return 0.5 * torch.log(alpha_bar) - 0.5 * torch.log(1.0 - alpha_bar)

    def to_alpha_bar(self, lambda_t):
        """Convert log-signal-to-noise ratio (lambda) back to alpha_bar."""
        return 1.0 / (1.0 + torch.exp(-2.0 * lambda_t))

    def dpm_solver_first_order_update(self, model_output, s, t, x_s):
        """First-order DPM-Solver update."""
        lambda_s = self.to_lambda(self.alphas_cumprod[s])
        lambda_t = self.to_lambda(self.alphas_cumprod[t])
        h = lambda_t - lambda_s
        
        alpha_bar_t = self.to_alpha_bar(lambda_t)
        sigma_t = torch.sqrt(1.0 - alpha_bar_t)
        
        x_t = (torch.sqrt(alpha_bar_t / self.alphas_cumprod[s])) * x_s - \
              sigma_t * (torch.exp(h) - 1.0) * model_output
        return x_t

    def multistep_dpm_solver_second_order_update(self, model_s_list, model_output, s, t, x_s):
        """Second-order DPM-Solver++ update."""
        if len(model_s_list) != 1:
            raise ValueError("model_s_list must have one element for second-order update.")
        
        s_prev = model_s_list[0]['s']
        model_prev = model_s_list[0]['output']

        lambda_s = self.to_lambda(self.alphas_cumprod[s])
        lambda_t = self.to_lambda(self.alphas_cumprod[t])
        lambda_s_prev = self.to_lambda(self.alphas_cumprod[s_prev])

        h = lambda_t - lambda_s
        h_prev = lambda_s - lambda_s_prev
        r = h_prev / h

        alpha_bar_t = self.to_alpha_bar(lambda_t)
        sigma_t = torch.sqrt(1.0 - alpha_bar_t)

        D = (1.0 + 1.0 / (2.0 * r)) * model_output - (1.0 / (2.0 * r)) * model_prev
        
        x_t = (torch.sqrt(alpha_bar_t / self.alphas_cumprod[s])) * x_s - \
              sigma_t * (torch.exp(h) - 1.0) * D
        return x_t


class Diffusion:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.device = device

        self.betas = self.prepare_noise_schedule(beta_start, beta_end).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(device)

    def prepare_noise_schedule(self, _beta_start, _beta_end):
        return torch.linspace(_beta_start, _beta_end, self.timesteps)

    def noise_images(self, x_0, t, mask_in):
        """
        Takes an original 3D volume x_0 and a timestep t, and returns a noisy version x_t.
        """
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t, None, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t, None, None, None, None]
        
        epsilon = torch.randn_like(x_0)
        
        # The mask will be broadcasted automatically
        masked_epsilon = epsilon * mask_in
        masked_x_0 = x_0 * mask_in
        x_t = sqrt_alpha_cumprod_t * masked_x_0 + sqrt_one_minus_alpha_cumprod_t * masked_epsilon
        return x_t, masked_epsilon

    def predict_x0_from_noise(self, x_t, t, epsilon_pred, mask_in):
        """
        Estimates the original clean 3D volume (x_0) from a noisy volume (x_t)
        and the predicted noise (epsilon_pred).
        """
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t, None, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t, None, None, None, None]
        
        # Apply mask via broadcasting
        masked_epsilon_pred = epsilon_pred * mask_in
        predicted_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * masked_epsilon_pred) / sqrt_alpha_cumprod_t
        return predicted_x0 * mask_in

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, mask_in, conditions, location_field):
        # The model call now includes the necessary conditional inputs.
        betas_t = self.betas[t_index, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index, None, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index, None, None, None, None]
        
        # Pass all required arguments to the model
        model_output_noise = model(x, t, mask_in, conditions=conditions, location_field=location_field)
        model_output_noise = model_output_noise * mask_in

        # Equation 11 from DDPM paper
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output_noise / sqrt_one_minus_alphas_cumprod_t)
        model_mean = model_mean * mask_in

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index, None, None, None, None]
            noise = torch.randn_like(x) * mask_in
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def ddim_sample(self, x_t, t, t_prev, x_0_pred, mask_in, eta=0.0):
        """
        Performs one step of the DDIM reverse process for 3D data.
        """
        if t_prev < 0:
            # This happens on the last step. The result should be the predicted x0.
            return x_0_pred

        alpha_bar_t = self.alphas_cumprod[t, None, None, None, None]
        alpha_bar_prev = self.alphas_cumprod[t_prev, None, None, None, None]
        
        predicted_noise = (x_t - torch.sqrt(alpha_bar_t) * x_0_pred) / torch.sqrt(1. - alpha_bar_t)
        predicted_noise *= mask_in

        # Equation 12 from DDIM paper
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        )
        direction_pointing_to_xt = torch.sqrt(1. - alpha_bar_prev - sigma_t**2) * predicted_noise
        noise = torch.randn_like(x_t) * mask_in if t[0] > 0 else torch.zeros_like(x_t)
        
        x_prev = torch.sqrt(alpha_bar_prev) * x_0_pred + direction_pointing_to_xt + sigma_t * noise
        return x_prev * mask_in