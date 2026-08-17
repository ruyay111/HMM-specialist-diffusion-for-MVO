import math
import torch


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPMSampler:
    def __init__(self, args, device):
        self.total_steps = args.total_steps
        self.device = device
        self.scheduler = args.scheduler
        self.alphas, self.betas = self.prepare_noise_schedule(scheduler=self.scheduler)
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def prepare_noise_schedule(self, scheduler):
        if scheduler == 'linear':
            betas = linear_beta_schedule(self.total_steps)
            alphas = 1.0 - betas
        elif scheduler == 'cosine':
            betas = cosine_beta_schedule(self.total_steps)
            alphas = 1.0 - betas
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        return alphas.to(self.device), betas.to(self.device)

    def sample_random_timesteps(self, n):
        t = torch.randint(low=1, high=self.total_steps, size=(n,), device=self.device)
        return t

    def add_gauss_noise(self, x, t):
        target_shape = [x.shape[0]] + [1] * (x.ndim - 1)
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t]).reshape(target_shape)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[t]).reshape(target_shape)
        noise_t = torch.randn_like(x)
        x_noise_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise_t
        return x_noise_t, noise_t

    def remove_gauss_noise(self, x_noise, noise_outputs, t):
        target_shape = [x_noise.shape[0]] + [1] * (x_noise.ndim - 1)
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t]).reshape(target_shape)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[t]).reshape(target_shape)
        denoised_x = (x_noise - sqrt_one_minus_alpha_hat * noise_outputs) / sqrt_alpha_hat
        return denoised_x

    def p_sample_gauss(self, e_t, x, t, temperature=1.0):
        target_shape = [x.shape[0]] + [1] * (x.ndim - 1)
        sqrt_alpha_t = torch.sqrt(self.alphas[t]).reshape(target_shape)
        betas_t = self.betas[t].reshape(target_shape)
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alphas_hat[t]).reshape(target_shape)
        epsilon_t = torch.sqrt(self.betas[t].reshape(target_shape))
        random_noise = torch.randn_like(x) * temperature
        random_noise[t == 0] = 0.0
        model_mean = ((1 / sqrt_alpha_t) * (x - (betas_t * e_t / sqrt_one_minus_alpha_hat_t)))
        x = model_mean + (epsilon_t * random_noise)
        return x