import torch
import numpy as np
from src.sampler.DDPMSampler import DDPMSampler
from src.utils.utils import dotdict


class DDIMSampler(DDPMSampler):
    def __init__(self, args, device, sample_step, n_steps, ddim_discretize="uniform", ddim_eta=0.):
        super().__init__(args, device)
        if ddim_discretize == 'uniform':
            c = sample_step // n_steps
            self.time_steps = np.asarray(list(range(0, sample_step, c))) + 1
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(sample_step * .8), n_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            alpha_hat = self.alphas_hat
            self.ddim_alpha = alpha_hat[self.time_steps].clone().to(torch.float32)
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat([alpha_hat[0:1], alpha_hat[self.time_steps[:-1]]])
            self.ddim_sigma = (ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
            self.ddim_sqrt_one_minus_alpha = ((1. - self.ddim_alpha) ** .5)

    def p_sample(self, e_t, x, index, temperature=1.0):
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t
        if sigma == 0:
            noise = torch.zeros_like(x)
        else:
            noise = torch.randn_like(x) * temperature
        noise = noise * temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev, pred_x0


if __name__ == '__main__':
    args = {
        'total_steps': 1000,
        'scheduler': 'linear'
    }
    args = dotdict(args)
    device = 'cuda'
    ddim_sampler = DDIMSampler(args, device, sample_step=500, n_steps=50, ddim_discretize="uniform", ddim_eta=0.)
    model_out = torch.randn(128, 128, 12).to(device)
    x = torch.randn(128, 128, 12).to(device)
    index = torch.tensor(0).to(device)
    ddim_sampler.p_sample(model_out, x, index)
