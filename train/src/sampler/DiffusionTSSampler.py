import torch
from src.sampler.DDPMSampler import DDPMSampler


class DiffusionTS(DDPMSampler):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.loss_weight = torch.sqrt(self.alphas) * torch.sqrt(1. - self.alphas_hat) / self.betas / 100