from src.exp.exp_basic_diffusion import Exp_Basic_Diffusion
from src.utils.utils import *
import torch
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class Exp_Diffusion_Direct_X(Exp_Basic_Diffusion):
    def __init__(self, args):
        super(Exp_Diffusion_Direct_X, self).__init__(args)

    def calc_loss(self, model, train_loss, criterion, batch_x, timesteps, batch_x_noise_t, noise_t, batch_cond=None, batch_con_prob=None):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, _ = model(batch_x_noise_t, timesteps)
                loss = criterion(x_outputs=outputs,
                                 x_targets=batch_x)
                train_loss.append(loss.item())
        else:
            outputs, _ = model(batch_x_noise_t, timesteps)
            loss = criterion(x_outputs=outputs,
                             x_targets=batch_x)
            train_loss.append(loss.item())
        return loss

    @torch.no_grad()
    def generate_data(self,
                      size,
                      sample_step,
                      dataset,
                      model,
                      sampler,
                      n_steps,
                      ddim_discretize,
                      ddim_eta,
                      method,
                      overlap_ratio,
                      temperature):
        samples = torch.randn((size, self.args.seq_len, dataset.data.shape[1])).float().to(self.device)
        if 'overlap' in method:
            samples = apply_overlap(samples, overlap_ratio)

        timesteps = torch.full((size,), (sample_step - 1)).to(self.device)
        model.eval()
        outputs, _ = model(samples, timesteps)

        if 'overlap' in method:
            outputs = apply_overlap(outputs, overlap_ratio)
        if method == 'discrete':
            outputs = outputs.reshape(-1, dataset.data.shape[1])
        elif method == 'overlap_discard':
            outputs = reconstruct_overlap(outputs, overlap_ratio, method='discard')
        elif method == 'overlap_average':
            outputs = reconstruct_overlap(outputs, overlap_ratio, method='average')
        outputs = outputs.detach().cpu().numpy()
        x_inv = dataset.scaler.inverse_transform(outputs)
        return pd.DataFrame(x_inv)
