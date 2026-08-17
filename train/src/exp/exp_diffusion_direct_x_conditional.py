from src.exp.exp_classifier_free_conditional_diffusion import Exp_Classifier_Free_Conditional_Diffusion
from src.utils.utils import *
import torch
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class Exp_Diffusion_Direct_X_Conditional(Exp_Classifier_Free_Conditional_Diffusion):
    def __init__(self, args):
        super(Exp_Diffusion_Direct_X_Conditional, self).__init__(args)

    def calc_loss(self,
                  model,
                  train_loss,
                  criterion,
                  batch_x,
                  timesteps,
                  batch_x_noise_t,
                  noise_t,
                  batch_cond=None,
                  batch_con_prob=None,
                  batch_prev_x=None):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, _ = model(batch_x_noise_t, timesteps, batch_cond, batch_con_prob, batch_prev_x)
                loss = criterion(x_outputs=outputs,
                                 x_targets=batch_x)
                train_loss.append(loss.item())
        else:
            outputs, _ = model(batch_x_noise_t, timesteps, batch_cond, batch_con_prob, batch_prev_x)
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
            stride = self.args.seq_len - int(self.args.seq_len * overlap_ratio)
        else:
            stride = self.args.seq_len

        regimes = np.zeros((size, self.args.seq_len))
        regimes_prob = np.zeros((size, self.args.seq_len, self.args.n_regimes))
        regimes_idx = repeat_array(np.arange(dataset.samples_len), size, stride=stride)
        prev_ts = np.zeros((size, self.args.prev_len, dataset.data.shape[1]))

        for i in range(size):
            regimes[i] = dataset.regimes[regimes_idx[i]]
            regimes_prob[i] = dataset.regimes_prob[regimes_idx[i]]
            prev_ts[i] = dataset.prev_ts[regimes_idx[i]]
        regimes = torch.tensor(regimes).int().to(self.device)
        regimes_prob = torch.cat([torch.zeros((size, self.args.seq_len, 1)), torch.tensor(regimes_prob)], dim=-1).float().to(self.device)
        prev_ts = torch.tensor(prev_ts).float().to(self.device)

        uncond_regimes = torch.zeros_like(regimes).to(self.device)
        uncond_regimes_prob = torch.ones((size, self.args.seq_len, 4))
        uncond_regimes_prob[:, :, 1:] = torch.tensor(0.0).float()
        uncond_regimes_prob = uncond_regimes_prob.to(self.device)
        uncond_prev_ts = torch.zeros_like(prev_ts).to(self.device)

        timesteps = torch.full((size,), (sample_step - 1)).to(self.device)
        model.eval()
        with torch.no_grad():
            cond_outputs, _ = model(samples, timesteps, regimes, regimes_prob, prev_ts).clone()
            uncond_outputs, _ = model(samples, timesteps, uncond_regimes, uncond_regimes_prob, uncond_prev_ts).clone()
            outputs = (1 + self.args.guidance_strength) * cond_outputs - self.args.guidance_strength * uncond_outputs
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