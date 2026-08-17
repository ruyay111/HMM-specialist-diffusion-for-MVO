from src.exp.exp_basic_diffusion import Exp_Basic_Diffusion
import torch
import warnings

warnings.filterwarnings('ignore')


class Exp_Diffusion_Denoised_X(Exp_Basic_Diffusion):
    def __init__(self, args):
        super(Exp_Diffusion_Denoised_X, self).__init__(args)

    def calc_loss(self, model, train_loss, criterion, batch_x, timesteps, batch_x_noise_t, noise_t, batch_cond=None, batch_con_prob=None):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                noise_outputs, _ = model(batch_x_noise_t, timesteps)
                x_outputs = self.diffuser.remove_gauss_noise(batch_x_noise_t, noise_outputs, timesteps)
                loss = criterion(noise_outputs=noise_outputs,
                                 noise_targets=noise_t,
                                 x_outputs=x_outputs,
                                 x_targets=batch_x)
                train_loss.append(loss.item())
        else:
            noise_outputs, _ = model(batch_x_noise_t, timesteps)
            x_outputs = self.diffuser.remove_gauss_noise(batch_x_noise_t, noise_outputs, timesteps)
            loss = criterion(noise_outputs=noise_outputs,
                             noise_targets=noise_t,
                             x_outputs=x_outputs,
                             x_targets=batch_x)
            train_loss.append(loss.item())
        return loss
