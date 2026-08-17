from src.exp.exp_basic_diffusion_latent_space import Exp_Basic_Diffusion_Latent_Space
import torch
import warnings

warnings.filterwarnings('ignore')


class Exp_Diffusion_Denoised_X_Latent_Space(Exp_Basic_Diffusion_Latent_Space):
    def __init__(self, args):
        super(Exp_Diffusion_Denoised_X_Latent_Space, self).__init__(args)

    def calc_loss(self, model, train_loss, criterion, batch_x, timesteps, batch_x_noise_t, noise_t, batch_cond=None, batch_con_prob=None):
        batch_x_noise_t = batch_x_noise_t.permute(0, 2, 1)
        noise_t = noise_t.permute(0, 2, 1)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                noise_outputs, _ = model.basic_diffusion_forward(batch_x_noise_t, timesteps)
                x_outputs = self.diffuser.remove_gauss_noise(batch_x_noise_t, noise_outputs, timesteps)
                self.ss_model.eval()
                with torch.no_grad():
                    x_outputs = self.ss_model.embed_predict(x_outputs)
                loss = criterion(noise_outputs=noise_outputs,
                                 noise_targets=noise_t,
                                 x_outputs=x_outputs,
                                 x_targets=batch_x)
                train_loss.append(loss.item())
        else:
            noise_outputs, _ = model.basic_diffusion_forward(batch_x_noise_t, timesteps)
            x_outputs = self.diffuser.remove_gauss_noise(batch_x_noise_t, noise_outputs, timesteps)
            self.ss_model.eval()
            with torch.no_grad():
                x_outputs = self.ss_model.embed_predict(x_outputs.permute(0, 2, 1))
            loss = criterion(noise_outputs=noise_outputs,
                             noise_targets=noise_t,
                             x_outputs=x_outputs,
                             x_targets=batch_x)
            train_loss.append(loss.item())
        return loss
