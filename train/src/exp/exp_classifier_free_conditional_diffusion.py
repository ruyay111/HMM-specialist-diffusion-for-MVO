from src.exp.exp_basic_diffusion import Exp_Basic_Diffusion
from src.sampler.DDIMSampler import DDIMSampler
from src.sampler.DDPMSampler import DDPMSampler
from src.utils.losses import GradNormLossWrapper
from src.utils.utils import *
from src.utils.plotting import plot_epoch_loss, plot_gradnorm_weights
from src.utils.pcgrad import PCGrad
import torch
import pandas as pd
import numpy as np
import copy
import pickle
import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_Classifier_Free_Conditional_Diffusion(Exp_Basic_Diffusion):
    def __init__(self, args):
        super(Exp_Classifier_Free_Conditional_Diffusion, self).__init__(args)

    def train(self):
        sys.stdout = self.logger
        pickle.dump(self.args, open(os.path.join(self.checkpoints_path, 'args.pkl'), 'wb'))  # save args

        if self.args.individual:
            for i in range(self.args.enc_in):
                self.train_i(i)
        else:
            self.train_m()

    def train_i(self, col):
        sys.stdout = self.logger
        best_model_path = self.checkpoints_path + '/' + f'checkpoint_model_{col}.pth'

        print(f'>>>>>>>start training model {col} : {self.setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        train_data, train_loader = self._get_data(col)

        time_now = time.time()
        time_start = copy.deepcopy(time_now)
        train_steps = len(train_loader)
        model_optim = self._select_optimizer(col)
        if self.args.pcgrad:
            model_optim = PCGrad(model_optim, reduction='sum')
        criterion = self._select_criterion()
        if self.args.grad_norm:
            criterion = GradNormLossWrapper(model=self.model_list[col],
                                            loss_object=criterion,
                                            alpha=self.args.gn_alpha,
                                            gradnorm_lr=self.args.gn_learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_loss = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model_list[col].train()
            epoch_time = time.time()
            iter_time = time.time()
            for i, batch_data in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_data['samples'].float().to(self.device)
                batch_cond = batch_data['regimes'].int().to(self.device)
                batch_con_prob = batch_data['regimes_prob'].float().to(self.device)
                batch_prev_x = batch_data['prev_ts'].float().to(self.device)
                uncond_mask = torch.rand(batch_cond.shape[0]) >= self.args.prob_uncond
                batch_cond = batch_cond * (uncond_mask[:, None].int().to(self.device))
                batch_prev_x = batch_prev_x * (uncond_mask[:, None, None].int().to(self.device))

                uncond_prob = torch.ones((batch_con_prob.shape[0], batch_con_prob.shape[1], 1)).float().to(self.device)
                batch_con_prob = torch.cat([uncond_prob, batch_con_prob], dim=-1)  # add probability for regime0/unconditional
                batch_con_prob[:, :, 1:] = batch_con_prob[:, :, 1:] * (uncond_mask[:, None, None].int().to(self.device))
                uncond_mask_inv = ~uncond_mask
                batch_con_prob[:, :, 0] = batch_con_prob[:, :, 0] * (uncond_mask_inv[:, None].int().to(self.device))

                timesteps = self.diffuser.sample_random_timesteps(n=len(batch_x))
                batch_x_noise_t, noise_t = self.diffuser.add_gauss_noise(batch_x, timesteps)

                loss = self.calc_loss(self.model_list[col],
                                      train_loss,
                                      criterion,
                                      batch_x,
                                      timesteps,
                                      batch_x_noise_t,
                                      noise_t,
                                      batch_cond,
                                      batch_con_prob,
                                      batch_prev_x)

                if (i + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | duration: {3:.2f}".format(i + 1,
                                                                                                epoch + 1,
                                                                                                loss.item(),
                                                                                                time.time() - iter_time))
                    iter_time = time.time()
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.pcgrad:
                        model_optim.pc_backward(criterion.losses)
                    else:
                        loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {:.2f}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            epoch_loss.append(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            torch.save(self.model_list[col].state_dict(), best_model_path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model_list[col].load_state_dict(torch.load(best_model_path))
        plot_epoch_loss(epoch_loss,
                        self.args.loss,
                        self.checkpoints_path + f'/train_loss_model_{col}.png',
                        log_scale=True if self.args.grad_norm else False)
        if self.args.grad_norm:
            plot_gradnorm_weights(criterion, self.checkpoints_path + f'/gradnorm_weights_{col}.png')
        print(f'>>>>>>>end training model {col}: {generate_elapsed_time(time_start)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return self.model_list[col]

    def train_m(self):
        sys.stdout = self.logger
        best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'

        print(f'>>>>>>>start training : {self.setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        train_data, train_loader = self._get_data()

        time_now = time.time()
        time_start = copy.deepcopy(time_now)
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        if self.args.pcgrad:
            model_optim = PCGrad(model_optim, reduction='sum')
        criterion = self._select_criterion()
        if self.args.grad_norm:
            criterion = GradNormLossWrapper(model=self.model,
                                            loss_object=criterion,
                                            alpha=self.args.gn_alpha,
                                            gradnorm_lr=self.args.gn_learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_loss = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            iter_time = time.time()
            for i, batch_data in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_data['samples'].float().to(self.device)
                batch_cond = batch_data['regimes'].int().to(self.device)
                batch_con_prob = batch_data['regimes_prob'].float().to(self.device)
                batch_prev_x = batch_data['prev_ts'].float().to(self.device)
                uncond_mask = torch.rand(batch_cond.shape[0]) >= self.args.prob_uncond
                batch_cond = batch_cond * (uncond_mask[:, None].int().to(self.device))
                batch_prev_x = batch_prev_x * (uncond_mask[:, None, None].int().to(self.device))

                uncond_prob = torch.ones((batch_con_prob.shape[0], batch_con_prob.shape[1], 1)).float().to(self.device)
                batch_con_prob = torch.cat([uncond_prob, batch_con_prob], dim=-1)  # add probability for regime0/unconditional
                batch_con_prob[:, :, 1:] = batch_con_prob[:, :, 1:] * (uncond_mask[:, None, None].int().to(self.device))
                uncond_mask_inv = ~uncond_mask
                batch_con_prob[:, :, 0] = batch_con_prob[:, :, 0] * (uncond_mask_inv[:, None].int().to(self.device))

                timesteps = self.diffuser.sample_random_timesteps(n=len(batch_x))
                batch_x_noise_t, noise_t = self.diffuser.add_gauss_noise(batch_x, timesteps)

                loss = self.calc_loss(self.model,
                                      train_loss,
                                      criterion,
                                      batch_x,
                                      timesteps,
                                      batch_x_noise_t,
                                      noise_t,
                                      batch_cond,
                                      batch_con_prob,
                                      batch_prev_x)

                if (i + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | duration: {3:.2f}".format(i + 1,
                                                                                                epoch + 1,
                                                                                                loss.item(),
                                                                                                time.time() - iter_time))
                    iter_time = time.time()
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.pcgrad:
                        model_optim.pc_backward(criterion.losses)
                    else:
                        loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {:.2f}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            epoch_loss.append(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            torch.save(self.model.state_dict(), best_model_path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(best_model_path))
        plot_epoch_loss(epoch_loss,
                        self.args.loss,
                        self.checkpoints_path + '/train_loss.png',
                        log_scale=True if self.args.grad_norm else False)
        if self.args.grad_norm:
            plot_gradnorm_weights(criterion, self.checkpoints_path + '/gradnorm_weights.png')
        print(f'>>>>>>>end training : {generate_elapsed_time(time_start)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return self.model

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
                outputs = model(batch_x_noise_t, timesteps, batch_cond, batch_con_prob, batch_prev_x)
                loss = criterion(noise_outputs=outputs,
                                 noise_targets=noise_t)
                train_loss.append(loss.item())
        else:
            outputs = model(batch_x_noise_t, timesteps, batch_cond, batch_con_prob, batch_prev_x)
            loss = criterion(noise_outputs=outputs,
                             noise_targets=noise_t)
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

        model.eval()
        if sampler == 'DDPM':
            diffuser = DDPMSampler(self.args, self.device)
            for step in reversed(range(0, sample_step)):
                timesteps = torch.full((size,), step).to(self.device)
                cond_outputs, _ = model(samples, timesteps, regimes, regimes_prob, prev_ts).clone()
                uncond_outputs, _ = model(samples, timesteps, uncond_regimes, uncond_regimes_prob, uncond_prev_ts).clone()
                outputs = (1 + self.args.guidance_strength) * cond_outputs - self.args.guidance_strength * uncond_outputs
                samples = diffuser.p_sample_gauss(outputs, samples, timesteps, temperature)
                if 'overlap' in method:
                    samples = apply_overlap(samples, overlap_ratio)
                del cond_outputs, uncond_outputs, outputs
                torch.cuda.empty_cache()
        elif sampler == 'DDIM':
            diffuser = DDIMSampler(self.args, self.device, sample_step, n_steps, ddim_discretize, ddim_eta)
            time_steps = np.flip(diffuser.time_steps)
            for i, step in enumerate(time_steps):
                index = len(time_steps) - i - 1
                timesteps = torch.full((size,), step).to(self.device)
                cond_outputs, _ = model(samples, timesteps, regimes, regimes_prob, prev_ts).clone()
                uncond_outputs, _ = model(samples, timesteps, uncond_regimes, uncond_regimes_prob, uncond_prev_ts).clone()
                outputs = (1 + self.args.guidance_strength) * cond_outputs - self.args.guidance_strength * uncond_outputs
                samples, _ = diffuser.p_sample(outputs, samples, index, temperature)
                if 'overlap' in method:
                    samples = apply_overlap(samples, overlap_ratio)
                del outputs
                torch.cuda.empty_cache()

        if method == 'discrete':
            samples = samples.reshape(-1, dataset.data.shape[1])
        elif method == 'overlap_discard':
            samples = reconstruct_overlap(samples, overlap_ratio, method='discard')
        elif method == 'overlap_average':
            samples = reconstruct_overlap(samples, overlap_ratio, method='average')
        samples = samples.detach().cpu().numpy()
        x_inv = dataset.scaler.inverse_transform(samples)
        return pd.DataFrame(x_inv)
