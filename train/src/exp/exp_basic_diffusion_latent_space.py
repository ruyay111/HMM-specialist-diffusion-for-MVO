from src.exp.exp_basic_diffusion import Exp_Basic_Diffusion
from src.utils.losses import GradNormLossWrapper
from src.utils.utils import *
from src.utils.plotting import *
from src.utils.pcgrad import PCGrad
from src.sampler.DDPMSampler import DDPMSampler
from src.sampler.DDIMSampler import DDIMSampler
import torch
from torch import nn
import pandas as pd
import numpy as np
import copy
import pickle
import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_Basic_Diffusion_Latent_Space(Exp_Basic_Diffusion):
    def __init__(self, args):
        super(Exp_Basic_Diffusion_Latent_Space, self).__init__(args)

    def train(self, load_ss_model=True, no_compile=True):
        sys.stdout = self.logger
        pickle.dump(self.args, open(os.path.join(self.checkpoints_path, 'args.pkl'), 'wb'))  # save args

        self.train_ss()

        if self.args.individual:
            for i in range(self.args.enc_in):
                self.model_list[i].update_d_model(self.ss_model.embed_size, self.device)
                self.train_i(i, load_ss_model=load_ss_model, no_compile=no_compile)
        else:
            self.model.update_d_model(self.ss_model.embed_size, self.device)
            self.train_m(load_ss_model=load_ss_model, no_compile=no_compile)

    def train_ss(self):
        sys.stdout = self.logger
        best_model_path = self.checkpoints_path + '/' + 'checkpoint_ss_model.pth'

        print(f'>>>>>>>start ss training : {self.setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        train_data, train_loader = self._get_data()

        time_now = time.time()
        time_start = copy.deepcopy(time_now)
        train_steps = len(train_loader)
        model_optim = self._select_optimizer(model=self.ss_model)
        criterion = nn.L1Loss()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_loss = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.ss_model.train()
            epoch_time = time.time()
            iter_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                batch_x_masked = batch_x.masked_fill(mask == 0, 0)
                timesteps = torch.zeros((B,), device=self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.ss_model.basic_diffusion_forward(batch_x_masked, timesteps)
                        loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                        train_loss.append(loss.item())
                else:
                    outputs, _ = self.ss_model.basic_diffusion_forward(batch_x_masked, timesteps)
                    loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                    train_loss.append(loss.item())

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
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {:.2f}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            epoch_loss.append(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            torch.save(self.ss_model.state_dict(), best_model_path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.ss_model.load_state_dict(torch.load(best_model_path))
        plot_epoch_loss(epoch_loss,
                        'Self-Supervised MAE Loss',
                        self.checkpoints_path + '/ss_train_loss.png',
                        log_scale=True if self.args.grad_norm else False)
        print(f'>>>>>>>end ss training : {generate_elapsed_time(time_start)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return self.model

    def train_i(self, col, load_ss_model=True, no_compile=True):
        sys.stdout = self.logger
        best_model_path = self.checkpoints_path + '/' + f'checkpoint_model_{col}.pth'

        if load_ss_model:
            print('loading ss model')
            if no_compile:
                self.ss_model.load_state_dict(process_model_dict(os.path.join(self.checkpoints_path, 'checkpoint_ss_model.pth')))
            else:
                self.ss_model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, 'checkpoint_ss_model.pth')))

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
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                # convert to latent space
                self.ss_model.eval()
                with torch.no_grad():
                    ss_timesteps = torch.zeros((batch_x.size(0),), device=self.device)
                    _, batch_x_latent = self.ss_model.basic_diffusion_forward(batch_x, ss_timesteps)

                timesteps = self.diffuser.sample_random_timesteps(n=len(batch_x_latent))
                batch_x_noise_t, noise_t = self.diffuser.add_gauss_noise(batch_x_latent, timesteps)

                loss = self.calc_loss(self.model_list[col], train_loss, criterion, batch_x, timesteps, batch_x_noise_t, noise_t)

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

    def train_m(self, load_ss_model=True, no_compile=True):
        sys.stdout = self.logger
        best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'

        if load_ss_model:
            print('loading ss model')
            if no_compile:
                self.ss_model.load_state_dict(process_model_dict(os.path.join(self.checkpoints_path, 'checkpoint_ss_model.pth')))
            else:
                self.ss_model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, 'checkpoint_ss_model.pth')))

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
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                # convert to latent space
                self.ss_model.eval()
                with torch.no_grad():
                    ss_timesteps = torch.zeros((batch_x.size(0),), device=self.device)
                    _, batch_x_latent = self.ss_model.basic_diffusion_forward(batch_x, ss_timesteps)

                timesteps = self.diffuser.sample_random_timesteps(n=len(batch_x_latent))
                batch_x_noise_t, noise_t = self.diffuser.add_gauss_noise(batch_x_latent, timesteps)

                loss = self.calc_loss(self.model, train_loss, criterion, batch_x, timesteps, batch_x_noise_t, noise_t)

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

    def calc_loss(self, model, train_loss, criterion, batch_x, timesteps, batch_x_noise_t, noise_t, batch_cond=None, batch_con_prob=None):
        # since embeddings has shape (B, N, E), we need to permute it to (B, E, N)
        batch_x_noise_t = batch_x_noise_t.permute(0, 2, 1)
        noise_t = noise_t.permute(0, 2, 1)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, _ = model.basic_diffusion_forward(batch_x_noise_t, timesteps)
                loss = criterion(noise_outputs=outputs,
                                 noise_targets=noise_t)
                train_loss.append(loss.item())
        else:
            outputs, _ = model.basic_diffusion_forward(batch_x_noise_t, timesteps)
            loss = criterion(noise_outputs=outputs,
                             noise_targets=noise_t)
            train_loss.append(loss.item())
        return loss

    def test(self, size=512, sample_step=None, method='discrete', overlap_ratio=0.25,
             sampler='DDPM', n_steps=20, ddim_discretize="uniform", ddim_eta=0.,
             temperature=1.0, load_model=True, no_compile=True, save_plot=True,
             absolute=True, lags=40):
        sys.stdout = self.logger
        if self.args.individual:
            self.test_i(size, sample_step, method, overlap_ratio,
                        sampler, n_steps, ddim_discretize, ddim_eta,
                        temperature, load_model, no_compile, save_plot,
                        absolute, lags)
        else:
            self.test_m(size, sample_step, method, overlap_ratio,
                        sampler, n_steps, ddim_discretize, ddim_eta,
                        temperature, load_model, no_compile, save_plot,
                        absolute, lags)

    def test_i(self, size=512, sample_step=None, method='discrete', overlap_ratio=0.25,
               sampler='DDPM', n_steps=20, ddim_discretize="uniform", ddim_eta=0.,
               temperature=1.0, load_model=True, no_compile=True, save_plot=True,
               absolute=True, lags=40):
        sys.stdout = self.logger
        print(f'>>>>>>>start testing - {method} : {self.setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        start_time = time.time()
        train_data, _ = self._get_data()

        if load_model:
            print('loading ss model')
            if no_compile:
                self.ss_model.load_state_dict(
                    process_model_dict(os.path.join(self.checkpoints_path, 'checkpoint_ss_model.pth')))
            else:
                self.ss_model.load_state_dict(
                    torch.load(os.path.join(self.checkpoints_path, 'checkpoint_ss_model.pth')))

            print('loading model')
            for i in range(self.args.enc_in):
                self.model_list[i].update_d_model(self.ss_model.embed_size, self.device)
                if no_compile:
                    self.model_list[i].load_state_dict(
                        process_model_dict(os.path.join(self.checkpoints_path, f'checkpoint_model_{i}.pth')))
                else:
                    self.model_list[i].load_state_dict(
                        torch.load(os.path.join(self.checkpoints_path, f'checkpoint_model_{i}.pth')))

        folder_path = self.args.test_results + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if sample_step is None:
            step_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            sample_step_list = [int(self.args.total_steps * step) for step in step_list]
        else:
            sample_step_list = [int(self.args.total_steps * step) for step in sample_step]

        benchmark = train_data.raw_data
        results_dict = {'Step': [],
                        'Mean_MSE': [],
                        'Std_MSE': [],
                        'Skewness_MSE': [],
                        'Kurtosis_MSE': [],
                        'Auto-Corr_DTW': [],
                        'Covariance_MSE': [],
                        'Correlation_Riemannian': []}

        train_data_list = []
        for i in range(self.args.enc_in):
            dataset, _ = self._get_data(col=i)
            train_data_list.append(dataset)

        for step in sample_step_list:
            if sampler == 'DDPM':
                folder_path = self.args.test_results + self.setting + '/' + f'{step}_{method}_{sampler}_{temperature}' + '/'
            elif sampler == 'DDIM':
                folder_path = self.args.test_results + self.setting + '/' + f'{step}_{method}_{sampler}_{n_steps}_{ddim_eta}_{temperature}' + '/'
            else:
                raise NotImplementedError(sampler)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            output = pd.DataFrame()
            for i in range(self.args.enc_in):
                output_i = self.generate_data(size, step, train_data_list[i], self.model_list[i],
                                              sampler, n_steps, ddim_discretize, ddim_eta,
                                              method, overlap_ratio, temperature)
                output = pd.concat([output, output_i], axis=1)

            if save_plot:
                kl_div = plot_generated_vs_benchmark_dist(benchmark, output, folder_path + 'dist.png')
                mse_mean, mse_std, mse_s, mse_kurt = plot_generated_vs_benchmark_moments(benchmark,
                                                                                         output,
                                                                                         folder_path + 'moments.png')
                dtw = plot_generated_vs_benchmark_autocorr(benchmark,
                                                           output,
                                                           folder_path + 'autocorr.png',
                                                           absolute=absolute,
                                                           lags=lags)
                output_cov, benchmark_cov, cov_diff, r_dist_cov = plot_generated_vs_benchmark_cov(benchmark,
                                                                                                  output,
                                                                                                  folder_path + 'cov.png')
                output_corr, benchmark_corr, corr_diff, r_dist_corr = plot_generated_vs_benchmark_corr(benchmark,
                                                                                                       output,
                                                                                                       folder_path + 'corr.png')
            else:
                kl_div = plot_generated_vs_benchmark_dist(benchmark, output)
                mse_mean, mse_std, mse_s, mse_kurt = plot_generated_vs_benchmark_moments(benchmark, output)
                dtw = plot_generated_vs_benchmark_autocorr(benchmark,
                                                           output,
                                                           absolute=absolute,
                                                           lags=lags)
                output_cov, benchmark_cov, cov_diff, r_dist_cov = plot_generated_vs_benchmark_cov(benchmark, output)
                output_corr, benchmark_corr, corr_diff, r_dist_corr = plot_generated_vs_benchmark_corr(benchmark,
                                                                                                       output)
            results_dict['Step'].append(step)
            results_dict['Mean_MSE'].append(mse_mean)
            results_dict['Std_MSE'].append(mse_std)
            results_dict['Skewness_MSE'].append(mse_s)
            results_dict['Kurtosis_MSE'].append(mse_kurt)
            results_dict['KL_Div'].append(kl_div)
            results_dict['Auto-Corr_DTW'].append(dtw)
            results_dict['Covariance_Riemannian'].append(r_dist_cov)
            results_dict['Correlation_Riemannian'].append(r_dist_corr)

        results_df = pd.DataFrame(results_dict)
        if sampler == 'DDPM':
            results_df.to_csv(self.args.test_results + self.setting + f'/results_{method}_{sampler}_{temperature}.csv',
                              index=False)
            plot_results_dict(results_dict,
                              self.args.test_results + self.setting + f'/results_{method}_{sampler}_{temperature}.png')
        elif sampler == 'DDIM':
            results_df.to_csv(
                self.args.test_results + self.setting + f'/results_{method}_{sampler}_{n_steps}_{ddim_eta}_{temperature}.csv',
                index=False)
            plot_results_dict(results_dict,
                              self.args.test_results + self.setting + f'/results_{method}_{sampler}_{n_steps}_{ddim_eta}_{temperature}.png')
        else:
            raise NotImplementedError(sampler)
        print(f'>>>>>>>end testing - {method} : {generate_elapsed_time(start_time)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def test_m(self, size=512, sample_step=None, method='discrete', overlap_ratio=0.25,
               sampler='DDPM', n_steps=20, ddim_discretize="uniform", ddim_eta=0.,
               temperature=1.0, load_model=True, no_compile=True, save_plot=True,
               absolute=True, lags=40):
        sys.stdout = self.logger
        print(f'>>>>>>>start testing - {method} : {self.setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        start_time = time.time()
        train_data, _ = self._get_data()
        if load_model:
            print('loading ss model')
            if no_compile:
                self.ss_model.load_state_dict(
                    process_model_dict(os.path.join(self.checkpoints_path, 'checkpoint_ss_model.pth')))
            else:
                self.ss_model.load_state_dict(
                    torch.load(os.path.join(self.checkpoints_path, 'checkpoint_ss_model.pth')))

            print('loading model')
            self.model.update_d_model(self.ss_model.embed_size, self.device)
            if no_compile:
                self.model.load_state_dict(process_model_dict(os.path.join(self.checkpoints_path, 'checkpoint.pth')))
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, 'checkpoint.pth')))
        folder_path = self.args.test_results + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if sample_step is None:
            step_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            sample_step_list = [int(self.args.total_steps * step) for step in step_list]
        else:
            sample_step_list = [int(self.args.total_steps * step) for step in sample_step]

        benchmark = train_data.raw_data
        results_dict = {'Step': [],
                        'Mean_MSE': [],
                        'Std_MSE': [],
                        'Skewness_MSE': [],
                        'Kurtosis_MSE': [],
                        'Auto-Corr_DTW': [],
                        'Covariance_MSE': [],
                        'Correlation_Riemannian': []}

        for step in sample_step_list:
            if sampler == 'DDPM':
                folder_path = self.args.test_results + self.setting + '/' + f'{step}_{method}_{sampler}_{temperature}' + '/'
            elif sampler == 'DDIM':
                folder_path = self.args.test_results + self.setting + '/' + f'{step}_{method}_{sampler}_{n_steps}_{ddim_eta}_{temperature}' + '/'
            else:
                raise NotImplementedError(sampler)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            output = self.generate_data(size, step, train_data, self.model,
                                        sampler, n_steps, ddim_discretize, ddim_eta,
                                        method, overlap_ratio, temperature)
            if save_plot:
                plot_generated_vs_benchmark_dist(benchmark, output, folder_path + 'dist.png')
                mse_mean, mse_std, mse_s, mse_kurt = plot_generated_vs_benchmark_moments(benchmark,
                                                                                         output,
                                                                                         folder_path + 'moments.png')
                dtw = plot_generated_vs_benchmark_autocorr(benchmark,
                                                           output,
                                                           folder_path + 'autocorr.png',
                                                           absolute=absolute,
                                                           lags=lags)
                output_cov, benchmark_cov, cov_diff, mse_cov = plot_generated_vs_benchmark_cov(benchmark,
                                                                                               output,
                                                                                               folder_path + 'cov.png')
                output_corr, benchmark_corr, corr_diff, r_dist_corr = plot_generated_vs_benchmark_corr(benchmark,
                                                                                                       output,
                                                                                                       folder_path + 'corr.png')
            else:
                plot_generated_vs_benchmark_dist(benchmark, output)
                mse_mean, mse_std, mse_s, mse_kurt = plot_generated_vs_benchmark_moments(benchmark, output)
                dtw = plot_generated_vs_benchmark_autocorr(benchmark,
                                                           output,
                                                           absolute=absolute,
                                                           lags=lags)
                output_cov, benchmark_cov, cov_diff, mse_cov = plot_generated_vs_benchmark_cov(benchmark, output)
                output_corr, benchmark_corr, corr_diff, r_dist_corr = plot_generated_vs_benchmark_corr(benchmark,
                                                                                                       output)
            results_dict['Step'].append(step)
            results_dict['Mean_MSE'].append(mse_mean)
            results_dict['Std_MSE'].append(mse_std)
            results_dict['Skewness_MSE'].append(mse_s)
            results_dict['Kurtosis_MSE'].append(mse_kurt)
            results_dict['Auto-Corr_DTW'].append(dtw)
            results_dict['Covariance_MSE'].append(mse_cov)
            results_dict['Correlation_Riemannian'].append(r_dist_corr)

        results_df = pd.DataFrame(results_dict)
        if sampler == 'DDPM':
            results_df.to_csv(self.args.test_results + self.setting + f'/results_{method}_{sampler}_{temperature}.csv',
                              index=False)
            plot_results_dict(results_dict,
                              self.args.test_results + self.setting + f'/results_{method}_{sampler}_{temperature}.png')
        elif sampler == 'DDIM':
            results_df.to_csv(
                self.args.test_results + self.setting + f'/results_{method}_{sampler}_{n_steps}_{ddim_eta}_{temperature}.csv',
                index=False)
            plot_results_dict(results_dict,
                              self.args.test_results + self.setting + f'/results_{method}_{sampler}_{n_steps}_{ddim_eta}_{temperature}.png')
        else:
            raise NotImplementedError(sampler)
        print(f'>>>>>>>end testing - {method} : {generate_elapsed_time(start_time)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

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

        samples = torch.randn((size, self.ss_model.embed_size, dataset.data.shape[1])).float().to(self.device)
        if 'overlap' in method:
            samples = apply_overlap(samples, overlap_ratio)
        self.ss_model.eval()
        model.eval()

        if sampler == 'DDPM':
            diffuser = DDPMSampler(self.args, self.device)
            for step in reversed(range(0, sample_step)):
                timesteps = torch.full((size,), step).to(self.device)
                outputs, _ = model.basic_diffusion_forward(samples, timesteps)
                samples = diffuser.p_sample_gauss(outputs, samples, timesteps, temperature)
                if 'overlap' in method:
                    samples = apply_overlap(samples, overlap_ratio)
                del outputs
                torch.cuda.empty_cache()
        elif sampler == 'DDIM':
            diffuser = DDIMSampler(self.args, self.device, sample_step, n_steps, ddim_discretize, ddim_eta)
            time_steps = np.flip(diffuser.time_steps)
            for i, step in enumerate(time_steps):
                index = len(time_steps) - i - 1
                timesteps = torch.full((size,), step).to(self.device)
                outputs, _ = model.basic_diffusion_forward(samples, timesteps)
                samples, _ = diffuser.p_sample(outputs, samples, index, temperature)
                if 'overlap' in method:
                    samples = apply_overlap(samples, overlap_ratio)
                del outputs
                torch.cuda.empty_cache()

        samples = self.ss_model.embed_predict(samples.permute(0, 2, 1))
        if method == 'discrete':
            samples = samples.reshape(-1, dataset.data.shape[1])
        elif method == 'overlap_discard':
            samples = reconstruct_overlap(samples, overlap_ratio, method='discard')
        elif method == 'overlap_average':
            samples = reconstruct_overlap(samples, overlap_ratio, method='average')
        samples = samples.detach().cpu().numpy()
        x_inv = dataset.scaler.inverse_transform(samples)
        return pd.DataFrame(x_inv)

