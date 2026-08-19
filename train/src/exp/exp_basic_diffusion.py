from src.exp.exp_basic import Exp_Basic
from src.utils.losses import GradNormLossWrapper
from src.utils.utils import *
from src.utils.plotting import *
from src.utils.pcgrad import PCGrad
from src.sampler.DDPMSampler import DDPMSampler
from src.sampler.DDIMSampler import DDIMSampler
import torch
import pandas as pd
import numpy as np
import copy
import pickle
import os
import time
import warnings
import json

warnings.filterwarnings('ignore')


class Exp_Basic_Diffusion(Exp_Basic):
    def __init__(self, args):
        super(Exp_Basic_Diffusion, self).__init__(args)

    def _output_dir(self, *parts):
        path = ensure_dir(os.path.join(self.args.test_results, self.setting, *parts))
        return extended_path(path) + os.sep

    def train(self):
        sys.stdout = self.logger
        pickle.dump(self.args, open(os.path.join(self.checkpoints_path, 'args.pkl'), 'wb'))  # save args
        json.dump(self.args.__dict__, open(os.path.join(self.checkpoints_path, 'args.json'), 'w'))  # save args

        if self.args.individual:
            for i in range(self.args.enc_in):
                self.train_i(i)
        else:
            self.train_m()

    def train_i(self, col):
        sys.stdout = self.logger
        best_model_path = os.path.join(self.checkpoints_path, f'checkpoint_model_{col}.pth')

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

        self._maybe_resume_weights(self.model_list[col], col=col, save_path=best_model_path)

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
                timesteps = self.diffuser.sample_random_timesteps(n=len(batch_x))
                batch_x_noise_t, noise_t = self.diffuser.add_gauss_noise(batch_x, timesteps)

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
                        os.path.join(self.checkpoints_path, f'train_loss_model_{col}.png'),
                        log_scale=True if self.args.grad_norm else False)
        if self.args.grad_norm:
            plot_gradnorm_weights(criterion, os.path.join(self.checkpoints_path, f'gradnorm_weights_{col}.png'))
        print(f'>>>>>>>end training model {col}: {generate_elapsed_time(time_start)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return self.model_list[col]

    def train_m(self):
        sys.stdout = self.logger
        best_model_path = os.path.join(self.checkpoints_path, 'checkpoint.pth')

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

        self._maybe_resume_weights(self.model, save_path=best_model_path)

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
                timesteps = self.diffuser.sample_random_timesteps(n=len(batch_x))
                batch_x_noise_t, noise_t = self.diffuser.add_gauss_noise(batch_x, timesteps)

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
                        os.path.join(self.checkpoints_path, 'train_loss.png'),
                        log_scale=True if self.args.grad_norm else False)
        if self.args.grad_norm:
            plot_gradnorm_weights(criterion, os.path.join(self.checkpoints_path, 'gradnorm_weights.png'))
        print(f'>>>>>>>end training : {generate_elapsed_time(time_start)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return self.model

    def calc_loss(self, model, train_loss, criterion, batch_x, timesteps, batch_x_noise_t, noise_t, batch_cond=None, batch_con_prob=None):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, _ = model(batch_x_noise_t, timesteps)
                loss = criterion(noise_outputs=outputs,
                                 noise_targets=noise_t)
                train_loss.append(loss.item())
        else:
            outputs, _ = model(batch_x_noise_t, timesteps)
            loss = criterion(noise_outputs=outputs,
                             noise_targets=noise_t)
            train_loss.append(loss.item())
        return loss

    def test(self, size=512, sample_step=None, method='discrete', overlap_ratio=0.25,
             sampler='DDPM', n_steps=20, ddim_discretize="uniform", ddim_eta=0.,
             temperature=1.0, load_model=True, no_compile=True, save_plot=True,
             absolute=True, lags=40, save_data=False, joint_kl=False):
        sys.stdout = self.logger
        if self.args.individual:
            return self.test_i(size, sample_step, method, overlap_ratio,
                               sampler, n_steps, ddim_discretize, ddim_eta,
                               temperature, load_model, no_compile, save_plot,
                               absolute, lags, save_data, joint_kl)
        else:
            return self.test_m(size, sample_step, method, overlap_ratio,
                               sampler, n_steps, ddim_discretize, ddim_eta,
                               temperature, load_model, no_compile, save_plot,
                               absolute, lags, save_data, joint_kl)

    def test_i(self, size=512, sample_step=None, method='discrete', overlap_ratio=0.25,
               sampler='DDPM', n_steps=20, ddim_discretize="uniform", ddim_eta=0.,
               temperature=1.0, load_model=True, no_compile=True, save_plot=True,
               absolute=True, lags=40, save_data=False, joint_kl=False):
        sys.stdout = self.logger
        print(f'>>>>>>>start testing - {method} : {self.setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        start_time = time.time()
        train_data, _ = self._get_data()

        if load_model:
            print('loading model')
            for i in range(self.args.enc_in):
                if no_compile:
                    self.model_list[i].load_state_dict(
                        process_model_dict(os.path.join(self.checkpoints_path, f'checkpoint_model_{i}.pth')))
                else:
                    self.model_list[i].load_state_dict(
                        torch.load(os.path.join(self.checkpoints_path, f'checkpoint_model_{i}.pth')))

        folder_path = self._output_dir()

        if sample_step is None:
            step_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            sample_step_list = [int(self.args.total_steps * step) for step in step_list]
        else:
            sample_step_list = [int(self.args.total_steps * step) for step in sample_step]

        benchmark = train_data.raw_data
        if save_data:
            benchmark.to_csv(folder_path + 'benchmark_data.csv', index=False)
        results_dict = {'Step': [],
                        'Mean_MSE': [],
                        'Std_MSE': [],
                        'Skewness_MSE': [],
                        'Kurtosis_MSE': [],
                        'M_KL_Div': [],   # Marginal KL Divergence
                        'J_KL_Div': [],   # Joint KL Divergence
                        'Auto-Corr_DTW': [],
                        'Covariance_Riemannian': [],
                        'Correlation_Riemannian': []}

        train_data_list = []
        for i in range(self.args.enc_in):
            dataset, _ = self._get_data(col=i)
            train_data_list.append(dataset)

        for step in sample_step_list:
            if sampler == 'DDPM':
                folder_path = self._output_dir(f'{step}_{method}_{sampler}_{temperature:.4f}')
            elif sampler == 'DDIM':
                folder_path = self._output_dir(
                    f'{step}_{method}_{sampler}_{n_steps}_{ddim_eta:.4f}_{temperature:.4f}')
            else:
                raise NotImplementedError(sampler)

            output = pd.DataFrame()
            for i in range(self.args.enc_in):
                output_i = self.generate_data(size, step, train_data_list[i], self.model_list[i],
                                              sampler, n_steps, ddim_discretize, ddim_eta,
                                              method, overlap_ratio, temperature)
                output = pd.concat([output, output_i], axis=1)
            if save_data:
                output.to_csv(folder_path + 'generated_data.csv', index=False)

            if save_plot:
                m_kl_div, j_kl_div = plot_generated_vs_benchmark_dist(benchmark,
                                                                      output,
                                                                      folder_path + 'dist.png',
                                                                      joint_kl=joint_kl,
                                                                      configs=self.args)
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
                m_kl_div, j_kl_div = plot_generated_vs_benchmark_dist(benchmark, output, joint_kl=joint_kl, configs=self.args)
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
            results_dict['M_KL_Div'].append(m_kl_div)
            results_dict['J_KL_Div'].append(j_kl_div)
            results_dict['Auto-Corr_DTW'].append(dtw)
            results_dict['Covariance_Riemannian'].append(r_dist_cov)
            results_dict['Correlation_Riemannian'].append(r_dist_corr)

        results_df = pd.DataFrame(results_dict)
        results_root = self._output_dir()
        if sampler == 'DDPM':
            stem = f'results_{method}_{sampler}_{temperature}'
        elif sampler == 'DDIM':
            stem = f'results_{method}_{sampler}_{n_steps}_{ddim_eta}_{temperature}'
        else:
            raise NotImplementedError(sampler)
        results_df.to_csv(results_root + stem + '.csv', index=False)
        plot_results_dict(results_dict, results_root + stem + '.png')
        print(f'>>>>>>>end testing - {method} : {generate_elapsed_time(start_time)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return results_df

    def test_m(self, size=512, sample_step=None, method='discrete', overlap_ratio=0.25,
               sampler='DDPM', n_steps=20, ddim_discretize="uniform", ddim_eta=0.,
               temperature=1.0, load_model=True, no_compile=True, save_plot=True,
               absolute=True, lags=40, save_data=False, joint_kl=False):
        sys.stdout = self.logger
        print(f'>>>>>>>start testing - {method} : {self.setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        start_time = time.time()
        train_data, _ = self._get_data()
        if load_model:
            print('loading model')
            if no_compile:
                self.model.load_state_dict(process_model_dict(os.path.join(self.checkpoints_path, 'checkpoint.pth')))
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, 'checkpoint.pth')))
        folder_path = self._output_dir()

        if sample_step is None:
            step_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            sample_step_list = [int(self.args.total_steps * step) for step in step_list]
        else:
            sample_step_list = [int(self.args.total_steps * step) for step in sample_step]

        benchmark = train_data.raw_data
        if save_data:
            benchmark.to_csv(folder_path + 'benchmark_data.csv', index=False)
        results_dict = {'Step': [],
                        'Mean_MSE': [],
                        'Std_MSE': [],
                        'Skewness_MSE': [],
                        'Kurtosis_MSE': [],
                        'M_KL_Div': [],  # Marginal KL Divergence
                        'J_KL_Div': [],  # Joint KL Divergence
                        'Auto-Corr_DTW': [],
                        'Covariance_Riemannian': [],
                        'Correlation_Riemannian': []}

        for step in sample_step_list:
            if sampler == 'DDPM':
                folder_path = self._output_dir(f'{step}_{method}_{sampler}_{temperature:.4f}')
            elif sampler == 'DDIM':
                folder_path = self._output_dir(
                    f'{step}_{method}_{sampler}_{n_steps}_{ddim_eta:.4f}_{temperature:.4f}')
            else:
                raise NotImplementedError(sampler)
            output = self.generate_data(size, step, train_data, self.model,
                                        sampler, n_steps, ddim_discretize, ddim_eta,
                                        method, overlap_ratio, temperature)
            if save_data:
                output.to_csv(folder_path + 'generated_data.csv', index=False)
            if save_plot:
                m_kl_div, j_kl_div = plot_generated_vs_benchmark_dist(benchmark,
                                                                      output,
                                                                      folder_path + 'dist.png',
                                                                      joint_kl=joint_kl,
                                                                      configs=self.args)
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
                m_kl_div, j_kl_div = plot_generated_vs_benchmark_dist(benchmark, output, joint_kl=joint_kl, configs=self.args)
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
            results_dict['M_KL_Div'].append(m_kl_div)
            results_dict['J_KL_Div'].append(j_kl_div)
            results_dict['Auto-Corr_DTW'].append(dtw)
            results_dict['Covariance_Riemannian'].append(r_dist_cov)
            results_dict['Correlation_Riemannian'].append(r_dist_corr)

        results_df = pd.DataFrame(results_dict)
        results_root = self._output_dir()
        if sampler == 'DDPM':
            stem = f'results_{method}_{sampler}_{temperature}'
        elif sampler == 'DDIM':
            stem = f'results_{method}_{sampler}_{n_steps}_{ddim_eta}_{temperature}'
        else:
            raise NotImplementedError(sampler)
        results_df.to_csv(results_root + stem + '.csv', index=False)
        plot_results_dict(results_dict, results_root + stem + '.png')
        print(f'>>>>>>>end testing - {method} : {generate_elapsed_time(start_time)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return results_df

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
        model.eval()

        if sampler == 'DDPM':
            diffuser = DDPMSampler(self.args, self.device)
            for step in reversed(range(0, sample_step)):
                timesteps = torch.full((size,), step).to(self.device)
                outputs, _ = model(samples, timesteps)
                samples = diffuser.p_sample_gauss(outputs, samples, timesteps, temperature)
                if 'overlap' in method:
                    samples = apply_overlap(samples, overlap_ratio)
                del outputs
                empty_accelerator_cache()
        elif sampler == 'DDIM':
            diffuser = DDIMSampler(self.args, self.device, sample_step, n_steps, ddim_discretize, ddim_eta)
            time_steps = np.flip(diffuser.time_steps)
            for i, step in enumerate(time_steps):
                index = len(time_steps) - i - 1
                timesteps = torch.full((size,), step).to(self.device)
                outputs, _ = model(samples, timesteps)
                samples, _ = diffuser.p_sample(outputs, samples, index, temperature)
                if 'overlap' in method:
                    samples = apply_overlap(samples, overlap_ratio)
                del outputs
                empty_accelerator_cache()

        if method == 'discrete':
            samples = samples.reshape(-1, dataset.data.shape[1])
        elif method == 'overlap_discard':
            samples = reconstruct_overlap(samples, overlap_ratio, method='discard')
        elif method == 'overlap_average':
            samples = reconstruct_overlap(samples, overlap_ratio, method='average')
        samples = samples.detach().cpu().numpy()
        x_inv = dataset.scaler.inverse_transform(samples)
        return pd.DataFrame(x_inv)
