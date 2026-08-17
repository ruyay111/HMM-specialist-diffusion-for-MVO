from src.exp.exp_basic_diffusion import Exp_Basic_Diffusion
from src.utils.utils import *
from src.utils.plotting import *
from src.sampler.DDPMSampler import DDPMSampler
from src.sampler.DDIMSampler import DDIMSampler
import torch
import pandas as pd
import numpy as np
import os
import time
import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')


class Exp_FinDiff(Exp_Basic_Diffusion):
    def __init__(self, args):
        super(Exp_FinDiff, self).__init__(args)

    def test(self, size=512, sample_step=None, method='discrete', overlap_ratio=0.25,
             sampler='DDPM', n_steps=20, ddim_discretize="uniform", ddim_eta=0.,
             temperature=1.0, load_model=True, no_compile=True, save_plot=True,
             absolute=True, lags=40, save_data=False, joint_kl=False):
        sys.stdout = self.logger
        if self.args.individual:
            raise NotImplementedError('Individual mode is not supported for FinDiff tabular data.')
        else:
            return self.test_m(size, sample_step, method, overlap_ratio,
                               sampler, n_steps, ddim_discretize, ddim_eta,
                               temperature, load_model, no_compile, save_plot,
                               absolute, lags, save_data, joint_kl)

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
        folder_path = self.args.test_results + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
                folder_path = self.args.test_results + self.setting + '/' + f'{step}_{method}_{sampler}_{temperature:.4f}' + '/'
            elif sampler == 'DDIM':
                folder_path = self.args.test_results + self.setting + '/' + f'{step}_{method}_{sampler}_{n_steps}_{ddim_eta:.4f}_{temperature:.4f}' + '/'
            else:
                raise NotImplementedError(sampler)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            output, output_scaled = self.generate_data(size, step, train_data, self.model,
                                                      sampler, n_steps, ddim_discretize, ddim_eta,
                                                      method, overlap_ratio, temperature)
            if save_data:
                output.to_csv(folder_path + 'generated_data.csv', index=False)
                output_scaled.to_csv(folder_path + 'generated_data_scaled.csv', index=False)
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

        samples = torch.randn((size, dataset.data.shape[1])).float().to(self.device)
        if 'overlap' in method:
            raise NotImplementedError('Overlap is not supported for FinDiff tabular data.')
        model.eval()

        if sampler == 'DDPM':
            diffuser = DDPMSampler(self.args, self.device)
            for step in reversed(range(0, sample_step)):
                timesteps = torch.full((size,), step).to(self.device)
                outputs, _ = model(samples, timesteps)
                samples = diffuser.p_sample_gauss(outputs, samples, timesteps, temperature)
                del outputs
                torch.cuda.empty_cache()
        elif sampler == 'DDIM':
            diffuser = DDIMSampler(self.args, self.device, sample_step, n_steps, ddim_discretize, ddim_eta)
            time_steps = np.flip(diffuser.time_steps)
            for i, step in enumerate(time_steps):
                index = len(time_steps) - i - 1
                timesteps = torch.full((size,), step).to(self.device)
                outputs, _ = model(samples, timesteps)
                samples, _ = diffuser.p_sample(outputs, samples, index, temperature)
                del outputs
                torch.cuda.empty_cache()

        samples = samples.reshape(-1, dataset.data.shape[1])
        samples = samples.detach().cpu().numpy()
        x_inv = dataset.scaler.inverse_transform(samples)
        return pd.DataFrame(x_inv), pd.DataFrame(samples)