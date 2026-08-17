from src.exp.exp_basic import Exp_Basic
from src.utils.utils import *
from src.utils.plotting import *
from src.utils.losses import neuralNDCG
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


class Exp_Two_Stage(Exp_Basic):
    def __init__(self, args):
        super(Exp_Two_Stage, self).__init__(args)

    def train(self):
        sys.stdout = self.logger
        pickle.dump(self.args, open(os.path.join(self.checkpoints_path, 'args.pkl'), 'wb'))  # save args
        json.dump(self.args.__dict__, open(os.path.join(self.checkpoints_path, 'args.json'), 'w'))  # save args

        if self.args.individual:
            raise NotImplementedError('Individual training is not available for two-stage generative models')
        else:
            self.train_m()

    def train_m(self):
        sys.stdout = self.logger
        best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'

        print(f'>>>>>>>start training : {self.setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        train_data, train_loader = self._get_data()

        time_now = time.time()
        time_start = copy.deepcopy(time_now)
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_loss = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            iter_time = time.time()
            for i, (batch_x, batch_score) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_score = batch_score.float().to(self.device).squeeze(-1)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = neuralNDCG(outputs, batch_score)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    loss = neuralNDCG(outputs, batch_score)
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

            torch.save(self.model.state_dict(), best_model_path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(best_model_path))
        plot_epoch_loss(epoch_loss,
                        self.args.loss,
                        self.checkpoints_path + '/train_loss.png',
                        log_scale=True if self.args.grad_norm else False)
        print(f'>>>>>>>end training : {generate_elapsed_time(time_start)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return self.model

    def test_2(self, findiff_test_path, load_model=True, no_compile=True, save_plot=True,
               absolute=True, lags=40, save_data=False, joint_kl=False):
        sys.stdout = self.logger
        print(f'>>>>>>>start reranking : {self.setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
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

        data_folders = [f for f in os.listdir(findiff_test_path) if os.path.isdir(os.path.join(findiff_test_path, f))]
        data_folders = sorted(data_folders, key=lambda x: int(x.split('_')[0]))
        sample_step_list = [int(folder.split('_')[0]) for folder in data_folders]
        method = data_folders[0].split('_')[1]
        sampler = data_folders[0].split('_')[2]
        temperature = float(data_folders[0].split('_')[3])
        data_folders = [os.path.join(findiff_test_path, folder) for folder in data_folders]

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

        for i, step in enumerate(sample_step_list):
            folder_path = self.args.test_results + self.setting + '/' + f'{step}_{method}_{sampler}_{temperature}' + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            output = self.generate_data_2(data_folders[i], train_data, self.model)
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
                m_kl_div, j_kl_div = plot_generated_vs_benchmark_dist(benchmark, output, joint_kl=joint_kl,
                                                                      configs=self.args)
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
    def generate_data_2(self,
                        data_path,
                        dataset,
                        model):

        generated_data = pd.read_csv(os.path.join(data_path, 'generated_data_scaled.csv'))
        generated_data = np.array(generated_data)
        sample_window = self.args.seq_len
        generated_data = generated_data.reshape(-1, sample_window, dataset.data.shape[1])
        generated_data = torch.tensor(generated_data).float().to(self.device)
        model.eval()

        outputs = model(generated_data)
        top_indices = torch.topk(outputs, k=self.args.seq_len, dim=1).indices
        top_k_data = torch.gather(generated_data, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, dataset.data.shape[1]))

        samples = top_k_data.detach().cpu().numpy().reshape(-1, dataset.data.shape[1])
        x_inv = dataset.scaler.inverse_transform(samples)
        return pd.DataFrame(x_inv)
