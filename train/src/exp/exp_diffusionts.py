from src.exp.exp_basic_diffusion import Exp_Basic_Diffusion
from src.utils.utils import *
from src.utils.plotting import plot_epoch_loss
import torch
import pandas as pd
import numpy as np
import copy
import pickle
import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_DiffusionTS(Exp_Basic_Diffusion):
    def __init__(self, args):
        super(Exp_DiffusionTS, self).__init__(args)

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
        criterion = self._select_criterion()

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
                timesteps = self.diffuser.sample_random_timesteps(n=len(batch_x))
                batch_x_noise_t, noise_t = self.diffuser.add_gauss_noise(batch_x, timesteps)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model_list[col](batch_x_noise_t, timesteps)
                        loss = criterion(x_outputs=outputs.squeeze(-1),
                                         x_targets=batch_x.squeeze(-1),
                                         t=timesteps)
                        train_loss.append(loss.item())
                else:
                    outputs, _ = self.model_list[col](batch_x_noise_t, timesteps)
                    loss = criterion(x_outputs=outputs.squeeze(-1),
                                     x_targets=batch_x.squeeze(-1),
                                     t=timesteps)
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

            torch.save(self.model_list[col].state_dict(), best_model_path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model_list[col].load_state_dict(torch.load(best_model_path))
        plot_epoch_loss(epoch_loss, self.args.loss, self.checkpoints_path + f'/train_loss_model_{col}.png')
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
        criterion = self._select_criterion()

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
                timesteps = self.diffuser.sample_random_timesteps(n=len(batch_x))
                batch_x_noise_t, noise_t = self.diffuser.add_gauss_noise(batch_x, timesteps)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(batch_x_noise_t, timesteps)
                        loss = criterion(x_outputs=outputs,
                                         x_targets=batch_x,
                                         t=timesteps)
                        train_loss.append(loss.item())
                else:
                    outputs, _ = self.model(batch_x_noise_t, timesteps)
                    loss = criterion(x_outputs=outputs,
                                     x_targets=batch_x,
                                     t=timesteps)
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
        plot_epoch_loss(epoch_loss, self.args.loss, self.checkpoints_path + '/train_loss.png')
        print(f'>>>>>>>end training : {generate_elapsed_time(time_start)}>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return self.model

    def generate_data(self, size, sample_step, dataset, model, method, overlap_ratio, temperature):
        samples = torch.randn((size, self.args.seq_len, dataset.data.shape[1])).float().to(self.device)
        if 'overlap' in method:
            samples = apply_overlap(samples, overlap_ratio)

        timestep = torch.full((size,), (sample_step - 1)).to(self.device)
        model.eval()
        with torch.no_grad():
            outputs, _ = model(samples, timestep)
        if method == 'discrete':
            outputs = outputs.reshape(-1, dataset.data.shape[1])
        elif method == 'overlap_discard':
            outputs = reconstruct_overlap(outputs, overlap_ratio, method='discard')
        elif method == 'overlap_average':
            outputs = reconstruct_overlap(outputs, overlap_ratio, method='average')
        outputs = outputs.detach().cpu().numpy()
        x_inv = dataset.scaler.inverse_transform(outputs)
        return pd.DataFrame(x_inv)
