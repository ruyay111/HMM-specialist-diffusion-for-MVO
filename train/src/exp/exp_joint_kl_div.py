import json
from src.data_provider.data_factory import data_provider_Joint_KL_Div
from src.models import MINE
import os
import pickle
import torch
from torch import optim
import numpy as np
import time
import copy


class Exp_Joint_KL_Div:
    def __init__(self, args, X, Y):
        self.args = args
        self.X = X
        self.Y = Y
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.setting = 'Joint_KL_Div_Model'
        self.checkpoints_path = os.path.join(self.args.checkpoints, self.setting)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = MINE.Model(self.args.input_size, self.args.hidden_size).float()
        if self.args.compile:
            return torch.compile(model, mode='max-autotune')
        else:
            return model

    def _get_data(self):
        data_set, data_loader = data_provider_Joint_KL_Div(self.X, self.Y, num_samples=self.args.num_samples, batch_size=self.args.batch_size)
        return data_set, data_loader

    def _select_optimizer(self, model):
        model_optim = optim.Adam(model.parameters(),
                                 lr=self.args.learning_rate)
        return model_optim

    def train(self):
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        # save args
        pickle.dump(self.args, open(os.path.join(self.checkpoints_path, 'args.pkl'), 'wb'))
        json.dump(self.args.__dict__, open(os.path.join(self.checkpoints_path, 'args.json'), 'w'))

        print('>>>>>>>start training Joint KL Div : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))

        KL_ma_ef = copy.deepcopy(self.args.ma_ef)
        data_set, data_loader = self._get_data()
        model_optim = self._select_optimizer(self.model)
        total_estimate = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            epoch_loss = []
            epoch_estimate = []

            epoch_time = time.time()
            for i, (X_ref, Y_ref) in enumerate(data_loader):
                self.model.train()
                iter_count += 1

                X_ref = X_ref.float().to(self.device)
                Y_ref = Y_ref.float().to(self.device)
                model_optim.zero_grad()
                mean_fX = self.model(X_ref).mean()
                mean_efY = torch.exp(self.model(Y_ref)).mean()
                KL_ma_ef = (1 - self.args.ma_rate) * KL_ma_ef + self.args.ma_rate * mean_efY
                KL_loss = - mean_fX + (1 / KL_ma_ef.mean()).detach() * mean_efY
                KL_loss.backward()
                model_optim.step()

                epoch_loss.append(KL_loss.item())

                if (i + 1) % 2 == 0:
                    KL_div = self.estimate()
                    epoch_estimate.append(KL_div)
                    print("Epoch: {} Iter: {} KL_Div: {:.7f}".format(epoch + 1, i + 1, KL_div))

            print("Epoch: {} cost time: {:.2f}".format(epoch + 1, time.time() - epoch_time))
            print("KL_Div Loss: {:.7f}".format(np.mean(epoch_loss)))
            print("KL_Div: {:.7f}".format(np.mean(epoch_estimate)))
            total_estimate.extend(epoch_estimate)
            torch.save(self.model.state_dict(), os.path.join(self.checkpoints_path, 'Joint_KL_Div_checkpoint.pth'))
            pickle.dump(total_estimate, open(os.path.join(self.checkpoints_path, 'total_estimate.pkl'), 'wb'))

        return self.estimate()

    def estimate(self):
        X = torch.tensor(self.X).float().to(self.device)
        Y = torch.tensor(self.Y).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            mean_fX = self.model(X).mean()
            log_mean_efY = torch.logsumexp(self.model(Y), dim=0) - np.log(Y.shape[0])
        return mean_fX.item() - log_mean_efY.item()