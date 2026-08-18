from src.data_provider.data_factory import data_provider
from src.models import UNet, iTransformer, SimpleMLP, PatchTST, UniTST, UniTST_NoFlatten, UniTST_Individual, UniTST_MultiPatch, FinDiff_MLP, LTR_Model_Attn
from src.sampler.DDPMSampler import DDPMSampler
from src.utils.losses import *
from src.utils.utils import generate_setting, PrintLogger, resolve_device, empty_accelerator_cache
import os
import pickle
import sys


class Exp_Basic(object):
    def __init__(self, args):
        print('Args in experiment:')
        print(args)
        self.__set_up__(args)

    def __set_up__(self, args):
        self.args = args
        self.model_dict = {
            'UNet': UNet,
            'iTransformer': iTransformer,
            'SimpleMLP': SimpleMLP,
            'PatchTST': PatchTST,
            'UniTST': UniTST,
            'UniTST_NF': UniTST_NoFlatten,
            'UniTST_I': UniTST_Individual,
            'UniTST_MP': UniTST_MultiPatch,
            'FinDiff_MLP': FinDiff_MLP,
            'LTR_Model_Attn': LTR_Model_Attn,
        }
        self.diffuser_dict = {
            'GaussianDiffusion': DDPMSampler
        }
        self.loss_dict = {
            'MSE_N': lambda: MSE_Loss(on='noise'),
            'MSE_X': lambda: MSE_Loss(on='x'),
            'L1_N': lambda: MAE_Loss(on='noise'),
            'L1_X': lambda: MAE_Loss(on='x'),
            'SmoothL1_N': lambda: Smooth_L1_Loss(on='noise'),
            'SmoothL1_X': lambda: Smooth_L1_Loss(on='x'),
            'KL_N': lambda: KL_Loss(on='noise'),  # softmax on channel dimension
            'KL_X': lambda: KL_Loss(on='x'),  # softmax on time dimension
            'KL2_N': lambda: KL_Loss_2(on='noise'),  # softmax on both channel and time dimension
            'KL2_X': lambda: KL_Loss_2(on='x'),  # softmax on both channel and time dimension
            'Wasserstein_N': lambda: Wasserstein_Loss(on='noise'),
            'Wasserstein_X': lambda: Wasserstein_Loss(on='x'),
            'Wasserstein2_N': lambda: Wasserstein_Loss_2(on='noise'),
            'Wasserstein2_X': lambda: Wasserstein_Loss_2(on='x'),
            'Sig': lambda: Signature_Loss(args),  # truncated signature
            'SigKer': lambda: SigKer_MMD(args),  # non-truncated signature kernel
            'AutoCorr': lambda: AutoCorr_Loss(args),
            'AutoCorrFFT': lambda: AutoCorr_Loss_FFT(args),
            'Corr': lambda: Corr_Loss(),
            'FFT': lambda: FFT_Loss(),  # real loss + imag loss
            'FFT2': lambda: FFT_Loss_2(),  # magnitude loss + phase loss
            'DTS': lambda: DTS_loss(args),
            'PlaceHolder': lambda: MSE_Loss(on='noise'),
        }
        self.device = self._acquire_device()
        if args.individual:
            self.model_list = self._build_model_list()
        else:
            self.model = self._build_model()
        if 'latent_space' in args.task_name:
            self.ss_model = self._build_model(self.args.ss_model)
        else:
            self.ss_model = None
        self.diffuser = self._build_diffuser(args)
        self._preprocess_loss()
        self._configs_check()

        # start_date and end_date should have form 'YYYY-MM-DD', e.g. '2000-01-01'
        self.setting = generate_setting(args)
        self.checkpoints_path = os.path.join(self.args.checkpoints, self.setting)
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        self.logger = PrintLogger(os.path.join(self.checkpoints_path, 'log.txt'))
        sys.stdout = self.logger

    def _acquire_device(self):
        device = resolve_device(
            use_gpu=getattr(self.args, 'use_gpu', True),
            gpu=getattr(self.args, 'gpu', 0),
            device=getattr(self.args, 'device', 'auto'),
        )
        print(f'Use device: {device}')
        return device

    def _build_model(self, model_name=None):
        if model_name is not None:
            model = self.model_dict[model_name].Model(self.args).float().to(self.device)
        else:
            model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        if self.args.compile:
            return torch.compile(model, mode='max-autotune')
        else:
            return model

    def _build_model_list(self):
        model_list = []
        for i in range(self.args.enc_in):
            model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
            if self.args.compile:
                model = torch.compile(model, mode='max-autotune')
            model_list.append(model)
        return model_list

    def _build_diffuser(self, args):
        diffuser = self.diffuser_dict[args.diffuser](args, self.device)
        return diffuser

    def _preprocess_loss(self):
        self.loss_list = []
        self.weight_list = []
        for loss in self.args.loss.split('+'):
            if '-' in loss:
                weight, loss = loss.split('-')
                weight = float(weight)
            else:
                weight = 1.0
            if loss not in self.loss_dict:
                raise ValueError('Loss function {} not supported'.format(loss))
            self.loss_list.append(self.loss_dict[loss]())
            self.weight_list.append(weight)

    def _configs_check(self):
        if self.args.pcgrad:
            assert not self.args.use_amp, 'PCGrad does not support AMP'
        if self.args.use_amp:
            assert not self.args.pcgrad, 'PCGrad does not support AMP'
        if len(self.loss_list) < 2:
            assert not self.args.grad_norm, 'GradNorm requires at least two loss functions'
            assert not self.args.pcgrad, 'PCGrad requires at least two loss functions'

    def _get_data(self, col=None):
        data_set, data_loader = data_provider(self.args, col=col)
        return data_set, data_loader

    def _select_optimizer(self, col=None, model=None):
        if model is None:
            if self.args.individual and col is not None:
                model_optim = optim.Adam(self.model_list[col].parameters(),
                                         lr=self.args.learning_rate,
                                         weight_decay=self.args.weight_decay)
            else:
                model_optim = optim.Adam(self.model.parameters(),
                                         lr=self.args.learning_rate,
                                         weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = Loss_Wrapper(self.loss_list, self.weight_list)
        return criterion

    def load_model(self, model_path):
        print('-' * 50)
        print('Load model from {}'.format(model_path))
        checkpoint_path = os.path.join(model_path, 'checkpoint.pth')
        args_path = os.path.join(model_path, 'args.pkl')
        new_args = pickle.load(open(args_path, 'rb'))
        self.__set_up__(new_args)
        self.model.load_state_dict(torch.load(checkpoint_path))
        print('Model Successfully Loaded!')
        print('-' * 50)

    def _resolve_resume_checkpoint(self, col=None):
        resume_path = getattr(self.args, 'resume_path', None)
        if not resume_path:
            return None
        resume_path = os.path.abspath(os.path.expanduser(resume_path))
        if os.path.isdir(resume_path):
            if col is not None:
                individual = os.path.join(resume_path, f'checkpoint_model_{col}.pth')
                if os.path.exists(individual):
                    return individual
            return os.path.join(resume_path, 'checkpoint.pth')
        return resume_path

    def _maybe_resume_weights(self, model, col=None, save_path=None):
        """Load pretrained weights before training. Optimizer / LR schedule are not restored."""
        path = self._resolve_resume_checkpoint(col=col)
        if path is None:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f'resume_path not found: {path}')
        if save_path is not None and os.path.abspath(save_path) == os.path.abspath(path):
            print(f'WARNING: resume_path points to the same file that train will overwrite: {path}')
            print('         Prefer a new --description so the original checkpoint is kept.')
        print(f'Resuming model weights from {path}')
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        print('Resume successful (weights only; optimizer starts fresh).')

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def calc_loss(self, model, train_loss, criterion, batch_x, timesteps, batch_x_noise_t, noise_t, batch_cond=None, batch_con_prob=None):
        raise NotImplementedError

    def generate_data(self,
                      size,
                      sample_step,
                      dataset,
                      model,
                      method,
                      overlap_ratio,
                      sampler,
                      n_steps,
                      ddim_discretize,
                      ddim_eta,
                      temperature):
        raise NotImplementedError