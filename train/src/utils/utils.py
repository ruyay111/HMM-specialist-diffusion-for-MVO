import atexit
import time
import sys
import os
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torch


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.5 ** ((epoch - 1) // args.lr_decay_rounds))
    if args.pcgrad:
        for param_group in optimizer.optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    print('Updating learning rate to {}'.format(lr))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def repeat_array(arr, size, stride=1):
    # Repeat the array enough times and then trim to the desired size
    output = []
    idx = 0

    while len(output) < size:
        output.append(arr[idx % len(arr)])
        idx = (idx + stride) % len(arr)

    return np.array(output)


def apply_overlap(tensor, overlap_ratio=0.25):
    """
    Apply overlap to the tensor by copying the last overlap_ratio portion
    of each batch's time steps to the next batch's first overlap_ratio portion.
    """
    B, L, _ = tensor.shape
    overlap_size = int(L * overlap_ratio)
    overlapped_tensor = tensor.clone()

    for i in range(B - 1):
        overlapped_tensor[i + 1, :overlap_size, :] = overlapped_tensor[i, -overlap_size:, :]

    return overlapped_tensor


def reconstruct_overlap(tensor, overlap_ratio=0.25, method='discard'):
    """
    Reconstruct the overlapped tensor by either discarding or averaging the overlapping regions.
    """
    B, L, _ = tensor.shape
    overlap_size = int(L * overlap_ratio)

    reconstructed = []

    for i in range(B):
        if i == 0:
            reconstructed.append(tensor[i])
        else:
            if method == 'discard':
                reconstructed.append(tensor[i, overlap_size:, :])
            elif method == 'average':
                overlap = (tensor[i, :overlap_size, :] + tensor[i - 1, -overlap_size:, :]) / 2
                non_overlap = tensor[i, overlap_size:, :]
                reconstructed.append(torch.cat((overlap, non_overlap), dim=0))

    final_sequence = torch.cat(reconstructed, dim=0)
    return final_sequence


def process_model_dict(path):
    # model may train using compile, which adds "_orig_mod." prefix to the keys
    checkpoint = torch.load(path)
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    return new_state_dict


class dotdict(dict):
    """dot.notation access to dictionary attributes with dict behavior."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # Ensure dotdict initializes like a dict
    def __init__(self, *args, **kwargs):
        super(dotdict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    # Ensure it represents like a dict
    def __repr__(self):
        return super(dotdict, self).__repr__()

    # Ensure it iterates like a dict
    def __iter__(self):
        return super(dotdict, self).__iter__()

    # Ensure that it can be pickled
    def __reduce__(self):
        return dotdict, (dict(self),)


def build_config(user_params):
    """
    Takes user-specified parameters and fills in default values for unspecified parameters.
    Sets stride equal to patch_len if not specified by the user.

    Args:
        user_params (dict): Dictionary containing user-specified parameters.

    Returns:
        dict: Complete configuration dictionary with all parameters.
    """
    # Default configuration
    default_config = {
        'task_name': 'basic_diffusion',
        'description': 'test_run',
        'model': 'iTransformer',
        'checkpoints': './checkpoints/',
        'test_results': './test_results/',
        'individual': False,
        'data': 'Benchmark',
        'data_path': './warehouse/processed/benchmark_data_log_ret.csv',
        'regime_path': './warehouse/processed/macro_regime_3.csv',
        'seq_len': 128,
        'scale': 'quantile',
        'num_workers': 10,
        'batch_size': 128,
        'sample_multiplier': 8,
        'diffuser': 'GaussianDiffusion',
        'total_steps': 1000,
        'scheduler': 'linear',
        'enc_in': 12,
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_ff': 512,
        'dropout': 0.1,
        'activation': 'gelu',
        'patch_len': 16,
        'stride': None,
        'dispatchers': 0,
        'causal_mask': False,
        'channel_embed': False,
        'ind_proj': False,
        'attn_proj': False,
        'RoPE': False,
        'use_fft': True,
        'dts_loss': 'L1',
        'use_tdt': False,
        'prob_uncond': 0.2,
        'guidance_strength': 0.2,
        'n_regimes': 3,
        'prev_len': 128,
        'cond_type': 'ridx',
        'ss_model': 'UniTST',
        'mask_rate': 0.2,
        'sig_depth': 3,
        'sig_loss': 'L1',
        'sig_uni': False,
        'sig_log': False,
        'kernel_type': 'rbf',
        'sigma': 1.0,
        'dyadic_order': 1,
        'seq_len_multiplier': 2,
        'start_date': 'None',
        'end_date': 'None',
        'train_epochs': 50,
        'learning_rate': 0.0001,
        'lr_decay_rounds': 10,
        'loss': 'KL2_N',
        'weight_decay': 0.0,
        'use_amp': False,
        'grad_norm': False,
        'gn_alpha': 0.5,
        'gn_learning_rate': 0.01,
        'pcgrad': False,
        'use_gpu': True,
        'gpu': 0,
        'device': 'auto',
        'compile': False,
    }

    # Merge user-provided parameters with defaults
    complete_config = default_config.copy()
    complete_config.update(user_params)

    # Set stride to patch_len if not explicitly specified
    if complete_config['stride'] is None:
        complete_config['stride'] = complete_config['patch_len']

    return complete_config


def generate_setting(args):
    setting = '{}_{}_{}_{}_{}_sl{}'.format(
        args.task_name,
        args.model,
        "IND" if args.individual else "ALL",
        args.data,
        args.loss,
        args.seq_len,
    )

    if getattr(args, 'start_date', 'None') not in (None, 'None'):
        setting += '_sd{}'.format(pd.to_datetime(args.start_date).strftime('%Y%m%d'))

    if getattr(args, 'end_date', 'None') not in (None, 'None'):
        setting += '_ed{}'.format(pd.to_datetime(args.end_date).strftime('%Y%m%d'))

    if args.diffuser != 'GaussianDiffusion':
        setting += '_diff{}'.format(args.diffuser)

    if args.scheduler != 'linear':
        setting += '_sch{}'.format(args.scheduler)

    if args.scale != 'quantile':
        setting += '_sca{}'.format(args.scale)

    if args.d_model != 128:
        setting += '_dm{}'.format(args.d_model)

    if args.n_heads != 8:
        setting += '_nh{}'.format(args.n_heads)

    if args.e_layers != 2:
        setting += '_el{}'.format(args.e_layers)

    if args.d_ff != 512:
        setting += '_df{}'.format(args.d_ff)

    if args.sample_multiplier != 8:
        setting += '_sm{}'.format(args.sample_multiplier)

    if 'latent_space' in args.task_name:
        setting += '_ss{}'.format(args.ss_model)

    if 'conditional' in args.task_name:
        setting += '_pu{}_gs{}_nr{}_ct{}'.format(
            args.prob_uncond,
            args.guidance_strength,
            args.n_regimes,
            args.cond_type)

    if args.model == 'PatchTST':
        setting += '_pl{}_st{}'.format(args.patch_len, args.stride)
    elif args.model == 'UniTST_MP':
        setting += '_ds{}_cm{}_ce{}_ip{}'.format(
            args.dispatchers, args.causal_mask, args.channel_embed, args.ind_proj)
        if args.RoPE:
            setting += '_RoPE'
    elif 'UniTST' in args.model:
        setting += '_pl{}_st{}_ds{}_cm{}_ce{}_ip{}'.format(
            args.patch_len, args.stride, args.dispatchers, args.causal_mask, args.channel_embed, args.ind_proj)
        if args.RoPE:
            setting += '_RoPE'
    elif args.model == 'iTransformer':
        setting += '_ip{}'.format(args.ind_proj)

    if args.loss == 'Sig' or 'Mix2' in args.loss or 'Mix5' in args.loss:
        setting += '_sd{}_sl{}_suni{}_slog{}'.format(args.sig_depth, args.sig_loss, args.sig_uni, args.sig_log)

    if args.loss == 'SigKer':
        setting += '_kt{}_sigma{}_do{}_suni{}'.format(args.kernel_type, args.sigma, args.dyadic_order, args.sig_uni)

    if args.task_name == 'diffusion_ts':
        setting += '_fft{}_dts{}_tdt{}'.format(args.use_fft, args.dts_loss, args.use_tdt)

    if args.task_name == 'two_stage':
        setting += '_slm{}'.format(args.seq_len_multiplier)

    if args.use_amp:
        setting += '_amp'

    if args.grad_norm:
        setting += f'_gna{args.gn_alpha}'

    if args.pcgrad:
        setting += '_pcgrad'

    if args.description != 'test_run':
        setting += '_{}'.format(args.description)

    return setting


def generate_elapsed_time(start_time, end_time=None):
    if end_time is None:
        end_time = time.time()
    elapsed_time = end_time - start_time

    if elapsed_time >= 3600:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int((elapsed_time % 3600) % 60)
        return f'{hours}h {minutes}m {seconds}s'
    elif elapsed_time >= 60:
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        return f'{minutes}m {seconds}s'
    else:
        seconds = int(elapsed_time)
        return f'{seconds}s'


class PrintLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'a')
        self.log.write(f"\n\n===== Log started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        self.log.flush()

        atexit.register(self.cleanup)

    def write(self, message):
        self.terminal.write(message)
        if self.log and not self.log.closed:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if self.log and not self.log.closed:
            self.log.flush()

    def cleanup(self):
        if self.log and not self.log.closed:
            self.log.write(f"===== Log ended on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            self.log.flush()
            self.log.close()

    def __del__(self):
        self.cleanup()


def resolve_device(use_gpu=True, gpu=0, device='auto'):
    """Pick torch device: cuda, mps (Apple GPU), or cpu."""
    if device == 'cpu' or not use_gpu:
        return torch.device('cpu')
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return torch.device(f'cuda:{gpu}')
    if device == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError('MPS requested but not available on this system')
        return torch.device('mps')
    # auto: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return torch.device(f'cuda:{gpu}')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def empty_accelerator_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fix_seed(seed=1999):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_torch_device():
    return resolve_device(use_gpu=True, gpu=0, device='auto')