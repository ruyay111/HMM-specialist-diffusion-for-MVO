from src.exp.exp_basic_diffusion import Exp_Basic_Diffusion
from src.exp.exp_diffusionts import Exp_DiffusionTS
from src.exp.exp_classifier_free_conditional_diffusion import Exp_Classifier_Free_Conditional_Diffusion
from src.exp.exp_diffusion_denoised_x import Exp_Diffusion_Denoised_X
from src.exp.exp_diffusion_denoised_x_conditional import Exp_Diffusion_Denoised_X_Conditional
from src.exp.exp_diffusion_direct_x import Exp_Diffusion_Direct_X
from src.exp.exp_diffusion_direct_x_conditional import Exp_Diffusion_Direct_X_Conditional
from src.exp.exp_basic_diffusion_latent_space import Exp_Basic_Diffusion_Latent_Space
from src.exp.exp_diffusion_denoised_x_latent_space import Exp_Diffusion_Denoised_X_Latent_Space
from src.exp.exp_findiff import Exp_FinDiff
from src.exp.exp_two_stage import Exp_Two_Stage
import argparse
import os
import torch
import random
import numpy as np
import warnings

# Change working directory to the root of the project
working_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_dir)
warnings.filterwarnings('ignore')
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    # fix random seed
    fix_seed = 1999
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Diffusion')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='basic_diffusion',
                        help='task name, options:[basic_diffusion, diffusion_ts, conditional_diffusion, '
                             'diffusion_denoised_x, diffusion_denoised_x_conditional, findiff, two_stage]')
    parser.add_argument('--description', type=str, default='test_run', help='exp description')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, SimpleMLP, PatchTST, UniTST, UniTST_NF]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--test_results', type=str, default='./test_results/', help='location of test results')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='train individual models for each asset')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='Benchmark',
                        help='dataset type, options:[Benchmark, Benchmark_Cond]')
    parser.add_argument('--data_path', type=str, default='./warehouse/processed/benchmark_data_log_ret.csv',
                        help='data file')
    parser.add_argument('--regime_path', type=str, default='./warehouse/processed/macro_regime_3.csv',
                        help='regime file')
    parser.add_argument('--seq_len', type=int, default=128, help='input sequence length')
    parser.add_argument('--scale', type=str, default='quantile',
                        help='scaler type, options:[z, minmax, quantile, none]')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--sample_multiplier', type=int, default=8,
                        help='sample multiple random timesteps for a batch, sample_multiplier * batch_size = total '
                             'samples')
    parser.add_argument('--start_date', type=str, default='None', help='start date of the dataset')
    parser.add_argument('--end_date', type=str, default='None', help='end date of the dataset (inclusive)')

    # model define
    parser.add_argument('--diffuser', type=str, default='GaussianDiffusion', help='diffusion model')
    parser.add_argument('--total_steps', type=int, default=1000, help='total steps of diffusion')
    parser.add_argument('--scheduler', type=str, default='linear', help='noise scheduler, options:[linear, cosine]')
    parser.add_argument('--enc_in', type=int, default=12, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # PatchTST specific
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=16, help='stride')

    # UniTST specific
    parser.add_argument('--dispatchers', type=int, default=0, help='number of dispatchers')
    parser.add_argument('--causal_mask', default=False, action='store_true',
                        help='Use causal mask if specified')
    parser.add_argument('--channel_embed', default=False, action='store_true',
                        help='Use channel-specific embedding')
    parser.add_argument('--ind_proj', default=False, action='store_true',
                        help='Use individual projection head for each channel')
    parser.add_argument('--attn_proj', default=False, action='store_true',
                        help='Use attentive projection head for each channel')
    parser.add_argument('--RoPE', default=False, action='store_true', help='Use RoPE')

    # DiffusionTS specific
    parser.add_argument('--use_fft', type=bool, default=True, help='use fft')
    parser.add_argument('--dts_loss', type=str, default='L1', help='DTS loss function, options:[L1, L2]')
    parser.add_argument('--use_tdt', type=bool, default=False, help='use TDT loss')

    # Classifier-Free Diffusion Guidance specific
    parser.add_argument('--prob_uncond', type=float, default=0.2, help='unconditional probability')
    parser.add_argument('--guidance_strength', type=float, default=0.2, help='guidance strength')
    parser.add_argument('--n_regimes', type=int, default=3, help='number of regimes')
    parser.add_argument('--prev_len', type=int, default=128, help='previous time steps for conditional info')
    parser.add_argument('--cond_type', type=str, default='ridx',
                        help='how to incorporate conditional info, options:[ridx, rprob, prevts]')

    # self-supervised learning
    parser.add_argument('--ss_model', type=str, default='UniTST',)
    parser.add_argument('--mask_rate', type=float, default=0.2, help='mask probability')

    # Signature Loss specific
    parser.add_argument('--sig_depth', type=int, default=3, help='signature depth')
    parser.add_argument('--sig_loss', type=str, default='L1',
                        help='signature loss function, options:[L1, L2, Kernel, Cosine]')
    parser.add_argument('--sig_uni', default=False, action='store_true',
                        help='Use univariate signature, also apply to SigKer')
    parser.add_argument('--sig_log', default=False, action='store_true',
                        help='Use log signature')

    # SigKer MMD specific
    parser.add_argument('--kernel_type', type=str, default='rbf', help='kernel type, options:[rbf, linear]')
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma for rbf kernel')
    parser.add_argument('--dyadic_order', type=int, default=1, help='dyadic order')

    # two-stage training FinDiff + LTR
    parser.add_argument('--seq_len_multiplier', type=int, default=2, help='during the second stage, randomly sample '
                                                                          'seq_len data out of multilier * seq_len '
                                                                          'data.')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lr_decay_rounds', type=int, default=10, help='half learning rate every n rounds')
    parser.add_argument('--loss', type=str, default='KL',
                        help='a string to specify the loss function, use + to combine multiple losses; use - to '
                             'multiply a weight to a loss; options:[MSE, L1, SmoothL1, KL, KL2, Wasserstein, '
                             'Wasserstein2, Sig, SigKer, AutoCorr, AutoCorrFFT, Corr, FFT, FFT2, DTS]')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--resume_path', type=str, default=None,
                        help='path to a checkpoint.pth file, or a checkpoint directory containing checkpoint.pth; '
                             'loads model weights before training continues (optimizer state is not restored). '
                             'Use a new --description so the original run is not overwritten.')

    # GradNorm
    parser.add_argument('--grad_norm', default=False, action='store_true', help='use grad norm')
    parser.add_argument('--gn_alpha', type=float, default=0.5, help='grad norm alpha')
    parser.add_argument('--gn_learning_rate', type=float, default=0.01, help='grad norm learning rate')

    # PCGrad
    parser.add_argument('--pcgrad', default=False, action='store_true', help='use pcgrad, not support AMP')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='device backend: auto (cuda>mps>cpu), cuda, mps, or cpu')
    parser.add_argument('--compile', default=False, action='store_true', help='use pytorch compile')

    args = parser.parse_args()
    if args.device == 'cpu':
        args.use_gpu = False
    else:
        args.use_gpu = True

    exp_dict = {
        'basic_diffusion': Exp_Basic_Diffusion,
        'basic_diffusion_latent_space': Exp_Basic_Diffusion_Latent_Space,
        'diffusion_ts': Exp_DiffusionTS,
        'conditional_diffusion': Exp_Classifier_Free_Conditional_Diffusion,
        'conditional_diffusion_latent_space': None,
        'diffusion_denoised_x': Exp_Diffusion_Denoised_X,
        'diffusion_denoised_x_latent_space': Exp_Diffusion_Denoised_X_Latent_Space,
        'diffusion_denoised_x_conditional': Exp_Diffusion_Denoised_X_Conditional,
        'diffusion_denoised_x_conditional_latent_space': None,
        'diffusion_direct_x': Exp_Diffusion_Direct_X,
        'diffusion_direct_x_conditional': Exp_Diffusion_Direct_X_Conditional,
        'findiff': Exp_FinDiff,
        'two_stage': Exp_Two_Stage,
    }

    if args.task_name in exp_dict:
        Exp = exp_dict[args.task_name]
    else:
        raise ValueError(f'Unsupported task name: {args.task_name}')

    exp = Exp(args)
    exp.train()
    exp.test(size=500, method='discrete', temperature=0.7473231454237225, save_data=True,
             sample_step=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0], joint_kl=False, lags=128)
    """exp.test(size=1024, method='discrete', temperature=1, save_data=True,
             sample_step=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0], joint_kl=False, lags=128)"""
    # exp.test_2(findiff_test_path='./test_results/findiff_FinDiff_MLP_ALL_Benchmark_FinDiff_MSE_N_sl128_10_mfs', save_data=True, lags=128)
    # exp.test(size=512, method='discrete', temperature=0.5)
    # exp.test(size=512, method='discrete', temperature=0.1)
    # exp.test(size=512, method='discrete', sampler='DDIM', n_steps=20)
    # exp.test(size=512, method='overlap_discard', temperature=0.1)
    # exp.test(size=512, method='overlap_average')