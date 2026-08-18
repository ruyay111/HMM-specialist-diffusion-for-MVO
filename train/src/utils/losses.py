import math
import torch
from torch import nn, optim
import torch.nn.functional as F
from einops import reduce
import geomloss
import numpy as np
from src.sampler.DiffusionTSSampler import DiffusionTS
from src.utils.utils import extract, get_torch_device, resolve_device
from src.utils.metrcis import dcg


class Loss_Wrapper(nn.Module):
    def __init__(self, loss_list, weight_list):
        super(Loss_Wrapper, self).__init__()
        self.loss_list = loss_list
        self.weight_list = weight_list
        self.name = [loss.name for loss in loss_list]
        self.losses = None

    def forward(self, noise_outputs=None, noise_targets=None, x_outputs=None, x_targets=None, **kwargs):
        losses = []
        for weight, loss_fn in zip(self.weight_list, self.loss_list):
            loss = loss_fn(noise_outputs, noise_targets, x_outputs, x_targets)
            losses.append(weight * loss)
        self.losses = losses
        return torch.sum(torch.stack(self.losses))

    def __len__(self):
        return len(self.loss_list)


class KL_Loss(nn.Module):
    def __init__(self, on='noise'):
        super(KL_Loss, self).__init__()
        self.name = 'KL_Div'
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.on = on
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.on == 'noise':
            return self.calc_loss(noise_outputs, noise_targets)
        elif self.on == 'x':
            return self.calc_loss(x_outputs, x_targets)

    def calc_loss(self, outputs, targets):
        loss = self.loss_fn(torch.log_softmax(outputs, dim=-1), torch.softmax(targets, dim=-1))
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class KL_Loss_2(nn.Module):
    def __init__(self, on='noise'):
        super(KL_Loss_2, self).__init__()
        self.name = 'KL_Div_2'
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.on = on
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.on == 'noise':
            return self.calc_loss(noise_outputs, noise_targets)
        elif self.on == 'x':
            return self.calc_loss(x_outputs, x_targets)

    def calc_loss(self, outputs, targets):
        outputs_flat = outputs.reshape(outputs.size(0), -1)
        outputs_flat = torch.log_softmax(outputs_flat, dim=-1)
        outputs_flat = outputs_flat.reshape(outputs.size(0), outputs.size(1), -1)
        targets_flat = targets.reshape(targets.size(0), -1)
        targets_flat = torch.softmax(targets_flat, dim=-1)
        targets_flat = targets_flat.reshape(targets.size(0), targets.size(1), -1)
        loss = self.loss_fn(outputs_flat, targets_flat)
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class Wasserstein_Loss(nn.Module):
    def __init__(self, on='noise'):
        super(Wasserstein_Loss, self).__init__()
        self.name = 'Wasserstein'
        self.loss_fn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
        self.on = on
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.on == 'noise':
            return self.calc_loss(noise_outputs, noise_targets)
        elif self.on == 'x':
            return self.calc_loss(x_outputs, x_targets)

    def calc_loss(self, outputs, targets, **kwargs):
        loss = self.loss_fn(torch.softmax(outputs, dim=-1), torch.softmax(targets, dim=-1)).mean()
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class Wasserstein_Loss_2(nn.Module):
    def __init__(self, on='noise'):
        super(Wasserstein_Loss_2, self).__init__()
        self.name = 'Wasserstein'
        self.loss_fn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
        self.on = on
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.on == 'noise':
            return self.calc_loss(noise_outputs, noise_targets)
        elif self.on == 'x':
            return self.calc_loss(x_outputs, x_targets)

    def calc_loss(self, outputs, targets):
        outputs_flat = outputs.reshape(outputs.size(0), -1)
        outputs_flat = torch.softmax(outputs_flat, dim=-1)
        outputs_flat = outputs_flat.reshape(outputs.size(0), outputs.size(1), -1)
        targets_flat = targets.reshape(targets.size(0), -1)
        targets_flat = torch.softmax(targets_flat, dim=-1)
        targets_flat = targets_flat.reshape(targets.size(0), targets.size(1), -1)
        loss = self.loss_fn(outputs_flat, targets_flat).mean()
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class Signature_Loss(nn.Module):
    def __init__(self, args):
        super(Signature_Loss, self).__init__()
        try:
            import signatory
        except ImportError as exc:
            raise ImportError(
                "signatory is required for Signature_Loss; "
                "it is not needed for KL2+Corr+FFT training."
            ) from exc
        self.name = f'Signature_{args.sig_loss}'
        self.sig_depth = args.sig_depth
        self.sig_loss = args.sig_loss
        self.sig_uni = args.sig_uni
        self.sig_func = signatory.signature if not args.sig_log else signatory.logsignature
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.sig_uni:
            B, L, C = x_outputs.shape
            x_outputs = x_outputs.permute(0, 2, 1).reshape(B * C, L, 1)
            x_targets = x_targets.permute(0, 2, 1).reshape(B * C, L, 1)

        if self.sig_loss == 'L1' or self.sig_loss == 'L2':
            return self.L1L2_forward(x_outputs, x_targets)
        elif self.sig_loss == 'Kernel':
            return self.Kernel_forward(x_outputs, x_targets)
        elif self.sig_loss == 'Cosine':
            return self.Cosine_forward(x_outputs, x_targets)

    def L1L2_forward(self, x_outputs, x_targets):
        loss_func = nn.L1Loss() if self.sig_loss == 'L1' else nn.MSELoss()
        sig_outputs = self.sig_func(x_outputs, self.sig_depth)
        sig_targets = self.sig_func(x_targets, self.sig_depth)
        loss = loss_func(sig_outputs, sig_targets)
        self.losses = [loss]
        return loss

    def Kernel_forward(self, x_outputs, x_targets):
        sig_outputs = self.sig_func(x_outputs, self.sig_depth)
        sig_targets = self.sig_func(x_targets, self.sig_depth)
        k_x_hat = torch.sum(sig_outputs * sig_outputs, dim=-1)
        k_x = torch.sum(sig_targets * sig_targets, dim=-1)
        k_x_x_hat = torch.sum(sig_targets * sig_outputs, dim=-1)
        loss = torch.mean(k_x + k_x_hat - 2 * k_x_x_hat)
        self.losses = [loss]
        return loss

    def Cosine_forward(self, x_outputs, x_targets):
        sig_outputs = self.sig_func(x_outputs, self.sig_depth)
        sig_targets = self.sig_func(x_targets, self.sig_depth)
        cosine = nn.CosineSimilarity(dim=-1)
        loss = torch.mean(1 - cosine(sig_outputs, sig_targets))
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class SigKer_MMD(nn.Module):
    def __init__(self, args):
        super(SigKer_MMD, self).__init__()
        try:
            import sigkernel
        except ImportError as exc:
            raise ImportError(
                "sigkernel is required for SigKer_MMD; "
                "it is not needed for KL2+Corr+FFT training."
            ) from exc
        self.name = 'SigKer_MMD'
        self.kernel_type = args.kernel_type
        self.sigma = args.sigma
        self.dyadic_order = args.dyadic_order
        self.sig_uni = args.sig_uni
        self.max_batch = 32

        if self.kernel_type == 'rbf':
            self.static_kernel = sigkernel.RBFKernel(self.sigma)
        elif self.kernel_type == 'linear':
            self.static_kernel = sigkernel.LinearKernel()

        self.signature_kernel = sigkernel.SigKernel(self.static_kernel, self.dyadic_order)
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.sig_uni:
            B, L, C = x_outputs.shape
            x_outputs = x_outputs.permute(0, 2, 1).reshape(B * C, L, 1)
            x_targets = x_targets.permute(0, 2, 1).reshape(B * C, L, 1)
        loss = self.signature_kernel.compute_mmd(x_outputs, x_targets, self.max_batch)
        self.losses = [loss]
        return loss


class AutoCorr_Loss(nn.Module):
    def __init__(self, args):
        super(AutoCorr_Loss, self).__init__()
        self.name = 'AutoCorr'
        self.loss_fn = nn.L1Loss()
        self.max_lag = int(10 * math.log10(args.seq_len))
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        total_loss = 0
        outputs_mean = x_outputs.mean(dim=1, keepdim=True)
        targets_mean = x_targets.mean(dim=1, keepdim=True)
        outputs_denom = ((x_outputs - outputs_mean)**2).sum(dim=1) + 1e-8
        targets_denom = ((x_targets - targets_mean)**2).sum(dim=1) + 1e-8

        for lag in range(1, self.max_lag + 1):
            outputs_lagged = x_outputs[:, :-lag, :]
            targets_lagged = x_targets[:, :-lag, :]

            outputs_original = x_outputs[:, lag:, :]
            targets_original = x_targets[:, lag:, :]

            outputs_centered = outputs_original - outputs_mean
            targets_centered = targets_original - targets_mean

            outputs_lagged_centered = outputs_lagged - outputs_mean
            targets_lagged_centered = targets_lagged - targets_mean

            outputs_autocorr = (outputs_centered * outputs_lagged_centered).sum(dim=1) / outputs_denom
            targets_autocorr = (targets_centered * targets_lagged_centered).sum(dim=1) / targets_denom

            lag_loss = self.loss_fn(outputs_autocorr, targets_autocorr)
            total_loss += lag_loss

        loss = total_loss / self.max_lag
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class AutoCorr_Loss_FFT(nn.Module):
    def __init__(self, args):
        super(AutoCorr_Loss_FFT, self).__init__()
        self.name = 'AutoCorrFFT'
        self.loss_fn = nn.L1Loss()
        self.max_lag = int(10 * math.log10(args.seq_len))
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        # [bs, seq_len, n_vars]
        x_fft = torch.fft.fft(x_outputs, dim=1)
        target_fft = torch.fft.fft(x_targets, dim=1)
        x_psd = x_fft * x_fft.conj()
        target_psd = target_fft * target_fft.conj()
        x_autocorr = torch.fft.ifft(x_psd, dim=1).real
        target_autocorr = torch.fft.ifft(target_psd, dim=1).real
        x_autocorr = x_autocorr / x_autocorr[:, :1, :]
        x_autocorr = x_autocorr[:, 1:self.max_lag, :]
        target_autocorr = target_autocorr / target_autocorr[:, :1, :]
        target_autocorr = target_autocorr[:, 1:self.max_lag, :]
        loss = self.loss_fn(x_autocorr, target_autocorr)
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class Corr_Loss(nn.Module):
    def __init__(self):
        super(Corr_Loss, self).__init__()
        self.name = 'Corr'
        self.loss_fn = nn.L1Loss()
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        _, L, C = x_outputs.shape

        pred_centered = x_outputs - x_outputs.mean(dim=1, keepdim=True)
        target_centered = x_targets - x_targets.mean(dim=1, keepdim=True)
        pred_cov = torch.bmm(pred_centered.transpose(1, 2), pred_centered) / (L - 1)
        target_cov = torch.bmm(target_centered.transpose(1, 2), target_centered) / (L - 1)
        pred_std = pred_centered.std(dim=1, keepdim=True)
        target_std = target_centered.std(dim=1, keepdim=True)
        pred_corr = pred_cov / (pred_std.transpose(1, 2) @ pred_std + 1e-8)
        target_corr = target_cov / (target_std.transpose(1, 2) @ target_std + 1e-8)
        mask = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
        pred_corr_upper = pred_corr[:, mask]
        target_corr_upper = target_corr[:, mask]
        loss = self.loss_fn(pred_corr_upper, target_corr_upper)
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class FFT_Loss(nn.Module):
    def __init__(self):
        super(FFT_Loss, self).__init__()
        self.name = 'FFT1'
        self.loss_fn = nn.L1Loss()
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        fft_outputs = torch.fft.fft(x_outputs.transpose(1, 2), norm='forward')
        fft_targets = torch.fft.fft(x_targets.transpose(1, 2), norm='forward')
        fft_outputs, fft_targets = fft_outputs.transpose(1, 2), fft_targets.transpose(1, 2)
        real_loss = self.loss_fn(torch.real(fft_outputs), torch.real(fft_targets))
        imag_loss = self.loss_fn(torch.imag(fft_outputs), torch.imag(fft_targets))
        loss = real_loss + imag_loss
        self.losses = [loss]
        return loss


class FFT_Loss_2(nn.Module):
    def __init__(self):
        super(FFT_Loss_2, self).__init__()
        self.name = 'FFT2'
        self.loss_fn = nn.L1Loss()
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        fft_outputs = torch.fft.fft(x_outputs.transpose(1, 2), norm='forward')
        fft_targets = torch.fft.fft(x_targets.transpose(1, 2), norm='forward')
        fft_outputs, fft_targets = fft_outputs.transpose(1, 2), fft_targets.transpose(1, 2)
        mag_loss = self.loss_fn(torch.abs(fft_outputs), torch.abs(fft_targets))
        phase_loss = self.loss_fn(torch.angle(fft_outputs), torch.angle(fft_targets))
        loss = mag_loss + phase_loss
        self.losses = [loss]
        return loss


class MSE_Loss(nn.Module):
    def __init__(self, on='noise'):
        super(MSE_Loss, self).__init__()
        self.name = 'MSE'
        self.loss_fn = nn.MSELoss()
        self.on = on
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.on == 'noise':
            return self.calc_loss(noise_outputs, noise_targets)
        elif self.on == 'x':
            return self.calc_loss(x_outputs, x_targets)

    def calc_loss(self, outputs, targets):
        loss = self.loss_fn(outputs, targets)
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class MAE_Loss(nn.Module):
    def __init__(self, on='noise'):
        super(MAE_Loss, self).__init__()
        self.name = 'MAE'
        self.loss_fn = nn.L1Loss()
        self.on = on
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.on == 'noise':
            return self.calc_loss(noise_outputs, noise_targets)
        elif self.on == 'x':
            return self.calc_loss(x_outputs, x_targets)

    def calc_loss(self, outputs, targets):
        loss = self.loss_fn(outputs, targets)
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


class Smooth_L1_Loss(nn.Module):
    def __init__(self, on='noise'):
        super(Smooth_L1_Loss, self).__init__()
        self.name = 'Smooth_L1'
        self.loss_fn = nn.SmoothL1Loss()
        self.on = on
        self.losses = None

    def forward(self, noise_outputs, noise_targets, x_outputs, x_targets, **kwargs):
        if self.on == 'noise':
            return self.calc_loss(noise_outputs, noise_targets)
        elif self.on == 'x':
            return self.calc_loss(x_outputs, x_targets)

    def calc_loss(self, outputs, targets):
        loss = self.loss_fn(outputs, targets)
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


# GradNorm Loss Wrapper
# Not compatible with DiffusionTS
class GradNormLossWrapper(nn.Module):
    def __init__(self, model, loss_object, alpha=0.5, initial_weights=None, gradnorm_lr=0.01):
        super(GradNormLossWrapper, self).__init__()
        self.model = model
        device = next(model.parameters()).device
        self.loss_object = loss_object
        self.alpha = alpha
        self.T = len(loss_object)
        if self.T < 2:
            raise ValueError("GradNorm requires at least two losses to optimize.")
        initial_weights = initial_weights if initial_weights else [1.0] * self.T
        self.task_weights = nn.Parameter(torch.tensor(initial_weights, device=device, requires_grad=True))
        self.eps = 1e-8

        self.loss_fn = nn.L1Loss()
        self.gradnorm_optimizer = optim.Adam([self.task_weights], lr=gradnorm_lr)

        self.initial_losses = None
        self.weights_history = []
        self.losses = None

    def forward(self, noise_outputs=None, noise_targets=None, x_outputs=None, x_targets=None):
        # Step 1: Compute individual task losses
        _ = self.loss_object(
            noise_outputs=noise_outputs,
            noise_targets=noise_targets,
            x_outputs=x_outputs,
            x_targets=x_targets,
        )
        task_losses = torch.stack(self.loss_object.losses)
        if self.initial_losses is None:
            self.initial_losses = task_losses.detach()

        # Step 2: Compute the weighted loss
        self.losses = [self.task_weights.clone()[i] * task_losses[i] for i in range(self.T)]
        weighted_loss = sum(self.losses)
        self.weights_history.append(self.task_weights.detach().cpu().numpy().tolist())

        # Step 3: Compute gradients of model parameters w.r.t. each task loss
        G = self.task_weights * self._compute_task_gradients(task_losses)

        # Step 4: Calculate the average gradient norm across tasks
        G_hat = torch.sum(G) / self.T

        # Step 5: Compute inverse training rates for each task
        L_tilde = task_losses / (self.initial_losses + self.eps)

        # Step 6: Compute relative inverse training rates for each task
        r = L_tilde / (L_tilde.mean() + self.eps)

        # Step 7: Compute the GradNorm loss for task weights
        gradnorm_loss = self.loss_fn(G, G_hat * (r ** self.alpha))
        self.gradnorm_optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        self.gradnorm_optimizer.step()

        # Step 8: Renormalize task weights so that their sum is T (number of tasks)
        with torch.no_grad():
            positive_weights = torch.nn.functional.relu(self.task_weights)
            normalized_weights = (positive_weights / (positive_weights.sum() + self.eps) * self.T).detach()
            self.task_weights.copy_(normalized_weights)

        return weighted_loss

    def _compute_task_gradients(self, task_losses):
        task_grads = []
        for i, task_loss in enumerate(task_losses):
            self.model.zero_grad()
            task_loss.backward(retain_graph=True)
            task_grad = torch.cat([param.grad.flatten() for param in self.model.parameters() if param.grad is not None])
            task_grads.append(torch.norm(task_grad, p=2))
        return torch.stack(task_grads)


class DTS_loss(nn.Module):
    def __init__(self, args):
        super(DTS_loss, self).__init__()
        self.use_tdt = args.use_tdt
        if args.use_tdt:
            self.loss_fn_1 = TDT_Loss(args)
            self.loss_fn_2 = F.l1_loss if args.dts_loss == 'L1' else F.mse_loss
        else:
            self.loss_fn_1 = F.l1_loss if args.dts_loss == 'L1' else F.mse_loss
        self.use_fft = args.use_fft
        self.ff_weight = math.sqrt(args.seq_len) / 5
        self.device = resolve_device(use_gpu=args.use_gpu, gpu=getattr(args, 'gpu', 0),
                                     device=getattr(args, 'device', 'auto'))
        self.loss_weight = DiffusionTS(args, self.device).loss_weight

    def forward(self, x_outputs, x_targets, t, **kwargs):
        train_loss = self.loss_fn_1(x_outputs,
                                    x_targets,
                                    reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_fft:
            fft1 = torch.fft.fft(x_outputs.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(x_targets.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            if self.use_tdt:
                fourier_loss = self.loss_fn_2(torch.real(fft1), torch.real(fft2), reduction='none') \
                               + self.loss_fn_2(torch.imag(fft1), torch.imag(fft2), reduction='none')
            else:
                fourier_loss = self.loss_fn_1(torch.real(fft1), torch.real(fft2), reduction='none') \
                               + self.loss_fn_1(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()


# https://arxiv.org/abs/2406.04777
class TDT_Loss(torch.nn.Module):
    def __init__(self, args):
        super(TDT_Loss, self).__init__()
        self.loss_func = F.l1_loss if args.dts_loss == 'L1' else F.mse_loss
        self.losses = None

    def forward(self, x_outputs, x_targets, reduction='none', **kwargs):
        prediction_loss = self.loss_func(x_outputs, x_targets, reduction=reduction)
        d_pred = torch.diff(x_outputs, dim=1)
        d_true = torch.diff(x_targets, dim=1)
        tdt_loss = self.loss_func(d_pred, d_true, reduction=reduction)
        tdt_mean = tdt_loss.mean(dim=1, keepdim=True).repeat(1, 1, 1)
        tdt_loss = torch.cat((tdt_mean, tdt_loss), dim=1)
        rho = (torch.sign(d_pred) != torch.sign(d_true)).float().mean()
        loss = rho * prediction_loss + (1 - rho) * tdt_loss
        self.losses = [loss]
        return loss

    def __len__(self):
        return 1


def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1e-10)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=1e-10)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat


def deterministic_neural_sort(s, tau, mask):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """
    dev = get_torch_device()

    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)

    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    temp = [n - m + 1 - 2 * (torch.arange(n - m, device=dev) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore

    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def sample_gumbel(samples_shape, device, eps=1e-10) -> torch.Tensor:
    """
    Sampling from Gumbel distribution.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param samples_shape: shape of the output samples tensor
    :param device: device of the output samples tensor
    :param eps: epsilon for the logarithm function
    :return: Gumbel samples tensor of shape samples_shape
    """
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """
    dev = get_torch_device()

    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat


def neuralNDCG(y_pred, y_true, eps=1e-10, mask=None, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True, **kwargs):
    """
    NeuralNDCG loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param mask: True/False mask indicating padded elements, shape [batch_size, slate_length]
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :return: loss value, a torch.Tensor
    """
    dev = get_torch_device()

    if k is None:
        k = y_true.shape[1]

    if mask is None:
        mask = torch.zeros_like(y_true, dtype=torch.bool).to(dev)

    # Choose the deterministic/stochastic variant
    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
    P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

    # Mask P_hat and apply to true labels, ie approximately sort them
    P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.

    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    discounted_gains = ground_truth * discounts

    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k], mask=mask).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x, mask=mask).permute(1, 0)

    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + eps)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG