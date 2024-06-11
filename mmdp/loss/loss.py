from topk.svm import SmoothTop1SVM

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .build import LOSS_REGISTRY


def nll_loss(logits, Y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # make sure these are ints
    batch_size = logits.shape[0]
    Y = Y.type(torch.int64)
    c = c.type(torch.int64)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1)  # censorship status, 0 or 1
    hazards = torch.sigmoid(logits)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # without padding, S(0) = S[0], h(0) = h[0]
    # print("S.shape", S.shape, S)
    # import pdb;pdb.set_trace()
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    
    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=Y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=Y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=Y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))
    
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        # surival is cumulative product of 1 - hazards
        S = torch.cumprod(1 - hazards, dim=1)
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) +
                      torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - \
        (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class SVMLoss(object):    
    def __init__(self, n_classes=2):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        svmloss = SmoothTop1SVM(n_classes=n_classes).cuda()
        if device.type == 'cuda':
            self.svmloss = svmloss.cuda()

    def __call__(self, X, Y):
        return self.svmloss(X, Y)

class CrossEntropyLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, X, Y):
        return F.cross_entropy(X, Y)


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


class NLLSurvLoss(object):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """

    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, logits, Y, c):
        # if alpha is None:
        #     return nll_loss(hazards, S, Y, c, alpha=self.alpha, eps=self.eps, reduction=self.reduction)
        # else:
        return nll_loss(logits, Y, c, alpha=self.alpha, eps=self.eps, reduction=self.reduction)

    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    # reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = - \
            torch.mean(
                (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))
        return loss_cox


class KLLoss(object):
    def __call__(self, y, y_hat):
        return F.kl_div(y_hat.softmax(dim=-1).log(), y.softmax(dim=-1), reduction="sum")


class CosineLoss(object):
    def __call__(self, y, y_hat):
        return 1 - F.cosine_similarity(y, y_hat, dim=1)


class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, P, P_hat, G, G_hat):
        pos_pairs = (1 - torch.abs(F.cosine_similarity(P.detach(), P_hat, dim=1))) + (
            1 - torch.abs(F.cosine_similarity(G.detach(), G_hat, dim=1))
        )
        neg_pairs = (
            torch.abs(F.cosine_similarity(P, G, dim=1))
            + torch.abs(F.cosine_similarity(P.detach(), G_hat, dim=1))
            + torch.abs(F.cosine_similarity(G.detach(), P_hat, dim=1))
        )

        loss = pos_pairs + self.gamma * neg_pairs
        return loss

@LOSS_REGISTRY.register()
def svmloss(**kwargs):
    return SVMLoss(**kwargs)

@LOSS_REGISTRY.register()
def nllsurvloss(**kwargs):
    return NLLSurvLoss(**kwargs)


@LOSS_REGISTRY.register()
def coxsurvloss(**kwargs):
    return CoxSurvLoss(**kwargs)


@LOSS_REGISTRY.register()
def cesurvloss(**kwargs):
    return CrossEntropySurvLoss(**kwargs)


@LOSS_REGISTRY.register()
def celoss(**kwargs):
    return CrossEntropyLoss(**kwargs)


