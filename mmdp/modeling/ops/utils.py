import os
import math
import numpy as np


import torch
import torch.nn as nn
import torch.distributed as dist
import timm.models.hub as timm_hub


from einops import rearrange, reduce

from typing import Optional
import torch.nn.functional as F





def sharpen_prob(p, temperature=2):
    """Sharpening probability with a temperature.

    Args:
        p (torch.Tensor): probability matrix (batch_size, n_classes)
        temperature (float): temperature.
    """
    p = p.pow(temperature)
    return p / p.sum(1, keepdim=True)


def reverse_index(data, label):
    """Reverse order."""
    inv_idx = torch.arange(data.size(0) - 1, -1, -1).long()
    return data[inv_idx], label[inv_idx]


def shuffle_index(data, label):
    """Shuffle order."""
    rnd_idx = torch.randperm(data.shape[0])
    return data[rnd_idx], label[rnd_idx]


def create_onehot(label, num_classes):
    """Create one-hot tensor.

    We suggest using nn.functional.one_hot.

    Args:
        label (torch.Tensor): 1-D tensor.
        num_classes (int): number of classes.
    """
    onehot = torch.zeros(label.shape[0], num_classes)
    onehot = onehot.scatter(1, label.unsqueeze(1).data.cpu(), 1)
    onehot = onehot.to(label.device)
    return onehot


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup.

    Args:
        current (int): current step.
        rampup_length (int): maximum step.
    """
    assert rampup_length > 0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current/rampup_length
    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup.

    Args:
        current (int): current step.
        rampup_length (int): maximum step.
    """
    assert rampup_length > 0
    ratio = np.clip(current / rampup_length, 0.0, 1.0)
    return float(ratio)


def ema_model_update(model, ema_model, alpha):
    """Exponential moving average of model parameters.

    Args:
        model (nn.Module): model being trained.
        ema_model (nn.Module): ema of the model.
        alpha (float): ema decay rate.
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


def exists(val):
    return val is not None


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 -
                                                   momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t()  # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs


def cluster_assignment_matrix(z, c):
    norm_z = F.normalize(z, dim=2)
    norm_c = F.normalize(c, dim=1)
    return (norm_z @ norm_c.unsqueeze(0).transpose(2, 1)).clamp(0)


def compute_self_distance_batch(x):
    dist = x.square().sum(dim=2, keepdims=True) + \
        x.square().sum(dim=2).unsqueeze(1) - 2 * (x @ x.transpose(2, 1))
    return torch.exp(-dist/x.shape[2])


def get_modularity_matrix_and_edge(x, mode='cos'):
    """
        getting W=(A-ddT/2m) and getting all edges (e)
    """
    if mode == 'cos':
        norm = F.normalize(x, dim=2)
        A = (norm @ norm.transpose(2, 1)).clamp(0)
    elif mode == 'l2':
        A = compute_self_distance_batch(x)

    A = A - A * torch.eye(A.shape[1]).cuda()
    d = A.sum(dim=2, keepdims=True)
    e = A.sum(dim=(1, 2), keepdims=True)
    W = A - (d / e) @ (d.transpose(2, 1) / e) * e
    return W, e


def transform(x):
    """
    B, P, D => B, D, root(P), root(P)

    Ex) 128, 400, 768 => 128, 768, 20, 20
    """
    B, P, D = x.shape
    return x.permute(0, 2, 1).view(B, D, int(math.sqrt(P)), int(math.sqrt(P)))


def untransform(x):
    """
    B, D, P, P => B, P*P, D,

    Ex) 128, 768, 20, 20 => 128, 400, 768
    """
    B, D, P, P = x.shape
    return x.view(B, D, -1).permute(0, 2, 1)


def stochastic_sampling(x, order=None, k=4):
    """
    pooling
    """
    N = x.shape[1]
    d = (int(math.sqrt(N))) ** 2  # drop

    x = x[:, :d, :]

    x = transform(x)
    x_patch = x.unfold(2, k, k).unfold(3, k, k)
    x_patch = x_patch.permute(0, 2, 3, 4, 5, 1)
    x_patch = x_patch.reshape(-1, x_patch.shape[3:5].numel(), x_patch.shape[5])

    if order == None:
        order = torch.randint(k ** 2, size=(x_patch.shape[0],))

    x_patch = x_patch[range(x_patch.shape[0]), order].reshape(
        x.shape[0], x.shape[2]//k, x.shape[3]//k, -1)
    x_patch = x_patch.permute(0, 3, 1, 2)
    x = untransform(x_patch)
    return x, order


def compute_modularity(c, x, temp=0.1, grid=False):

    # detach
    x = x.detach().clone()
    # pooling for reducing GPU memory allocation
    if grid:
        x, _ = stochastic_sampling(x)

    # modularity matrix and its edge matrix
    W, e = get_modularity_matrix_and_edge(x)

    # cluster assignment matrix
    C = cluster_assignment_matrix(x, c.T)

    # tanh with temperature
    # D = C.transpose(2, 1)
    # import pdb;pdb.set_trace()

    E = torch.tanh(C @ C.transpose(2, 3) / temp)
    delta, _ = E.max(dim=1)
    Q = (W / e) @ delta

    # trace
    diag = Q.diagonal(offset=0, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)

    return -trace.mean()


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output