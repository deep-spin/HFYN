# author: vlad niculae and andre f. t. martins
# license: simplified bsd

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from entmax import sparsemax, entmax15, normmax_bisect
from lpsmap import TorchFactorGraph, Budget
from lpsmap import Sequence, SequenceBudget
import numpy as np

def constrained_sparsemax(z, u):
    '''Solve the problem:

    min_p 0.5*\|p - z\|^2
    s.t. p <= u
         p in simplex
    '''
    l = np.zeros_like(u)
    return double_constrained_sparsemax(z, l, u)

def double_constrained_sparsemax(z, l, u):
    '''Solve the problem:

    min_p 0.5*\|p - z\|^2
    s.t. l <= p <= u
         p in simplex

    This maps to Pardalos' canonical problem by making the transformations
    below.
    '''
    assert (u>=0).all(), "Invalid: u[i]<0 for some i"

    # Look for -inf entries in z, create due to padding and masking.
    ind = np.nonzero(z != -np.inf)[0]
    if len(ind) < len(z):
        p = np.zeros(len(z))
        regions = np.zeros(len(z), dtype=int)
        p[ind], regions[ind], tau, val = double_constrained_sparsemax(
            z[ind], l[ind], u[ind])
        return p, regions, tau, val

    dtype = z.dtype
    z = z.astype('float64')
    l = l.astype('float64')
    u = u.astype('float64')
    a = .5 * (l - z)
    b = .5 * (u - z)
    c = np.ones_like(z)
    d = .5 * (1 - z.sum())
    x, tau, regions = solve_quadratic_problem(a, b, c, d)
    tau = -2*tau
    p = z - tau
    ind = np.nonzero(regions == 0)[0]
    p[ind] = l[ind]
    ind = np.nonzero(regions == 2)[0]
    p[ind] = u[ind]
    p = p.astype(dtype)
    return p, regions, tau, .5*np.dot(p-z, p-z)

def solve_quadratic_problem(a, b, c, d):
    '''Solve the problem:

    min_x sum_i c_i x_i^2
    s.t. sum_i c_i x_i = d
         a_i <= x_i <= b_i, for all i.

    by using Pardalos' algorithm:

    Pardalos, Panos M., and Naina Kovoor.
    "An algorithm for a singly constrained class of quadratic programs subject
    to upper and lower bounds." Mathematical Programming 46.1 (1990): 321-328.
    '''
    K = np.shape(c)[0]

    # Check for tight constraints.
    ind = np.nonzero(a == b)[0]
    if len(ind):
        x = np.zeros(K)
        regions = np.zeros(K, dtype=int)
        x[ind] = a[ind]
        regions[ind] = 0 # By convention.
        dd = d - c[ind].dot(x[ind])
        ind = np.nonzero(a < b)[0]
        if len(ind):
            x[ind], tau, regions[ind] = \
                solve_quadratic_problem(a[ind], b[ind], c[ind], dd)
        else:
            tau = 0. # By convention.
        return x, tau, regions

    # Sort lower and upper bounds and keep the sorted indices.
    sorted_lower = np.argsort(a)
    sorted_upper = np.argsort(b)
    slackweight = 0.
    tightsum = np.dot(a, c)
    k, l, level = 0, 0, 0
    right = -np.inf
    found = False
    while k < K or l < K:
        # Compute the estimate for tau.
        if level:
            tau = (d - tightsum) / slackweight
        if k < K:
            index_a = sorted_lower[k]
            val_a = a[index_a]
        else:
            val_a = np.inf
        if l < K:
            index_b = sorted_upper[l]
            val_b = b[index_b]
        else:
            val_b = np.inf

        left = right
        if val_a < val_b:
            # Next value comes from the a-list.
            right = val_a
        else:
            # Next value comes from the b-list.
            left = right
            right = val_b

        assert not level or tau >= left
        if (not level and d == tightsum) or (level and left <= tau <= right):
            # Found the right split-point!
            found = True
            break

        if val_a < val_b:
            tightsum -= a[index_a] * c[index_a]
            slackweight += c[index_a]
            level += 1
            k += 1
        else:
            tightsum += b[index_b] * c[index_b]
            slackweight -= c[index_b]
            level -= 1
            l += 1

    x = np.zeros(K)
    if not found:
        left = right
        right = np.inf

    regions = -np.ones(K, dtype=int)
    for i in range(K):
        if a[i] >= right:
            x[i] = a[i]
            regions[i] = 0
        elif b[i] <= left:
            x[i] = b[i]
            regions[i] = 2
        else:
            assert found and level
            x[i] = tau
            regions[i] = 1

    return x, tau, regions

class ConstrainedSparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, input1, input2):
        z = input1.cpu().numpy()
        u = input2.cpu().numpy()
        probs = np.zeros_like(z)
        regions = np.zeros_like(z)
        for i in range(z.shape[0]):
            probs[i,:], regions[i,:], _, _ = constrained_sparsemax(z[i], u[i])
            assert np.all(probs[i, :] == probs[i, :])
        probs = torch.from_numpy(probs)
        regions = torch.from_numpy(regions)
        if input1.is_cuda:
            probs = probs.cuda()
            regions = regions.cuda()
        ctx.save_for_backward(probs)
        ctx.saved_intermediate = regions, # Not sure this is safe.
        return probs

    @classmethod
    def backward(cls, ctx, grad_output):
        output, = ctx.saved_tensors
        regions, = ctx.saved_intermediate
        probs = output
        regions = regions.cpu().numpy() # TODO: do everything with tensors.
        r0 = np.array(regions == 0, dtype=regions.dtype)
        r1 = np.array(regions == 1, dtype=regions.dtype)
        r2 = np.array(regions == 2, dtype=regions.dtype)
        np_grad_output = grad_output.cpu().numpy()
        avg = np.sum(np_grad_output * r1, 1) / np.sum(r1, 1)
        np_grad_input1 = r1 * (np_grad_output - np.tile(avg[:,None],
                                                        [1, r1.shape[1]]))
        np_grad_input2 = r2 * (np_grad_output - np.tile(avg[:,None],
                                                        [1, r2.shape[1]]))
        ind = np.nonzero(np.sum(r1, 1) == 0)[0]
        for i in ind:
            np_grad_input1[i, :] = 0.
            np_grad_input2[i, :] = 0.
        assert np.all(np_grad_input1 == np_grad_input1)
        assert np.all(np_grad_input2 == np_grad_input2)
        grad_input1 = torch.from_numpy(np_grad_input1)
        grad_input2 = torch.from_numpy(np_grad_input2)
        if grad_output.is_cuda:
            grad_input1 = grad_input1.cuda()
            grad_input2 = grad_input2.cuda()
        return grad_input1, grad_input2


def constrained_sparsemax_function(z, u):
    return ConstrainedSparsemaxFunction.apply(z, u)

class ConstrainedSparsemax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, u):
        return constrained_sparsemax(z, u)
    
class Flatten(object):

    def __call__(self, tensor):
        return torch.flatten(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
def tsallis_unif(n, alpha, dtype=torch.double):
    if alpha == 1:
        return math.log(n)
    return (1 - n**(1-alpha)) / (alpha * (alpha-1))


def tsallis(p, alpha, dim=-1):
    if alpha == 1:
        return torch.special.entr(p).sum(dim)
    else:
        return ((p - p ** alpha) / (alpha * (alpha-1))).sum(dim=dim)
def normmaxentropy(p, alpha, dim=-1):

    return 1 - torch.sqrt((p**2).sum(dim=-1))

def normmax_unif(n, alpha, dim=-1):
    return 1 - math.sqrt(n)/n

def entmax(Z, alpha, dim):
    """only exact cases; raise otherwise; for toy experiments"""
    if alpha == 1:
        return torch.softmax(Z, dim=dim)
    elif alpha == 1.5:
        return entmax15(Z, dim=dim)
    elif alpha == 2:
        return sparsemax(Z, dim=dim)
    else:
        raise NotImplementedError()

def SparseMAP_exactly_k(scores, k=2):
    marginals = torch.zeros_like(scores)
    for j in range(scores.shape[1]):
        print(j)
        fg = TorchFactorGraph()
        u = fg.variable_from(scores[j, :])
        fg.add(Budget(u, k, force_budget=True))
        fg.solve(verbose=0)
        marginals[j, :] = u.value[:]
    return marginals

def SparseMAP_sequence_exactly_k(scores, edge_score, k=2):
    n = scores.shape[0]
    transition = torch.zeros((n+1,2,2), requires_grad=True)
    transition.data[1:n, 0, 0] = edge_score
    # Only one state in the beginning and in the end for start / stop symbol.
    transition = transition.reshape(-1)[2:-2]
    marginals = torch.zeros_like(scores)
    for j in range(scores.shape[1]):
        s = torch.zeros((n, 2))
        s[:, 0] = scores[:, j]
        fg = TorchFactorGraph()
        u = fg.variable_from(s)
        fg.add(SequenceBudget(u, transition, k, force_budget=True))
        fg.solve(verbose=0)
        marginals[:, j] = u.value[:, 0]
    return marginals

def indicator_function(q, r):
    norm_q = torch.norm(q, dim=1)  # Assuming q is a 2D tensor with each row as a vector in the batch
    indicator = torch.where(norm_q <= r, torch.tensor(0.0), torch.tensor(float('inf')))
    return indicator

def indicator_function_ln(q, r):
    N = q.shape[1]
    norm_q = torch.norm(q, dim=1)  # Assuming q is a 2D tensor with each row as a vector in the batch
    outside_set = (torch.sum(q, dim=1) == 0)  # Check if the sum of elements in each row is 0
    indicator = torch.where((norm_q <= r*math.sqrt(N-1)) & outside_set, torch.tensor(0.0), torch.tensor(float('inf')))
    return indicator

def energy(Q, X, alpha=1.0, beta=1.0, normmax = False, normalize = False, layer_normalize=False):
    # Q: tensor dim (m, d)
    # X: tensor dim (n, d)
    n = X.shape[0]

    # m is batch dim, so bXQ[i] = bXq_i
    bXQ = beta * Q @ X.T
    if normmax:
        phat = normmax_bisect(bXQ, alpha, -1, 500)
        fy_term_gold = -normmax_unif(n, alpha) - bXQ.mean(dim=-1)
        fy_term_pred = -normmaxentropy(phat, alpha) - (bXQ * phat).sum(dim=-1)
    else:
        phat = entmax(bXQ, alpha, dim = -1)
        fy_term_gold = -tsallis_unif(n, alpha) - bXQ.mean(dim=-1)
        fy_term_pred = -tsallis(phat, alpha) - (bXQ * phat).sum(dim=-1)
    
    fy = fy_term_gold - fy_term_pred
    mx = X.mean(dim=0)
    q_nrmsq = (Q**2).sum(dim=-1)
    Msq = (X ** 2).sum(dim=-1).max()
    if normalize:
       return -fy/beta - Q @ mx + indicator_function(Q, 1) + Msq/2
    elif layer_normalize:
       return -fy/beta - Q @ mx + indicator_function_ln(Q, 1) + Msq/2 
    else:
        return -fy/beta - Q @ mx + q_nrmsq/2 + Msq/2

class HopfieldNet(nn.Module):

    def __init__(self, 
                 in_features,
                 yomega="entmax",
                 alpha=1.0,
                 ypsi = "none",
                 beta=1.0,
                 inner_beta = 0.05,
                 max_iter=128,
                 device = "cpu",
                 k = 2):
        
        super(HopfieldNet, self).__init__()
        self.X = in_features
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.yomega = yomega
        self.ypsi = ypsi
        self.inner_beta = inner_beta
        self.device = device
     
    def _none(self, tensor, axis=1):
        return tensor

    def _normalization(self, tensor, axis=1):
        return tensor / torch.norm(tensor, p=2, dim=axis, keepdim=True)

    def _layer_normalization(self, tensor, axis=1):
        layer_norm = torch.nn.LayerNorm(normalized_shape=tensor.shape[axis], elementwise_affine=False).to(self.device)
        if axis==0:
            return layer_norm(tensor.T).T 
        else: 
            return layer_norm(tensor)

    def _tanh(self, tensor, axis=1):
        return torch.tanh(self.beta * tensor)

    def _sparsemap(self, tensor, axis=1):
        return SparseMAP_exactly_k(self.beta * tensor, self.k)

    def _entmax(self, tensor, axis=1):
        return entmax(self.beta * tensor, self.alpha, dim=0)

    def _normmax(self, tensor, axis=1):
        return normmax_bisect(self.beta * tensor, self.alpha, dim=0)

    def _identity(self, tensor, axis=1):
        return tensor

    def _5_poly(self, tensor, axis=1):
        return (tensor)**5

    def _10_poly(self, tensor, axis=1):
        return (tensor)**10

    def _exp(self, tensor, axis=1):
        return torch.exp(tensor * self.inner_beta)
    
    def run(self, Q, return_p=True):
        ypsi = getattr(self, f"_{self.ypsi}")
        yomega = getattr(self, f"_{self.yomega}")
        if self.ypsi != "tanh":
            self.X = ypsi(self.X)
            Q = ypsi(Q, axis = 0)
        for i in range(self.max_iter):
            p = yomega(self.X.mm(Q))
            Q = ypsi(self.X.T.mm(p), axis=0)
        if return_p:
            return p
        else:
            return Q
     
    def forward(self, Q, return_p):
        
        return self.run(Q, return_p)