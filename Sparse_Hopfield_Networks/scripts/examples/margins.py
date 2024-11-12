import torch
import numpy as np
import matplotlib.pyplot as plt
from entmax import sparsemax, entmax15, entmax_bisect, EntmaxBisectLoss, NormmaxBisectLoss, normmax_bisect

# Define entropy functions using PyTorch
def tsallis_entropy(p, alpha):
    if alpha == 1:
        return torch.sum(p * torch.log(p))
    else:
        return -1*(1 - torch.sum(p**alpha)) / (alpha - 1)

def norm_entropy(p, gamma):
    return -1 + torch.norm(p, p=gamma)

# Generate values for t and s
t = torch.linspace(0, 1, 100)
s = torch.linspace(-3, 3, 100)

# Entropy H(t, 1-t)
H_t_1_tsallis = [tsallis_entropy(torch.tensor([t_val, 1-t_val]), alpha=1).item() for t_val in t]
H_t_1_5_tsallis = [tsallis_entropy(torch.tensor([t_val, 1-t_val]), alpha=1.5).item() for t_val in t]
H_t_2_tsallis = [tsallis_entropy(torch.tensor([t_val, 1-t_val]), alpha=2).item() for t_val in t]
H_t_argmax_tsallis = [tsallis_entropy(torch.tensor([t_val, 1-t_val]), alpha=9e3).item() for t_val in t]
H_t_argmax_norm = [norm_entropy(torch.tensor([t_val, 1-t_val]), gamma=1.0000000000000000001).item() for t_val in t]
H_t_2_norm = [norm_entropy(torch.tensor([t_val, 1-t_val]), gamma=2).item() for t_val in t]
H_t_5_norm = [norm_entropy(torch.tensor([t_val, 1-t_val]), gamma=5).item() for t_val in t]
H_t_inf_norm = [norm_entropy(torch.tensor([t_val, 1-t_val]), gamma=100).item() for t_val in t]

# Predictive distribution yhat(s, 0)
yhat_1_tsallis = [torch.softmax(torch.tensor([s_val, 0]), dim=-1).detach().numpy() for s_val in s]
yhat_1_5_tsallis = [entmax_bisect(torch.tensor([s_val, 0]), alpha=1.5).detach().numpy() for s_val in s]
yhat_2_tsallis = [entmax_bisect(torch.tensor([s_val, 0]), alpha=2).detach().numpy() for s_val in s]
yhat_argmax_tsallis = [entmax_bisect(torch.tensor([s_val, 0]), alpha=100).detach().numpy() for s_val in s]
yhat_argmax_norm = [normmax_bisect(torch.tensor([s_val, 0]), alpha=1.00000000000000001).detach().numpy() for s_val in s]
yhat_2_norm = [normmax_bisect(torch.tensor([s_val, 0]), alpha=2).detach().numpy() for s_val in s]
yhat_5_norm = [normmax_bisect(torch.tensor([s_val, 0]), alpha=5).detach().numpy() for s_val in s]
yhat_inf_norm = [normmax_bisect(torch.tensor([s_val, 0]), alpha=1000).detach().numpy() for s_val in s]

# Loss L_H([s, 0]; e1)
L_H_1_tsallis = [torch.nn.CrossEntropyLoss()(torch.tensor([[s_val, 0]]), torch.tensor([0])).item() for s_val in s]
L_H_1_5_tsallis = [EntmaxBisectLoss(alpha=1.5)(torch.tensor([[s_val, 0]]), torch.tensor([0])).item() for s_val in s]
L_H_2_tsallis = [EntmaxBisectLoss(alpha=2)(torch.tensor([[s_val, 0]]), torch.tensor([0])).item() for s_val in s]
L_H_argmax_tsallis = [EntmaxBisectLoss(alpha=100)(torch.tensor([[s_val, 0]]), torch.tensor([0])).item() for s_val in s]
L_H_argmax_norm = [NormmaxBisectLoss(alpha=1.000000000001)(torch.tensor([[s_val, 0]]), torch.tensor([0])).item() for s_val in s]
L_H_2_norm = [NormmaxBisectLoss(alpha=2)(torch.tensor([[s_val, 0]]), torch.tensor([0])).item() for s_val in s]
L_H_5_norm = [NormmaxBisectLoss(alpha=5)(torch.tensor([[s_val, 0]]), torch.tensor([0])).item() for s_val in s]
L_H_inf_norm = [NormmaxBisectLoss(alpha=100)(torch.tensor([[s_val, 0]]), torch.tensor([0])).item() for s_val in s]

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(28, 14))

# Plot for H(t, 1-t)
axes[0, 0].plot(t, H_t_1_tsallis, label='$1$-entmax (softmax)', color="dodgerblue", linewidth=3)
axes[0, 0].plot(t, H_t_1_5_tsallis, label='$1.5$-entmax', color="red", linestyle="--", linewidth=3)
axes[0, 0].plot(t, H_t_2_tsallis, label='$2$-entmax (sparsemax)', color="green", linewidth=3)
axes[0, 0].plot(t, H_t_argmax_tsallis, linestyle='--', color='black', label='α = $\infty$-entmax (argmax)', linewidth=3)
axes[0, 0].set_title('Negentropy $\Omega([p, 1-p])$', fontsize=30, pad=15)
axes[0, 0].tick_params(axis='y', labelsize=20, which='major')
axes[0, 0].set_xticklabels([])

# Plot for Entropy (Norm) with reversed color order
axes[1, 0].plot(t, H_t_inf_norm, label='γ = 9e9 (Norm)', color='dodgerblue', linewidth=3)
axes[1, 0].plot(t, H_t_5_norm, label='γ = 5 (Norm)', color="red", linestyle="--", linewidth=3)
axes[1, 0].plot(t, H_t_2_norm, label='γ = 2 (Norm)', color='green', linewidth=3)
axes[1, 0].plot(t, H_t_argmax_norm, linestyle='--', color='black', label='γ = 1 (Norm)', linewidth=3)
axes[1, 0].tick_params(axis='y', labelsize=20, which='major')
axes[1, 0].tick_params(axis='x', labelsize=20, which='major')
axes[1, 0].set_xlabel('p', fontsize=25)

# Plot for Predictive Distribution (Tsallis) with reversed color order
axes[0, 1].plot(s, [y[0] for y in yhat_argmax_tsallis], linestyle='--', color='black', label='α = 9e9 (Tsallis)', linewidth=3)
axes[0, 1].plot(s, [y[0] for y in yhat_2_tsallis], label='α = 2 (Tsallis)',  color="green", linewidth=3)
axes[0, 1].plot(s, [y[0] for y in yhat_1_5_tsallis], label='α = 1.5 (Tsallis)', linestyle="--",  color="red", linewidth=3)
axes[0, 1].plot(s, [y[0] for y in yhat_1_tsallis], label='α = 1 (Tsallis)', color="dodgerblue", linewidth=3)
axes[0, 1].set_title('Predictive Distribution $\mathbf{\hat{y}}_\Omega([s, 0])$', fontsize=30, pad=15)
axes[0, 1].tick_params(axis='y', labelsize=20, which='major')
axes[0, 1].set_xticklabels([])

# Plot for Predictive Distribution (Norm) with reversed color order
axes[1, 1].plot(s, [y[0] for y in yhat_inf_norm], label='γ = 9e9 (Norm)', color='dodgerblue', linewidth=3)
axes[1, 1].plot(s, [y[0] for y in yhat_5_norm], label='γ = 5 (Norm)', color="red", linestyle="--", linewidth=3)
axes[1, 1].plot(s, [y[0] for y in yhat_2_norm], label='γ = 2 (Norm)', color='green', linewidth=3)
axes[1, 1].plot(s, [y[0] for y in yhat_argmax_norm], linestyle='--', color='black', label='γ = 1 (Norm)', linewidth=3)
axes[1, 1].set_xlabel('s', fontsize=25)
axes[1, 1].tick_params(axis='y', labelsize=20, which='major')
axes[1, 1].tick_params(axis='x', labelsize=20, which='major')


# Plot for Loss (Tsallis) with reversed color order
axes[0, 2].plot(s, L_H_argmax_tsallis, linestyle='--', color='black', label='$\infty$-entmax (argmax)', linewidth=3)
axes[0, 2].plot(s, L_H_2_tsallis, label='$2$-entmax (sparsemax)', color="green", linewidth=3)
axes[0, 2].plot(s, L_H_1_5_tsallis, label='$1.5$-entmax', color="red", linestyle="--", linewidth=3)
axes[0, 2].plot(s, L_H_1_tsallis, label='$1$-entmax (softmax)', color="dodgerblue", linewidth=3)
axes[0, 2].set_title('FY Loss $L_\Omega([s, 0]; \mathbf{e}_1)$', fontsize=30, pad=15)
legend = axes[0, 2].legend(fontsize=25, title="Tsallis negentropy $\Omega_{\\alpha}^T$")
plt.setp(legend.get_title(), fontsize=27)
axes[0, 2].tick_params(axis='y', labelsize=20, which='major')
axes[0, 2].set_xticklabels([])

# Plot for Loss (Norm) with reversed color order
axes[1, 2].plot(s, L_H_inf_norm, label='$\infty$-normmax', color='dodgerblue', linewidth=3)
axes[1, 2].plot(s, L_H_5_norm, label='$5$-normmax', color="red", linestyle="--", linewidth=3)
axes[1, 2].plot(s, L_H_2_norm, label='$2$-normmax', color='green', linewidth=3)
axes[1, 2].plot(s, L_H_argmax_norm, linestyle='--', color='black', label='$1$-normmax (argmax)', linewidth=3)
axes[1, 2].set_xlabel('s', fontsize=25)
legend = axes[1, 2].legend(fontsize=25, title="Norm negentropy $\Omega_{\gamma}^N$")
plt.setp(legend.get_title(), fontsize=27)
axes[1, 2].tick_params(axis='y', labelsize=20, which='major')
axes[1, 2].tick_params(axis='x', labelsize=20, which='major')

plt.tight_layout()
plt.show()
plt.savefig("margins.pdf")