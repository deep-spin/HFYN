import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import energy, entmax, normmax_bisect

def main(normalize=False, layer_normalize=False):
    D = 2
    N = 5
    temp = 1
    n_samples = 100

    torch.random.manual_seed(42)

    which = [0, 3, 8, 9]
    nplots = len(which)

    fig, axes = plt.subplots(nplots, 5, figsize=(10, 6),
                             constrained_layout=True)

    patterns = []
    queries = []

    for i in range(10):
        pattern = torch.randn(N, D, dtype=torch.float64)
        query = torch.randn(D, 1, dtype=torch.float64)
        if layer_normalize:
            #Convex conjugate restriction
            patterns.append(torch.cat((pattern, - pattern[:, 0].unsqueeze(1) - pattern[:, 1].unsqueeze(1)), dim=1))
            queries.append(torch.cat((query, - query[0].unsqueeze(0) - query[1].unsqueeze(0)), dim=0))
        
        else:
            patterns.append(pattern)
            queries.append(query)
    
    if layer_normalize:
        point = torch.tensor([
            [-1, -1, 2],
            [-1, +1, 0],
            [+1, -1, 0]], dtype=torch.float64)
    else:
        point =torch.tensor([
        [-1, -1],
        [-1, +1],
        [+1, -1]], dtype=torch.float64)
    patterns[0] = point
    queries[0].zero_()


    for i in range(nplots):
        ii = which[i]
        X = patterns[ii]
        
        query = queries[ii]
            #print(torch.sqrt(torch.sum(X*X, dim=1)))
        if layer_normalize:
            X = (X - torch.mean(X, dim=1, keepdim=True))/torch.sqrt(torch.std(X,dim=1, keepdim=True)**2 + 1e-8)
        else:
            X = X / torch.sqrt(torch.sum(X*X, dim=1)).unsqueeze(1)
        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), X[:, 1].max()

        xmin -= .1
        ymin -= .1
        xmax += .1
        ymax += .1
        if i==0:
            ymin-=0.2
            ymax+=0.2
        xx = np.linspace(xmin, xmax, n_samples)
        yy = np.linspace(ymin, ymax, n_samples)
        
        mesh_x, mesh_y = np.meshgrid(xx, yy)
        if layer_normalize:
            mesh_z = - mesh_x - mesh_y
            Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()])
        else:
            Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        Q = torch.from_numpy(Q)
        # cmap = 'OrRd_r'
        cmap = 'viridis'

        E1 = energy(Q, X, alpha=1, beta=1/temp, normalize=normalize, layer_normalize=layer_normalize).reshape(*mesh_x.shape)
        axes[i,0].contourf(mesh_x, mesh_y, E1, cmap=cmap)
        E15 = energy(Q, X, alpha=1.5, beta=1/temp, normalize=normalize, layer_normalize=layer_normalize).reshape(*mesh_x.shape)
        axes[i,1].contourf(mesh_x, mesh_y, E15, cmap=cmap)
        E2 = energy(Q, X, alpha=2, beta=1/temp, normalize=normalize, layer_normalize=layer_normalize).reshape(*mesh_x.shape)
        axes[i,2].contourf(mesh_x, mesh_y, E2, cmap=cmap)
        E3 = energy(Q, X, alpha=2, beta=1/temp, normmax=True, normalize=normalize, layer_normalize=layer_normalize).reshape(*mesh_x.shape)
        axes[i,3].contourf(mesh_x, mesh_y, E3, cmap=cmap)
        E4 = energy(Q, X, alpha=5, beta=1/temp, normmax=True, normalize=normalize, layer_normalize=layer_normalize).reshape(*mesh_x.shape)
        axes[i,4].contourf(mesh_x, mesh_y, E4, cmap=cmap)
        p = torch.softmax(X.mm(query), dim=0)
        query = X.T @ p

        for k, alpha in enumerate([1, 1.5, 2, "normmax2", "normmax5"]):
            num_iters = 1000
            if layer_normalize:
                D=3
            xis = np.zeros((num_iters, D))
            xi = query
            for j in range(num_iters):
                xis[j, :] = xi[:, 0]
                if "normmax" in str(alpha):
                    p = normmax_bisect(X.mm(xi)/temp, alpha=int(alpha[-1]), n_iter=100, dim=0)
                else:
                    p = entmax(X.mm(xi)/temp, alpha, dim=0)
                xi = X.T.mm(p)
                if normalize:
                    xi = xi/torch.sqrt(torch.sum(xi*xi, dim=0)).unsqueeze(1)
                if layer_normalize:
                    xi = (xi - torch.mean(xi, dim=0))/torch.sqrt(torch.std(xi,dim=0, keepdim=True)**2 + 1e-8)
            first_point = xis[0]

# Plot a marker at the first point to represent the circumference
            axes[i, k].scatter(first_point[0], first_point[1], marker='o',facecolors='none', s=75, edgecolors='C1', linewidths=1.5, label='$q_0$')
            axes[i, k].plot(xis[0:, 0], xis[0:, 1],
                            lw=2,
                            marker='.',
                            color='C1',
                            label='$q_t$')

        for i, ax in enumerate(axes[i]):
            ax.plot(X[:, 0], X[:, 1], 's', markerfacecolor='w',
                    markeredgecolor='k', markeredgewidth=1, markersize=5,
                    label='$x_i$')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            # ax.set_xlim(xmin-.2, xmax+.2)
            # ax.set_ylim(ymin-.2, ymax+.2)
            ax.set_xticks(())
            ax.set_yticks(())

    axes[0,0].set_title("$1$-entmax",fontsize=22)
    axes[0,1].set_title("$1.5$-entmax",fontsize=22)
    axes[0,2].set_title("$2$-entmax", fontsize=22)
    axes[0,3].set_title("$2$-normmax", fontsize=22)
    axes[0,4].set_title("$5$-normmax", fontsize=22)
    if normalize:
        for i in range(5):
            axes[0, i].set_ylim(ymin-0.5, ymax)
    axes[0, 0].legend(fontsize=12)

    plt.savefig("contours.pdf", transparent=False)
    plt.show()


if __name__ == '__main__':
    main(False, False)

