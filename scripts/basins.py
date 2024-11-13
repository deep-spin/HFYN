import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import energy, entmax, normmax_bisect

def main(normalize=False, layer_normalize=False):


    D = 2
    N = 5
    # temp = 0.1
    temp = .25
    n_samples = 100
    thresh = 0.001

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

        xx = np.linspace(xmin, xmax, n_samples)
        yy = np.linspace(ymin, ymax, n_samples)

        mesh_x, mesh_y = np.meshgrid(xx, yy)

        if layer_normalize:
            mesh_z = - mesh_x - mesh_y
            Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()])
        else:
            Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        Q = torch.from_numpy(Q)

        for k, alpha in enumerate([1, 1.5, 2, "normmax2", "normmax5"]):
            num_iters = 50

            # X is n by d. Xi is m by d.
            Xi=Q
            for j in range(num_iters):
                if "normmax" in str(alpha):
                    p = normmax_bisect(Xi @ X.T / temp, alpha=int(alpha[-1]), n_iter=100, dim=-1)
                else:
                    p = entmax(Xi @ X.T / temp, alpha=alpha, dim=-1)
                Xi = p @ X
                if normalize:
                    mask = torch.sqrt(torch.sum(Xi*Xi, dim=1)) > 10**-6
                    Xi[mask] = Xi[mask]/(10**-6 + torch.sqrt(torch.sum(Xi[mask]*Xi[mask], dim=1)).unsqueeze(1))
                if layer_normalize:
                    Xi = (Xi - torch.mean(Xi, dim=1, keepdim=True))/torch.sqrt(torch.std(Xi,dim=1, keepdim=True)**2 + 1e-8)

            dists = torch.cdist(Xi, X)
            
            response = torch.zeros_like(dists[:, 0])
            for pp in range(len(X)):
                response[dists[:, pp] < thresh] = pp+1

            cols = ['w', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5'][:len(X)+1]
            cmap = matplotlib.colors.ListedColormap(cols)

            for pp in range(len(X)):
                response = response.reshape(*mesh_x.shape)
                axes[i,k].pcolormesh(mesh_x, mesh_y, response,
                                     vmin=0, vmax=len(X)+1,
                                     cmap=cmap)

                # axes[i,k].contourf(mesh_x, mesh_y, response,
                                   # levels=np.array([0, 1, 2, 3, 4, 5]) - .1,
                                   # colors=['w', 'C0', 'C1', 'C2', 'C3', 'C4',
                                   # 'C5']
                                   # )


        for ax in axes[i]:
            for pp in range(len(X)):
                ax.plot(X[pp, 0], X[pp, 1],
                        's',
                        markerfacecolor=f'C{pp}',
                        markeredgecolor='k',
                        markeredgewidth=1,
                        markersize=5)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(())
            ax.set_yticks(())
    plt.show()
    axes[0,0].set_title("$1$-entmax",fontsize=22)
    axes[0,1].set_title("$1.5$-entmax",fontsize=22)
    axes[0,2].set_title("$2$-entmax", fontsize=22)
    axes[0,3].set_title("$2$-normmax", fontsize=22)
    axes[0,4].set_title("$5$-normmax", fontsize=22)
    plt.savefig("basins4.png", dpi=600, format='png')


if __name__ == '__main__':
    main(False, False)




