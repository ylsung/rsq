import time
import torch


def KMeans(x, K=10, Niter=10, verbose=True, using_loop=True):
    # Modified from https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    rand_indices = torch.randperm(K).to(x.device)
    
    c = x[rand_indices, :].clone()  # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        if using_loop:
            D_ij = torch.zeros(N, K, device=x.device)
            for i in range(K):
                D_ij[:, i] = torch.sum((x - c[i])**2, dim=1)
        else:
            # D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
            D_ij = - 2 * x.matmul(c.transpose(0, 1)) + (x ** 2).sum(-1)[:, None] + (c ** 2).sum(-1)[None, :]
        
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= (Ncl + 1e-8)  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c
