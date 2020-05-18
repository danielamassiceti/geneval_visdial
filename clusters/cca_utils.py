import torch
import numpy as np
import scipy.linalg as la

def topk_corr(view1, view2, k, dim=1):
 
    assert (dim >= 1)

    denom = torch.norm(view1, p=2, dim=dim) * torch.norm(view2.expand_as(view1), p=2, dim=dim)
    if dim == 1: 
        corrs = torch.mm(view1, view2.t()).squeeze().div_(denom)
    else:
        corrs = torch.matmul(view1, view2.transpose(dim-1,dim)).squeeze(-1).div_(denom)
    
    return corrs, torch.topk(corrs, k=k, dim=dim-1, sorted=True) # indices: top k 

def diagonal(a):
    return a.as_strided((a.size(0),), (a.size(0)+1,))

def mean_center(X, mu):
    return X - mu.t().expand_as(X)

# Bach, F. R. and Jordan, M. I. Kernel independent component analysis. J. Mach. Learn. Res., 3:1-48, 2002
# https://www.di.ens.fr/~fbach/kernelICA-jmlr.pdf
def compute_cca(views, k=300, eps=1e-12):

    """
    views: list of views, each N x v_i_emb where N is the number of observations and v_i_emb is the embedding dimensionality of that view
    k: integer for the dimensionality of the joint projection space
    eps: float added to diagonals of matrices A and B for stability
    """

    m = views[0].size(0)
    t = views[0].type()
    o = [v.size(1) for v in views]
    os = sum(o)
    A = torch.zeros(os, os).type(t) 
    B = torch.zeros(os, os).type(t)

    print('doing generalised eigendecomposition...')
   
    row_i = 0
    for i, V_i in enumerate(views):
        V_i = V_i.t()
        o_i = V_i.size(0)
        mu_i = V_i.mean(dim=1, keepdim=True)

        # mean center view i
        V_i_bar = V_i - mu_i.expand_as(V_i) # o_i x N 

        col_i = 0
        for j, V_j in enumerate(views):

            V_j = V_j.t()
            o_j = V_j.size(0) 
            
            if i>j:
                col_i += o_j
                continue
            mu_j = V_j.mean(dim=1, keepdim=True)

            # mean center view j
            V_j_bar = V_j - mu_j.expand_as(V_j) # o_j x N 
            
            C_ij = (1.0 / (m - 1)) * torch.mm(V_i_bar, V_j_bar.t()) # o_i x o_j
            
            A[row_i:row_i+o_i,col_i:col_i+o_j] = C_ij
            A[col_i:col_i+o_j,row_i:row_i+o_i] = C_ij.t()
            if i == j:
                B[row_i:row_i+o_i,col_i:col_i+o_j] = C_ij.clone()
            
            col_i += o_j
        row_i += o_i

    diagonal(A).add_(eps)
    diagonal(B).add_(eps)

    A = A.cpu().numpy()
    B = B.cpu().numpy()

    l, v = la.eig(A, B)
    idx = l.argsort()[-k:][::-1]
    l = l[idx] # eigenvalues
    v = v[:,idx] # eigenvectors

    l = torch.from_numpy(l.real)
    v = torch.from_numpy(v.real)

    # extracting projection matrices
    proj_matrices = [v[sum(o[:i]):sum(o[:i])+views[i].size(1)].type(t) for i in range(len(views))] 
    return l.type(t), proj_matrices

def get_projection(x, b, l, p=0):
    return torch.mm(x, torch.mm(b, torch.diag(l ** p)))

