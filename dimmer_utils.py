import numpy as np
import jax.numpy as jnp
from jax import jit,vmap

def find_unique_index1_points(U,Vm,gradV,hessV,iflm,eig_eps,eig_gap,grad_eps):
    Xdx1=jnp.array(U)
    S,N_a,d=Xdx1.shape
    index = jnp.arange(0, S)
    total_d=N_a*d
    # first remove points with large gradient
    dv_x  = gradV(Xdx1)
    dv_norm = jnp.linalg.norm(dv_x.reshape(S, total_d), axis=1)
    norm_index=dv_norm < grad_eps
    have_small_grad=norm_index.any()
    if have_small_grad:
        Xdx1=Xdx1[norm_index]
        S=Xdx1.shape[0]
        index = jnp.arange(0, S)
        ddv_x = hessV(Xdx1).reshape(S,total_d,total_d)
        evals, evecs = jnp.linalg.eigh(ddv_x)
        # first_pos_index = total_d - np.sum(evals[:] > eig_eps, axis=1)
        # first_pos_values = iflm(index, first_pos_index, evals)
        # particles that satisfy index1 condition
        index1_cond = jnp.logical_and(evals[:, 1+eig_gap] > eig_eps, evals[:, 0] < -eig_eps)
        Ndim_prtcls = len(Xdx1[index1_cond])
        # found index-1 points, return the unique ones
        if Ndim_prtcls > 0:
            # Xdx1 = jnp.round(Xdx1, decimals=5)
            fEval=jnp.round(Vm(Xdx1), decimals=6)
            unique_val, unique_idx = jnp.unique(fEval, return_index=True)
            return True,Xdx1[unique_idx]
        # failed to find any index-1 points
        else:
            return False,None
    # failed to find any
    else:
        return False,None
# Among the N particles, select the top S according to weight
# compute the eigevalues of the S best particles
# find all that are index-1
# the best one is the one with smallest gradient norm
# X (all particle states
# Wgts -- weights
# gradV/hessV gradient and hessain
# eig_eps>0 parameter if m>eig_eps then m is considered positive
# N_p number of particles
# total_d total number of dimensions (number of atoms x d)
# iflm helper function
def find_best_particles(X,Wgts,gradV,hessV,eig_eps,eig_gap,S,N_p,total_d,iflm,rule):
    index = jnp.arange(0, S)
    # find the S top weights
    W=Wgts.flatten()
    highest_weights = np.argpartition(W, -S)[-S:]
    # argpartition doesnt have them sorted, so sort them
    highest_weights=highest_weights[np.argsort(W[highest_weights])]
    # flip the sort so that it is highest first
    highest_weights=np.flip(highest_weights)
    # selct the best S
    Z = X[highest_weights]
    W = W[highest_weights]
    # calculate the hessian of the best particles
    ddv_hw = hessV(Z).reshape(S, total_d, total_d)
    # find the eigenvlaues of their hessian
    evals, evecs = jnp.linalg.eigh(ddv_hw)
    # find the index of the first positive eigevalue
    # first_pos_index = total_d - np.sum(evals[:] > eig_eps, axis=1)
    # first_pos_values = iflm(index, first_pos_index, evals)
    # particles that satisfy index1 condition
    index1_cond = jnp.logical_and(evals[:, 1+eig_gap] > eig_eps ,  evals[:, 0] < -eig_eps)
    Ndim_prtcls=len(Z[index1_cond])
    # if a particle exists that is index-1
    if Ndim_prtcls > 0:
        U = Z[index1_cond]
        W = W[index1_cond]
        # calculate gradeint norms
        dvu = gradV(U)
        gd_norm = jnp.linalg.norm(dvu,axis=2)
        # if selecting smallest gradient
        if rule == "grad":
            min_norm_ind = np.argmin(gd_norm)
            return Ndim_prtcls,U,U[min_norm_ind]
        # otherwise pick the largest weight
        else:
            max_weight_ind = np.argmax(W)
            return Ndim_prtcls, U, U[max_weight_ind]
    else:
        return 0,None,None

def get_weight_stats(Weights):
    maxw_ind = np.argmax(Weights)
    minw_ind = np.argmin(Weights)
    max_weight = Weights[maxw_ind][0]
    min_weight = Weights[minw_ind][0]
    return maxw_ind, minw_ind, max_weight, min_weight

def single_particle_stats(X,gradV,hessV,N):
    dx=gradV(X)
    ddx=hessV(X).reshape(N,N)
    gd_norm=jnp.linalg.norm(dx)
    eigs=jnp.linalg.eigvalsh(ddx)
    return gd_norm,eigs

def setup_particle_dimmer():
    dist_adj_mat=lambda i,a: 1.0/jnp.exp(jnp.linalg.norm(a[i]-a,axis=2))
    dist_adj_mat=jit(dist_adj_mat)
    dist_adj_mat_map=jit(vmap(dist_adj_mat,(0,None)))
    scale_adj_mat = (lambda i, A, A_sum: A[i, :] / A_sum[i])
    scale_adj_mat_map = (vmap(scale_adj_mat, (0, None, None)))
    dot_product_dimmer = lambda i, a, b: jnp.sum(a[i] * b[i])
    dot_product_dimmermap = jit(vmap(dot_product_dimmer, (0, None, None)))
    mult_dimmer = lambda i, a, b: (a[i] * b[i])
    mult_dimmermap = jit(vmap(mult_dimmer, (0, None, None)))

    return dist_adj_mat_map,scale_adj_mat_map,dot_product_dimmermap,mult_dimmermap