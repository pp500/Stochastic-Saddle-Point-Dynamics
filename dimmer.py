import numpy as np
import jax.numpy as jnp
from jax import vmap,jit,random
import jax.ops
import scipy
import networkx as nx
import pickle
import dimmer_utils

dist_adj_mat=lambda i,a: 1.0/jnp.exp(jnp.linalg.norm(a[i]-a,axis=2))
dist_adj_mat=jit(dist_adj_mat)
dist_adj_mat_map=jit(vmap(dist_adj_mat,(0,None)))

def dimmer_eig_calc(hess,d,k=2):
    if d <= 2:
        mu_t, v_t = scipy.linalg.eigh(hess)
    else:
        mu_t, v_t = scipy.sparse.linalg.eigsh(hess, k=k, which="SA")
    # v1=v_t[:,0]
    v1=v_t[:,0].reshape(d,1)
    return mu_t,v1

def getLaplacian(prms,U_t,dist_adj_mat_map,scale_adj_mat_map):
    N_prtcls = U_t.shape[0]
    dimmer_prtcl_index = np.arange(0, N_prtcls)
    # # compute adjencncy matrix
    adj_mat = dist_adj_mat_map(dimmer_prtcl_index, U_t)[:, :, 0]
    # # any node more than prms.d_r is set to 0 amd this is the (unscaled) adjency matrix
    adj_mat = jax.ops.index_update(adj_mat, adj_mat < prms.d_r, 0)
    # # scale adjecency matrix to sum to 1
    adj_mat_sum = jnp.sum(adj_mat, axis=1)
    adj_mat = scale_adj_mat_map(dimmer_prtcl_index, adj_mat, adj_mat_sum)
    G = nx.Graph(np.array(adj_mat))
    num_connected_comp=nx.number_connected_components(G)
    lap_mat = (nx.laplacian_matrix(G, weight='weight')).todense()
    return N_prtcls,lap_mat,num_connected_comp

def particle_ideal_dimmer(prms,key,U_t,Vm,dV,ddV,index_func_lambda_map,dist_adj_mat_map,scale_adj_mat_map,dot_product_dimmermap,mult_dimmermap,debug=False):
    N_prtcls,lap_mat,num_connected_comp = getLaplacian(prms,U_t,dist_adj_mat_map,scale_adj_mat_map)
    dimmer_prtcl_index=np.arange(0,N_prtcls)
    index = jnp.arange(0, N_prtcls)
    total_d=prms.N_a*prms.d
    # 0:success,1:hit max iter,2:failed because out of index-1
    exit_code = 0
    grad_t = dV(U_t)
    dv_norm = jnp.linalg.norm(grad_t.reshape(N_prtcls, total_d), axis=1)
    dv_norm_max = jnp.max(dv_norm)
    for t in range(0, prms.dimmer_max_iter):

        if debug:
            file = open("../debug/particle_dimer_" + str(t) + ".pickle", "wb")
            pickle.dump([t,U_t,lap_mat], file)
            print(t,dv_norm_max)
            file.close()

        grad_t = dV(U_t)
        hess_t = ddV(U_t).reshape(N_prtcls, total_d, total_d)
        # compute terms for update
        inter_term = jnp.atleast_3d(lap_mat @ U_t.reshape(N_prtcls, total_d)).reshape(N_prtcls, prms.N_a, prms.d)
        mu_t, v_t = jnp.linalg.eigh(hess_t)
        v_1 = v_t[:, :, 0].reshape(N_prtcls, prms.N_a, prms.d)
        v1_grad = dot_product_dimmermap(dimmer_prtcl_index, v_1, grad_t)
        v_term = mult_dimmermap(dimmer_prtcl_index, v_1, v1_grad)
        dV_term = (grad_t - 2.0 * v_term).reshape(N_prtcls, prms.N_a, prms.d)
        # do the update
        U_t = U_t - prms.dimmer_dt * (dV_term + prms.idimmer_lambda * inter_term)


        # rebalance
        # rebalance= (number_out > 0) or (t % 100 == 0) or (t == prms.dimmer_max_iter - 1)
        rebalance =  (t % 100 == 0) or (t == prms.dimmer_max_iter - 1)
        # rebalance = False
        if rebalance:
            # check if any partciles are not index-1
            # first_pos_index = total_d - np.sum(mu_t[:] > prms.eig_eps, axis=1)
            # forst_pos_value = index_func_lambda_map(index, first_pos_index, mu_t)
            index1_cond = jnp.logical_and(mu_t[:, 1+prms.eig_gap] > prms.eig_eps, mu_t[:, 0] < -prms.eig_eps)

            index1_pos = dimmer_prtcl_index[np.array(index1_cond)]
            not_index1_prtcls = np.setdiff1d(dimmer_prtcl_index, index1_pos)

            number_out = len(not_index1_prtcls)
            number_in = len(index1_pos)
            if number_in==0:
                # 0:success,1:hit max iter,2:failed because out of index-1
                exit_code = 2
                return exit_code,t,None,U_t
            elif number_out>0:
                key, subkey = random.split(key)
                new_index = np.array(random.randint(subkey, (number_out,), 0, number_in))
                # randomly select another state
                U_t = jax.ops.index_update(U_t, not_index1_prtcls, U_t[index1_pos[new_index]])
                #     rebalance adj mat
                N_prtcls, lap_mat,num_connected_comp = getLaplacian(prms, U_t, dist_adj_mat_map, scale_adj_mat_map)
        # check if *all* particles converged
        dv_norm = jnp.linalg.norm(grad_t.reshape(N_prtcls, total_d), axis=1)
        dv_norm_max = jnp.max(dv_norm)
        # if t%10000==0:
        #     print("\t\t",t,dv_norm_max)
        if dv_norm_max<prms.dimmer_eps:
            succeed, U_index1 = dimmer_utils.find_unique_index1_points(U_t, Vm, dV, ddV, index_func_lambda_map, prms.eig_eps,prms.eig_gap, prms.dimmer_eps)
            if succeed:
                exit_code = 0
                return exit_code, t, U_index1, U_t
            else:
                exit_code = 2
                return exit_code, t, None, U_t

    # if here then reached max iters but check if any points are good anyway
    succeed, U_index1 = dimmer_utils.find_unique_index1_points(U_t, Vm, dV, ddV, index_func_lambda_map, prms.eig_eps, prms.eig_gap,prms.dimmer_eps)
    if succeed:
        exit_code = 0
        return exit_code, t, U_index1, U_t
    # max-iter
    else:
        exit_code = 1
        return exit_code, t, None, U_t

    # exit_code = 1
    return exit_code, t, None, U_t






def ideal_dimmer(x,V,dV,hess,beta=2.0,dt=0.001,eig_eps=10e-12,eps=10e-4,max_iter=100,debug=False,eig_gap=2):

    t=0
    N_a = x.shape[0]
    N_d = x.shape[1]
    total_d = N_a * N_d
    converge = False
    fail=False
    while converge is False:
    # calculate smallest eigenvector

        hess_t=np.array(hess(x))
        hess_t = hess_t.reshape(total_d,total_d)
        mu_t, v1_t = dimmer_eig_calc(hess_t,total_d,k=2+1+eig_gap)
        grad_t = dV(x)
        grad_t = grad_t.reshape(total_d, 1)
        vdf = v1_t.T @ grad_t
        dX = 2 * v1_t * vdf - grad_t
        x = x + beta * dt * (dX).reshape(N_a, N_d)
        norm_gd = jnp.linalg.norm(grad_t)
        if mu_t[0]>-  eig_eps:
            fail = True
        t = t + 1
        converge = (norm_gd < eps) or (t > max_iter) or fail

        if debug and (t%1==0 or t==1):
            print(t,V(x),norm_gd,mu_t[0],mu_t[1+eig_gap] )


    # V_d = V(x)
    grad_t = dV(x)
    norm_gd = jnp.linalg.norm(grad_t)
    hess_t = np.array(hess(x))
    hess_t = hess_t.reshape(total_d, total_d)

    mu_t, v1_t = dimmer_eig_calc(hess_t, total_d, k=2+eig_gap)
    # 0:success,1:hit max iter,2:failed because out of index-1
    exit_code = 0
    U_index1=None
    if mu_t[0] < - eig_eps and mu_t[1+eig_gap] > eig_eps and norm_gd<eps:
        exit_code=0
        U_index1=x
    if t > max_iter:
        exit_code = 1
    if fail:
        exit_code = 2

    # print(t,exit_code,V(x))
    return exit_code,t,U_index1,x
