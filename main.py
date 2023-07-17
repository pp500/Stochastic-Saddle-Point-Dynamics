from types import SimpleNamespace
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
# import jax.scipy.stats.norm as jaxnm
import jax.ops as jo
import numpy as np
from jax import grad, hessian, vmap, jit, jvp, random  # ,value_and_grad
from functools import partial
import time
from jax_md import energy, space
import sys
import importlib
import pickle

import dimmer
import potentials_2d
import resampling
import func_utils
import dimmer_utils
import opt_algorithms

importlib.reload(dimmer)
importlib.reload(potentials_2d)
importlib.reload(resampling)
importlib.reload(func_utils)
importlib.reload(dimmer_utils)

#  parameters problem/algorithm
alg_params = {
    # problem parameters
    "prb_index": 1 ,  # problem index
    "str_file": "",  # file to save figures
    "d": 2,  # dimensions set below depending on problem
    "N_a": 1,  # number of atoms, dimesnions are dxN_a, set N_a for "euclidian problems"
    # parameters for saddle point search
    "seed": 0,  # seed
    "epsilon": 0.0,  # noise strength \sqrt{2\epsilon}
    "h_x": 0.01,  # stepszie for forward euler
    "h_y": 0.01,  # stepszie for euler step in y equation
    "T_real": 10,  # real time to run the system for
    "ess_th": 0.95,  # ess threshold
    "N_p": np.int32(2000),  # number of particles
    "M": 1000,  # run dimmer every M iterations
    # parameters for dimmer
    "eig_eps": 10e-8,  # if m>eig_eps then considered positive
    "S": 2000,  # how many particles to go into dimmer or particle dimmer
    "best_particle":"grad", # metric to select best particle, in the paper we only use weight (alternative is to use norm of gradient)
    "dimmer_dt": 0.001,
    "dimmer_eps": 10e-5,  # accuracy to solve dimmer
    # parameters for particle dimmer
    "run_dimmer": True,
    "run_particle_dimmer": True,
    "idimmer_lambda": 1,  # interaction strength
    "d_r":0.9,# for node i if node j is more than d_r then no interaction
    # parapeters for checking if particles are in basin
    "run_basin_check": False,# check if a point belongs to a basin of attraction
    "m_basin": 1000, # check m particles are in basin
    "basin_gd_tol": 0.001,# tolerance for checking if a point is in a basin
    "basin_f_tol":0.001, # kill all particles that are diffrent tolerance
    "gd_dt": 0.01, # step-size of gradient descent
    "gd_max_iter": 1000, # maximum iterations of gradient descent
    "save_file":True,# save the iterates in a pickle file

}
prms = SimpleNamespace(**alg_params)
displacement_fn, shift_fn = space.free()

prms, V, V_plot, x_s, y_s, l_s, X0, limits = potentials_2d.setup_problem(prms)
key = random.PRNGKey(prms.seed)
np.random.seed(prms.seed)

x_stats = open(str(prms.str_file) + "x_iter_stats.tsv", 'w+')
d_stats = open(str(prms.str_file) + "d_iter_stats.tsv", 'w+')

out_s = '{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'.format("Time", "ESS", "W_min", "W_max", "Gd Norm", "eig-1", "eig-2", "#Changes", "#TotalDimmer","#SuccDimmer ", "DimmerCode", "Out","CPU-Time")
print(out_s)
print(out_s, file=x_stats)
V_m,dV_s, ddV_s, dV, hvp_vec, ddV, index_func_lambda, index_func_lambda_map,gd_vmap = func_utils.get_func_evals(V,prms)
# setup
dist_adj_mat_map, scale_adj_mat_map, dot_product_dimmermap, mult_dimmermap = dimmer_utils.setup_particle_dimmer()

if prms.save_file:
    file = open(prms.str_file + "prms.pickle", "wb")
    pickle.dump(prms, file)
    file.close()

prtcl_index = np.arange(0, prms.N_p)
# initial condition for X
X_t = jnp.ones((prms.N_p, prms.N_a, prms.d), dtype=jnp.float64) * X0  # stores state
# key, subkey = random.split(key)
# X_t = X0+0.5*random.normal(key, ((prms.N_p, prms.N_a, prms.d)), dtype=np.float64)

# initial value
VInit=V(X_t[0])
key, subkey = random.split(key)
# initial condition for Y
Y_t =random.uniform(key, ((prms.N_p, prms.N_a, prms.d)), minval=0.0, maxval=1.0, dtype=np.float64)
noise_coeff = jnp.sqrt(2.0 * prms.epsilon * prms.h_x)
T = np.int32(prms.T_real / prms.h_x)
Nd = prms.N_a * prms.d
fl_s = 0
dim_search = 0
dim_search_succs = 0
# T=1000
total_changes = 0
number_out = 0
print(Nd,prms.M)


# T=0
for t in range(0, T):

    exit_code=-1
    t0 = time.perf_counter()
    # evaluate gradient: \nabla V(X_t)
    grad_t = dV(X_t)
    # evaluate hessian vector product: \nabla V^2(X_t)Y_t
    hvp_t = hvp_vec(X_t, Y_t)
    # convert Y_t into weights
    W_t = vmap(jnp.linalg.norm)(Y_t).reshape(prms.N_p, 1)

    # store log of weights and compute ESS
    wgts = resampling.Weights(lw=np.log((W_t)))
    run_dimmer = (t % prms.M == 0) and (prms.run_dimmer or prms.run_particle_dimmer)
    collect_stats = (t == 0 or t % prms.M == 0 or t == T - 1)
    # particle with highest weight
    # collect stats - only for output
    maxw_ind, minw_ind, max_weight, min_weight = dimmer_utils.get_weight_stats(wgts.W)
    X_best = X_t[maxw_ind]


    gd_norm, eigs_best = dimmer_utils.single_particle_stats(X_best, dV_s, ddV_s, Nd)
    ess = wgts.ESS / prms.N_p

    # if it is time to check basin
    if prms.run_basin_check and t%prms.m_basin==0:
        number_out, prtcl_out = opt_algorithms.check_basin_fVal(gd_vmap, X_t, prms, VInit, prtcl_index)
        if number_out > 0:
            # all particles inside
            prtcl_in = np.setdiff1d(prtcl_index, prtcl_out)
            number_in = len(prtcl_in)
            if number_in == 0:
                sys.exit(" Error: All particles are out!")
                # randomly sample from the particles that are in
            else:
                key, subkey = random.split(key)
                new_index = np.array(random.randint(subkey, (number_out,), 0, number_in))
                # randomly select another state
                X_t = jo.index_update(X_t, prtcl_out, X_t[prtcl_in[new_index]])
                Y_t = jo.index_update(Y_t, prtcl_out, Y_t[prtcl_in[new_index]])

    # if it is time to run dimmer
    S_card = 0
    if run_dimmer:

        S_card, U, U_0 = dimmer_utils.find_best_particles(X_t, wgts.W, dV, ddV, prms.eig_eps,prms.eig_gap, prms.S, prms.N_p, Nd, index_func_lambda_map,prms.best_particle)

        # run particle dimmer with the the single particle
        run_single_particle_dimmer=(S_card > 0 and prms.run_particle_dimmer is False) or S_card==1
        if run_single_particle_dimmer:
            # collect some stats
            gd_0 = dV_s(U_0)
            hess_0 = ddV_s(U_0).reshape(Nd, Nd)
            eigval_0, eigvec_0 = jnp.linalg.eigh(hess_0)
            gd_norm, eigs_best = jnp.linalg.norm(gd_0), eigval_0
            dim_search += 1
            # print("dimmer-init:",gd_0,eigval_0[0],eigval_0[1])
            exit_code, dimmer_iters, x_index1, x_final = \
                dimmer.ideal_dimmer(U_0, V, dV_s, ddV_s, beta=2.0, dt=prms.dimmer_dt, eig_eps=prms.eig_eps, eps=prms.dimmer_eps, max_iter=prms.dimmer_max_iter,eig_gap=prms.eig_gap)
            if prms.save_file:
                file = open(prms.str_file + "d_stats_" + str(fl_s) + ".pickle", "wb")
            # print(x_final)
                pickle.dump([U_0,t, exit_code,dimmer_iters,x_index1,x_final], file)
            # debuging dimer
            V_final = V(x_final)
            dV_final = jnp.linalg.norm(dV_s(x_final))
            ddV_final = np.array(ddV_s(x_final).reshape(prms.N_a * prms.d, prms.N_a * prms.d))
            mu_t, v1_t = dimmer.dimmer_eig_calc(ddV_final, prms.N_a * prms.d, k=2 + prms.eig_gap)
            out_s = '{:>12.6f}{:>12d}{:>12d}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}'\
                .format((t) * prms.h_x, exit_code, dimmer_iters, V(x_final), dV_final, mu_t[0], mu_t[1 + prms.eig_gap])
            print(out_s, file=d_stats)
            d_stats.flush()

        # otherwise run particle dimmer
        elif S_card > 0 and prms.run_particle_dimmer is True:
            dim_search += 1
            exit_code,dimmer_iters,x_index1,x_final = dimmer.particle_ideal_dimmer(prms, key, U, V_m, dV, ddV, index_func_lambda_map, dist_adj_mat_map, scale_adj_mat_map, dot_product_dimmermap, mult_dimmermap, debug=False)
            if prms.save_file:
                file = open(prms.str_file + "d_stats_" + str(fl_s) + ".pickle", "wb")
                pickle.dump([U,t, exit_code,dimmer_iters,x_index1,x_final], file)

            # , V(x_final))
        if exit_code == 0:
            dim_search_succs += 1

        # print("Dimmer="+str(dim_search_succs)+"/"+str(dim_search),x_d)
    if collect_stats:
        t1 = time.perf_counter()
        out_s = '{:>12.3f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12d}{:>12d}{:>12d}{:>12d}{:>12d}{:>12.6f}' \
            .format((t) * prms.h_x, ess, min_weight, max_weight, gd_norm, eigs_best[0], eigs_best[1+prms.eig_gap], total_changes, dim_search,dim_search_succs, exit_code, number_out,t1 - t0)
        out_s2 = 't={:>.2e},ESS={:>.2e},min_w={:>.2e},max_w={:>.2e},gd_norm={:>.2e},l1={:>.2e},l2={:>.2e},J={:>12d},out={:>12d},dimr={:>12d}/{:>12d},{:>12d},cpu={:>.2e}' \
            .format((t) * prms.h_x, ess, min_weight, max_weight, gd_norm, eigs_best[0], eigs_best[1+prms.eig_gap], total_changes, dim_search_succs,number_out,dim_search ,exit_code, t1 - t0)

        total_changes = 0
        if prms.save_file:
            file = open(prms.str_file + "x_stats_" + str(fl_s) + ".pickle", "wb")
            pickle.dump([out_s2, t, X_t, Y_t, wgts, run_dimmer, S_card], file)
        # file.close()

        print(out_s, file=x_stats)
        x_stats.flush()
        fl_s += 1
        print(out_s)

    # check if resampling is needed
    num_changes = 0
    if ess < prms.ess_th:
        rs_index, fail = resampling.stratified_sample(wgts.W, prms.N_p)
        if fail is True:
            sys.exit("Error: Resampling failed")
        else:
            X_t = X_t[rs_index]
            Y_t = Y_t[rs_index]
            # rescale weights
            Y_t = (Y_t.reshape(prms.N_p, Nd) / W_t[rs_index]).reshape(prms.N_p, prms.N_a, prms.d)
            # #
            # keep track of the number of changes
            change = (rs_index != prtcl_index)
            num_changes = prtcl_index[change].shape[0]
            total_changes += num_changes
    # update the state
    key, subkey = random.split(key)
    Z = random.normal(subkey, ((prms.N_p, prms.N_a, prms.d)))


    X_t = shift_fn(X_t, -grad_t * prms.h_x + Z * noise_coeff)
    Y_t = Y_t - hvp_t * (prms.h_y)

