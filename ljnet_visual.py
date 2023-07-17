import sys

from jax.config import config
config.update("jax_enable_x64", True)
import matplotlib
import numpy as np
from scipy import stats
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import importlib
import matplotlib.cm as cm
import visualise_lj
from jax import random
import jax.numpy as jnp
from datetime import datetime
from scipy.spatial import ConvexHull
import matplotlib.ticker as ticker
import dimmer_utils
import func_utils
import potentials_2d
import dimmer
import opt_algorithms
importlib.reload(potentials_2d)
importlib.reload(func_utils)
importlib.reload(dimmer_utils)
importlib.reload(dimmer)
importlib.reload(visualise_lj)


# file_name="../lj7_1p/lj7_prms.pickle"
file_name="../lj7_0/lj7_prms.pickle"
file = open(file_name, 'rb')

name="T4_4.svg"
fs = 4
idx = 4

plot_initial=True
plot_ts=True


prms=pickle.load(file)
file.close()
prms,V,V_plot,x_s, y_s, l_s,X0,limits=potentials_2d.setup_problem(prms)
T=np.int32(prms.T_real/prms.h_x)
Nd=prms.N_a*prms.d


total_i1=11
index1=np.zeros(total_i1)
# TS points from C1
index1[0]=-11.03733448
index1[1]=-10.93522342
index1[2] =-10.91978916
index1[3]= -10.89848533
index1[4]= -10.79874588
index1[5]=-11.0402529
index1[6]=-10.80730445
index1[7]=-10.84132249
index1[8]=-10.88257484
index1[9]=-10.77139919
index1[10]=-10.79803943
index1= jnp.sort(index1)

print("all index1=",index1)


prms, V, V_plot, x_s, y_s, l_s, X0, limits = potentials_2d.setup_problem(prms)
plt.rcParams.update({'font.size': 3})
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
plt.figure(figsize=(1200, 1200))
fig, axs = plt.subplots(1, 1)
axs=[axs]
axs[0].set(xlim=(-3, 3), ylim=(-3, 3))
axs[0].set_aspect('equal', 'box')
fig.tight_layout()


file_x = open(prms.str_file + "x_stats_" + str(fs) + ".pickle", "rb")
file_d = open(prms.str_file + "d_stats_" + str(fs) + ".pickle", "rb")
[out_s, t, X_t, Y_t, wgts, run_dimmer, S_card] = pickle.load(file_x)
U_0, t, exit_code, dimmer_iters, x_index1, x_final = pickle.load(file_d)

V_m, dV_s, ddV_s, dV, hvp_vec, ddV, index_func_lambda, index_func_lambda_map, gd_map = func_utils.get_func_evals(V, prms)
succeed, U_index1 = dimmer_utils.find_unique_index1_points(x_final, V_m, dV, ddV, index_func_lambda_map, prms.eig_eps, prms.eig_gap, prms.dimmer_eps)

f_value = (V(U_index1[idx]))
print(fs, V_m(U_index1))
print(idx, f_value)
total_d = prms.N_a * prms.d
# intial configuarion
dv_norm_0, hess_eval, mu_0, v_0 = visualise_lj.get_stats(X0, dV_s, ddV_s, total_d)
if plot_initial:
    axs[0] = visualise_lj.plot_2dLJ(X0, prms, axs[0], cols=None, t=0, fEval=V(X0), grad_norm=dv_norm_0, first_egv=mu_0[0], first_pos=mu_0[1 + prms.eig_gap])
    plt.text(-2,2.8,'V='+str(V(X0)),fontsize=15)
    plt.axis('off')



if plot_ts:
# saddle point
    x_index1 = U_index1[idx]
    dv_norm_x_index1, hess_eval, mu_x_index1, v_x_index1 = visualise_lj.get_stats(x_index1, dV_s, ddV_s, total_d)
    axs[0] = visualise_lj.plot_2dLJ(x_index1, prms, axs[0], cols=None, t=0, fEval=V(x_index1), grad_norm=dv_norm_x_index1, first_egv=mu_x_index1[0], first_pos=mu_x_index1[1 + prms.eig_gap])
    plt.text(-2,2.8,'V='+str(V(x_index1)),fontsize=15)
    plt.axis('off')

plt.savefig(name, dpi=600,format='svg')
# import networkx as nx
# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('../lj7_visualization/ljmatrix.csv',header=None)
# Graphtype = nx.Graph()
# G = nx.from_pandas_adjacency(df)
# mapping = {0:"C0",1:"C1",2:"C2",3:"C3",4:"T0",5:"T1",6:"T1_1",7:"T1_2",8:"T1_3",9:"T1_4",10:"T2",11:"T3",12:"T4",13:"T4_1"}
# G = nx.relabel_nodes(G, mapping)
# options = {
#     "font_size": 12,
#     "node_size": 1000,
#     "node_color": "white",
#     "edgecolors": "black",
#     "linewidths": 1,
#     "width": 1,
# }
# plt.clf()
# pos = nx.kamada_kawai_layout(G)
# # pos = pos = nx.spring_layout(G)
# # pos = nx.shell_layout(G)
# nx.draw_networkx(G,pos,**options)
# plt.savefig("net.png", dpi=300)
# plt.draw()