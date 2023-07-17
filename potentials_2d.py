import jax.numpy as jnp
import numpy as np
import sys
# import jax.ops as jo
from jax_md import space, smap, energy, minimize, quantity, simulate
from types import SimpleNamespace
import pickle

########## Example 0 #########################
# toy symmetric double well, very easy to solve
def symmetric_dw_vec(X):
    x = X[0, 0]
    y = X[0, 1]
    alpha=2
    f=(1-x**2)**2+alpha*y**2
    return f
def symmetric_dw(x,y):
    Z = jnp.array([[x, y]])
    return symmetric_dw_vec(Z)


def symmetric_dw_index0():
    x = [-1,1]
    y = [0,0]
    l = ['M1', 'M2']
    return x, y, l
# symmetric double well-saddle points
def symmetric_dw_index1():
    x = [0]
    y = [0]
    l = ['S1']
    return x, y, l

########## Example 1 #########################
# From Example 2 (Fig2 SIAM J. Num. Anal. Convergence and Cycling in Walker-Type Saddle Search
# degenerate case with no saddlpe points
def degenerate_vec(X):
    x = X[0, 0]
    y = X[0, 1]
    V= (x**2+y**2)**2+x**2-y**2-x+y
    V = V / 4000
    return V
def degenerate(x,y):
    Z = jnp.array([[x, y]])
    return degenerate_vec(Z)
def degenerate_index0():
    x = [0.19]
    y = [-0.86]
    l = ['M1']
    return x, y, l
# degenerate case as no saddles
def degenerate_index1():
    return [], [], []



# Modifed Example 2 (Fig2 SIAM J. Num. Anal. Convergence and Cycling in Walker-Type Saddle Search
# degenerate case with no saddlpe points near minimum 1 but some saddles far away
# also index-1 regions are connected
def example2_vec(X):
    x = X[0, 0]
    y = X[0, 1]
    V = (x ** 2 + y ** 2) ** 2 + x ** 2 - y ** 2 - x + y
    beta = 1.0
    V = V / 4000
    V = V - jnp.exp(-((x - 2.5) ** 2 + (y - 2.5) ** 2) / beta)
    return V

def example2(x,y):
    Z = jnp.array([[x, y]])
    return example2_vec(Z)
def example2_index0():
    x = [0.19,2.5]
    y = [-0.86,2.5]
    l = ['M1','M2']
    return x, y, l
# degenerate case has a saddle
def example2_index1():
    return [], [], []

# Modifed Example 3 (Fig2 SIAM J. Num. Anal. Convergence and Cycling in Walker-Type Saddle Search
# degenerate case with no saddlpe points near minimum 1 but some saddles far away
# also index-1 regions are disconnected
def example3_vec(X):
    x = X[0, 0]
    y = X[0, 1]
    V = (x ** 2 + y ** 2) ** 2 + x ** 2 - y ** 2 - x + y
    beta = 1
    V =V/4000 - jnp.exp(-((x - 5) ** 2 + (y - 5) ** 2) / beta)
    return V

def example3(x,y):
    Z = jnp.array([[x, y]])
    return example3_vec(Z)
def example3_index0():
    x = [0.19,5]
    y = [-0.86,5]
    l = ['M1','M2']
    return x, y, l
# degenerate case has a saddle
def example3_index1():
    return [], [], []



##### muller brown
def muller_brown_vec(X):
    x = X[0, 0]
    y = X[0, 1]
    A = jnp.array((-200, -100, -170, 15))/100
    b = jnp.array((0, 0, 11, 0.6))
    x0 = jnp.array((1, 0, -0.5, -1))
    a = jnp.array((-1, -1, -6.5, 0.7))
    c = jnp.array((-10, -10, -6.5, 0.7))
    y0 = jnp.array((0, 0.5, 1.5, 1))
    f = 0
    for i in range(0, 4):
        a_term = a[i] * (x - x0[i]) ** 2
        b_term = b[i] * (x - x0[i]) * (y - y0[i])
        c_term = c[i] * (y - y0[i]) ** 2
        f = f + A[i] * jnp.exp(a_term + b_term + c_term)
    return f

def muller_brown(x,y):
    Z = jnp.array([[x, y]])
    return muller_brown_vec(Z)

def muller_brown_index0():
    x = [-0.558, 0.623, -0.050]
    y = [1.442, 0.028, 0.467]
    l = ['SP1', 'SP2', 'SP3']
    return x, y, l
# muller b saddles
def muller_brown_index1():
    x = [0.212, -0.822]
    y = [0.2930, 0.624]
    l = ['SP4', 'SP5']
    return x, y, l


def setup_problem(prms):
    # symmetric double well
    if prms.prb_index == 1:
        prms.str_file = "../example_1/sdw_"
        V = symmetric_dw_vec
        V_plot = symmetric_dw
        prms.d = 2
        prms.N_a = 1
        prms.T_real = 1
        prms.N_p=np.int32(25000) #25000 2000 0.002
        prms.M=100
        prms.h_x = 0.0001 # 0.001 0.0001 0.002
        prms.h_y = 0.0001 # 0.001 0.005    0.005
        prms.ess_th = 0.99  # 0.99 0.95 ess threshold
        prms.epsilon = 0.07 # 0.07 0.05
        # dimmer
        prms.dimmer_dt= 0.01
        prms.dimmer_max_iter=5000
        prms.dimmer_eps=10e-5
        prms.best_particle = "weight"
        prms.run_dimmer = False
        prms.run_particle_dimmer=False
        x_s, y_s, l_s = symmetric_dw_index0()
        # start from one of the local minima (-1,-1)
        X0 = np.array([x_s[0], y_s[0]])
        limits= np.array([-1.5, 1.5,-1.5,1.5])
        prms.eig_gap=0
    # degenerate problem with no saddles
    elif prms.prb_index == 2:
        prms.str_file = "../example_2/dgn1_"
        V = degenerate_vec
        V_plot = degenerate
        prms.T_real = 100
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.dimmer_dt = 0.001
        prms.M = 1000
        prms.epsilon = 0.001
        prms.ess_th = 0.95
        prms.dimmer_dt = 0.1
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        prms.best_particle = "grad"
        prms.run_dimmer = False
        prms.run_particle_dimmer = True
        x_s, y_s, l_s = degenerate_index0()
        # start from one of the local minima (-1,-1)
        X0 = np.array([x_s[0], y_s[0]])
        limits= np.array([-1.5, 1.5,-1.5,1.5])
    # same as 2 but with a saddle close byy
    # now it works
    elif prms.prb_index == 3:
        prms.str_file = "../example_4/dgn3_"
        V = example2_vec
        V_plot = example2
        prms.T_real = 100
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.M = 1000
        prms.epsilon = 0.001
        prms.ess_th = 0.95
        prms.run_dimmer = False
        prms.run_particle_dimmer = True
        prms.dimmer_dt = 0.1
        prms.dimmer_eps = 10e-5
        prms.dimmer_max_iter = 50000

        prms.best_particle = "grad"
        x_s, y_s, l_s = degenerate_index0()
        # start from one of the local minima (-1,-1)
        X0 = np.array([x_s[0], y_s[0]])
        limits = np.array([-2, 5, -2, 5])
    # same as degenerate case but with saddle far away
    # with this level of noise it doesnt work
    elif prms.prb_index == 4:
        prms.str_file = "../example_4/dgn3_"
        V = example3_vec
        V_plot = example3
        prms.T_real = 100
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.dimmer_dt = 0.001
        prms.M = 1000
        prms.epsilon = 0.001
        prms.ess_th = 0.95
        prms.dimmer_dt = 0.1
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        x_s, y_s, l_s = example3_index0()
        prms.best_particle = "weight"
        prms.run_dimmer = False
        prms.run_particle_dimmer = True
        # start from one of the local minima (-1,-1)
        X0 = np.array([x_s[0], y_s[0]])
        limits = np.array([-2, 7, -2, 7])
    # same 4 but with higher noise
    # 0.1 (works)
    elif prms.prb_index == 5:
        prms.str_file = "../example_dgn/dgn3_"
        V = example3_vec
        V_plot = example3
        prms.T_real = 100
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.dimmer_dt = 0.001
        prms.M = 1000
        prms.epsilon = 0.1
        prms.ess_th = 0.95
        prms.dimmer_dt = 0.1
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        prms.best_particle = "weight"
        prms.run_dimmer = False
        prms.run_particle_dimmer = True
        x_s, y_s, l_s = example3_index0()
        # start from one of the local minima (-1,-1)
        X0 = np.array([x_s[0], y_s[0]])
        limits = np.array([-2, 7, -2, 7])

    elif prms.prb_index == 6:
        prms.str_file = "../muller_brown_01p/mb1_"
        V = muller_brown_vec
        V_plot = muller_brown
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.T_real = 100
        prms.dimmer_dt = 0.001
        prms.epsilon = 0.05
        prms.M = 1000

        prms.ess_th = 0.9  # ess threshold
        prms.dimmer_dt = 0.01
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        prms.best_particle = "weight"
        prms.run_dimmer = True
        prms.run_particle_dimmer = True
        prms.eig_gap = 0
        x_s, y_s, l_s = muller_brown_index0()
        # start from one of the local minima
        X0 = np.array([x_s[2], y_s[2]])
        limits = np.array([-2, 1, -0.75, 2.25])



    elif prms.prb_index == 7:
        prms.str_file = "../lj7_1/lj7_"
        prms.d=2
        prms.N_a=7
        prms.h_x =0.0001
        prms.h_y =0.0001
        prms.dimmer_dt=0.0001
        prms.eig_gap=2
        prms.epsilon = 0.1
        prms.ess_th = -0.99  # ess threshold
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter =20000
        prms.M = 100  # run dimmer every M iterations
        prms.N_p=2000
        prms.S=100
        prms.idimmer_lambda=0
        prms.best_particle = "weight"
        prms.run_dimmer = True
        prms.run_particle_dimmer = False
        params_dict_LJ = {
            "sigma_LJ": 1,  # sigma of LJ
            "epsilon_LJ": 1,  # epsilon of LJ
            "r_onset": 25,  # cut off LJ
            "r_cutoff": 25.1,
        }
        prms_lj = SimpleNamespace(**params_dict_LJ)
        displacement_fn, shift_fn = space.free()
        V = energy.lennard_jones_pair(displacement_fn,sigma=prms_lj.sigma_LJ,epsilon=prms_lj.epsilon_LJ, r_onset=prms_lj.r_onset,r_cutoff=prms_lj.r_cutoff)
        V_plot=V
        file = open('lj72_init/lj72_C1.pickle', 'rb')

        X0 = pickle.load(file)
        # print(V(X0))

        file.close()
        x_s, y_s, l_s=None,None,None
        limits = None
    elif prms.prb_index == 8:
        prms.str_file = "../lj7_1p/lj7_"
        prms.d = 2
        prms.N_a = 7
        prms.h_x = 0.0001
        prms.h_y = 0.0001
        prms.dimmer_dt = 0.0001
        prms.eig_gap = 3
        prms.eig_eps=1e-8
        prms.epsilon = 0.1
        prms.ess_th = 0.99  # ess threshold
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 20000
        prms.M = 100  # run dimmer every M iterations
        prms.N_p = 2000
        prms.S = 100
        prms.idimmer_lambda = 0
        prms.best_particle = "weight"
        prms.run_dimmer = False
        prms.run_particle_dimmer = True
        params_dict_LJ = {
            "sigma_LJ": 1,  # sigma of LJ
            "epsilon_LJ": 1,  # epsilon of LJ
            "r_onset": 25,  # cut off LJ
            "r_cutoff": 25.1,
        }
        prms_lj = SimpleNamespace(**params_dict_LJ)
        displacement_fn, shift_fn = space.free()
        V = energy.lennard_jones_pair(displacement_fn, sigma=prms_lj.sigma_LJ, epsilon=prms_lj.epsilon_LJ, r_onset=prms_lj.r_onset, r_cutoff=prms_lj.r_cutoff)
        V_plot = V
        file = open('lj72_init/lj72_C1.pickle', 'rb')

        X0 = pickle.load(file)
        print(V(X0))

        file.close()
        x_s, y_s, l_s = None, None, None
        limits = None

    elif prms.prb_index == 9:
        prms.str_file = "../lj7_3p/lj7_"
        prms.d = 2
        prms.N_a = 7
        prms.h_x = 0.0001
        prms.h_y = 0.0001
        prms.dimmer_dt = 0.0001
        prms.eig_gap = 3
        prms.eig_eps=1e-8
        prms.epsilon = 0.1
        prms.ess_th = 0.99  # ess threshold
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 20000
        prms.M = 100  # run dimmer every M iterations
        prms.N_p = 2000
        prms.S = 100
        prms.idimmer_lambda = 0
        prms.best_particle = "weight"
        prms.run_dimmer = False
        prms.run_particle_dimmer = True
        params_dict_LJ = {
            "sigma_LJ": 1,  # sigma of LJ
            "epsilon_LJ": 1,  # epsilon of LJ
            "r_onset": 25,  # cut off LJ
            "r_cutoff": 25.1,
        }
        prms_lj = SimpleNamespace(**params_dict_LJ)
        displacement_fn, shift_fn = space.free()
        V = energy.lennard_jones_pair(displacement_fn, sigma=prms_lj.sigma_LJ, epsilon=prms_lj.epsilon_LJ, r_onset=prms_lj.r_onset, r_cutoff=prms_lj.r_cutoff)
        V_plot = V
        file = open('lj72_init/lj72_C3.pickle', 'rb')

        X0 = pickle.load(file)
        print(V(X0))

        file.close()
        x_s, y_s, l_s = None, None, None
        limits = None

    elif prms.prb_index == 10:
        prms.str_file = "../lj7_0p/lj7_"
        prms.d = 2
        prms.N_a = 7
        prms.h_x = 0.0001
        prms.h_y = 0.0001
        prms.dimmer_dt = 0.0001
        prms.eig_gap = 3
        prms.eig_eps = 1e-8
        prms.epsilon = 0.1
        prms.ess_th = 0.99  # ess threshold
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 20000
        prms.M = 100  # run dimmer every M iterations
        prms.N_p = 2000
        prms.S = 100
        prms.idimmer_lambda = 0
        prms.best_particle = "weight"
        prms.run_dimmer = False
        prms.run_particle_dimmer = True
        params_dict_LJ = {
            "sigma_LJ": 1,  # sigma of LJ
            "epsilon_LJ": 1,  # epsilon of LJ
            "r_onset": 25,  # cut off LJ
            "r_cutoff": 25.1,
        }
        prms_lj = SimpleNamespace(**params_dict_LJ)
        displacement_fn, shift_fn = space.free()
        V = energy.lennard_jones_pair(displacement_fn, sigma=prms_lj.sigma_LJ, epsilon=prms_lj.epsilon_LJ, r_onset=prms_lj.r_onset, r_cutoff=prms_lj.r_cutoff)
        V_plot = V
        file = open('lj72_init/lj72_C0.pickle', 'rb')

        X0 = pickle.load(file)
        print(V(X0))

        file.close()
        x_s, y_s, l_s = None, None, None
        limits = None

        # load initial condition
    elif prms.prb_index == 11:
        init_sys = 1
        file = open("vd_init/vd" + str(init_sys) + ".pickle", "rb")
        prms.str_file = "../vd"+str(init_sys)+"_0/vd_"+str(init_sys)+"_"  # file to save figures
        prms.d = 2
        data = pickle.load(file)
        X0_full = jnp.array(data[0])
        Xfi = data[1]
        Xfixed_idx = np.setdiff1d(range(0, X0_full.shape[0]), Xfi)
        X0 = X0_full[Xfi]
        prms.N_a = Xfi.size
        file.close()
        print("vd_init/vd" + str(init_sys) + ".pickle",prms.N_a*2,X0_full.shape[0])

        prms.h_x = 0.0001
        prms.h_y = 0.0001
        prms.dimmer_dt = 0.0001
        prms.eig_gap = 3
        prms.eig_eps = 1e-8
        prms.epsilon = 0.1
        prms.ess_th = 0.99  # ess threshold
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 100000
        prms.M = 100  # run dimmer every M iterations
        prms.N_p = 2000
        prms.S = 100
        prms.idimmer_lambda = 0
        prms.best_particle = "weight"
        prms.run_dimmer = True
        prms.run_particle_dimmer = False


        displacement_fn, shift_fn = space.free()
        V_full = energy.morse_pair(displacement_fn, sigma=1.0, epsilon=1.0, alpha=4.0, r_onset=2000, r_cutoff=2500)

        def Vf(Xfree, V, free_index, Xfixed):
            X_new = Xfixed.at[free_index].set(Xfree)
            f = V(X_new)
            return f

        V = lambda _X: Vf(_X, V_full, Xfi, X0_full)
        V_plot = V
        x_s, y_s, l_s = None, None, None
        limits = None

    elif prms.prb_index == 12:
        prms.str_file = "../muller_brown_01/mb1_"
        V = muller_brown_vec
        V_plot = muller_brown
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.T_real = 100
        prms.dimmer_dt = 0.001
        prms.epsilon = 0.05
        prms.M = 1000  # run dimmer every M iterations
        prms.ess_th = 0.9  # ess threshold
        prms.dimmer_dt = 0.01
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        prms.best_particle = "weight"
        prms.run_dimmer = True
        prms.run_particle_dimmer = False
        x_s, y_s, l_s = muller_brown_index0()
        # start from one of the local minima
        X0 = np.array([x_s[2], y_s[2]])
        prms.eig_gap = 0
        limits = np.array([-2, 1, -0.75, 2.25])



    elif prms.prb_index == 12:
        prms.str_file = "../muller_brown_01/mb1_"
        V = muller_brown_vec
        V_plot = muller_brown
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.T_real = 100
        prms.dimmer_dt = 0.001
        prms.epsilon = 0.05
        prms.M = 1000  # run dimmer every M iterations
        prms.ess_th = 0.9  # ess threshold
        prms.dimmer_dt = 0.01
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        prms.best_particle = "weight"
        prms.run_dimmer = True
        prms.run_particle_dimmer = False
        x_s, y_s, l_s = muller_brown_index0()
        # start from one of the local minima
        X0 = np.array([x_s[2], y_s[2]])
        prms.eig_gap = 0
        limits = np.array([-2, 1, -0.75, 2.25])



    elif prms.prb_index == 13:
        prms.str_file = "../muller_brown_02p/mb1_"
        V = muller_brown_vec
        V_plot = muller_brown
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.T_real = 100
        prms.dimmer_dt = 0.001
        prms.epsilon = 0.1
        prms.M = 1000

        prms.ess_th = 0.9  # ess threshold
        prms.dimmer_dt = 0.01
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        prms.best_particle = "weight"
        prms.run_dimmer = True
        prms.run_particle_dimmer = True
        prms.eig_gap = 0
        x_s, y_s, l_s = muller_brown_index0()
        # start from one of the local minima
        X0 = np.array([x_s[0], y_s[0]])
        limits = np.array([-2, 1, -0.75, 2.25])

    elif prms.prb_index == 14:
        prms.str_file = "../muller_brown_02/mb1_"
        V = muller_brown_vec
        V_plot = muller_brown
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.T_real = 100
        prms.dimmer_dt = 0.001
        prms.epsilon = 0.1
        prms.M = 1000

        prms.ess_th = 0.9  # ess threshold
        prms.dimmer_dt = 0.01
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        prms.best_particle = "weight"
        prms.run_dimmer = True
        prms.run_particle_dimmer = False
        prms.eig_gap = 0
        x_s, y_s, l_s = muller_brown_index0()
        # start from one of the local minima
        X0 = np.array([x_s[0], y_s[0]])
        limits = np.array([-2, 1, -0.75, 2.25])


    elif prms.prb_index == 15:
        prms.str_file = "../lj7_1/lj7_"
        prms.d = 2
        prms.N_a = 7
        prms.h_x = 0.0001
        prms.h_y = 0.0001
        prms.dimmer_dt = 0.0001
        prms.eig_gap = 3
        prms.eig_eps = 1e-8
        prms.epsilon = 0.1
        prms.ess_th = 0.99  # ess threshold
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 20000
        prms.M = 100  # run dimmer every M iterations
        prms.N_p = 2000
        prms.S = 100
        prms.idimmer_lambda = 0
        prms.best_particle = "weight"
        prms.run_dimmer = True
        prms.run_particle_dimmer = False
        params_dict_LJ = {
            "sigma_LJ": 1,  # sigma of LJ
            "epsilon_LJ": 1,  # epsilon of LJ
            "r_onset": 25,  # cut off LJ
            "r_cutoff": 25.1,
        }
        prms_lj = SimpleNamespace(**params_dict_LJ)
        displacement_fn, shift_fn = space.free()
        V = energy.lennard_jones_pair(displacement_fn, sigma=prms_lj.sigma_LJ, epsilon=prms_lj.epsilon_LJ,
                                      r_onset=prms_lj.r_onset, r_cutoff=prms_lj.r_cutoff)
        V_plot = V
        file = open('lj72_init/lj72_C1.pickle', 'rb')

        X0 = pickle.load(file)
        print(V(X0))

        file.close()
        x_s, y_s, l_s = None, None, None
        limits = None

        # same 4 but with higher noise
        # 0.1 (works)
    elif prms.prb_index == 16:
        prms.str_file = "../example_challenge_case/dgn3_"
        V = example3_vec
        V_plot = example3
        prms.T_real = 100000
        prms.d = 2
        prms.N_a = 1
        prms.h_x = 0.001
        prms.h_y = 0.001
        prms.dimmer_dt = 0.001
        prms.M = 1000
        prms.epsilon = 0.1
        prms.ess_th = 0.95
        prms.dimmer_dt = 0.1
        prms.dimmer_eps = 10e-6
        prms.dimmer_max_iter = 50000
        prms.best_particle = "weight"
        prms.run_dimmer = True
        # prms.d_r=0.01
        prms.eig_gap=0
        prms.run_particle_dimmer = False
        x_s, y_s, l_s = example3_index0()
        # start from one of the local minima (-1,-1)
        X0 = np.array([x_s[0], y_s[0]])
        limits = np.array([-2, 7, -2, 7])


    else:
        sys.exit("Check problem index prb_index=")
    return prms,V,V_plot,x_s, y_s, l_s,X0,limits