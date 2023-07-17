from jax.config import config
config.update("jax_enable_x64", True)


# import math
import numpy as np
# import jax 
import jax.numpy as jnp
from jax import jit,hessian,grad
# import matplotlib.pyplot as plt
#%matplotlib qt
#%matplotlib inline
import pandas as pd
from jax_md import energy,space
# import jax.ops as jo
import scipy
import pickle
# import sdd
# import timeit


N=1

result=pd.read_csv("vd_init/X"+str(N)+".csv",header=None) 
X=np.array(result.to_numpy())

result=pd.read_csv("vd_init/Xf"+str(N)+".csv",header=None) 
Xfi=np.array(result.to_numpy())-1
Xfi=Xfi.reshape(Xfi.shape[0],)
Xf=X[Xfi]

N_a=X.shape[0]
X=jnp.array(X)


displacement_fn, shift_fn=space.free()

V_full = energy.morse_pair(displacement_fn,sigma=1.0, epsilon=1.0, alpha=4.0, r_onset=2000, r_cutoff=2500)
# normdV=np.linalg.norm(dV(X))
def Vf(Xfree,V,free_index,Xfixed):
  X_new=Xfixed.at[free_index].set(Xfree)
  f=V(X_new)
  return f
Xfix=X
V=  lambda _X: Vf( _X,V_full,Xfi,Xfix)

X_t=Xf


V=jit(V)
dV=jit(grad(V))
ddV=jit(hessian(V))

# # minimize
max_iter=5000
dt=0.001
total_d= Xf.shape[0]* Xf.shape[1]
eps=1e-6
if max_iter>0:
  for i in range(max_iter):
          grad_t=dV(X_t)
          X_t= X_t-grad_t*dt
          fEval=V(X_t)
          
          if i%1000==0 or i==max_iter-1:
              grad_norm=np.linalg.norm(grad_t)
              hess_eval=ddV(X_t).reshape(total_d,total_d)
              hess_eval=np.array(hess_eval)
              mu,v=scipy.linalg.eigh(hess_eval)
              count_neg=np.sum(mu < -eps)
              count_pos=np.sum(mu > eps)
              count_zero=total_d-count_neg-count_pos
              print('{:>6d} {:>12.6f}{:>12.6f}{:>6d}{:>6d}{:>6d}'.format(i+1,fEval,grad_norm,count_neg,count_zero,count_pos))

file = open( "vd_init/vd"+str(N)+".pickle", "wb" )
X=X.at[Xfi].set(X_t)
pickle.dump( [X,Xfi], file )
file.close()

# V=jit(V)
# dV=jit(grad(V))
# ddV=jit(hessian(V))

# Va=jit((V_full))
# dVa=jit(grad(V_full))
# ddVa=jit(hessian(V_full))


# gd_all=grad(V_full)(X)
# print(V(Xf)-V_full(X))
# print(np.linalg.norm(dV(Xf)-gd_all[Xfi]))

# total_d= Xf.shape[0]* Xf.shape[1]
# X_t=Xf

# # hess_eval=ddV(X_t).reshape(total_d,total_d)
# # gh=jit(jnp.linalg.eigh)
# # mu,v=scipy.linalg.eigh(hess_eval)
# # mu,v=jnp.linalg.eigh(hess_eval) 
# # mu,v=gh(hess_eval) 

# t1=ddVa(X)
# print("Done comp 1")
# t2=ddV(X_t)
# print("Done comp 2")


# starttime = timeit.default_timer()
# for ii in range(0,100):
#   t1=ddVa(X)
# endtime=timeit.default_timer()
# print("Full-",endtime-starttime)


# starttime = timeit.default_timer()
# for ii in range(0,100):
#   t2=ddV(X_t)
# endtime=timeit.default_timer()
# print("Partial-",endtime-starttime)

# print(np.linalg.norm(t2-t1))


# %timeit ddVa(X)
# %timeit ddV(X_t)
# %timeit temp=np.array(hess_eval)
# %timeit mu,v=scipy.linalg.eigh(hess_eval)
# %timeit mu,v=gh(hess_eval)        



# 
# # t,x,V_d,norm_gd,mu_t,v_t=sdd.ideal_dimmer(X_t,V,dV,ddV,beta=2.0,dt=0.001,eps=10e-4,max_iter=1000,eig_calc=8)

# print(V_d)




# plt.figure(figsize=(60,25))
# plt.rcParams.update({'font.size': 30})
# plt.clf()
# ax_b = plt.subplot(1,2, 1,aspect='equal')
# ax_b.set_ylim(-10, 10)
# ax_b.set_xlim(-10, 10)


# ax_a = plt.subplot(1,2, 2,aspect='equal')
# ax_a.set_ylim(-10, 10)
# ax_a.set_xlim(-10, 10)


# ax_b.scatter( *zip(*X),c='k' )
# ax_b.scatter( *zip(*X[Xfi]),c='r' )
# ax_b.scatter(X[0,0],X[0,1],c='b' )

# ax_a.scatter( *zip(*x),c='r' )
# # ax_a.scatter( *zip(*X_t[Xfi]),c='r' )
# ax_a.scatter(x[0,0],x[0,1],c='b' )
# plt.savefig("vd.png")
# plt.show()

# # # print(V(X))





# # def triangular_lattice(point_distance ,expansion_level = 4, starting_point = (0, 0)):

# #     set_of_points_on_the_plane = set()

# #     for current_level in range(expansion_level+1):

# #         temporary_hexagon_coordinates = {}

# #         equilateral_triangle_side = current_level * point_distance
# #         equilateral_triangle__half_side = equilateral_triangle_side / 2
# #         equilateral_triangle_height = (math.sqrt(3) * equilateral_triangle_side) / 2
# #         if current_level != 0:
# #             point_distance_as_triangle_side_percentage = point_distance / equilateral_triangle_side

# #         temporary_hexagon_coordinates['right'] = (starting_point[0] + point_distance * current_level, starting_point[1]) #right
# #         temporary_hexagon_coordinates['left'] = (starting_point[0] - point_distance * current_level, starting_point[1]) #left
# #         temporary_hexagon_coordinates['top_right'] = (starting_point[0] + equilateral_triangle__half_side, starting_point[1] + equilateral_triangle_height) #  top_right
# #         temporary_hexagon_coordinates['top_left'] = (starting_point[0] - equilateral_triangle__half_side, starting_point[1] + equilateral_triangle_height) #  top_left
# #         temporary_hexagon_coordinates['bottom_right'] = (starting_point[0] + equilateral_triangle__half_side, starting_point[1] - equilateral_triangle_height) #  bottom_right
# #         temporary_hexagon_coordinates['bottom_left'] = (starting_point[0] - equilateral_triangle__half_side, starting_point[1] - equilateral_triangle_height) # bottom_left

# #         if current_level > 1:
# #             for intermediate_points in range(1, current_level):

# #                 set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['left'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['left'][0]) , temporary_hexagon_coordinates['left'][1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][1] - temporary_hexagon_coordinates['left'][1])  ))        #from left to top left
# #                 print(intermediate_points)
# #                 print((temporary_hexagon_coordinates['left'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['left'][0]) , temporary_hexagon_coordinates['left'][1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][1] - temporary_hexagon_coordinates['left'][1])  ))
# #                 set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['left'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['bottom_left'][0] - temporary_hexagon_coordinates['left'][0]) , temporary_hexagon_coordinates['left'][1] - intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['bottom_left'][1] - temporary_hexagon_coordinates['left'][1]) ))  # from left to bottom left

# #                 set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['top_right'][0] - intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['top_right'][0]) , temporary_hexagon_coordinates['top_right'][1] ))  #from top right to top left
# #                 set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['top_right'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['right'][0] - temporary_hexagon_coordinates['top_right'][0]) , temporary_hexagon_coordinates['top_left'][1] - intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['right'][1] - temporary_hexagon_coordinates['top_right'][1]) ))    # from top right to right

# #                 set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['bottom_right'][0] - intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['bottom_left'][0] - temporary_hexagon_coordinates['bottom_right'][0]) , temporary_hexagon_coordinates['bottom_right'][1] ))   #apo bottom right pros aristera
# #                 set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['bottom_right'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['right'][0] - temporary_hexagon_coordinates['bottom_right'][0]) , temporary_hexagon_coordinates['bottom_right'][1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['right'][1] - temporary_hexagon_coordinates['bottom_right'][1]) ))         # from bottom right to right

# #         # dictionary to set

# #         set_of_points_on_the_plane.update( temporary_hexagon_coordinates.values() )

# #     return list(set_of_points_on_the_plane)


# # triangular_lattice =0.3* jnp.array( triangular_lattice(3, expansion_level = 10) ) # returns a list with points on the Euclidean plane (2d space)

# # pyplot.scatter( *zip(*triangular_lattice) )
# # pyplot.axis(aspect='equal')
# # pyplot.show()