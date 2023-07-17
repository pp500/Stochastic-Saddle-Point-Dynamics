#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:06:43 2020

@author: p.parpas
"""
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
import numpy as np
import math
import jax.numpy as jnp
from jax import random
from jax import jit,hessian,grad
import jax.ops as jo
from jax_md import space, smap, energy, minimize, quantity, simulate
from types import SimpleNamespace 
import pickle
import scipy


def min_gd(X_t,dt,V,dV,ddV,eps,max_iter):
  converge=False
  k=0
  grad_t=dV(X_t)
  while converge is False:
    grad_t=dV(X_t)    
    grad_norm=np.linalg.norm(grad_t)
    if (grad_norm<eps or k>max_iter):
      converge=True
    else:
      X_t=X_t-dt*grad_t
      k=k+1
    
  fEval=V(X_t)
  return X_t,fEval,grad_norm,k

# parameters choosen to compute the global min
# of LJ-7 in 2d
# should produce configuration C0 in
# A  generalized  parallel  replica  dynamics Andrew Binder, Tony Lelièvre, Gideon Simpson 
# Journal of Computational Physics 284 (2015) 595–616
# saves the point computed into a pickle
# def compute_initial():
parameter_dict={
#file to save figures
"N_a":7, # number of atoms
"d": 2, # dimensions
"seed":0, #seed
"LJ_bnd": "Free", #boundary condition for LJ
"sigma_LJ":1, # sigma of LJ
"epsilon_LJ":1, # epsilon of LJ
"r_onset":25, #cut off LJ
"r_cutoff":25.1,
"max_iter":5000,
"dt":0.0001
}



prms = SimpleNamespace(**parameter_dict)  
displacement_fn, shift_fn=space.free()
total_d=prms.N_a*prms.d
V = energy.lennard_jones_pair(displacement_fn,sigma=prms.sigma_LJ,epsilon=prms.epsilon_LJ, r_onset=prms.r_onset,r_cutoff=prms.r_cutoff)
V=jit(V)
dV=jit(grad(V))
ddV=jit(hessian(V))


lc=math.pi/3.0


R2d7_0=np.array([[0,0],
                 [lc,0],
                 [lc/2,lc/2],
                 [-lc/2,lc/2],
                 [-lc,0],
                 [-lc/2,-lc/2],
                 [lc/2,-lc/2]
                 ])

# initial close to global min
# X_t=R2d7_0
# or load from file
file = open('lj72_init/lj_C3.pickle', 'rb')
X_t = pickle.load(file)
file.close()

X_t = X_t-X_t[0]
# theta=-math.pi/2
# for i in range(1,7):
#     X_t[i,0]=X_t[i,0]*jnp.cos(theta)-X_t[i,1]*jnp.sin(theta)
#     X_t[i, 1] = X_t[i,0] * jnp.sin(theta)+ X_t[i, 1] * jnp.cos(theta)

print(V(X_t))

fEval=V(X_t)
grad_t=dV(X_t)    
grad_norm=np.linalg.norm(grad_t)
total_d=prms.N_a*prms.d
hess_eval=hessian(V)(X_t).reshape(total_d,total_d)
hess_eval=np.array(hess_eval)
mu,v=np.linalg.eigh(hess_eval)
eps=1e-8
#print(mu)
count_neg=np.sum(mu < -eps)
count_pos=np.sum(mu > eps)
count_zero=total_d-count_neg-count_pos

i=0
print('{:>6s}{:>12s}{:>12s}{:>6s}{:>6s}{:>6s}'.format("t","V","|dV|","-ve","#0","#+ve"))
print('{:>6d}{:>12.6f}{:>12.6f}{:>6d}{:>6d}{:>6d}'.format(i,fEval,grad_norm,count_neg,count_zero,count_pos))
      
#print(fEval,grad_norm,count_neg,count_zero,count_pos)


for i in range(prms.max_iter):
        grad_t=dV(X_t)    
        X_t = X_t-grad_t*prms.dt
        fEval=V(X_t)
        
        if i%100==0 or i==prms.max_iter-1:
            grad_norm=np.linalg.norm(grad_t)
            hess_eval=ddV(X_t).reshape(total_d,total_d)
            # hess_eval=onp.array(hess_eval)
            mu,v=scipy.linalg.eigh(hess_eval)

            count_neg=np.sum(mu < -eps)
            count_pos=np.sum(mu > eps)
            count_zero=total_d-count_neg-count_pos
            print('{:>6d}{:>12.6f}{:>12.6f}{:>6d}{:>6d}{:>6d}'.format(i+1,fEval,grad_norm,count_neg,count_zero,count_pos))


fig = plt.figure(figsize = (8,8))
axs= fig.add_subplot(111, aspect='equal')
# ##fig, axs  = plt.subplots(1,1,aspect='equal') #
# #
colors = cm.gist_rainbow(np.linspace(0, 1, prms.N_a))
for p in range(0,prms.N_a):
    at = Circle((X_t[:,0][p], X_t[:,1][p]),math.pi/6,alpha=1,color=colors[p])
    axs.add_patch(at)
    tx = plt.text(X_t[:,0][p], X_t[:,1][p], str(p), fontsize=32,fontweight='bold', color='k')
axs.set_xlim(-2, 2)
axs.set_ylim(-2,2)
# title='V={:>12.6f},|dV|={:>12.6f}, #< {:>6d},#0={:>6d},#>0 {:>6d}'.format(fEval,grad_norm,count_neg,count_zero,count_pos)
            
# axs.title.set_text(title)  
            
           

plt.savefig("lj72_init/lj72_C3.png", dpi=300)
# fig.tight_layout()
# plt.show()

file = open( "lj72_init/lj72_C3.pickle", "wb" )
pickle.dump( X_t, file )
file.close()



##R=np.array([ -0.9523699364 ,       0.0159052548   ,    -0.0840802250,
##         -0.2528949024    ,    0.8598484003      , -0.3332183372,
##         -0.3357005052     ,  -0.8500181306      ,  0.2812584200,
##          0.7960710440      ,  0.5155096551      , -0.1218613860,
##          0.7448954577     ,  -0.5412454414       , 0.2579017563,
##         -0.0442130155      ,  0.1953980103       , 0.5377653956,
##          0.0442118576      , -0.1953977484      , -0.5377656238])
#
#key = random.PRNGKey(prms.seed)
#
#key, subkey = random.split(key)
##R=random.normal(subkey,((prms.N_a,prms.d)))
#R=R.reshape(prms.N_a,prms.d)
#fEval=V(R)    
#gdEval=grad(V)(R)
#grad_norm=np.linalg.norm(gdEval)
#hess_eval=hessian(V)(R).reshape(total_d,total_d)
#hess_eval=onp.array(hess_eval)
#mu,v=onp.linalg.eigh(hess_eval)
#count_neg=np.sum(mu < 0)
#count_pos=np.sum(mu > 0)
#print(0,count_neg,count_pos, mu[0:count_neg],grad_norm,fEval)

#i=0
#j=0
#h=1e-6
#Rp_i=jo.index_update(R,jo.index[i,j],R[i,j]+h)
#Rm_i=jo.index_update(R,jo.index[i,j],R[i,j]-h)
#

#fEval_pi=V(Rp_i)
#fEval_mi=V(Rm_i)
#
#grad_i= (fEval_pi-fEval_mi)/(2*h)  
#
#force_fn = quantity.force(V)
#frc=force_fn(R)
#
#print(grad_i,gdEval[i,j])
#
#i_1=1
#j_1=0
#i_2=4
#j_2=0
#
#R1=jo.index_update(R,jo.index[i_1,j_1],R[i_1,j_1]+h)
#R1=jo.index_update(R1,jo.index[i_2,j_2],R1[i_2,j_2]+h)
#fEval_1=V(R1)
#
#R2=jo.index_update(R,jo.index[i_1,j_1],R[i_1,j_1]+h)
#fEval_2=V(R2)
#
#R3=jo.index_update(R,jo.index[i_2,j_2],R[i_2,j_2]+h)
#fEval_3=V(R3)
#
#R4=jo.index_update(R,jo.index[i_1,j_1],R[i_1,j_1]-h)
#fEval_4=V(R4)
#
#R5=jo.index_update(R,jo.index[i_2,j_2],R[i_2,j_2]-h)
#fEval_5=V(R5)
#
#R6=jo.index_update(R,jo.index[i_1,j_1],R[i_1,j_1]-h)
#R6=jo.index_update(R6,jo.index[i_2,j_2],R6[i_2,j_2]-h)
#fEval_6=V(R6)
#
#ddf=fEval_1-fEval_2-fEval_3+2*fEval-fEval_4-fEval_5+fEval_6
#ddf=ddf/(2*h**2)
#
#hess_eval=hessian(V)(R)
#print(ddf,hess_eval[i_1,j_1,i_2,j_2])


#fire_init, fire_apply = minimize.fire_descent(V, shift_fn)
#fire_apply = jit(fire_apply)
#
#E = []
#for k in range(1):
#    fire_state = fire_init(R)
#    for i in range(1):
##        fire_state = fire_apply(fire_state)
#        fEval=V(fire_state.position)
#        print(i,fEval)
##        E += [fEval]
##        R = fire_state.position
    
#    key, subkey = random.split(key)
#    Z=random.normal(subkey,((prms.N_a,prms.d)))
#    R=R+Z*0.01
#
#i1=0
#h=0.0001
#Rp=np.array([ -0.9523699364 +h,       0.0159052548   ,    -0.0840802250,
#         -0.2528949024    ,    0.8598484003      , -0.3332183372,
#         -0.3357005052     ,  -0.8500181306      ,  0.2812584200,
#          0.7960710440      ,  0.5155096551      , -0.1218613860,
#          0.7448954577     ,  -0.5412454414       , 0.2579017563,
#         -0.0442130155      ,  0.1953980103       , 0.5377653956,
#          0.0442118576      , -0.1953977484      , -0.5377656238])
#
#Rm=np.array([ -0.9523699364 -h,       0.0159052548   ,    -0.0840802250,
#         -0.2528949024    ,    0.8598484003      , -0.3332183372,
#         -0.3357005052     ,  -0.8500181306      ,  0.2812584200,
#          0.7960710440      ,  0.5155096551      , -0.1218613860,
#          0.7448954577     ,  -0.5412454414       , 0.2579017563,
#         -0.0442130155      ,  0.1953980103       , 0.5377653956,
#          0.0442118576      , -0.1953977484      , -0.5377656238])
#
#Rp=Rp.reshape(prms.N_a,prms.d)
#Rm=Rm.reshape(prms.N_a,prms.d)
#f_p=V(Rp)
#f_m=V(Rm)
#f_b=V(R)
#hess_p=(f_p-2*f_b+f_p)/(h**2)
#print(hess_p,hess_eval[i1,i1])
#
#


