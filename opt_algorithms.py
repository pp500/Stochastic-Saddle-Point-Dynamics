from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def gd_constant(V,dV,dt,n_iter,x_k):
  for k in range(0,n_iter):
    x_k=x_k-dt*dV(x_k)
  vEval=V(x_k)
  dVNorm=jnp.linalg.norm(dV(x_k))
  return x_k,vEval,dVNorm

def gd_vanilla(V,dV,dt,max_iter,epsilon,x_k):
  converge=False
  grd=dV(x_k)
  k=0
  while converge is False:
    x_k=x_k-dt*grd
    grd = dV(x_k)
    dVNorm = jnp.linalg.norm(grd)
    k=k+1
    if dVNorm<epsilon or k>max_iter:
        converge=True
  vEval=V(x_k)
  success=False
  if dVNorm<epsilon:
      success=True

  return success,x_k,vEval,dVNorm


def check_basin_fVal(gd_vmap, X_t, prms, VInit, prtcl_index):
    # print("Start bc")

    gd_data = gd_vmap(X_t)
    # X_rs=gd_data[0]
    f_values = gd_data[1]
    norms = gd_data[2]
    # check for convergence
    max_norm = jnp.max(norms)
    converge_gd = max_norm < prms.basin_gd_tol
    if converge_gd == False:
        print("Basin GD not converged, increase number of iters?")
        print("Min GD Norm=", jnp.min(norms), "Max=", max_norm)

    # kill ones that are not in the basin
    prtcl_out = prtcl_index[np.array(jnp.abs(f_values - VInit) > prms.basin_f_tol)]
    number_out = len(prtcl_out)
    # out_basin=number_out
    return number_out, prtcl_out