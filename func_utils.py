import numpy as np
import jax.numpy as jnp
from jax import grad,hessian,vmap,jit,jvp,random #,value_and_grad
from functools import partial
import opt_algorithms
import importlib

importlib.reload(opt_algorithms)



def hvp_particle(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]

def get_func_evals(V,prms):
    @partial(vmap, in_axes=(0, 0))
    def hvp(_X, _W):
        return hvp_particle(V, (_X,), (_W,))

    dV_s=jit(grad(V)) #gradient single particle
    ddV_s=jit(hessian(V))
    dV=jit(vmap(grad(V))) #gradient vectorized
    hvp_vec=jit(hvp)#hessian/vector product vectorized
    ddV=jit(vmap(hessian(V)))#hessian vectorized
    index_func_lambda=lambda i,j,a: a[i,j]
    index_func_lambda_map=jit(vmap(index_func_lambda,(0,0,None)))
    Vm = jit(vmap(V))

    gd_vmap = vmap(partial(opt_algorithms.gd_constant, V, dV_s, prms.gd_dt, prms.gd_max_iter))

    return Vm,dV_s,ddV_s,dV,hvp_vec,ddV,index_func_lambda,index_func_lambda_map,gd_vmap
