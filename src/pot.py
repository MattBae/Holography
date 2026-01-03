import torch
import math
import ot
import jax.numpy as jnp
from ott.geometry import geometry
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem
import numpy as np

def compute_cost_matrix(n):
    "compute cost matrix for optimal transport"
    coords = []
    for i in range(len(n)):
        coords.append(torch.linspace(-math.floor((n[i]) /2), math.floor((n[i]-1)/2),n[i]))
    lattices = list(torch.meshgrid(*coords, indexing='ij'))
    for i in range(len(lattices)):
        lattices[i] = lattices[i].flatten()
    N = lattices[0].shape[0]
    c = torch.zeros((N,N))
    for i in lattices:
        c+=(i[:,None]-i[None,:])**2
    c = c/c.max()
    return c,lattices


def line_integrate(gammas):
    "integrate the gamma gradients values using a line integral and trapezoidal rule"
    integrals = []
    ndim = gammas[0].ndim
    shape = gammas[0].shape
    result = torch.zeros_like(gammas[0])
    for i in range(len(gammas)):
        slicer = [slice(None)] * (i + 1) + [0] * (ndim - i - 1)
        sliced = gammas[i][tuple(slicer)]
        integrated = torch.cumulative_trapezoid(sliced, dim=i,dx = (gammas[i].shape[i])**-1)
        zero_pad = torch.zeros(*sliced.shape[:-1], 1)
        integrated_full = torch.cat([zero_pad, integrated], dim=-1).view(*(sliced.shape + (1,)*((ndim - i - 1)))).expand(*shape)
        result += integrated_full
    return result

def ot_phase_retrieval(input_intensity, target_intensity, reg_param=0.0001,n_iters = 100):
    """Retrieve phase using Optimal Transport (OT) method."""
    n = input_intensity.shape
    u = torch.flatten(input_intensity)
    v = torch.flatten(target_intensity)
    c,lattices = compute_cost_matrix(n)
    # tau = ot.sinkhorn(u, v, c, reg=reg_param,method = "sinkhorn", numItermax=n_iters,verbose=True)
    uj = jnp.array(u)  # source distribution
    vj = jnp.array(v)  # target distribution
    c = jnp.array(c)
    # Create geometry object with entropic regularization
    geom = geometry.Geometry(cost_matrix=c, epsilon=reg_param)
    ot_problem = linear_problem.LinearProblem(geom, uj, vj)
    solver = sinkhorn.Sinkhorn(max_iterations = n_iters,inner_iterations=1)
    out = solver(ot_problem)
    tau = torch.tensor(np.array(out.matrix))
    print(tau.shape)
    gammas = [((tau@i)/u).reshape(target_intensity.shape) for i in lattices]
    phi = line_integrate(gammas)
    return phi,out    

