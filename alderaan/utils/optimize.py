"""MAP optimization of PyMC models"""

import numpy as np
import pymc as pm
from pymc.blocking import DictToArrayBijection, RaveledVars
from scipy.optimize import minimize


def optimize(start=None, vars=None, model=None):
    """
    Maximize the log probability of a PyMC model over vars with scipy BFGS,
    holding all other variables fixed at start

    Unlike pm.find_MAP, the objective includes the Jacobians of the variable
    transforms, which keeps bounded variables off their boundaries, and the
    result is kept only if the log probability is finite and improved
    (the behavior of pymc3-ext's optimize)

    Parameters
    ----------
    start : dict
        starting point; default model.initial_point()
    vars : list
        free random variables to optimize; default all

    Returns
    -------
    point : dict
        value of every unobserved variable, including deterministics
    info : scipy.optimize.OptimizeResult
    """
    model = pm.modelcontext(model)
    if start is None:
        start = model.initial_point()
    start = {v.name: start[v.name] for v in model.value_vars}
    vars = model.free_RVs if vars is None else vars
    value_vars = [model.rvs_to_values[v] for v in vars]

    x0 = DictToArrayBijection.map({v.name: start[v.name] for v in value_vars})
    logp = DictToArrayBijection.mapf(model.compile_logp(jacobian=True), start)
    dlogp = DictToArrayBijection.mapf(model.compile_dlogp(vars, jacobian=True), start)

    def objective(x):
        x = RaveledVars(x, x0.point_map_info)
        return -logp(x), -dlogp(x)

    info = minimize(objective, x0.data, jac=True, method="BFGS")
    if not (np.isfinite(info.fun) and info.fun < objective(x0.data)[0]):
        info.x = x0.data

    point = DictToArrayBijection.rmap(RaveledVars(info.x, x0.point_map_info), start)
    outs = model.unobserved_value_vars
    values = model.compile_fn(inputs=model.value_vars, outs=outs)(point)
    return {v.name: val for v, val in zip(outs, values)}, info
