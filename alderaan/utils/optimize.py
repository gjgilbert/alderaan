"""MAP optimization of PyMC models"""

import numpy as np
import pymc as pm
from pymc.blocking import DictToArrayBijection, RaveledVars
from scipy.optimize import minimize


def _bfgs(logp_fn, dlogp_fn, point, value_vars):
    """
    One BFGS stage over value_vars, others held at point; the result is kept
    only if the log probability is finite and improved
    """
    x0 = DictToArrayBijection.map({v.name: point[v.name] for v in value_vars})
    logp = DictToArrayBijection.mapf(logp_fn, point)
    dlogp = DictToArrayBijection.mapf(dlogp_fn, point)

    def objective(x):
        x = RaveledVars(x, x0.point_map_info)
        return -logp(x), -dlogp(x)

    info = minimize(objective, x0.data, jac=True, method="BFGS")
    if not (np.isfinite(info.fun) and info.fun < objective(x0.data)[0]):
        info.x = x0.data

    point = DictToArrayBijection.rmap(RaveledVars(info.x, x0.point_map_info), point)
    return point, info


def optimize(stages=None, start=None, model=None):
    """
    Maximize the log probability of a PyMC model with scipy BFGS, in stages:
    each stage optimizes its variables, holding all others at the current point

    Unlike pm.find_MAP, the objective includes the Jacobians of the variable
    transforms, which keeps bounded variables off their boundaries, and a
    stage's result is kept only if the log probability is finite and improved
    (the behavior of pymc3-ext's optimize)

    The log probability and each distinct gradient are compiled once per call

    Parameters
    ----------
    stages : list of lists
        free random variables to optimize, one list per stage, in order;
        default one stage over all free random variables
    start : dict
        starting point; default model.initial_point()

    Returns
    -------
    point : dict
        value of every unobserved variable, including deterministics
    info : scipy.optimize.OptimizeResult
        result of the last stage
    """
    model = pm.modelcontext(model)
    if start is None:
        start = model.initial_point()
    point = {v.name: start[v.name] for v in model.value_vars}
    stages = [model.free_RVs] if stages is None else stages

    logp_fn = model.compile_logp(jacobian=True)
    dlogp_fns = {}
    for vars in stages:
        key = tuple(v.name for v in vars)
        if key not in dlogp_fns:
            dlogp_fns[key] = model.compile_dlogp(vars, jacobian=True)
        value_vars = [model.rvs_to_values[v] for v in vars]
        point, info = _bfgs(logp_fn, dlogp_fns[key], point, value_vars)

    outs = model.unobserved_value_vars
    values = model.compile_fn(inputs=model.value_vars, outs=outs)(point)
    return {v.name: val for v, val in zip(outs, values)}, info
