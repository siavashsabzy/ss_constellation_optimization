# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:31:06 2023
@author: Siavash Sabzy
"""
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA,\
    MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
import numpy as np
from ss_astro_lib import *
from pymoo.algorithms.moo.sms import SMSEMOA


class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "inc": Real(bounds=(0, 90)),
            "n_p": Integer(bounds=(1, 20)),
            "n_s": Integer(bounds=(1, 20)),
        }
        super().__init__(vars=vars, n_obj=3, n_ieq_constr=0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        n_p, n_s, inc =  X["n_p"], X["n_s"], X["inc"]
        epoch = np.array([2022, 11, 1, 0, 3, 24])
        refsat = np.append(epoch, np.array([20200 + 6378.137, 0, inc, 0, 0, 0]))
        constellation = ss_walker(n_p, n_s, 1, refsat)
        mesh_grid = ss_gridsphere(256)
        dops = ss_get_dops_multi(mesh_grid, constellation, 15) # elevation mask of 15 degrees for GPS
        
        if np.max(dops[:,0]) >= 500.0:
            f1 = 10e10
            f2 = 10e10
            f3 = 10e10
        else:
            f3 = np.sum(dops[:,0])/len(dops[:,0])
            f1 = n_p*n_s
            f2 = n_p

        out["F"] = [f1, f2, f3]


problem = MultiObjectiveMixedVariableProblem()

algorithm = SMSEMOA(pop_size=5,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),
                  )

res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=100,
               verbose=True)

print(res.F)
print(res.X)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()