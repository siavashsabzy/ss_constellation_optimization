# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:31:06 2023
@author: Siavash Sabzy

GPS altitude: 20200 inc: 55
Glonass altitude: 19100  inc: 64.8 
Galileo altitude: 23222 inc: 56

input: 
altitude, mask angle, inclination [0-90],  \
number of sats, number of planes, \
epoch, n_grid, and algorithm settings


About the elevation mask: it is mentioned that for GPs constellation the elevation mask is
considered to be between 15-20 degrees, however, in these simulations it considered to be
23 degrees for the sake of redundancy.



"""
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.visualization.scatter import Scatter
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
import numpy as np
from ss_astro_lib import *
from pymoo.algorithms.moo.sms import SMSEMOA

class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "inc": Real(bounds=(0, 90)),
            "n_p": Integer(bounds=(1, 10)),
            "n_s": Integer(bounds=(1, 10)),
        }
        super().__init__(vars=vars, n_obj=3, n_ieq_constr=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        n_p, n_s, inc =  X["n_p"], X["n_s"], X["inc"]
        grid_n = 400
        elmask = 8 
        altitude = 20200
        dops = ss_get_all_dops(n_p, n_s, inc, grid_n, elmask, altitude)
        f1 = n_p*n_s
        f2 = n_p
        f3 = 0
        g1 = 0
        for i in range(len(dops)):
            f3 += dops[i][0]
            if dops[i][0] > g1:
                g1 = dops[i][0]
        f3 = f3/len(dops)
        g1 = g1 - 500
        g2 = f3 - 10
        out["F"] = [f1,f2,f3]
        out["G"] = [g1]


problem = MultiObjectiveMixedVariableProblem()

algorithm = SMSEMOA(pop_size=35,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),
                  normalize=True,
                  )

res = minimize(problem,
               algorithm,
               ('n_gen', 30),
               seed=100,
               verbose=True)

print(res.F)
print(res.X)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()