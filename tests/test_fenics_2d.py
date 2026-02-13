import os
import sys

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "pde"))
)
from solvers.fenics import Poisson2DSolver

poisson = Poisson2DSolver(resolution=16)
solution = poisson.solve_poisson(seed=15)
bound_val, dom_val, bound_pt, dom_pt = poisson.get_pointwise_solution(solution)
poisson.poisson_pde.plot_solution(dom_val, bound_val, dom_pt, bound_pt)
print(poisson.sample_params())
