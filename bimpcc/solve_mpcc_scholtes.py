import numpy as np
import pandas as pd
import cyipopt
import traceback
from .AbstractMPCC import AbstractMPCC

def solve_mpcc_scholtes(
    problem_class: AbstractMPCC,
    x0,
    lb,
    ub,
    cl,
    cu,
    *args,
    pi_init=1.0,
    sigma=10,
    k_max=100,
    gamma=0.4,
    kappa=0.2,
    nu=10,
    tol=1e-6,
    **kwargs
):
    print(f'Solving MPCC with {problem_class.__name__} with size {len(x0)}')
    k = 1
    pi = pi_init
    x = x0
    last_obj = np.inf
    general_info = {}
    infos = []
    print(f'{"Iter": >5}\t{"Termination_status": >15}\t{"Objective": >15}\t{
          "MPCC_compl": >15}\t{"π": >15}\t{"comment"}\n')
    while k < k_max:
        # print(f'Iteration {k}, mu={mu}, pi={pi}')
        problem_instance = problem_class(pi, *args, **kwargs)
        # print(f'Gradient norm {np.linalg.norm(problem_instance.gradient(x))}')
        nlp = cyipopt.Problem(
            n=len(x),
            m=len(cl),
            problem_obj=problem_instance,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
        )
        # Add configuration to ipopt
        # nlp.add_option('nlp_scaling_method', 'none')
        # nlp.add_option('mu_init', mu)
        # nlp.add_option('mu_strategy', 'monotone')
        # nlp.add_option('dual_inf_tol', tol_p)
        # nlp.add_option('constr_viol_tol', tol_p)
        # nlp.add_option('compl_inf_tol', tol_p)
        nlp.add_option('print_level', 5)
        # nlp.add_option('check_derivatives_for_naninf', 'yes')
        nlp.add_option('max_iter', 10000)
        # nlp.add_option('nlp_scaling_method', 'none')
        nlp.add_option('acceptable_tol', tol)
        nlp.add_option('tol', tol)
        nlp.add_option('jacobian_approximation', 'exact')
        nlp.add_option('hessian_approximation', 'exact')
        # nlp.add_option('jacobian_regularization_value', 1e-3)
        # nlp.add_option('fast_step_computation', 'yes')
        nlp.add_option('sb', 'yes')
        nlp.add_option('derivative_test', 'second-order')
        # nlp.add_option('jacobian_approximation', 'finite-difference-values')
        x_, info = nlp.solve(x)
        # x_ = np.clip(x_, lb, ub)
        # print(x_)
        comp = problem_instance.min_complementarity(x_)
        # comp = problem_instance.min_complementarity(x_)
        # info['extra'] = {'k': k, 'comp': comp, 'mu': mu, 'pi': pi}
        info['k'] = k
        info['comp'] = comp
        info['pi'] = pi
        info['num_constraints'] = len(cl)
        # info['jac_nz'] = len(problem_instance.jacobianstructure()[0])
        # print(info)
        print(f'{k: > 5}\t{info["status"]: > 15}\t{info["obj_val"]: > 15}\t{
              comp: > 15}\t{pi: > 15}\t{info['status_msg'][:50]}')
        
        if (np.abs(comp) < tol):
            print(f'Obtained solution satisfies the complementarity condition at {comp} at {k} iterations')
            infos.append(info)
            break
        else:
            last_obj = info['obj_val']

        k += 1

        if (np.abs(comp) <= tol):  # & (info['status'] >= -1):
            x = x_
            break
        else:
            pi *= sigma

        infos.append(info)
        k = k_max
    v = problem_instance.getvars(x)

    return (v), pd.DataFrame().from_records(infos)
