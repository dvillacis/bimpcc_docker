import numpy as np
import pandas as pd
import cyipopt
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
    mu_init=1.0,
    sigma=0.1,
    k_max=10,
    gamma=0.4,
    kappa=0.2,
    nu=0.1,
    tol=1e-6,
    **kwargs
):
    print(f'Solving MPCC with {problem_class.__name__} with size {len(x0)}')
    k = 1
    pi = pi_init
    mu = mu_init
    x = x0
    last_obj = np.inf
    general_info = {}
    infos = []
    print(f'{"Iter": >5}\t{"Termination_status": >15}\t{"Objective": >15}\t{
          "MPCC_compl": >15}\t{"lg(mu)": >15}\t{"π": >15}\t{"comment"}\n')
    while k < k_max:
        # print(f'Iteration {k}, mu={mu}, pi={pi}')
        tol_c = mu**gamma
        tol_p = nu*pi
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
        nlp.add_option('dual_inf_tol', tol_p)
        nlp.add_option('constr_viol_tol', tol_p)
        nlp.add_option('compl_inf_tol', tol_p)
        nlp.add_option('print_level', 0)
        # nlp.add_option('check_derivatives_for_naninf', 'yes')
        # nlp.add_option('max_iter', 10)
        # nlp.add_option('nlp_scaling_method', 'none')
        nlp.add_option('tol', tol)
        nlp.add_option('hessian_approximation', 'limited-memory')
        # nlp.add_option('fast_step_computation', 'yes')
        nlp.add_option('sb', 'yes')
        # nlp.add_option('derivative_test', 'first-order')
        nlp.add_option('jacobian_approximation', 'finite-difference-values')
        x_, info = nlp.solve(x)
        x_ = np.clip(x_, lb, ub)
        # print(x_)
        comp = problem_instance.complementarity(x_)
        # comp = problem_instance.min_complementarity(x_)
        # info['extra'] = {'k': k, 'comp': comp, 'mu': mu, 'pi': pi}
        info['k'] = k
        info['comp'] = comp
        info['mu'] = mu
        info['pi'] = pi
        info['num_constraints'] = len(cl)
        # info['jac_nz'] = len(problem_instance.jacobianstructure()[0])
        # print(info)
        print(f'{k: > 5}\t{info["status"]: > 15}\t{info["obj_val"]: > 15}\t{comp: > 15}\t{np.log10(mu): > 15}\t{pi: > 15}\t{info['status_msg'][:50]}')
        
        if (np.abs(comp) < tol) or k >= k_max:
            print(f'Obtained solution satisfies the complementarity condition at {comp} at {k} iterations')
            infos.append(info)
            break
        else:
            last_obj = info['obj_val']
            pi *= sigma
            # mu *= kappa

        # if (np.abs(comp) <= tol_p):  # & (info['status'] >= -1):
        #     k += 1
        #     x = x_
        #     # if (np.abs(last_obj-info['obj_val']) < tol ):
        #     mu *= kappa
        # else:
        #     if pi < 1e10:
        #         pi *= sigma
        #         # mu = mu_init
        #     else:
        #         print('The problem is unbounded')
        #         break

        infos.append(info)
        # k = k_max
    v = problem_instance.getvars(x)

    return (v), pd.DataFrame().from_records(infos)
