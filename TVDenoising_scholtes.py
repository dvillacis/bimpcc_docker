import numpy as np
from scipy.sparse import diags,identity
from bimpcc.L2TVMPCC_PD_scholtes import L2TVMPCC_PD_scholtes
from bimpcc.solve_mpcc_scholtes import solve_mpcc_scholtes
from bimpcc.Dataset import get_dataset
from bimpcc.img_utils import chambolle_pock_rof_denoising, sparse_gradient, sparse_divergence

def test(size=5, dataset_name='cameraman'):
    utrue, unoisy = get_dataset(dataset_name, size).get_training_data()

    alpha0 = np.array([0.01])
    u0,q0,Ku0 = chambolle_pock_rof_denoising(unoisy, 0.01, 0.1, lambda_tv=alpha0[0])
    u0 = u0.ravel()
    theta0 = np.arctan2(q0[:,:,1], q0[:,:,0]).ravel()
    r0 = np.sqrt(q0[:,:,0]**2 + q0[:,:,1]**2).ravel()
    delta0 = np.sqrt(Ku0[:,:,0]**2 + Ku0[:,:,1]**2).ravel()
    qx0 = q0[:,:,0].ravel()
    qy0 = q0[:,:,1].ravel()

    print(f'u0: {u0.shape}, qx0: {qx0.shape}, qy0: {qy0.shape}, alpha0: {alpha0.shape}, r0: {r0.shape}, delta0: {delta0.shape}, theta0: {theta0.shape}')

    # print(f'r0: {r0}')

    x0 = np.concatenate((u0, qx0, qy0, alpha0, r0, delta0, theta0))

    N = size**2

    Kx,Ky = sparse_gradient(size,size)
    P = 1
    A = identity(N, format='coo')
    Q = np.ones((N, P))
    
    lb_u = np.zeros(N)
    lb_qx = -1e20*np.ones(N)
    lb_qy = -1e20*np.ones(N)
    lb_alpha = 1e-10*np.ones(P)
    lb_r = 1e-10*np.ones(N)
    lb_delta = 1e-20*np.ones(N)
    lb_theta = 1e-20*np.pi*np.ones(N)
    lb = np.concatenate(
        (lb_u, lb_qx, lb_qy, lb_alpha, lb_r, lb_delta, lb_theta))

    ub_u = 1e20*np.ones(N)
    ub_qx = 1e20*np.ones(N)
    ub_qy = 1e20*np.ones(N)
    ub_alpha = 1e20*np.ones(P)
    ub_r = 1e20*np.ones(N)
    ub_delta = 1e20*np.ones(N)
    ub_theta = 2*np.pi*np.ones(N)
    ub = np.concatenate(
        (ub_u, ub_qx, ub_qy, ub_alpha, ub_r, ub_delta, ub_theta))
    
    cl_1 = np.zeros(N)
    cl_2 = np.zeros(N)
    cl_3 = np.zeros(N)
    cl_4 = np.zeros(N)
    cl_5 = np.zeros(N)
    cl_6 = np.zeros(N)
    cl_7 = np.zeros(N)
    cl = np.concatenate((cl_1, cl_2, cl_3, cl_4, cl_5, cl_6, cl_7))

    cu_1 = np.zeros(N)
    cu_2 = np.zeros(N)
    cu_3 = np.zeros(N)
    cu_4 = np.zeros(N)
    cu_5 = np.zeros(N)
    cu_6 = 1e20*np.ones(N)
    cu_7 = 1e20*np.ones(N)
    cu = np.concatenate((cu_1, cu_2, cu_3, cu_4, cu_5, cu_6, cu_7))

    vars, info = solve_mpcc_scholtes(
        L2TVMPCC_PD_scholtes,
        x0,
        lb,
        ub,
        cl,
        cu,
        A,  
        Kx,
        Ky,
        Q,  
        utrue,  
        unoisy,  
        pi_init=0.1,
        tol=1e-3
    )
    
    return vars, info


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='cameraman')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    vars,info = test(args.N, args.dataset)
    
    if args.save:
        print(f'Saving to results_ScalarTVDenoising_global/{args.dataset}_{args.N}.pkl')
        info.to_pickle(
            f'results_ScalarTVDenoising_global/{args.dataset}_{args.N}.pkl')
    #print(vars)