"""
Utils for training SVMp+
Author: Fan Feng
"""
import numpy as np
import time
import os
import pandas as pd
from copy import deepcopy
from data_utils import get_kernel, get_Hessian, generate_start, get_obj_func_value, get_gradient_at_point, normalize_features
from direction_utils import permn, find_feasible_directions, clipping_func
from direction_utils import process_direction_1, process_direction_41, process_direction_42, process_direction_51, process_direction_52
from direction_utils import process_direction_61, process_direction_62, process_direction_71, process_direction_72
from direction_utils import process_direction_81, process_direction_82, process_direction_91, process_direction_92


_start_time = time.time()


def tic():
    global _start_time
    _start_time = time.time()


def toc():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(' - Time elapsed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


def train_mixture_model(x, xp, y, pi, gamma, C=0.1, Cs=0.9,
                        tau=0., kappa=0., rho=1., tol=1e-8, tol_steps=100,
                        max_iter=1e5, kernel='linear'):
    """
    Parameters
    ------------
        x: numpy.array
            training data (n * d)
        xp: numpy.array
            training privileged data (m * d)
        y: numpy.array
            training labels (n-dim)
        pi: numpy.array
            confidence (n-dim)
        gamma: float
            gamma in the objective function
        C: float
            C in the objective function
        Cs: float
            C^(star) in the objective function
        tau: float
            cosine of desired minimal angle (0 or 0.14?)
        kappa: float
            minimum distance to the constraint boundaries
        rho: float
            rho in the objective function
        tol: float
        tol_steps: int
            if lambda is less than tol for more than tol_steps consecutive steps, the iteration ends
        max_iter: int
            max # of iterations
        kernel: str
            kernel type

    Returns
    ------------
        p: numpy.array
            the concatenation of alpha and beta
        val_history: list
            the objective function value at each step
        """
    n, m = len(x), len(xp)
    print('n=', n, ', m=', m)
    print('Calculating kernels and Hessian...')

    x = normalize_features(x)
    xp = normalize_features(xp)

    # Two kernels
    x_kernel = get_kernel(x, kernel=kernel)
    xp_kernel = get_kernel(xp, kernel=kernel)

    # Hessian mat
    H = get_Hessian(y, x_kernel, xp_kernel, gamma)
    print(H)

    # start point
    p = generate_start(n, m, Cs)
    # np.savetxt('start_p.txt', p)
    # print('p', p)

    # groups of feasible directions
    print('Initializing searching directions...')
    if m != 0:  # If not SVM (so only SVM+ and SVMp+)
        I1 = permn(list(range(n, n+m)), 2)

        I2 = permn(list(range(m)), 2)
        I2 = [elm for elm in I2 if y[elm[0]] == y[elm[1]]]

        I5 = permn(list(range(m)), 2)
        I5 = [elm for elm in I5 if y[elm[0]] != y[elm[1]]]
    else:
        I1, I2, I5 = [], [], []

    if m != n:  # If not SVM+ (so only SVM and SVMp+)
        I3 = permn(list(range(m, n)), 2)
        I3 = [elm for elm in I3 if y[elm[0]] == y[elm[1]]]

        I4 = permn(list(range(m, n)), 2)
        I4 = [elm for elm in I4 if y[elm[0]] != y[elm[1]]]
    else:
        I3, I4 = [], []

    if m != 0 and m != n:  # If only SVMp+
        I6 = permn(list(range(n)), 2)
        I6 = [elm for elm in I6 if y[elm[0]] == y[elm[1]] and elm[0] < m <= elm[1]]

        I7 = permn(list(range(n)), 2)
        I7 = [elm for elm in I7 if y[elm[0]] != y[elm[1]] and elm[0] < m <= elm[1]]

        I8 = permn(list(range(m)), 2)
        I8 = [elm for elm in I8 if y[elm[0]] != y[elm[1]]]
        I8_plus = [elm for elm in I8 if y[elm[1]] > 0]
        I8_minus = [elm for elm in I8 if y[elm[1]] < 0]
        I8c_plus = [elm for elm in range(m, n) if y[elm] > 0]
        I8c_minus = [elm for elm in range(m, n) if y[elm] < 0]
    else:
        I6, I7, I8 = [], [], []
        I8_plus, I8_minus, I8c_plus, I8c_minus = [], [], [], []

    val_history = [get_obj_func_value(p, y, x_kernel, xp_kernel, gamma, Cs)]
    pos_history = [deepcopy(p)]
    ending_count = 0
    for i in range(max_iter):
        if i % 100 == 0:
            print(' - Iteration:', i+1)
            if i != 0:
                toc()
            tic()

        # val_prev = val_history[-1]
        # val = get_obj_func_value(p, y, x_kernel, xp_kernel, gamma, Cs)
        grad = get_gradient_at_point(p, y, x_kernel, xp_kernel, gamma, Cs)
        # print(grad)
        # np.savetxt('grad.txt', grad)

        # print('I1', I1)
        # print('I2', I2)
        # print('I5', I5)
        # print('I3', I3)
        # print('I4', I4)

        # Calculate the indices of all the directions to be searched over at this point
        I1_tilde, I2_tilde, I3_tilde, I4_tilde_1, I4_tilde_2, \
        alpha1_51, alpha2s_51, beta_51, alpha1_52, alpha2s_52, beta_52, \
        alpha1_61, alpha2s_61, beta_61, alpha1_62, alpha2s_62, beta_62, \
        alpha1_71, alpha2s_71, beta_71, alpha1_72, alpha2s_72, beta_72, \
        alpha1_811, alpha2s_811, alpha3_811, alpha1_812, alpha2s_812, alpha3_812, \
        alpha1_821, alpha2s_821, alpha3_821, alpha1_822, alpha2s_822, alpha3_822, \
        alpha1_911, alpha2s_911, alpha3_911, alpha1_912, alpha2s_912, alpha3_912, \
        alpha1_921, alpha2s_921, alpha3_921, alpha1_922, alpha2s_922, alpha3_922 = \
            find_feasible_directions(I1, I2, I3, I4, I5, I6, I7, I8_plus, I8c_plus, I8_minus, I8c_minus,
                                     p, n, m, pi, grad, tau, kappa, C, rho * Cs)

        print([len(x) for x in [I1, I2, I3, I4, I5, I6, I7, I8_plus, I8c_plus, I8_minus, I8c_minus]])
        print(sum([len(x) for x in [I1, I2, I3, I4, I5, I6, I7, I8_plus, I8c_plus, I8_minus, I8c_minus]]))
        # print('I1_tilde', I1_tilde)
        # print('I2_tilde', I2_tilde)
        # print('I5', alpha1_51, beta_51, alpha2s_51)
        # print('I5', alpha1_52, beta_52, alpha2s_52)
        # print('I3_tilde', I3_tilde)
        # print('I4_tilde_1', I4_tilde_1)
        # print('I4_tilde_2', I4_tilde_2)

        [s1, d1, L1] = process_direction_1(I1_tilde, H, grad)
        [s2, d2, L2] = process_direction_1(I2_tilde, H, grad)
        [s3, d3, L3] = process_direction_1(I3_tilde, H, grad)
        [s_41, d_41, L_41] = process_direction_41(I4_tilde_1, H, grad)
        [s_42, d_42, L_42] = process_direction_42(I4_tilde_2, H, grad)
        [s_51, d_51, L_51] = process_direction_51(alpha1_51, alpha2s_51, beta_51, H, grad)
        # print(s_51, d_51, L_51)
        [s_52, d_52, L_52] = process_direction_52(alpha1_52, alpha2s_52, beta_52, H, grad)
        [s_61, d_61, L_61] = process_direction_61(alpha1_61, alpha2s_61, beta_61, H, grad)
        [s_62, d_62, L_62] = process_direction_62(alpha1_62, alpha2s_62, beta_62, H, grad)
        [s_71, d_71, L_71] = process_direction_71(alpha1_71, alpha2s_71, beta_71, H, grad)
        [s_72, d_72, L_72] = process_direction_72(alpha1_72, alpha2s_72, beta_72, H, grad)
        [s_811, d_811, L_811] = process_direction_81(alpha1_811, alpha2s_811, alpha3_811, H, grad)
        [s_812, d_812, L_812] = process_direction_81(alpha1_812, alpha2s_812, alpha3_812, H, grad)
        [s_821, d_821, L_821] = process_direction_82(alpha1_821, alpha2s_821, alpha3_821, H, grad)
        [s_822, d_822, L_822] = process_direction_82(alpha1_822, alpha2s_822, alpha3_822, H, grad)
        [s_911, d_911, L_911] = process_direction_91(alpha1_911, alpha2s_911, alpha3_911, H, grad)
        [s_912, d_912, L_912] = process_direction_91(alpha1_912, alpha2s_912, alpha3_912, H, grad)
        [s_921, d_921, L_921] = process_direction_92(alpha1_921, alpha2s_921, alpha3_921, H, grad)
        [s_922, d_922, L_922] = process_direction_92(alpha1_922, alpha2s_922, alpha3_922, H, grad)

        s = [s1, s2, s3, s_41, s_42, s_51, s_52, s_61, s_62, s_71, s_72,
             s_811, s_812, s_821, s_822, s_911, s_912, s_921, s_922]
        t = int(np.argmax(s))

        if t == 0:  # d_1
            u, d, L = np.array([1, -1]), list(d1), L1
            if len(d) == 0:
                print('d1, termination')
                break
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 1:  # d_2
            u, d, L = np.array([1, -1]), list(d2), L2
            if len(d) == 0:
                print('d2, termination')
                break
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 2:  # d_3
            u, d, L = np.array([1, -1]), list(d3), L3
            if len(d) == 0:
                print('d3, termination')
                break
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 3:  # d_41
            u, d, L = np.array([1, 1]), list(d_41), L_41
            if len(d) == 0:
                print('d41, termination')
                break
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 4:  # d_42
            u, d, L = np.array([-1, -1]), list(d_42), L_42
            if len(d) == 0:
                print('d42, termination')
                break
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 5:  # d_51
            if isinstance(d_51, list) and len(d_51) == 0:
                print('d51, termination')
                break
            u, d, L = np.array([1, 1, -2]), [alpha1_51, d_51, beta_51], L_51
            # print(u, d, L)
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 6:  # d_52
            if isinstance(d_52, list) and len(d_52) == 0:
                print('d52, termination')
                break
            u, d, L = np.array([-1, -1, 2]), [alpha1_52, d_52, beta_52], L_52
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 7:  # d_61
            if isinstance(d_61, list) and len(d_61) == 0:
                print('d61, termination')
                break
            u, d, L = np.array([1, -1, -1]), [alpha1_61, d_61, beta_61], L_61
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 8:  # d_62
            if isinstance(d_62, list) and len(d_62) == 0:
                print('d62, termination')
                break
            u, d, L = np.array([-1, 1, 1]), [alpha1_62, d_62, beta_62], L_62
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 9:  # d_71
            if isinstance(d_71, list) and len(d_71) == 0:
                print('d71, termination')
                break
            u, d, L = np.array([1, 1, -1]), [alpha1_71, d_71, beta_71], L_71
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 10:  # d_72
            if isinstance(d_72, list) and len(d_72) == 0:
                print('d72, termination')
                break
            u, d, L = np.array([-1, -1, 1]), [alpha1_72, d_72, beta_72], L_72
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 11:  # d_811
            if isinstance(d_811, list) and len(d_811) == 0:
                print('d811, termination')
                break
            u, d, L = np.array([1, -1, 2]), [alpha1_811, d_811, alpha3_811], L_811
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 12:  # d_812
            if isinstance(d_812, list) and len(d_812) == 0:
                print('d812, termination')
                break
            u, d, L = np.array([1, -1, 2]), [alpha1_812, d_812, alpha3_812], L_812
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 13:  # d_821
            if isinstance(d_821, list) and len(d_821) == 0:
                print('d821, termination')
                break
            u, d, L = np.array([-1, 1, -2]), [alpha1_821, d_821, alpha3_821], L_821
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 14:  # d_822
            if isinstance(d_822, list) and len(d_822) == 0:
                print('d822, termination')
                break
            u, d, L = np.array([-1, 1, -2]), [alpha1_822, d_822, alpha3_822], L_822
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 15:  # d_911
            if isinstance(d_911, list) and len(d_911) == 0:
                print('d911, termination')
                break
            u, d, L = np.array([1, -1, -2]), [alpha1_911, d_911, alpha3_911], L_911
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 16:  # d_912
            if isinstance(d_912, list) and len(d_912) == 0:
                print('d912, termination')
                break
            u, d, L = np.array([1, -1, -2]), [alpha1_912, d_912, alpha3_912], L_912
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 17:  # d_921
            if isinstance(d_921, list) and len(d_921) == 0:
                print('d921, termination')
                break
            u, d, L = np.array([-1, 1, 2]), [alpha1_921, d_921, alpha3_921], L_921
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        elif t == 18:  # d_922
            if isinstance(d_922, list) and len(d_922) == 0:
                print('d922, termination')
                break
            u, d, L = np.array([-1, 1, 2]), [alpha1_922, d_922, alpha3_922], L_922
            L = clipping_func(p[d], pi, d, L, u, n, m, C, rho * Cs)
            p[d] = p[d] + L * u

        else:
            raise ValueError
        # print(p)
        diff = np.linalg.norm(p - pos_history[-1])
        pos_history.append(deepcopy(p))

        if diff < tol:
            ending_count += 1
        else:
            ending_count = 0

        val = get_obj_func_value(p, y, x_kernel, xp_kernel, gamma, Cs)
        # if val < val_history[-1]:
        #     print([t, L, val_history[-1], diff])

        if i % 100 == 0:
            print('Iter', i + 1)
            print([t, L, val_history[-1], diff])
            # print(' - Objective function:', val)
        val_history.append(val)

        if ending_count >= tol_steps:
            print('Ending iteration!')
            break

    return p, val_history


if __name__ == '__main__':
    # path = '../../MNIST+/'
    # x = pd.read_csv(os.path.join(path, 'train_features.txt')).to_numpy()
    # y = pd.read_csv(os.path.join(path, 'train_labels.txt')).to_numpy()
    # y = np.where(y == 5, 1, -1)
    # xp = pd.read_csv(os.path.join(path, 'train_PFfeatures.txt')).to_numpy()
    x = np.load('../../MNIST+/final_data/train_x.npy')
    xp = np.load('../../MNIST+/final_data/train_xp_0.6.npy')
    y = np.load('../../MNIST+/final_data/train_y.npy')
    x, y, xp = np.array(x), np.array(y).flatten(), np.array(xp)
    print(x.shape)
    print(xp.shape)
    print(y.shape)

    pi = np.ones((len(x),))
    gamma = 1
    t1 = time.time()
    p, hist = train_mixture_model(x, xp, y, pi, gamma, C=1., Cs=1.,
                                  tau=0., kappa=0., rho=1., tol=1e-8, tol_steps=1,
                                  max_iter=100000, kernel='linear')
    t2 = time.time()
    duration = t2 - t1
    print('Total Time:', duration)

    # np.savetxt('final_p_Yuanhao_data.txt', p)
    # np.save('loss_hist.npy', hist)

# 4 6 7 9
