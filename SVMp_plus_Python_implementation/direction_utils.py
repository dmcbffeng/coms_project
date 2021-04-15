"""
Utils for searching directions
Author: Fan Feng
"""
import numpy as np
from copy import deepcopy
from itertools import permutations


def permn(vals, N, K=None):
    """
    Permutations without repetition, for generating feasible directions
    e.g, permn([1, 2, 3], 2) returns:
    [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

    Parameters
    ------------
        vals: list or numpy.array
            candidate directions
        N: int
            length
        K: list (optional)
            only return items with these indices
            (it avoids memory issues that may occur when there are too many combinations.
            This is particularly useful when you only need a few permutations at a given time.)

    Returns
    ------------
        cand: list
            all feasible permutations
    """
    ite = permutations(vals, N)

    if K is None:
        cand = [elm for elm in ite]
    else:
        K = sorted(list(set(deepcopy(K))), reverse=True)
        cand = []
        for i, elm in enumerate(ite):
            if i == K[-1]:
                cand.append(elm)
                K.pop()
                if len(K) == 0:
                    break
    return cand


def find_feasible_directions(I1, I2, I3, I4, I5, I6, I7,
                             I8_plus, I8c_plus, I8_minus, I8c_minus,
                             p, n, m, pi, grad, tau, kappa, C, rCs):
    """
    Using all candidate directions to find the feasible ones
    :param p: current point
    :param n:
    :param m:
    :param pi: confidence
    :param grad: gradient
    :param tau: tolerance of gradient change
    :param kappa: min distance to the constraint
    :param C: weight C
    :param rCs: rho * C^(star)
    """
    ind_vec = list(range(n, n+m))  # indices for betas

    if m != 0:
        ############
        # I1
        ############
        I1 = [elm for elm in I1 if p[elm[1]] > kappa]
        if len(I1) > 0:
            # Find the direction maximally aligned with the gradient
            _grad_align = [grad[elm[0]] - grad[elm[1]] for elm in I1]
            max_dir = I1[int(np.argmax(_grad_align))]
            # Find the directions with the same positive direction index as that maximally aligned with the gradient
            I1_tilde = [elm for elm in I1 if elm[0] == max_dir[0] and grad[elm[0]] - grad[elm[1]] >= tau]
        else:
            I1_tilde = []

        ############
        # I2
        ############
        I2 = [elm for elm in I2 if p[elm[1]] > kappa and p[elm[0]] < rCs * pi[elm[0]]]
        if len(I2) > 0:
            _grad_align = [grad[elm[0]] - grad[elm[1]] for elm in I2]
            max_dir = I2[int(np.argmax(_grad_align))]
            I2_tilde = [elm for elm in I2 if elm[0] == max_dir[0] and grad[elm[0]] - grad[elm[1]] >= tau]
        else:
            I2_tilde = []

        ############
        # I5 first half
        ############
        I5p = [elm for elm in I5 if p[elm[0]] < rCs * pi[elm[0]] and p[elm[1]] < rCs * pi[elm[1]]]
        ind_vec5 = [elm for elm in ind_vec if p[elm] > kappa]
        if len(I5p) > 0 and len(ind_vec5) > 0:
            b5min = int(np.argmin(grad[ind_vec5]))
            _grad_align = [grad[elm[0]] + grad[elm[1]] for elm in I5p]
            max_dir = I5p[int(np.argmax(_grad_align))]
            alpha1_51 = max_dir[0]  # Save alpha1
            beta_51 = ind_vec5[b5min]  # Save beta
            alpha2s_51 = [elm[1] for elm in I5p if
                          elm[0] == alpha1_51 and grad[elm[0]] + grad[elm[1]] - 2 * grad[beta_51] > tau]  # All possible alpha2
        else:
            alpha1_51, alpha2s_51, beta_51 = [], [], []

        ############
        # I5 second half
        ############
        I5p = [elm for elm in I5 if p[elm[0]] > kappa and p[elm[1]] > kappa]
        b5max = int(np.argmax(grad[ind_vec]))
        if len(I5p) > 0:
            _grad_align = [grad[elm[0]] + grad[elm[1]] for elm in I5p]
            max_dir = I5p[int(np.argmin(_grad_align))]
            alpha1_52 = max_dir[0]
            beta_52 = ind_vec[b5max]
            alpha2s_52 = [elm[1] for elm in I5p if
                          elm[0] == alpha1_52 and - grad[elm[0]] - grad[elm[1]] + 2 * grad[beta_52] > tau]
        else:
            alpha1_52, alpha2s_52, beta_52 = [], [], []

    else:
        I1_tilde, I2_tilde = [], []
        alpha1_51, alpha2s_51, beta_51 = [], [], []
        alpha1_52, alpha2s_52, beta_52 = [], [], []

    if m != n:
        ############
        # I3
        ############
        I3 = [elm for elm in I3 if p[elm[0]] < C * pi[elm[0]] and p[elm[1]] > kappa]
        if len(I3) > 0:
            _grad_align = [grad[elm[0]] - grad[elm[1]] for elm in I3]
            max_dir = I3[int(np.argmax(_grad_align))]
            I3_tilde = [elm for elm in I3 if elm[0] == max_dir[0] and grad[elm[0]] - grad[elm[1]] >= tau]
        else:
            I3_tilde = []

        ############
        # I4 first half
        ############
        I4p = [elm for elm in I4 if p[elm[0]] < C * pi[elm[0]] and p[elm[1]] < C * pi[elm[1]]]
        if len(I4p) > 0:
            _grad_align = [grad[elm[0]] + grad[elm[1]] for elm in I4p]
            max_dir = I4p[int(np.argmax(_grad_align))]
            I4_tilde_1 = [elm for elm in I4p if elm[0] == max_dir[0] and grad[elm[0]] + grad[elm[1]] >= tau]
        else:
            I4_tilde_1 = []

        ############
        # I4 second half
        ############
        I4p = [elm for elm in I4 if p[elm[0]] > kappa and p[elm[1]] > kappa]
        if len(I4p) > 0:
            _grad_align = [- grad[elm[0]] - grad[elm[1]] for elm in I4p]
            max_dir = I4p[int(np.argmax(_grad_align))]
            I4_tilde_2 = [elm for elm in I4p if elm[0] == max_dir[0] and - grad[elm[0]] - grad[elm[1]] >= tau]
        else:
            I4_tilde_2 = []

    else:
        I3_tilde, I4_tilde_1, I4_tilde_2 = [], [], []

    if m != 0 and m != n:
        ############
        # I6 first half
        ############
        I6p = [elm for elm in I6 if p[elm[0]] < rCs * pi[elm[0]] and p[elm[1]] > kappa]
        # ind_vec5 = [elm for elm in ind_vec if p[elm] > kappa]
        # b5min = int(np.argmin(grad[ind_vec5]))
        if len(I6p) > 0 and len(ind_vec5) > 0:
            _grad_align = [grad[elm[0]] - grad[elm[1]] for elm in I6p]
            max_dir = I6p[int(np.argmax(_grad_align))]
            alpha1_61 = max_dir[0]
            beta_61 = ind_vec5[b5min]
            alpha2s_61 = [elm[1] for elm in I6p if
                          elm[0] == alpha1_61 and grad[elm[0]] - grad[elm[1]] - grad[beta_61] > tau]
        else:
            alpha1_61, alpha2s_61, beta_61 = [], [], []

        ############
        # I6 second half
        ############
        I6p = [elm for elm in I6 if p[elm[0]] > kappa and p[elm[1]] < C * pi[elm[1]]]
        if len(I6p) > 0:
            _grad_align = [- grad[elm[0]] + grad[elm[1]] for elm in I6p]
            max_dir = I6p[int(np.argmax(_grad_align))]
            alpha1_62 = max_dir[0]
            beta_62 = ind_vec[b5max]
            alpha2s_62 = [elm[1] for elm in I6p if
                          elm[0] == alpha1_62 and - grad[elm[0]] + grad[elm[1]] + grad[beta_62] > tau]
        else:
            alpha1_62, alpha2s_62, beta_62 = [], [], []

        ############
        # I7 first half
        ############
        I7p = [elm for elm in I7 if p[elm[0]] < rCs * pi[elm[0]] and p[elm[1]] < C * pi[elm[1]]]
        if len(I7p) > 0 and len(ind_vec5) > 0:
            _grad_align = [grad[elm[0]] + grad[elm[1]] for elm in I7p]
            max_dir = I7p[int(np.argmax(_grad_align))]
            alpha1_71 = max_dir[0]
            beta_71 = ind_vec5[b5min]
            alpha2s_71 = [elm[1] for elm in I7p if
                          elm[0] == alpha1_71 and grad[elm[0]] + grad[elm[1]] - grad[beta_71] > tau]
        else:
            alpha1_71, alpha2s_71, beta_71 = [], [], []

        ############
        # I7 second half
        ############
        I7p = [elm for elm in I7 if p[elm[0]] > kappa and p[elm[1]] > kappa]
        if len(I7p) > 0:
            _grad_align = [grad[elm[0]] + grad[elm[1]] for elm in I7p]
            max_dir = I7p[int(np.argmin(_grad_align))]
            alpha1_72 = max_dir[0]
            beta_72 = ind_vec[b5max]
            alpha2s_72 = [elm[1] for elm in I7p if
                          elm[0] == alpha1_72 and - grad[elm[0]] - grad[elm[1]] + grad[beta_72] > tau]
        else:
            alpha1_72, alpha2s_72, beta_72 = [], [], []

        ############
        # I8 first half of the first half
        ############
        I8p11 = [elm for elm in I8_plus if p[elm[0]] < rCs * pi[elm[0]] and p[elm[1]] > kappa]
        I8cp = [elm for elm in I8c_plus if p[elm] < C * pi[elm]]
        if len(I8p11) > 0 and len(I8cp) > 0:
            _grad_align = [grad[elm[0]] - grad[elm[1]] for elm in I8p11]
            max_dir = I8p11[int(np.argmin(_grad_align))]
            alpha1_811 = max_dir[0]
            alpha3_811 = I8cp[int(np.argmax(grad[I8cp]))]
            # alpha2s_811911 = [elm[1] for elm in I8p11 if elm[0] == alpha1_811]
            alpha2s_811 = [elm[1] for elm in I8p11 if elm[0] == alpha1_811 and
                           grad[elm[0]] - grad[elm[1]] + 2 * grad[alpha3_811] > tau]
        else:
            alpha1_811, alpha2s_811, alpha3_811 = [], [], []
            # alpha2s_811911 = []

        ############
        # I8 second half of the first half
        ############
        I8p12 = [elm for elm in I8_minus if p[elm[0]] < rCs * pi[elm[0]] and p[elm[1]] > kappa]
        I8cp = [elm for elm in I8c_minus if p[elm] < C * pi[elm]]
        if len(I8p12) > 0 and len(I8cp) > 0:
            _grad_align = [grad[elm[0]] - grad[elm[1]] for elm in I8p12]
            max_dir = I8p12[int(np.argmin(_grad_align))]
            alpha1_812 = max_dir[0]
            alpha3_812 = I8cp[int(np.argmax(grad[I8cp]))]
            # alpha2s_812912 = [elm[1] for elm in I8p12 if elm[0] == alpha1_812]
            alpha2s_812 = [elm[1] for elm in I8p12 if elm[0] == alpha1_812 and
                           grad[elm[0]] - grad[elm[1]] + 2 * grad[alpha3_812] > tau]
        else:
            alpha1_812, alpha2s_812, alpha3_812 = [], [], []
            # alpha2s_812912 = []

        ############
        # I8 first half of the second half
        ############
        I8p21 = [elm for elm in I8_plus if p[elm[0]] > kappa and p[elm[1]] < rCs * pi[elm[0]]]
        I8cp = [elm for elm in I8c_plus if p[elm] > kappa]
        if len(I8p21) > 0 and len(I8cp) > 0:
            _grad_align = [- grad[elm[0]] + grad[elm[1]] for elm in I8p21]
            max_dir = I8p21[int(np.argmin(_grad_align))]
            alpha1_821 = max_dir[0]
            alpha3_821 = I8cp[int(np.argmin(grad[I8cp]))]
            # alpha2s_821921 = [elm[1] for elm in I8p21 if elm[0] == alpha1_821]
            alpha2s_821 = [elm[1] for elm in I8p21 if elm[0] == alpha1_821 and
                           - grad[elm[0]] + grad[elm[1]] - 2 * grad[alpha3_821] > tau]
        else:
            alpha1_821, alpha2s_821, alpha3_821 = [], [], []
            # alpha2s_821921 = []

        ############
        # I8 second half of the second half
        ############
        I8p22 = [elm for elm in I8_minus if p[elm[0]] > kappa and p[elm[1]] < rCs * pi[elm[0]]]
        I8cp = [elm for elm in I8c_minus if p[elm] > kappa]
        if len(I8p22) > 0 and len(I8cp) > 0:
            _grad_align = [- grad[elm[0]] + grad[elm[1]] for elm in I8p22]
            max_dir = I8p22[int(np.argmin(_grad_align))]
            alpha1_822 = max_dir[0]
            alpha3_822 = I8cp[int(np.argmin(grad[I8cp]))]
            # alpha2s_822922 = [elm[1] for elm in I8p22 if elm[0] == alpha1_822]
            alpha2s_822 = [elm[1] for elm in I8p22 if elm[0] == alpha1_822 and
                           - grad[elm[0]] + grad[elm[1]] - 2 * grad[alpha3_822] > tau]
        else:
            alpha1_822, alpha2s_822, alpha3_822 = [], [], []
            # alpha2s_821921 = []

        ############
        # I9 first half of the first half
        ############
        I9cp = [elm for elm in I8c_minus if p[elm] > kappa]
        if len(I8p11) > 0 and len(I9cp) > 0:
            alpha1_911 = alpha1_811
            alpha3_911 = I9cp[int(np.argmin(grad[I9cp]))]
            alpha2s_911 = [elm[1] for elm in I8p11 if elm[0] == alpha1_911 and
                           grad[elm[0]] - grad[elm[1]] - 2 * grad[alpha3_911] > tau]
        else:
            alpha1_911, alpha2s_911, alpha3_911 = [], [], []

        ############
        # I9 second half of the first half
        ############
        I9cp = [elm for elm in I8c_plus if p[elm] > kappa]
        if len(I8p12) > 0 and len(I9cp) > 0:
            alpha1_912 = alpha1_812
            alpha3_912 = I9cp[int(np.argmin(grad[I9cp]))]
            alpha2s_912 = [elm[1] for elm in I8p12 if elm[0] == alpha1_912 and
                           grad[elm[0]] - grad[elm[1]] - 2 * grad[alpha3_912] > tau]
        else:
            alpha1_912, alpha2s_912, alpha3_912 = [], [], []

        ############
        # I9 first half of the second half
        ############
        I9cp = [elm for elm in I8c_minus if p[elm] < C * pi[elm]]
        if len(I8p21) > 0 and len(I9cp) > 0:
            alpha1_921 = alpha1_821
            alpha3_921 = I9cp[int(np.argmax(grad[I9cp]))]
            alpha2s_921 = [elm[1] for elm in I8p21 if elm[0] == alpha1_921 and
                           - grad[elm[0]] + grad[elm[1]] + 2 * grad[alpha3_921] > tau]
        else:
            alpha1_921, alpha2s_921, alpha3_921 = [], [], []

        ############
        # I9 second half of the second half
        ############
        I9cp = [elm for elm in I8c_plus if p[elm] < C * pi[elm]]
        if len(I8p22) > 0 and len(I9cp) > 0:
            alpha1_922 = alpha1_822
            alpha3_922 = I9cp[int(np.argmax(grad[I9cp]))]
            alpha2s_922 = [elm[1] for elm in I8p22 if elm[0] == alpha1_922 and
                           - grad[elm[0]] + grad[elm[1]] + 2 * grad[alpha3_922] > tau]
        else:
            alpha1_922, alpha2s_922, alpha3_922 = [], [], []

    else:
        alpha1_61, alpha2s_61, beta_61 = [], [], []
        alpha1_62, alpha2s_62, beta_62 = [], [], []
        alpha1_71, alpha2s_71, beta_71 = [], [], []
        alpha1_72, alpha2s_72, beta_72 = [], [], []
        alpha1_811, alpha2s_811, alpha3_811 = [], [], []
        alpha1_812, alpha2s_812, alpha3_812 = [], [], []
        alpha1_821, alpha2s_821, alpha3_821 = [], [], []
        alpha1_822, alpha2s_822, alpha3_822 = [], [], []
        alpha1_911, alpha2s_911, alpha3_911 = [], [], []
        alpha1_912, alpha2s_912, alpha3_912 = [], [], []
        alpha1_921, alpha2s_921, alpha3_921 = [], [], []
        alpha1_922, alpha2s_922, alpha3_922 = [], [], []

    return I1_tilde, I2_tilde, I3_tilde, I4_tilde_1, I4_tilde_2, \
        alpha1_51, alpha2s_51, beta_51, alpha1_52, alpha2s_52, beta_52, \
        alpha1_61, alpha2s_61, beta_61, alpha1_62, alpha2s_62, beta_62, \
        alpha1_71, alpha2s_71, beta_71, alpha1_72, alpha2s_72, beta_72, \
        alpha1_811, alpha2s_811, alpha3_811, alpha1_812, alpha2s_812, alpha3_812, \
        alpha1_821, alpha2s_821, alpha3_821, alpha1_822, alpha2s_822, alpha3_822, \
        alpha1_911, alpha2s_911, alpha3_911, alpha1_912, alpha2s_912, alpha3_912, \
        alpha1_921, alpha2s_921, alpha3_921, alpha1_922, alpha2s_922, alpha3_922


def process_direction_1(I1_tilde, H, grad):
    """
    Choose a direction and the corresponding lambda for each group.
    Also gives the s value to choose which direction is the best between groups.

    Returns
    -----------
        s: float
            increase of the objective function
        d: tuple
            direction
        L: float
            step length (optimal lambda)
    """
    if len(I1_tilde) == 0:
        s, d, L = 0, [], 0
    else:
        dim1, dim2 = [elm[0] for elm in I1_tilde], [elm[1] for elm in I1_tilde]
        mult = np.diag(H[dim1, :][:, dim1] - 2 * H[dim1, :][:, dim2] + H[dim2, :][:, dim2])
        grad_diff = np.array([grad[elm[0]] - grad[elm[1]] for elm in I1_tilde])

        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [I1_tilde[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_41(I4_tilde_1, H, grad):
    """
    Choose a direction and the corresponding lambda for each group.
    Also gives the s value to choose which direction is the best between groups.

    Returns
    -----------
        s: float
            increase of the objective function
        d: tuple
            direction
        L: float
            step length (optimal lambda)
    """
    if len(I4_tilde_1) == 0:
        s, d, L = 0, [], 0
    else:
        dim1, dim2 = [elm[0] for elm in I4_tilde_1], [elm[1] for elm in I4_tilde_1]
        mult = np.diag(H[dim1, :][:, dim1] + 2 * H[dim1, :][:, dim2] + H[dim2, :][:, dim2])
        # print(mult)
        grad_diff = np.array([grad[elm[0]] + grad[elm[1]] for elm in I4_tilde_1])
        # print(grad_diff)

        s_prime_results = - grad_diff * grad_diff / mult
        # print(s_prime_results)
        lambda_primes = - grad_diff / mult
        # print(lambda_primes)
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [I4_tilde_1[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_42(I4_tilde_2, H, grad):
    if len(I4_tilde_2) == 0:
        s, d, L = 0, [], 0
    else:
        dim1, dim2 = [elm[0] for elm in I4_tilde_2], [elm[1] for elm in I4_tilde_2]
        mult = np.diag(H[dim1, :][:, dim1] + 2 * H[dim1, :][:, dim2] + H[dim2, :][:, dim2])
        grad_diff = np.array([- grad[elm[0]] - grad[elm[1]] for elm in I4_tilde_2])

        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [I4_tilde_2[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_51(alpha1, alpha2s, beta, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) - 4 * H[alpha2s, beta]
        mult = mult + H[alpha1, alpha1] - 4 * H[alpha1, beta] + 4 * H[beta, beta]
        # print(mult)

        grad_diff = np.array([grad[alpha1] + grad[elm] - 2 * grad[beta] for elm in alpha2s])
        # print(grad_diff)

        s_prime_results = - grad_diff * grad_diff / mult
        # print(s_prime_results)
        lambda_primes = - grad_diff / mult
        # print(lambda_primes)
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_52(alpha1, alpha2s, beta, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) - 4 * H[alpha2s, beta]
        mult = mult + H[alpha1, alpha1] - 4 * H[alpha1, beta] + 4 * H[beta, beta]

        grad_diff = np.array([- grad[alpha1] - grad[elm] + 2 * grad[beta] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_61(alpha1, alpha2s, beta, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = - 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) + 2 * H[alpha2s, beta]
        mult = mult + H[alpha1, alpha1] - 2 * H[alpha1, beta] + H[beta, beta]

        grad_diff = np.array([grad[alpha1] - grad[elm] - grad[beta] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_62(alpha1, alpha2s, beta, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = - 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) + 2 * H[alpha2s, beta]
        mult = mult + H[alpha1, alpha1] - 2 * H[alpha1, beta] + H[beta, beta]

        grad_diff = np.array([- grad[alpha1] + grad[elm] + grad[beta] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_71(alpha1, alpha2s, beta, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) - 2 * H[alpha2s, beta]
        mult = mult + H[alpha1, alpha1] - 2 * H[alpha1, beta] + H[beta, beta]

        grad_diff = np.array([grad[alpha1] + grad[elm] - grad[beta] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_72(alpha1, alpha2s, beta, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) - 2 * H[alpha2s, beta]
        mult = mult + H[alpha1, alpha1] - 2 * H[alpha1, beta] + H[beta, beta]

        grad_diff = np.array([- grad[alpha1] - grad[elm] + grad[beta] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_81(alpha1, alpha2s, alpha3, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = - 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) - 4 * H[alpha2s, alpha3]
        mult = mult + H[alpha1, alpha1] + 4 * H[alpha1, alpha3] + 4 * H[alpha3, alpha3]

        grad_diff = np.array([grad[alpha1] - grad[elm] + 2 * grad[alpha3] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_82(alpha1, alpha2s, alpha3, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = - 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) - 4 * H[alpha2s, alpha3]
        mult = mult + H[alpha1, alpha1] + 4 * H[alpha1, alpha3] + 4 * H[alpha3, alpha3]

        grad_diff = np.array([- grad[alpha1] + grad[elm] - 2 * grad[alpha3] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_91(alpha1, alpha2s, alpha3, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = - 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) + 4 * H[alpha2s, alpha3]
        mult = mult + H[alpha1, alpha1] - 4 * H[alpha1, alpha3] + 4 * H[alpha3, alpha3]

        grad_diff = np.array([grad[alpha1] - grad[elm] - 2 * grad[alpha3] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def process_direction_92(alpha1, alpha2s, alpha3, H, grad):
    if len(alpha2s) == 0:
        s, d, L = 0, [], 0
    else:
        mult = - 2 * H[alpha2s, alpha1] + np.diag(H[alpha2s, :][:, alpha2s]) + 4 * H[alpha2s, alpha3]
        mult = mult + H[alpha1, alpha1] - 4 * H[alpha1, alpha3] + 4 * H[alpha3, alpha3]

        grad_diff = np.array([- grad[alpha1] + grad[elm] + 2 * grad[alpha3] for elm in alpha2s])
        s_prime_results = - grad_diff * grad_diff / mult
        lambda_primes = - grad_diff / mult
        feasible_idx = [i for i, elm in enumerate(lambda_primes) if elm is not None and elm > 0 and not np.isinf(elm)]

        if len(feasible_idx) > 0:
            indices = [alpha2s[elm] for elm in feasible_idx]
            s_prime_results = s_prime_results[feasible_idx]
            lambda_primes = lambda_primes[feasible_idx]
            t = int(np.argmax(s_prime_results))
            s, d, L = s_prime_results[t], indices[t], lambda_primes[t]
        else:
            s, d, L = 0, [], 0
    return s, d, L


def clipping_func(p, pi, d, L, u, n, m, C, rCs):
    """
    General Comments: This function clips the step size that the
    algorithm is about to take to ensure that the next point is
    within the constraints of the objective function

    Parameters
    ------------
        p: numpy.array
            current point (in the direction d) (2-dim or 3-dim)
        pi: numpy.array
            confidence level (n-dim)
        d: tuple
            direction (2-dim or 3-dim)
        L: float
            optimal lambda
        u: numpy.array
            unit direction?
        n: int
        m: int
        C: float
        rCs: float
            rho * C^(star)

    Returns
    ------------
        L_new: float
            clipped version of lambda
    """
    assert len(u) == len(d)
    L_new = L
    # Lower bound
    for i in range(len(u)):
        mini = abs(p[i] / u[i])
        if u[i] < 0 and L_new > mini:
            L_new = mini

    # upper bound
    for i in range(len(u)):
        if d[i] < m:  # rho * Cs * pi_i
            mini = (rCs * pi[i] - p[i]) / abs(u[i])
            if u[i] > 0 and L_new > mini:
                L_new = mini
        elif d[i] < n:  # m+1 ~ n, constraint = C * pi_i
            mini = (C * pi[i] - p[i]) / abs(u[i])
            if u[i] > 0 and L_new > mini:
                L_new = mini
        else:
            pass

    return L_new




