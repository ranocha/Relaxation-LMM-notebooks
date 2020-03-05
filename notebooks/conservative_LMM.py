import numpy as np
from scipy.optimize import root, fsolve, newton, brentq, bisect


def compute_eocs(dts, errors):
    eocs = np.zeros(len(errors) - 1)
    for i in np.arange(len(errors) - 1):
        eocs[i] = np.log(errors[i+1] / errors[i]) / np.log(dts[i+1] / dts[i])
    return eocs

def compute_eoc(dts_, errors_):
    dts = np.array(dts_)
    errors = np.array(errors_)
    idx = ~np.isnan(np.array(errors))
    if np.any(idx):
        return np.mean(compute_eocs(dts[idx], errors[idx]))
    else:
        return np.nan


def etaL2(u):
    """
    The standard inner product norm (L^2) entropy.
    """
    return 0.5 * np.dot(u, u)

def detaL2(u):
    """
    The derivative of the standard inner product norm (L^2) entropy.
    """
    return u

def compute_single_result(f, u, t_final, dt, scheme, num_steps, **kwargs):
    """
    Compute the numerical solution obtained by the LMM `scheme` which
    uses `num_steps` previous step/derivative values for the ODE
    given by the right hand side `f` with analytical solution or
    starting procedure `u` and a time step `dt`.
    """
    t0 = 0.
    u0 = u(t0)
    t1 = dt
    u1 = u(t1)
    t2 = 2*dt
    u2 = u(t2)
    t3 = 3*dt
    u3 = u(t3)
    t4 = 4*dt
    u4 = u(t4)

    if num_steps == 2:
        tt, uu, gamma = scheme(f, t_final, t0, u0, t1, u1,
                               return_gamma=True, **kwargs)
    elif num_steps == 3:
        tt, uu, gamma = scheme(f, t_final, t0, u0, t1, u1, t2, u2,
                               return_gamma=True, **kwargs)
    elif num_steps == 4:
        tt, uu, gamma = scheme(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                               return_gamma=True, **kwargs)
    elif num_steps == 5:
        tt, uu, gamma = scheme(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3, t4, u4,
                               return_gamma=True, **kwargs)
    else:
        raise Exception("num_steps == %d not implemented yet." % (num_steps))

    return tt, uu, gamma


def compute_convergence_data(f, u, t_final, dts, scheme, num_steps,
                             error_idx=None, fixed_coefficients_twice=False,
                             **kwargs):
    """
    Compute the numerical errors obtained by the LMM `scheme` which
    uses `num_steps` previous step/derivative values for the ODE
    given by the right hand side `f` with analytical solution or
    starting procedure `u` and time steps `dts`.
    """
    error_b = []
    gammaM1_b = []
    error_p = []
    gammaM1_p = []
    error_rf = []
    gammaM1_rf = []
    error_rff = []
    gammaM1_rff = []
    error_ra = []
    gammaM1_ra = []
    error_idt = []
    gammaM1_idt = []

    for dt in dts:
        try:
            tt, uu, gamma = compute_single_result(f, u, t_final, dt, scheme, num_steps,
                                                  projection=False, relaxation=False,
                                                  adapt_dt=False, adapt_coefficients=False,
                                                  **kwargs)
            if np.any(error_idx == None):
                error_b.append( np.linalg.norm(uu[-1] - u(tt[-1])) )
            else:
                error_b.append( np.linalg.norm(uu[-1][error_idx] - u(tt[-1])[error_idx]) )
            gammaM1_b.append( np.linalg.norm(gamma - 1, ord=np.inf) )
        except:
            error_b.append(np.nan)
            gammaM1_b.append(np.nan)

        try:
            tt, uu, gamma = compute_single_result(f, u, t_final, dt, scheme, num_steps,
                                                  projection=True, relaxation=False,
                                                  adapt_dt=False, adapt_coefficients=False,
                                                  **kwargs)
            if np.any(error_idx == None):
                error_p.append( np.linalg.norm(uu[-1] - u(tt[-1])) )
            else:
                error_p.append( np.linalg.norm(uu[-1][error_idx] - u(tt[-1])[error_idx]) )
            gammaM1_p.append( np.linalg.norm(gamma - 1, ord=np.inf) )
        except:
            error_p.append(np.nan)
            gammaM1_p.append(np.nan)

        try:
            tt, uu, gamma = compute_single_result(f, u, t_final, dt, scheme, num_steps,
                                                  projection=False, relaxation=True,
                                                  adapt_dt=True, adapt_coefficients=False,
                                                  **kwargs)
            if np.any(error_idx == None):
                error_rf.append( np.linalg.norm(uu[-1] - u(tt[-1])) )
            else:
                error_rf.append( np.linalg.norm(uu[-1][error_idx] - u(tt[-1])[error_idx]) )
            gammaM1_rf.append( np.linalg.norm(gamma - 1, ord=np.inf) )
        except:
            error_rf.append(np.nan)
            gammaM1_rf.append(np.nan)

        if fixed_coefficients_twice:
            try:
                tt, uu, gamma = compute_single_result(f, u, t_final, dt, scheme, num_steps,
                                                      projection=False, relaxation=True,
                                                      adapt_dt=True, adapt_coefficients=False,
                                                      fixed_coefficient_fix=True,
                                                      **kwargs)
                if np.any(error_idx == None):
                    error_rff.append( np.linalg.norm(uu[-1] - u(tt[-1])) )
                else:
                    error_rff.append( np.linalg.norm(uu[-1][error_idx] - u(tt[-1])[error_idx]) )
                gammaM1_rff.append( np.linalg.norm(gamma - 1, ord=np.inf) )
            except:
                error_rff.append(np.nan)
                gammaM1_rff.append(np.nan)

        try:
            tt, uu, gamma = compute_single_result(f, u, t_final, dt, scheme, num_steps,
                                                  projection=False, relaxation=True,
                                                  adapt_dt=True, adapt_coefficients=True,
                                                  **kwargs)
            if np.any(error_idx == None):
                error_ra.append( np.linalg.norm(uu[-1] - u(tt[-1])) )
            else:
                error_ra.append( np.linalg.norm(uu[-1][error_idx] - u(tt[-1])[error_idx]) )
            gammaM1_ra.append( np.linalg.norm(gamma - 1, ord=np.inf) )
        except:
            error_ra.append(np.nan)
            gammaM1_ra.append(np.nan)

        try:
            tt, uu, gamma = compute_single_result(f, u, t_final, dt, scheme, num_steps,
                                                  projection=False, relaxation=True,
                                                  adapt_dt=False, adapt_coefficients=False,
                                                  **kwargs)
            if np.any(error_idx == None):
                error_idt.append( np.linalg.norm(uu[-1] - u(tt[-1])) )
            else:
                error_idt.append( np.linalg.norm(uu[-1][error_idx] - u(tt[-1])[error_idx]) )
            gammaM1_idt.append( np.linalg.norm(gamma - 1, ord=np.inf) )
        except:
            error_idt.append(np.nan)
            gammaM1_idt.append(np.nan)

    if fixed_coefficients_twice:
        return error_b, gammaM1_b, error_p, gammaM1_p, error_rf, gammaM1_rf, error_rff, gammaM1_rff, error_ra, gammaM1_ra, error_idt, gammaM1_idt
    else:
        return error_b, gammaM1_b, error_p, gammaM1_p, error_rf, gammaM1_rf, error_ra, gammaM1_ra, error_idt, gammaM1_idt



class SolveForGammaException(BaseException):
    def __init__(self, message, data):
        self.message = message
        self.data = data


def conservative_relaxation_solve(eta, deta, u_old, eta_old, u_new, old_gamma, method, tol, maxiter):
    """
    Compute the relaxation factor `gamma` for a step from `u_old` to `u_new`
    and the invariant `eta` with derivative `deta`.
    The initial guess of `gamma` is `old_gamma` and the solution is obtained
    by `method` using the tolerance `tol` and not more than `maxiter` iterations.
    Possible `method`s are
    - "newton"
    - "simplified Newton"
    - "brentq"
    - "bisect"
    - "hybr"
    - "lm"
    - "broyden1"
    - "broyden2"
    - "anderson"
    - "linearmixing"
    - "diagbroyden"
    - "excitingmixing"
    - "krylov"
    - "df-sane"
    """
    if eta == etaL2:
        # assume eta == squared Euclidean inner product
        # gamma = -2 * np.dot(u_old, u_new - u_old) / np.dot(u_new - u_old, u_new - u_old)
        a = eta(u_old) - eta_old
        b = np.dot(u_old, u_new - u_old)
        c = eta(u_new - u_old)
        if np.abs(a) < 1.0e-14:
            gamma = -b / c
        else:
            gamma = (-b + np.sqrt(b*b - 4*a*c)) / (2*c)
        return gamma

    r = lambda gamma: eta(u_old + gamma * (u_new - u_old)) - eta_old
    if method == "newton":
        gamma = newton(r, old_gamma, tol=tol, maxiter=maxiter)
        success = True
        msg = "Newton method did not converge"
    elif method == "simplified Newton":
        eta_prime = deta(u_new)
        denominator = np.dot(eta_prime, u_new - u_old)
        gamma = old_gamma
        delta_gamma = 10. * tol
        iter = 0
        val = r(gamma)
        while np.abs(val) > tol and iter < maxiter:
            delta_gamma = -val / denominator
            gamma += delta_gamma
            iter += 1
            val = r(gamma)

        u_new = u_old + gamma * (u_new - u_old)
        success = iter < maxiter
        msg = "'simplified Newton' method did not converge"
    elif method == "brentq" or method == "bisect":
        left = 0.9 * old_gamma
        right = 1.1 * old_gamma
        left_right_iter = 0
        while r(left) * r(right) > 0:
            left *= 0.9
            right *= 1.1
            left_right_iter += 1
            if left_right_iter > 100:
                raise SolveForGammaException(
                    "No suitable bounds found after %d iterations.\nLeft = %e; r(left) = %e\nRight = %e; r(right) = %e\n"%(
                        left_right_iter, left, r(left), right, r(right)),
                    u_old)

        if method == "brentq":
            gamma = brentq(r, left, right, xtol=tol, maxiter=maxiter)
        else:
            gamma = bisect(r, left, right, xtol=tol, maxiter=maxiter)
        success = True
        msg = "%s method did not converge"%method
    else:
        # Possible methods:
        # hybr, lm, broyden1, broyden2, anderson, linearmixing, diagbroyden
        # excitingmixing, krylov, df-sane
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
        sol = root(r, old_gamma, method=method, tol=tol,
                    options={'xtol': tol, 'maxiter': maxiter})
        gamma = np.sum(sol.x); success = sol.success; msg = sol.message

    if success == False:
        print('Warning: fsolve did not converge.')
        print(gamma)
        print(msg)

    if gamma <= 0:
        print('Warning: gamma is negative.')

    return gamma

def cons_or_diss_relaxation_solve(eta, deta, eta_est, u_old, eta_old, u_new, old_gamma, method, tol, maxiter):
    """
    Compute the relaxation factor `gamma` for a step from `u_old` to `u_new`
    and the general (conserved or dissipated) quantity of interest `eta`
    with derivative `deta`. The previous value of `eta` is `eta_old`, the
    desired estimate is `eta_est`.
    The initial guess of `gamma` is `old_gamma` and the solution is obtained
    by `method` using the tolerance `tol` and not more than `maxiter` iterations.
    Possible `method`s are
    - "newton"
    - "simplified Newton"
    - "brentq"
    - "bisect"
    - "hybr"
    - "lm"
    - "broyden1"
    - "broyden2"
    - "anderson"
    - "linearmixing"
    - "diagbroyden"
    - "excitingmixing"
    - "krylov"
    - "df-sane"
    """
    if eta == etaL2:
        # assume eta == squared Euclidean inner product
        # gamma = 2 * ( eta_est - eta_old - np.dot(u_old, u_new - u_old) ) / np.dot(u_new - u_old, u_new - u_old)
        a = eta(u_old) - eta_old
        b = np.dot(u_old, u_new - u_old) - eta_est + eta_old
        c = eta(u_new - u_old)
        if np.abs(a) < 1.0e-14:
            gamma = -b / c
        else:
            gamma = (-b + np.sqrt(b*b - 4*a*c)) / (2*c)
        return gamma

    r = lambda gamma: eta(u_old + gamma * (u_new - u_old)) - eta_old - gamma * (eta_est - eta_old)
    if method == "newton":
        gamma = newton(r, old_gamma, tol=tol, maxiter=maxiter)
        success = True
        msg = "Newton method did not converge"
    elif method == "simplified Newton":
        eta_prime = deta(u_new)
        denominator = np.dot(eta_prime, u_new - u_old) - (eta_est - eta_old)
        gamma = old_gamma
        delta_gamma = 10. * tol
        iter = 0
        val = r(gamma)
        while np.abs(val) > tol and iter < maxiter:
            delta_gamma = -val / denominator
            gamma += delta_gamma
            iter += 1
            val = r(gamma)

        u_new = u_old + gamma * (u_new - u_old)
        success = iter < maxiter
        msg = "'simplified Newton' method did not converge"
    elif method == "brentq" or method == "bisect":
        left = 0.9 * old_gamma
        right = 1.1 * old_gamma
        left_right_iter = 0
        while r(left) * r(right) > 0:
            left *= 0.9
            right *= 1.1
            left_right_iter += 1
            if left_right_iter > 100:
                raise SolveForGammaException(
                    "No suitable bounds found after %d iterations.\nLeft = %e; r(left) = %e\nRight = %e; r(right) = %e\n"%(
                        left_right_iter, left, r(left), right, r(right)),
                    u_old)

        if method == "brentq":
            gamma = brentq(r, left, right, xtol=tol, maxiter=maxiter)
        else:
            gamma = bisect(r, left, right, xtol=tol, maxiter=maxiter)
        success = True
        msg = "%s method did not converge"%method
    else:
        # Possible methods:
        # hybr, lm, broyden1, broyden2, anderson, linearmixing, diagbroyden
        # excitingmixing, krylov, df-sane
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
        sol = root(r, old_gamma, method=method, tol=tol,
                    options={'xtol': tol, 'maxiter': maxiter})
        gamma = np.sum(sol.x); success = sol.success; msg = sol.message

    if success == False:
        print('Warning: fsolve did not converge.')
        print(gamma)
        print(msg)

    if gamma <= 0:
        print('Warning: gamma is negative.')

    return gamma


def conservative_projection_solve(eta, deta, u_old, eta_old, u_new, method, tol, maxiter):
    """
    Compute the projection factor `gamma` and the projected value for a step from
    `u_old` to `u_new` and the invariant `eta` with derivative `deta`.
    The solution is obtained by `method` using the tolerance `tol` and not more
    than `maxiter` iterations. Possible `method`s are
    - "simplified Newton"
    """
    if eta == etaL2:
        # assume eta == 1/2 * squared Euclidean inner product
        factor = np.sqrt( eta_old / eta(u_new) )
        gamma = factor
        u_new = factor * u_new
        return gamma, u_new

    if method == "simplified Newton":
        eta_prime = deta(u_new)
        denominator = (np.dot(eta_prime, eta_prime) + 1.e-16)
        gamma_p1 = 0.
        delta_gamma = 10.
        iter = 0
        while delta_gamma > tol and iter < maxiter:
            delta_gamma = -(eta(u_new + gamma_p1*eta_prime) - eta_old) / denominator
            gamma_p1 += delta_gamma
            iter += 1

        gamma = gamma_p1 + 1.0
        u_new = u_new + gamma_p1 * eta_prime
        success = iter < maxiter
        msg = "'simplified Newton' method did not converge"
    else:
        raise Exception("Method %s not implemented yet." % (method))

    if success == False:
        print('Warning: fsolve did not converge.')
        print(msg)

    return gamma, u_new

def cons_or_diss_projection_solve(eta, deta, eta_est, u_old, eta_old, u_new, method, tol, maxiter):
    """
    Compute the projection factor `gamma` and the projected value for a step from
    `u_old` to `u_new` and the general (conserved or dissipated) quantity of interest
    `eta` with derivative `deta`. The previous value of `eta` is `eta_old`, the
    desired estimate is `eta_est`.
    The solution is obtained by `method` using the tolerance `tol` and not more
    than `maxiter` iterations. Possible `method`s are
    - "simplified Newton"
    """
    if eta == etaL2:
        # assume eta == 1/2 * squared Euclidean inner product
        factor = np.sqrt( eta_est / eta(u_new) )
        gamma = factor
        u_new = factor * u_new
        return gamma, u_new

    if method == "simplified Newton":
        eta_prime = deta(u_new)
        denominator = (np.dot(eta_prime, eta_prime) + 1.e-16)
        gamma_p1 = 0.
        delta_gamma = 10.
        iter = 0
        while delta_gamma > tol and iter < maxiter:
            delta_gamma = -(eta(u_new + gamma_p1*eta_prime) - eta_est) / denominator
            gamma_p1 += delta_gamma
            iter += 1

        gamma = gamma_p1 + 1.0
        u_new = u_new + gamma_p1 * eta_prime
        success = iter < maxiter
        msg = "'simplified Newton' method did not converge"
    else:
        raise Exception("Method %s not implemented yet." % (method))

    if success == False:
        print('Warning: fsolve did not converge.')
        print(msg)

    return gamma, u_new


def conservative_LMM(f, t_final, initial_t, initial_u,
                     fixed_step, adaptive_step,
                     idx_u_old=-1,
                     eta=etaL2, deta=detaL2,
                     return_gamma=False,
                     projection=False, relaxation=False,
                     adapt_dt=False, adapt_coefficients=False,
                     fixed_coefficient_fix=False,
                     method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    uu = [u for u in initial_u]
    ff = [f(u) for u in initial_u]
    tt = [t for t in initial_t]
    if len(uu) != len(tt):
        raise Exception("You must provide the same number of initial values for `t` and `u`.")
    if len(uu) < 2:
        raise Exception("You must provide at least 2 initial values for `t` and `u`.")

    h = tt[1] - tt[0]
    old_omega = [(tt[i+1] - tt[i]) / h for i in np.arange(len(tt)-1)]
    old_gamma = [1.0 for i in np.arange(len(tt)-1)]

    old_eta = [eta(uu[i]) for i in np.arange(len(uu))]

    if relaxation and projection:
        raise Exception("Use either relaxation or projection, not both.")

    if relaxation and method == None:
        method = "brentq"
    elif projection and method == None:
        method = "simplified Newton"

    if callable(idx_u_old):
        old_weights_func = idx_u_old
    elif not hasattr(idx_u_old, '__iter__'):
        old_weights = [0.0 for u in uu]
        old_weights[idx_u_old] = 1.0
        old_weights_func = lambda old_omega: old_weights
    else:
        old_weights_func = lambda old_omega: idx_u_old

    t = tt[-1]
    gammas = [1.0 for t in initial_t]
    step = 0
    while t < t_final and step < maxsteps:
        step += 1

        if relaxation and adapt_coefficients:
            u_new = adaptive_step(uu, ff, h, old_omega)
        else:
            u_new = fixed_step(uu, ff, h)

        old_weights = old_weights_func(old_omega)
        u_old = sum(old_weights[idx]*uu[idx] for idx in np.arange(-len(old_weights), 0))
        eta_old = sum(old_weights[idx]*old_eta[idx] for idx in np.arange(-len(old_weights), 0))
        if projection:
            gamma, u_new = conservative_projection_solve(eta, deta, u_old, eta_old, u_new, method, tol, maxiter)
        elif relaxation:
            gamma = conservative_relaxation_solve(eta, deta, u_old, eta_old, u_new, old_gamma[-1], method, tol, maxiter)
            u_new = u_old + gamma * (u_new - u_old)
            for i in np.arange(-len(old_gamma), -1):
                old_gamma[i] = old_gamma[i+1]
            old_gamma[-1] = gamma
        else:
            gamma = 1.0

        if return_gamma:
            gammas.append(gamma)

        uu.append(u_new)
        if relaxation and adapt_dt:
            t_old = np.sum([old_weights[idx]*tt[idx] for idx in np.arange(-len(old_weights), 0)])
            if fixed_coefficient_fix and not adapt_coefficients:
                t_diff = -h * np.sum([idx*old_weights[idx] for idx in np.arange(-len(old_weights), 0)])
            else:
                t_diff = tt[-1] + h - t_old
            t = t_old + gamma * t_diff

            if adapt_coefficients:
                # new_omega = -idx_u_old*gamma - np.sum([old_omega[i] for i in np.arange(-1, idx_u_old, -1)])
                new_omega = (t - tt[-1]) / h
                for i in np.arange(-len(old_omega), -1):
                    old_omega[i] = old_omega[i+1]
                old_omega[-1] = new_omega

            if gamma < 1.0e-14:
                raise Exception("gamma = %.2e is too small in step %d!" % (gamma, step))
        else:
            t += h
        tt.append(t)

        for i in np.arange(-len(ff), -1):
            ff[i] = ff[i+1]
        ff[-1] = f(u_new)

        for i in np.arange(-len(old_eta), -1):
            old_eta[i] = old_eta[i+1]
        old_eta[-1] = eta(u_new)

    if return_gamma:
        return np.array(tt), uu, np.array(gammas)
    else:
        return np.array(tt), uu


def cons_or_diss_LMM(f, t_final, initial_t, initial_u,
                     fixed_step, adaptive_step,
                     fixed_estimate, adaptive_estimate,
                     idx_u_old=-1,
                     eta=etaL2, deta=detaL2,
                     return_gamma=False,
                     projection=False, relaxation=False,
                     adapt_dt=False, adapt_coefficients=False,
                     fixed_coefficient_fix=False,
                     method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    uu = [u for u in initial_u]
    ff = [f(u) for u in initial_u]
    tt = [t for t in initial_t]
    if len(uu) != len(tt):
        raise Exception("You must provide the same number of initial values for `t` and `u`.")
    if len(uu) < 2:
        raise Exception("You must provide at least 2 initial values for `t` and `u`.")

    h = tt[1] - tt[0]
    old_omega = [(tt[i+1] - tt[i]) / h for i in np.arange(len(tt)-1)]
    old_gamma = [1.0 for i in np.arange(len(tt)-1)]

    old_eta = [eta(uu[i]) for i in np.arange(len(uu))]
    old_deta_f = [np.dot(deta(uu[i]), ff[i]) for i in np.arange(len(uu))]

    if relaxation and projection:
        raise Exception("Use either relaxation or projection, not both.")

    if relaxation and method == None:
        method = "brentq"
    elif projection and method == None:
        method = "simplified Newton"

    if callable(idx_u_old):
        old_weights_func = idx_u_old
    elif not hasattr(idx_u_old, '__iter__'):
        old_weights = [0.0 for u in uu]
        old_weights[idx_u_old] = 1.0
        old_weights_func = lambda old_omega: old_weights
    else:
        old_weights_func = lambda old_omega: idx_u_old

    t = tt[-1]
    gammas = [1.0 for t in initial_t]
    step = 0
    while t < t_final and step < maxsteps:
        step += 1

        if relaxation and adapt_coefficients:
            u_new = adaptive_step(uu, ff, h, old_omega)
            eta_est = adaptive_estimate(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega)
        else:
            u_new = fixed_step(uu, ff, h)
            eta_est = fixed_estimate(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h)

        old_weights = old_weights_func(old_omega)
        u_old = sum(old_weights[idx]*uu[idx] for idx in np.arange(-len(old_weights), 0))
        eta_old = sum(old_weights[idx]*old_eta[idx] for idx in np.arange(-len(old_weights), 0))
        if projection:
            gamma, u_new = cons_or_diss_projection_solve(eta, deta, eta_est, u_old, eta_old, u_new, method, tol, maxiter)
        elif relaxation:
            gamma = cons_or_diss_relaxation_solve(eta, deta, eta_est, u_old, eta_old, u_new, old_gamma[-1], method, tol, maxiter)
            u_new = u_old + gamma * (u_new - u_old)
            for i in np.arange(-len(old_gamma), -1):
                old_gamma[i] = old_gamma[i+1]
            old_gamma[-1] = gamma
        else:
            gamma = 1.0

        if return_gamma:
            gammas.append(gamma)

        uu.append(u_new)
        if relaxation and adapt_dt:
            t_old = np.sum([old_weights[idx]*tt[idx] for idx in np.arange(-len(old_weights), 0)])
            if fixed_coefficient_fix and not adapt_coefficients:
                t_diff = -h * np.sum([idx*old_weights[idx] for idx in np.arange(-len(old_weights), 0)])
            else:
                t_diff = tt[-1] + h - t_old
            t = t_old + gamma * t_diff

            if adapt_coefficients:
                # new_omega = -idx_u_old*gamma - np.sum([old_omega[i] for i in np.arange(-1, idx_u_old, -1)])
                new_omega = (t - tt[-1]) / h
                for i in np.arange(-len(old_omega), -1):
                    old_omega[i] = old_omega[i+1]
                old_omega[-1] = new_omega

            if gamma < 1.0e-14:
                raise Exception("gamma = %.2e is too small in step %d!" % (gamma, step))
        else:
            t += h
        tt.append(t)

        for i in np.arange(-len(ff), -1):
            ff[i] = ff[i+1]
        ff[-1] = f(u_new)

        for i in np.arange(-len(old_eta), -1):
            old_eta[i] = old_eta[i+1]
        old_eta[-1] = eta(u_new)

        for i in np.arange(-len(old_deta_f), -1):
            old_deta_f[i] = old_deta_f[i+1]
        old_deta_f[-1] = np.dot(deta(u_new), ff[-1])

    if return_gamma:
        return np.array(tt), uu, np.array(gammas)
    else:
        return np.array(tt), uu


# explicit Adams methods
def fixed_step_AB2(uu, ff, h):
    du_new = (
            1.5
        ) * ff[-1] + (
            -0.5
        ) * ff[-2]
    u_new = uu[-1] + h * du_new
    return u_new

def fixed_estimate_AB2(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    du1int = (
            0.625
        ) * ff[-1] + (
            -0.125
        ) * ff[-2]
    u1int = uu[-1] + h * du1int

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def adaptive_step_AB2(uu, ff, h, old_omega):
    om1 = old_omega[-1]
    du_new = (
            1 + 1/(2.*om1)
        ) * ff[-1] + (
            -1/(2.*om1)
        ) * ff[-2]

    u_new = uu[-1] + h * du_new
    return u_new

def adaptive_estimate_AB2(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om1 = old_omega[-1]
    du1int = (
            (4 + 1./om1)/8.
        ) * ff[-1] + (
            -1/(8.*om1)
        ) * ff[-2]
    u1int = uu[-1] + h * du1int

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def conservative_AB2(f, t_final, t0, u0, t1, u1,
                     **kwargs):
    return conservative_LMM(f, t_final, [t0, t1], [u0, u1],
                            fixed_step_AB2, adaptive_step_AB2,
                            **kwargs)

def cons_or_diss_AB2(f, t_final, t0, u0, t1, u1,
                     **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1], [u0, u1],
                            fixed_step_AB2, adaptive_step_AB2,
                            fixed_estimate_AB2, adaptive_estimate_AB2,
                            **kwargs)


def fixed_step_AB3(uu, ff, h):
    u_new = uu[-1] + h * ( (23./12.)*ff[-1] - (16./12.)*ff[-2] + (5./12.)*ff[-3])
    return u_new

def fixed_estimate_AB3(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    du1int = (
            0.24639141243202012
        ) * ff[-1] + (
            -0.04780399468440579
        ) * ff[-2] + (
            0.012737447657572745
        ) * ff[-3]
    u1int = uu[-1] + h * du1int
    du2int = (
            1.3369419209013131
        ) * ff[-1] + (
            -0.7855293386489275
        ) * ff[-2] + (
            0.23726255234242727
        ) * ff[-3]
    u2int = uu[-1] + h * du2int

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_AB3(uu, ff, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    du_new = (
            (2 + 3*om2 + 6*om1 * (1 + om1 + om2)) / (6*om1 * (om1 + om2))
        ) * ff[-1] + (
            -(2 + 3 * (om1 + om2)) / (6 * om1 * om2)
        ) * ff[-2] + (
            (2 + 3*om1) / (6 * om2 * (om1 + om2))
        ) * ff[-3]

    u_new = uu[-1] + h * du_new
    return u_new

def adaptive_estimate_AB3(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    du1int = (
            -(-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om2 + 18*om1*(-2 + np.sqrt(3) + (-3 + np.sqrt(3))*om1 + (-3 + np.sqrt(3))*om2))/(108.*om1*(om1 + om2))
        ) * ff[-1] + (
            (-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om1 + 9*(-2 + np.sqrt(3))*om2)/(108.*om1*om2)
        ) * ff[-2] + (
            (9 - 5*np.sqrt(3) - 9*(-2 + np.sqrt(3))*om1)/(108.*om2*(om1 + om2))
        ) * ff[-3]
    u1int = uu[-1] + h * du1int
    du2int = (
            (9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om2 + 18*om1*(2 + np.sqrt(3) + (3 + np.sqrt(3))*om1 + (3 + np.sqrt(3))*om2))/(108.*om1*(om1 + om2))
        ) * ff[-1] + (
            -(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om1 + 9*(2 + np.sqrt(3))*om2)/(108.*om1*om2)
        ) * ff[-2] + (
            (9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om1)/(108.*om2*(om1 + om2))
        ) * ff[-3]
    u2int = uu[-1] + h * du2int

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_AB3(f, t_final, t0, u0, t1, u1, t2, u2,
                     **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_AB3, adaptive_step_AB3,
                            **kwargs)

def cons_or_diss_AB3(f, t_final, t0, u0, t1, u1, t2, u2,
                     **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_AB3, adaptive_step_AB3,
                            fixed_estimate_AB3, adaptive_estimate_AB3,
                            **kwargs)


def fixed_step_AB4(uu, ff, h):
    du_new = (
            2.2916666666666665
        ) * ff[-1] + (
            -2.4583333333333335
        ) * ff[-2] + (
            1.5416666666666667
        ) * ff[-3] + (
            -0.375
        ) * ff[-4]
    u_new = uu[-1] + h * du_new
    return u_new

def fixed_estimate_AB4(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    du1int = (
            0.2554904416411566
        ) * ff[-1] + (
            -0.07510108231181513
        ) * ff[-2] + (
            0.04003453528498212
        ) * ff[-3] + (
            -0.009099029209136444
        ) * ff[-4]
    u1int = uu[-1] + h * du1int
    du2int = (
            1.5384910398403246
        ) * ff[-1] + (
            -1.3901766954659625
        ) * ff[-2] + (
            0.8419099091594622
        ) * ff[-3] + (
            -0.2015491189390117
        ) * ff[-4]
    u2int = uu[-1] + h * du2int

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_AB4(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    du_new = (
            (3 + 8*om2 + 4*om3 + 6*(2*(om1*om1*om1) + om2*(om2 + om3) + 2*om1*(1 + om2)*(1 + om2 + om3) + om1*om1*(3 + 4*om2 + 2*om3)))/(12.*om1*(om1 + om2)*(om1 + om2 + om3))
        ) * ff[-1] + (
            -(3 + 6*(om1*om1) + 8*om2 + 4*om3 + 6*om2*(om2 + om3) + 2*om1*(4 + 6*om2 + 3*om3))/(12.*om1*om2*(om2 + om3))
        ) * ff[-2] + (
            (3 + 4*om2 + 4*om3 + 2*om1*(4 + 3*om1 + 3*om2 + 3*om3))/(12.*om2*(om1 + om2)*om3)
        ) * ff[-3] + (
            -(3 + 4*om2 + 2*om1*(4 + 3*om1 + 3*om2))/(12.*om3*(om2 + om3)*(om1 + om2 + om3))
        ) * ff[-4]

    u_new = uu[-1] + h * du_new
    return u_new

def adaptive_estimate_AB4(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    du1int = (
            (21 - 12*np.sqrt(3) + 8*(9 - 5*np.sqrt(3))*om2 + 4*(9 - 5*np.sqrt(3))*om3 + 12*(-6*(-3 + np.sqrt(3))*(om1*om1*om1) - 3*(-2 + np.sqrt(3))*om2*(om2 + om3) + 3*(om1*om1)*(6 - 3*np.sqrt(3) - 4*(-3 + np.sqrt(3))*om2 - 2*(-3 + np.sqrt(3))*om3) + om1*(9 - 5*np.sqrt(3) - 6*(-2 + np.sqrt(3))*om3 - 6*om2*(2*(-2 + np.sqrt(3)) + (-3 + np.sqrt(3))*om2 + (-3 + np.sqrt(3))*om3))))/(432.*om1*(om1 + om2)*(om1 + om2 + om3))
        ) * ff[-1] + (
            (3*(-7 + 4*np.sqrt(3)) + 8*(-9 + 5*np.sqrt(3))*om2 + 4*(-9 + 5*np.sqrt(3))*om3 + 4*(9*(-2 + np.sqrt(3))*(om1*om1) + 9*(-2 + np.sqrt(3))*om2*(om2 + om3) + om1*(2*(-9 + 5*np.sqrt(3)) + 18*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3)))/(432.*om1*om2*(om2 + om3))
        ) * ff[-2] + (
            (21 - 12*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om2 + 4*(9 - 5*np.sqrt(3))*om3 + 4*om1*(18 - 10*np.sqrt(3) - 9*(-2 + np.sqrt(3))*om1 - 9*(-2 + np.sqrt(3))*om2 - 9*(-2 + np.sqrt(3))*om3))/(432.*om2*(om1 + om2)*om3)
        ) * ff[-3] + (
            (3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om2 + 4*om1*(2*(-9 + 5*np.sqrt(3)) + 9*(-2 + np.sqrt(3))*om1 + 9*(-2 + np.sqrt(3))*om2))/(432.*om3*(om2 + om3)*(om1 + om2 + om3))
        ) * ff[-4]
    u1int = uu[-1] + h * du1int
    du2int = (
            (3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 12*(6*(3 + np.sqrt(3))*(om1*om1*om1) + 3*(2 + np.sqrt(3))*om2*(om2 + om3) + 3*(om1*om1)*(3*(2 + np.sqrt(3)) + 4*(3 + np.sqrt(3))*om2 + 2*(3 + np.sqrt(3))*om3) + om1*(9 + 5*np.sqrt(3) + 6*(2 + np.sqrt(3))*om3 + 6*om2*(2*(2 + np.sqrt(3)) + (3 + np.sqrt(3))*om2 + (3 + np.sqrt(3))*om3))))/(432.*om1*(om1 + om2)*(om1 + om2 + om3))
        ) * ff[-1] + (
            -(3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(9*(2 + np.sqrt(3))*(om1*om1) + 9*(2 + np.sqrt(3))*om2*(om2 + om3) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3)))/(432.*om1*om2*(om2 + om3))
        ) * ff[-2] + (
            (3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*om1*(2*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om1 + 9*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3))/(432.*om2*(om1 + om2)*om3)
        ) * ff[-3] + (
            -(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om2 + 4*om1*(2*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om1 + 9*(2 + np.sqrt(3))*om2))/(432.*om3*(om2 + om3)*(om1 + om2 + om3))
        ) * ff[-4]
    u2int = uu[-1] + h * du2int

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_AB4(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                     **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_AB4, adaptive_step_AB4,
                            **kwargs)

def cons_or_diss_AB4(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                     **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_AB4, adaptive_step_AB4,
                            fixed_estimate_AB4, adaptive_estimate_AB4,
                            **kwargs)


def fixed_step_AB5(uu, ff, h):
    du_new = (
            2.640277777777778
        ) * ff[-1] + (
            -3.852777777777778
        ) * ff[-2] + (
            3.6333333333333333
        ) * ff[-3] + (
            -1.7694444444444444
        ) * ff[-4] + (
            0.3486111111111111
        ) * ff[-5]
    u_new = uu[-1] + h * du_new
    return u_new

def adaptive_step_AB5(uu, ff, h, old_omega):
    om4 = old_omega[-4]
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    du_new = (
            (3*(4 + 15*om2 + 10*om3 + 5*om4) + 10*(6*(om1*om1*om1*om1) + 3*(om2*om2*om2) + 4*om2*om4 + 2*om3*(om3 + om4) + 6*om1*(1 + om2)*(1 + om2 + om3)*(1 + om2 + om3 + om4) + 3*(om2*om2)*(2 + 2*om3 + om4) + 6*(om1*om1*om1)*(2 + 3*om2 + 2*om3 + om4) + om2*om3*(8 + 3*om3 + 3*om4) + 3*(om1*om1)*(4 + 6*(om2*om2) + 3*om4 + 2*om3*(3 + om3 + om4) + om2*(9 + 8*om3 + 4*om4))))/(60.*om1*(om1 + om2)*(om1 + om2 + om3)*(om1 + om2 + om3 + om4))
        ) * ff[-1] + (
            -(30*(om1*om1*om1) + 30*(om1*om1)*(2 + 3*om2 + 2*om3 + om4) + 3*(4 + 10*om3 + 5*om4) + 5*om1*(9 + 16*om3 + 8*om4 + 6*(3*(om2*om2) + om3*(om3 + om4) + 2*om2*(2 + 2*om3 + om4))) + 5*(6*(om2*om2*om2) + 4*om3*(om3 + om4) + 6*(om2*om2)*(2 + 2*om3 + om4) + om2*(9 + 8*om4 + 2*om3*(8 + 3*om3 + 3*om4))))/(60.*om1*om2*(om2 + om3)*(om2 + om3 + om4))
        ) * ff[-2] + (
            (3*(4 + 10*om2 + 10*om3 + 5*om4) + 5*(6*(om1*om1*om1) + 4*(om2 + om3)*(om2 + om3 + om4) + 6*(om1*om1)*(2*(1 + om2 + om3) + om4) + om1*(9 + 6*(om2*om2) + 16*om3 + 8*om4 + 6*om3*(om3 + om4) + 2*om2*(8 + 6*om3 + 3*om4))))/(60.*om2*(om1 + om2)*om3*(om3 + om4))
        ) * ff[-3] + (
            -(3*(4 + 10*om2 + 5*om3 + 5*om4) + 5*(6*(om1*om1*om1) + 4*om2*(om2 + om3 + om4) + 6*(om1*om1)*(2 + 2*om2 + om3 + om4) + om1*(9 + 8*om3 + 8*om4 + 2*om2*(8 + 3*om2 + 3*om3 + 3*om4))))/(60.*om3*(om2 + om3)*(om1 + om2 + om3)*om4)
        ) * ff[-4] + (
            (3*(4 + 10*om2 + 5*om3) + 5*(6*(om1*om1*om1) + 4*om2*(om2 + om3) + 6*(om1*om1)*(2 + 2*om2 + om3) + om1*(9 + 8*om3 + 2*om2*(8 + 3*om2 + 3*om3))))/(60.*om4*(om3 + om4)*(om2 + om3 + om4)*(om1 + om2 + om3 + om4))
        ) * ff[-5]

    u_new = uu[-1] + h * du_new
    return u_new

def conservative_AB5(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3, t4, u4,
                     **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3, t4], [u0, u1, u2, u3, u4],
                            fixed_step_AB5, adaptive_step_AB5,
                            **kwargs)


# Nyström methods based on the idea u_{n} = u_{n-2} + \int_{t_{n-2}}^{t_{n}} f
def fixed_step_Nyström2(uu, ff, h):
    u_new = uu[-2] + 2 * h * ff[-1]
    return u_new

def fixed_estimate_Nyström2(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = uu[-2] + h * (
            (
                1.125
            ) * ff[-1] + (
                0.375
            ) * ff[-2]
        )

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def adaptive_step_Nyström2(uu, ff, h, old_omega):
    om1 = old_omega[-1]
    u_new = uu[-2] + h * (
            (
                2./om1
            ) * ff[-1] + (
                2 - 2./om1
            ) * ff[-2]
        )

    return u_new

def adaptive_estimate_Nyström2(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om1 = old_omega[-1]
    u1int = uu[-2] + h * (
            (
                ((1 + 2*om1)*(1 + 2*om1))/(8.*om1)
            ) * ff[-1] + (
                -1/(8.*om1) + om1/2.
            ) * ff[-2]
        )

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def conservative_Nyström2(f, t_final, t0, u0, t1, u1,
                          idx_u_old=-2,
                          **kwargs):
    return conservative_LMM(f, t_final, [t0, t1], [u0, u1],
                            fixed_step_Nyström2, adaptive_step_Nyström2,
                            idx_u_old=idx_u_old,
                            **kwargs)

def cons_or_diss_Nyström2(f, t_final, t0, u0, t1, u1,
                          idx_u_old=-2,
                          **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1], [u0, u1],
                            fixed_step_Nyström2, adaptive_step_Nyström2,
                            fixed_estimate_Nyström2, adaptive_estimate_Nyström2,
                            idx_u_old=idx_u_old,
                            **kwargs)


def fixed_step_Nyström3(uu, ff, h):
    u_new = uu[-2] + h * (
            (
                2.3333333333333335
            ) * ff[-1] + (
                -0.6666666666666666
            ) * ff[-2] + (
                0.3333333333333333
            ) * ff[-3]
        )
    return u_new

def fixed_estimate_Nyström3(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = uu[-2] + h * (
            (
                0.6630580790986869
            ) * ff[-1] + (
                0.618862671982261
            ) * ff[-2] + (
                -0.07059588567576056
            ) * ff[-3]
        )
    u2int = uu[-2] + h * (
            (
                1.7536085875679797
            ) * ff[-1] + (
                -0.11886267198226091
            ) * ff[-2] + (
                0.15392921900909387
            ) * ff[-3]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_Nyström3(uu, ff, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = uu[-2] + h * (
            (
                (8 + 6*om2)/(3*(om1*om1) + 3*om1*om2)
            ) * ff[-1] + (
                (2*(-4 - 3*om2 + 3*om1*(1 + om2)))/(3.*om1*om2)
            ) * ff[-2] + (
                (8 - 6*om1)/(3*om1*om2 + 3*(om2*om2))
            ) * ff[-3]
        )

    return u_new

def adaptive_estimate_Nyström3(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int =uu[-2] + h * (
            (
                (9 - 5*np.sqrt(3) - 9*(-2 + np.sqrt(3))*om2 + 18*om1*(2 - np.sqrt(3) - (-3 + np.sqrt(3))*om2 + om1*(3 - np.sqrt(3) + 2*om1 + 3*om2)))/(108.*om1*(om1 + om2))
            ) * ff[-1] + (
                (-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om2 + 9*om1*(-2 + np.sqrt(3) + 2*om1*(om1 + 3*om2)))/(108.*om1*om2)
            ) * ff[-2] + (
                (9 - 5*np.sqrt(3) - 9*om1*(-2 + np.sqrt(3) + 2*(om1*om1)))/(108.*om2*(om1 + om2))
            ) * ff[-3]
        )
    u2int = uu[-2] + h * (
            (
                (9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om2 + 18*om1*(2 + np.sqrt(3) + (3 + np.sqrt(3))*om2 + om1*(3 + np.sqrt(3) + 2*om1 + 3*om2)))/(108.*om1*(om1 + om2))
            ) * ff[-1] + (
                -(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om2 + 9*om1*(2 + np.sqrt(3) - 2*om1*(om1 + 3*om2)))/(108.*om1*om2)
            ) * ff[-2] + (
                (9 + 5*np.sqrt(3) + 9*om1*(2 + np.sqrt(3) - 2*(om1*om1)))/(108.*om2*(om1 + om2))
            ) * ff[-3]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_Nyström3(f, t_final, t0, u0, t1, u1, t2, u2,
                          idx_u_old=-2,
                          **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_Nyström3, adaptive_step_Nyström3,
                            idx_u_old=idx_u_old,
                            **kwargs)

def cons_or_diss_Nyström3(f, t_final, t0, u0, t1, u1, t2, u2,
                          idx_u_old=-2,
                          **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_Nyström3, adaptive_step_Nyström3,
                            fixed_estimate_Nyström3, adaptive_estimate_Nyström3,
                            idx_u_old=idx_u_old,
                            **kwargs)


def fixed_step_Nyström4(uu, ff, h):
    u_new = uu[-2] + h * (
            (
                2.6666666666666665
            ) * ff[-1] + (
                -1.6666666666666667
            ) * ff[-2] + (
                1.3333333333333333
            ) * ff[-3] + (
                -0.3333333333333333
            ) * ff[-4]
        )
    return u_new

def fixed_estimate_Nyström4(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = uu[-2] + h * (
            (
                0.6304904416411566
            ) * ff[-1] + (
                0.7165655843548515
            ) * ff[-2] + (
                -0.1682987980483512
            ) * ff[-3] + (
                0.03256763745753022
            ) * ff[-4]
        )
    u2int = uu[-2] + h * (
            (
                1.9134910398403246
            ) * ff[-1] + (
                -0.5985100287992959
            ) * ff[-2] + (
                0.6335765758261289
            ) * ff[-3] + (
                -0.15988245227234502
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_Nyström4(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = uu[-2] + h * (
            (
                (2*(6 + 4*om3 + om2*(8 + 3*om2 + 3*om3)))/(3.*om1*(om1 + om2)*(om1 + om2 + om3))
            ) * ff[-1] + (
                (2*(-6 - 4*om3 - om2*(8 + 3*om2 + 3*om3) + om1*(4 + 3*om3 + 3*om2*(2 + om2 + om3))))/(3.*om1*om2*(om2 + om3))
            ) * ff[-2] + (
                (2*(6 + 4*om2 + 4*om3 - om1*(4 + 3*om2 + 3*om3)))/(3.*om2*(om1 + om2)*om3)
            ) * ff[-3] + (
                (2*(-6 - 4*om2 + om1*(4 + 3*om2)))/(3.*om3*(om2 + om3)*(om1 + om2 + om3))
            ) * ff[-4]
        )

    return u_new

def adaptive_estimate_Nyström4(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = uu[-2] + h * (
            (
                (21 - 12*np.sqrt(3) + 8*(9 - 5*np.sqrt(3))*om2 + 4*(9 - 5*np.sqrt(3))*om3 + 12*(9*(om1*om1*om1*om1) - 3*(-2 + np.sqrt(3))*om2*(om2 + om3) + 6*(om1*om1*om1)*(3 - np.sqrt(3) + 4*om2 + 2*om3) + 3*(om1*om1)*(6 - 3*np.sqrt(3) - 2*(-3 + np.sqrt(3))*om3 + 2*om2*(6 - 2*np.sqrt(3) + 3*om2 + 3*om3)) + om1*(9 - 5*np.sqrt(3) - 6*(-2 + np.sqrt(3))*om3 - 6*om2*(2*(-2 + np.sqrt(3)) + (-3 + np.sqrt(3))*om2 + (-3 + np.sqrt(3))*om3))))/(432.*om1*(om1 + om2)*(om1 + om2 + om3))
            ) * ff[-1] + (
                (3*(-7 + 4*np.sqrt(3)) + 8*(-9 + 5*np.sqrt(3))*om2 + 4*(-9 + 5*np.sqrt(3))*om3 + 4*(9*(om1*om1*om1*om1) + 9*(-2 + np.sqrt(3))*om2*(om2 + om3) + 18*(om1*om1*om1)*(2*om2 + om3) + om1*(2*(-9 + 5*np.sqrt(3)) + 18*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3) + 9*(om1*om1)*(-2 + np.sqrt(3) + 6*om2*(om2 + om3))))/(432.*om1*om2*(om2 + om3))
            ) * ff[-2] + (
                -(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om2 + 4*(-9 + 5*np.sqrt(3))*om3 + 4*om1*(2*(-9 + 5*np.sqrt(3)) + 9*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3 + 9*om1*(-2 + np.sqrt(3) + om1*om1 + 2*om1*(om2 + om3))))/(432.*om2*(om1 + om2)*om3)
            ) * ff[-3] + (
                (3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om2 + 4*om1*(2*(-9 + 5*np.sqrt(3)) + 9*(-2 + np.sqrt(3))*om2 + 9*om1*(-2 + np.sqrt(3) + om1*om1 + 2*om1*om2)))/(432.*om3*(om2 + om3)*(om1 + om2 + om3))
            ) * ff[-4]
        )
    u2int = uu[-2] + h * (
            (
                (3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 12*(9*(om1*om1*om1*om1) + 3*(2 + np.sqrt(3))*om2*(om2 + om3) + 6*(om1*om1*om1)*(3 + np.sqrt(3) + 4*om2 + 2*om3) + 3*(om1*om1)*(3*(2 + np.sqrt(3)) + 2*(3 + np.sqrt(3))*om3 + 2*om2*(6 + 2*np.sqrt(3) + 3*om2 + 3*om3)) + om1*(9 + 5*np.sqrt(3) + 6*(2 + np.sqrt(3))*om3 + 6*om2*(2*(2 + np.sqrt(3)) + (3 + np.sqrt(3))*om2 + (3 + np.sqrt(3))*om3))))/(432.*om1*(om1 + om2)*(om1 + om2 + om3))
            ) * ff[-1] + (
                -(3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(-9*(om1*om1*om1*om1) + 9*(2 + np.sqrt(3))*om2*(om2 + om3) - 18*(om1*om1*om1)*(2*om2 + om3) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3) + 9*(om1*om1)*(2 + np.sqrt(3) - 6*om2*(om2 + om3))))/(432.*om1*om2*(om2 + om3))
            ) * ff[-2] + (
                (3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*om1*(2*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3 + 9*om1*(2 + np.sqrt(3) - om1*(om1 + 2*(om2 + om3)))))/(432.*om2*(om1 + om2)*om3)
            ) * ff[-3] + (
                -(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om2 + 4*om1*(2*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om2 + 9*om1*(2 + np.sqrt(3) - om1*(om1 + 2*om2))))/(432.*om3*(om2 + om3)*(om1 + om2 + om3))
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_Nyström4(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                          idx_u_old=-2,
                          **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_Nyström4, adaptive_step_Nyström4,
                            idx_u_old=idx_u_old,
                            **kwargs)

def cons_or_diss_Nyström4(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                          idx_u_old=-2,
                          **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_Nyström4, adaptive_step_Nyström4,
                            fixed_estimate_Nyström4, adaptive_estimate_Nyström4,
                            idx_u_old=idx_u_old,
                            **kwargs)


#NOTE: This method does not work well with relaxation
def conservative_Nyström2mod(f, t_final, t0, u0, t1, u1,
                             eta=etaL2, deta=detaL2,
                             return_gamma=False,
                             projection=False, relaxation=False,
                             adapt_dt=False, adapt_coefficients=False,
                             method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    uu = [u0, u1]
    ff = [f(u0), f(u1)]
    tt = [t0, t1]
    old_omega = [1.0, 1.0]
    old_gamma = [1.0, 1.0]
    h = t1 - t0

    if relaxation and projection:
        raise Exception("Use either relaxation or projection, not both.")

    if relaxation and method == None:
        method = "brentq"
    elif projection and method == None:
        method = "simplified Newton"

    t = t1
    gammas = [1.0, 1.0]
    step = 0
    while t < t_final and step < maxsteps:
        step += 1

        if relaxation and adapt_coefficients:
            om1 = old_omega[-1]
            du_new = (2.0) / (om1) * ff[-1]
            du_new = du_new + 2*(om1 - 1) / (om1) * ff[-2]
            u_new = uu[-2] + h * du_new
        else:
            u_new = uu[-2] + 2 * h * ff[-1]

        u_old = uu[-1]
        eta_old = eta(u_old)
        if projection:
            gamma, u_new = conservative_projection_solve(eta, deta, u_old, eta_old, u_new, method, tol, maxiter)
        elif relaxation:
            gamma = conservative_relaxation_solve(eta, deta, u_old, eta_old, u_new, old_gamma[-1], method, tol, maxiter)
            u_new = u_old + gamma * (u_new - u_old)
            old_gamma[-2] = old_gamma[-1]
            old_gamma[-1] = gamma
        else:
            gamma = 1.0

        if return_gamma:
            gammas.append(gamma)

        uu.append(u_new)
        if relaxation and adapt_dt:
            t = tt[-1] + gamma * h
            old_omega[-2] = old_omega[-1]
            old_omega[-1] = gamma
            if gamma < 1.0e-14:
                raise Exception("gamma = %.2e is too small in step %d!" % (gamma, step))
        else:
            t = tt[-1] + h
        tt.append(t)

        ff[-2] = ff[-1]
        ff[-1] = f(u_new)

    if return_gamma:
        return np.array(tt), uu, np.array(gammas)
    else:
        return np.array(tt), uu


#NOTE: This method does not work well with relaxation
def conservative_Nyström3mod(f, t_final, t0, u0, t1, u1, t2, u2,
                             eta=etaL2, deta=detaL2,
                             return_gamma=False,
                             projection=False, relaxation=False,
                             adapt_dt=False, adapt_coefficients=False,
                             method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    uu = [u0, u1, u2]
    ff = [f(u0), f(u1), f(u2)]
    tt = [t0, t1, t2]
    old_omega = [1.0, 1.0, 1.0]
    old_gamma = [1.0, 1.0, 1.0]
    h = t1 - t0
    np.testing.assert_approx_equal(h, t2 - t1)

    if relaxation and projection:
        raise Exception("Use either relaxation or projection, not both.")

    if relaxation and method == None:
        method = "brentq"
    elif projection and method == None:
        method = "simplified Newton"

    t = t2
    gammas = [1.0, 1.0, 1.0]
    step = 0
    while t < t_final and step < maxsteps:
        step += 1

        if relaxation and adapt_coefficients:
            om2 = old_omega[-2]
            om1 = old_omega[-1]
            du_new = 2 * (4 + 3*om2) / (3 * om1 * (om1 + om2)) * ff[-1]
            du_new = du_new - 2 * (4 + 3*om2 - 3*om1 * (1 + om2)) / (3 * om1 * om2) * ff[-2]
            du_new = du_new + (8 - 6*om1) / (3 * om2 * (om1 + om2)) * ff[-3]
            u_new = uu[-2] + h * du_new
        else:
            u_new = uu[-2] + h * ((7.0/3.0) * ff[-1] - (2.0/3.0) * ff[-2] + (1.0/3.0) * ff[-3])

        u_old = uu[-1]
        eta_old = eta(u_old)
        if projection:
            gamma, u_new = conservative_projection_solve(eta, deta, u_old, eta_old, u_new, method, tol, maxiter)
        elif relaxation:
            gamma = conservative_relaxation_solve(eta, deta, u_old, eta_old, u_new, old_gamma[-1], method, tol, maxiter)
            u_new = u_old + gamma * (u_new - u_old)
            old_gamma[-3] = old_gamma[-2]
            old_gamma[-2] = old_gamma[-1]
            old_gamma[-1] = gamma
        else:
            gamma = 1.0

        if return_gamma:
            gammas.append(gamma)

        uu.append(u_new)
        if relaxation and adapt_dt:
            t = tt[-1] + gamma * h
            old_omega[-3] = old_omega[-2]
            old_omega[-2] = old_omega[-1]
            old_omega[-1] = gamma
            if gamma < 1.0e-14:
                raise Exception("gamma = %.2e is too small in step %d!" % (gamma, step))
        else:
            t = tt[-1] + h
        tt.append(t)

        ff[-3] = ff[-2]
        ff[-2] = ff[-1]
        ff[-1] = f(u_new)

    if return_gamma:
        return np.array(tt), uu, np.array(gammas)
    else:
        return np.array(tt), uu


#NOTE: This method does not work well with relaxation
def conservative_Nyström4mod(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                             eta=etaL2, deta=detaL2,
                             return_gamma=False,
                             projection=False, relaxation=False,
                             adapt_dt=False, adapt_coefficients=False,
                             method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    uu = [u0, u1, u2, u3]
    ff = [f(u0), f(u1), f(u2), f(u3)]
    tt = [t0, t1, t2, t3]
    old_omega = [1.0, 1.0, 1.0, 1.0]
    old_gamma = [1.0, 1.0, 1.0, 1.0]
    h = t1 - t0
    np.testing.assert_approx_equal(h, t2 - t1)
    np.testing.assert_approx_equal(h, t3 - t2)

    if relaxation and projection:
        raise Exception("Use either relaxation or projection, not both.")

    if relaxation and method == None:
        method = "brentq"
    elif projection and method == None:
        method = "simplified Newton"

    t = t3
    gammas = [1.0, 1.0, 1.0, 1.0]
    step = 0
    while t < t_final and step < maxsteps:
        step += 1

        if relaxation and adapt_coefficients:
            om3 = old_omega[-3]
            om2 = old_omega[-2]
            om1 = old_omega[-1]
            du_new = 2 * (6 + 4*om3 + om2 * (8 + 3*om2 + 3*om3)) / (3*om1 * (om1 + om2) * (om1 + om2 + om3)) * ff[-1]
            du_new = du_new - 2 * (6 + 4*om3 + om2 * (8 + 3*om2 + 3*om3) - om1 * (4 + 3*om3 + 3*om2 * (2 + om2 + om3))) / (3 * om1 * om2 * (om2 + om3)) * ff[-2]
            du_new = du_new + 2 * (6 + 4*om2 + 4*om3 - om1 * (4 + 3*om2 + 3*om3)) / (3 * om2 * (om1 + om2) * om3) * ff[-3]
            du_new = du_new - 2 * (6 + 4*om2 - om1 * (4 + 3*om2)) / (3 * om3 * (om2 + om3) * (om1 + om2 + om3)) * ff[-4]
            u_new = uu[-2] + h * du_new
        else:
            u_new = uu[-2] + h * ((8.0/3.0) * ff[-1] - (5.0/3.0) * ff[-2] + (4.0/3.0) * ff[-3] - (1.0/3.0) * ff[-4])

        u_old = uu[-1]
        eta_old = eta(u_old)
        if projection:
            gamma, u_new = conservative_projection_solve(eta, deta, u_old, eta_old, u_new, method, tol, maxiter)
        elif relaxation:
            gamma = conservative_relaxation_solve(eta, deta, u_old, eta_old, u_new, old_gamma[-1], method, tol, maxiter)
            u_new = u_old + gamma * (u_new - u_old)
            old_gamma[-4] = old_gamma[-3]
            old_gamma[-3] = old_gamma[-2]
            old_gamma[-2] = old_gamma[-1]
            old_gamma[-1] = gamma
        else:
            gamma = 1.0

        if return_gamma:
            gammas.append(gamma)

        uu.append(u_new)
        if relaxation and adapt_dt:
            t = tt[-1] + gamma * h
            old_omega[-4] = old_omega[-3]
            old_omega[-3] = old_omega[-2]
            old_omega[-2] = old_omega[-1]
            old_omega[-1] = gamma
            if gamma < 1.0e-14:
                raise Exception("gamma = %.2e is too small in step %d!" % (gamma, step))
        else:
            t = tt[-1] + h
        tt.append(t)

        ff[-4] = ff[-3]
        ff[-3] = ff[-2]
        ff[-2] = ff[-1]
        ff[-1] = f(u_new)

    if return_gamma:
        return np.array(tt), uu, np.array(gammas)
    else:
        return np.array(tt), uu


# Nyström methods with extension to variable stepsizes by Arévalo & Söderlind (2017)
def fixed_step_Nyström2AS(uu, ff, h):
    u_new = (
            0.
        ) * uu[-1] + (
            1.
        ) * uu[-2] + h * (
            (
                2.
            ) * ff[-1]
        )
    return u_new

def fixed_estimate_Nyström2AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            0.75
        ) * uu[-1] + (
            0.25
        ) * uu[-2] + h * (
            (
                0.75
            ) * ff[-1]
        )

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def adaptive_step_Nyström2AS(uu, ff, h, old_omega):
    om1 = old_omega[-1]
    u_new = (
            1 - 1./(om1*om1)
        ) * uu[-1] + (
            1./(om1*om1)
        ) * uu[-2] + h * (
            (
                1 + 1./om1
            ) * ff[-1]
        )

    return u_new

def adaptive_estimate_Nyström2AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om1 = old_omega[-1]
    u1int = (
            1 - 1/(4.*(om1*om1))
        ) * uu[-1] + (
            1/(4.*(om1*om1))
        ) * uu[-2] + h * (
            (
                (2 + 1/om1)/4.
            ) * ff[-1]
        )

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def conservative_Nyström2AS(f, t_final, t0, u0, t1, u1,
                            **kwargs):
    return conservative_LMM(f, t_final, [t0, t1], [u0, u1],
                            fixed_step_Nyström2AS, adaptive_step_Nyström2AS,
                            **kwargs)

def cons_or_diss_Nyström2AS(f, t_final, t0, u0, t1, u1,
                            **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1], [u0, u1],
                            fixed_step_Nyström2AS, adaptive_step_Nyström2AS,
                            fixed_estimate_Nyström2AS, adaptive_estimate_Nyström2AS,
                            **kwargs)


def fixed_step_Nyström3AS(uu, ff, h):
    u_new = (
            0.
        ) * uu[-1] + (
            1.
        ) * uu[-2] + h * (
            (
                2.3333333333333335
            ) * ff[-1] + (
                -0.6666666666666666
            ) * ff[-2] + (
                0.3333333333333333
            ) * ff[-3]
        )
    return u_new

def fixed_estimate_Nyström3AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            0.9641470039866955
        ) * uu[-1] + (
            0.03585299601330434
        ) * uu[-2] + h * (
            (
                0.26133016077089677
            ) * ff[-1] + (
                -0.023901997342202896
            ) * ff[-2] + (
                0.009749697989797416
            ) * ff[-3]
        )
    u2int = (
            0.41085299601330433
        ) * uu[-1] + (
            0.5891470039866956
        ) * uu[-2] + h * (
            (
                1.5824198392291033
            ) * ff[-1] + (
                -0.39276466932446374
            ) * ff[-2] + (
                0.18816696867686925
            ) * ff[-3]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_Nyström3AS(uu, ff, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            (-2 - 3*om2 + om1*(-3 + om1*om1 + 7*om1*om2))/(om1*om1*(om1 + 7*om2))
        ) * uu[-1] + (
            (2 + 3*om1 + 3*om2)/(om1*om1*(om1 + 7*om2))
        ) * uu[-2] + h * (
            (
                (3*om1*((1 + om1)*(1 + om1)) + 2*(5 + 3*om1*(5 + 4*om1))*om2 + 3*(5 + 7*om1)*(om2*om2))/(3.*om1*(om1 + om2)*(om1 + 7*om2))
            ) * ff[-1] + (
                -((4 + 6*om1 + 6*om2)/(3*(om1*om1) + 21*om1*om2))
            ) * ff[-2] + (
                (7 + 9*om1)/(3.*(om1 + om2)*(om1 + 7*om2))
            ) * ff[-3]
        )

    return u_new

def adaptive_estimate_Nyström3AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            (-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om2 + 9*om1*(-2 + np.sqrt(3) + 2*om1*(om1 + 7*om2)))/(18.*(om1*om1)*(om1 + 7*om2))
        ) * uu[-1] + (
            -(-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om1 + 9*(-2 + np.sqrt(3))*om2)/(18.*(om1*om1)*(om1 + 7*om2))
        ) * uu[-2] + h * (
            (
                -(18*(-3 + np.sqrt(3))*(om1*om1*om1) + 36*(om1*om1)*(-2 + np.sqrt(3) + 4*(-3 + np.sqrt(3))*om2) + 10*om2*(-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om2) + 3*om1*(-9 + 5*np.sqrt(3) + 60*(-2 + np.sqrt(3))*om2 + 42*(-3 + np.sqrt(3))*(om2*om2)))/(108.*om1*(om1 + om2)*(om1 + 7*om2))
            ) * ff[-1] + (
                (-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om1 + 9*(-2 + np.sqrt(3))*om2)/(27.*om1*(om1 + 7*om2))
            ) * ff[-2] + (
                (63 - 35*np.sqrt(3) - 54*(-2 + np.sqrt(3))*om1)/(108.*(om1 + om2)*(om1 + 7*om2))
            ) * ff[-3]
        )
    u2int = (
            -(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om2 + 9*om1*(2 + np.sqrt(3) - 2*om1*(om1 + 7*om2)))/(18.*(om1*om1)*(om1 + 7*om2))
        ) * uu[-1] + (
            (9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om1 + 9*(2 + np.sqrt(3))*om2)/(18.*(om1*om1)*(om1 + 7*om2))
        ) * uu[-2] + h * (
            (
                (18*(3 + np.sqrt(3))*(om1*om1*om1) + 10*om2*(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om2) + 36*(om1*om1)*(2 + np.sqrt(3) + 4*(3 + np.sqrt(3))*om2) + 3*om1*(9 + 5*np.sqrt(3) + 60*(2 + np.sqrt(3))*om2 + 42*(3 + np.sqrt(3))*(om2*om2)))/(108.*om1*(om1 + om2)*(om1 + 7*om2))
            ) * ff[-1] + (
                -(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om1 + 9*(2 + np.sqrt(3))*om2)/(27.*om1*(om1 + 7*om2))
            ) * ff[-2] + (
                (7*(9 + 5*np.sqrt(3)) + 54*(2 + np.sqrt(3))*om1)/(108.*(om1 + om2)*(om1 + 7*om2))
            ) * ff[-3]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_Nyström3AS(f, t_final, t0, u0, t1, u1, t2, u2,
                            **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_Nyström3AS, adaptive_step_Nyström3AS,
                            **kwargs)

def cons_or_diss_Nyström3AS(f, t_final, t0, u0, t1, u1, t2, u2,
                            **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_Nyström3AS, adaptive_step_Nyström3AS,
                            fixed_estimate_Nyström3AS, adaptive_estimate_Nyström3AS,
                            **kwargs)


def fixed_step_Nyström4AS(uu, ff, h):
    u_new = (
            0.
        ) * uu[-1] + (
            1.
        ) * uu[-2] + h * (
            (
                2.6666666666666665
            ) * ff[-1] + (
                -1.6666666666666667
            ) * ff[-2] + (
                1.3333333333333333
            ) * ff[-3] + (
                -0.3333333333333333
            ) * ff[-4]
        )
    return u_new

def fixed_estimate_Nyström4AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            0.9694504071951937
        ) * uu[-1] + (
            0.030549592804806153
        ) * uu[-2] + h * (
            (
                0.26694653894295894
            ) * ff[-1] + (
                -0.050915988008010254
            ) * ff[-2] + (
                0.03367003678398088
            ) * ff[-3] + (
                -0.007826129508936195
            ) * ff[-4]
        )
    u2int = (
            0.4345043950646931
        ) * uu[-1] + (
            0.5654956049353068
        ) * uu[-2] + h * (
            (
                1.7505518916910652
            ) * ff[-1] + (
                -0.9424926748921779
            ) * ff[-2] + (
                0.7240983247979401
            ) * ff[-3] + (
                -0.17798680206670725
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_Nyström4AS(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            (-3 + om1*om1*om1*om1 - 8*om2 - 4*om3 - 6*om2*(om2 + om3) + 2*(om1*om1*om1)*(2*om2 + om3) - 2*om1*(4 + 6*om2 + 3*om3) + om1*om1*(-6 + 26*om2*(om2 + om3)))/(om1*om1*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-1] + (
            (3 + 6*(om1*om1) + 8*om2 + 4*om3 + 6*om2*(om2 + om3) + 2*om1*(4 + 6*om2 + 3*om3))/(om1*om1*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-2] + h * (
            (
                (2*(3*(om1*om1)*((1 + om1)*(1 + om1)*(1 + om1)) + 9*om1*((1 + om1)*(1 + om1))*(1 + 2*om1)*om2 + 3*(8 + om1*(38 + 5*om1*(12 + 7*om1)))*(om2*om2) + 8*(8 + 3*om1*(8 + 7*om1))*(om2*om2*om2) + 6*(8 + 13*om1)*(om2*om2*om2*om2)) + 3*(1 + 2*om1 + 2*om2)*(3*om1*((1 + om1)*(1 + om1)) + 2*(8 + om1*(19 + 16*om1))*om2 + 4*(8 + 13*om1)*(om2*om2))*om3 + 4*(3*om1*((1 + om1)*(1 + om1)) + 2*(8 + 3*om1*(8 + 7*om1))*om2 + 3*(8 + 13*om1)*(om2*om2))*(om3*om3))/(6.*om1*(om1 + om2)*(om1 + om2 + om3)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-1] + (
                (-5*(3 + 6*(om1*om1) + 8*om2 + 4*om3 + 6*om2*(om2 + om3) + 2*om1*(4 + 6*om2 + 3*om3)))/(3.*om1*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-2] + (
                (3*(om1*om1*om1) + 13*(om2 + om3)*(3 + 4*om2 + 4*om3) + om1*om1*(6 + 75*om2 + 75*om3) + om1*(3 + 72*(om2*om2) + 8*om3*(13 + 9*om3) + 8*om2*(13 + 18*om3)))/(6.*(om1 + om2)*om3*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-3] + (
                -(3*om1*((1 + om1)*(1 + om1)) + (39 + om1*(104 + 75*om1))*om2 + 4*(13 + 18*om1)*(om2*om2))/(6.*om3*(om1 + om2 + om3)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-4]
        )

    return u_new

def adaptive_estimate_Nyström4AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            (3*(-7 + 4*np.sqrt(3)) + 8*(-9 + 5*np.sqrt(3))*om2 + 4*(-9 + 5*np.sqrt(3))*om3 + 4*(9*(om1*om1*om1*om1) + 9*(-2 + np.sqrt(3))*om2*(om2 + om3) + 18*(om1*om1*om1)*(2*om2 + om3) + om1*(2*(-9 + 5*np.sqrt(3)) + 18*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3) + 9*(om1*om1)*(-2 + np.sqrt(3) + 26*om2*(om2 + om3))))/(36.*(om1*om1)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-1] + (
            (21 - 12*np.sqrt(3) + 8*(9 - 5*np.sqrt(3))*om2 + 4*(9 - 5*np.sqrt(3))*om3 + 4*(-9*(-2 + np.sqrt(3))*(om1*om1) - 9*(-2 + np.sqrt(3))*om2*(om2 + om3) + om1*(18 - 10*np.sqrt(3) - 18*(-2 + np.sqrt(3))*om2 - 9*(-2 + np.sqrt(3))*om3)))/(36.*(om1*om1)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-2] + h * (
            (
                -(36*(-3 + np.sqrt(3))*(om1*om1*om1*om1*om1) + 108*(om1*om1*om1*om1)*(-2 + np.sqrt(3) + 2*(-3 + np.sqrt(3))*om2 + (-3 + np.sqrt(3))*om3) + 18*(om1*om1*om1)*(-9 + 5*np.sqrt(3) + 30*(-2 + np.sqrt(3))*om2 + 70*(-3 + np.sqrt(3))*(om2*om2) + 15*(-2 + np.sqrt(3))*om3 + 70*(-3 + np.sqrt(3))*om2*om3 + 4*(-3 + np.sqrt(3))*(om3*om3)) + 16*om2*(om2 + om3)*(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om3 + 4*om2*(2*(-9 + 5*np.sqrt(3)) + 9*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3)) + 6*(om1*om1)*(-7 + 4*np.sqrt(3) + 336*(-3 + np.sqrt(3))*(om2*om2*om2) + 72*(om2*om2)*(5*(-2 + np.sqrt(3)) + 7*(-3 + np.sqrt(3))*om3) + 6*om3*(-9 + 5*np.sqrt(3) + 4*(-2 + np.sqrt(3))*om3) + 12*om2*(-9 + 5*np.sqrt(3) + 30*(-2 + np.sqrt(3))*om3 + 14*(-3 + np.sqrt(3))*(om3*om3))) - 3*om1*(-312*(-3 + np.sqrt(3))*(om2*om2*om2*om2) + om3*(21 - 12*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om3) - 48*(om2*om2*om2)*(16*(-2 + np.sqrt(3)) + 13*(-3 + np.sqrt(3))*om3) - 4*(om2*om2)*(19*(-9 + 5*np.sqrt(3)) + 288*(-2 + np.sqrt(3))*om3 + 78*(-3 + np.sqrt(3))*(om3*om3)) + om2*(42 - 24*np.sqrt(3) + 4*om3*(171 - 95*np.sqrt(3) - 96*(-2 + np.sqrt(3))*om3))))/(216.*om1*(om1 + om2)*(om1 + om2 + om3)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-1] + (
                (5*(3*(-7 + 4*np.sqrt(3)) + 8*(-9 + 5*np.sqrt(3))*om2 + 4*(-9 + 5*np.sqrt(3))*om3 + 4*(9*(-2 + np.sqrt(3))*(om1*om1) + 9*(-2 + np.sqrt(3))*om2*(om2 + om3) + om1*(2*(-9 + 5*np.sqrt(3)) + 18*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3))))/(108.*om1*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-2] + (
                (-18*(-2 + np.sqrt(3))*(om1*om1*om1) + 39*(7 - 4*np.sqrt(3))*(om2 + om3) - 52*(-9 + 5*np.sqrt(3))*((om2 + om3)*(om2 + om3)) - 6*(om1*om1)*(-9 + 5*np.sqrt(3) + 75*(-2 + np.sqrt(3))*om2 + 75*(-2 + np.sqrt(3))*om3) + om1*(21 - 12*np.sqrt(3) - 432*(-2 + np.sqrt(3))*(om2*om2) + 8*om2*(117 - 65*np.sqrt(3) - 108*(-2 + np.sqrt(3))*om3) + 8*om3*(117 - 65*np.sqrt(3) - 54*(-2 + np.sqrt(3))*om3)))/(216.*(om1 + om2)*om3*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-3] + (
                (18*(-2 + np.sqrt(3))*(om1*om1*om1) + 6*(om1*om1)*(-9 + 5*np.sqrt(3) + 75*(-2 + np.sqrt(3))*om2) + 13*om2*(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om2) + om1*(3*(-7 + 4*np.sqrt(3)) + 8*om2*(13*(-9 + 5*np.sqrt(3)) + 54*(-2 + np.sqrt(3))*om2)))/(216.*om3*(om1 + om2 + om3)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-4]
        )
    u2int = (
            -(3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(-9*(om1*om1*om1*om1) + 9*(2 + np.sqrt(3))*om2*(om2 + om3) - 18*(om1*om1*om1)*(2*om2 + om3) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3) + 9*(om1*om1)*(2 + np.sqrt(3) - 26*om2*(om2 + om3))))/(36.*(om1*om1)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-1] + (
            (3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(9*(2 + np.sqrt(3))*(om1*om1) + 9*(2 + np.sqrt(3))*om2*(om2 + om3) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3)))/(36.*(om1*om1)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-2] + h * (
            (
                (36*(3 + np.sqrt(3))*(om1*om1*om1*om1*om1) + 108*(om1*om1*om1*om1)*(2 + np.sqrt(3) + 2*(3 + np.sqrt(3))*om2 + (3 + np.sqrt(3))*om3) + 18*(om1*om1*om1)*(9 + 5*np.sqrt(3) + 30*(2 + np.sqrt(3))*om2 + 70*(3 + np.sqrt(3))*(om2*om2) + 15*(2 + np.sqrt(3))*om3 + 70*(3 + np.sqrt(3))*om2*om3 + 4*(3 + np.sqrt(3))*(om3*om3)) + 16*om2*(om2 + om3)*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3 + 4*om2*(2*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3)) + 6*(om1*om1)*(7 + 4*np.sqrt(3) + 336*(3 + np.sqrt(3))*(om2*om2*om2) + 6*om3*(9 + 5*np.sqrt(3) + 4*(2 + np.sqrt(3))*om3) + 72*(om2*om2)*(5*(2 + np.sqrt(3)) + 7*(3 + np.sqrt(3))*om3) + 12*om2*(9 + 5*np.sqrt(3) + 30*(2 + np.sqrt(3))*om3 + 14*(3 + np.sqrt(3))*(om3*om3))) + 3*om1*(312*(3 + np.sqrt(3))*(om2*om2*om2*om2) + 48*(om2*om2*om2)*(16*(2 + np.sqrt(3)) + 13*(3 + np.sqrt(3))*om3) + om3*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3) + om2*(6*(7 + 4*np.sqrt(3)) + 76*(9 + 5*np.sqrt(3))*om3 + 384*(2 + np.sqrt(3))*(om3*om3)) + 4*(om2*om2)*(19*(9 + 5*np.sqrt(3)) + 288*(2 + np.sqrt(3))*om3 + 78*(3 + np.sqrt(3))*(om3*om3))))/(216.*om1*(om1 + om2)*(om1 + om2 + om3)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-1] + (
                (-5*(3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(9*(2 + np.sqrt(3))*(om1*om1) + 9*(2 + np.sqrt(3))*om2*(om2 + om3) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3))))/(108.*om1*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-2] + (
                (18*(2 + np.sqrt(3))*(om1*om1*om1) + 39*(7 + 4*np.sqrt(3))*(om2 + om3) + 52*(9 + 5*np.sqrt(3))*((om2 + om3)*(om2 + om3)) + 6*(om1*om1)*(9 + 5*np.sqrt(3) + 75*(2 + np.sqrt(3))*om2 + 75*(2 + np.sqrt(3))*om3) + om1*(3*(7 + 4*np.sqrt(3)) + 432*(2 + np.sqrt(3))*(om2*om2) + 8*om3*(13*(9 + 5*np.sqrt(3)) + 54*(2 + np.sqrt(3))*om3) + 8*om2*(13*(9 + 5*np.sqrt(3)) + 108*(2 + np.sqrt(3))*om3)))/(216.*(om1 + om2)*om3*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-3] + (
                -(18*(2 + np.sqrt(3))*(om1*om1*om1) + 6*(om1*om1)*(9 + 5*np.sqrt(3) + 75*(2 + np.sqrt(3))*om2) + 13*om2*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om2) + om1*(3*(7 + 4*np.sqrt(3)) + 8*om2*(13*(9 + 5*np.sqrt(3)) + 54*(2 + np.sqrt(3))*om2)))/(216.*om3*(om1 + om2 + om3)*(om1*om1 + 26*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_Nyström4AS(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                            **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_Nyström4AS, adaptive_step_Nyström4AS,
                            **kwargs)

def cons_or_diss_Nyström4AS(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                            **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_Nyström4AS, adaptive_step_Nyström4AS,
                            fixed_estimate_Nyström4AS, adaptive_estimate_Nyström4AS,
                            **kwargs)


# methods based on the idea $u_{n+k} = u_{n-k} + \int_{t_{n-k}}^{t_{n+k}} f$
def fixed_step_Leapfrog4(uu, ff, h):
    u_new = uu[-4] + h * (
            (
                2.6666666666666665
            ) * ff[-1] + (
                -1.3333333333333333
            ) * ff[-2] + (
                2.6666666666666665
            ) * ff[-3]
        )
    return u_new

def fixed_estimate_Leapfrog4(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = uu[-4] + h * (
            (
                0.6304904416411566
            ) * ff[-1] + (
                1.0498989176881848
            ) * ff[-2] + (
                1.165034535284982
            ) * ff[-3] + (
                0.36590097079086353
            ) * ff[-4]
        )
    u2int = uu[-4] + h * (
            (
                1.9134910398403246
            ) * ff[-1] + (
                -0.2651766954659626
            ) * ff[-2] + (
                1.9669099091594622
            ) * ff[-3] + (
                0.1734508810609883
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_Leapfrog4(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = uu[-4] + h * (
            (
                -((2 + om2 + om3)*(2 + om2 + om3)*(-12 + (-4 + om2)*om2 - (-4 + om3)*om3))/(12.*om1*(om1 + om2)*(om1 + om2 + om3))
            ) * ff[-1] + (
                ((2 + om2 + om3)*(2 + om2 + om3)*((2 + om2)*(-6 + 4*om1 + om2) - 2*(-2 + om1)*om3 - om3*om3))/(12.*om1*om2*(om2 + om3))
            ) * ff[-2] + (
                ((2 + om2 + om3)*(2 + om2 + om3)*(2*om1*(-4 + om2 + om3) - 4*(-3 + om2 + om3) + (om2 + om3)*(om2 + om3)))/(12.*om2*(om1 + om2)*om3)
            ) * ff[-3] + (
                (-16*(3 + 2*om2) - (om2 - 3*om3)*((om2 + om3)*(om2 + om3)*(om2 + om3)) + om1*(-2*(om2*om2*om2) + 6*om2*(4 + om3*om3) + 4*(8 + om3*om3*om3)))/(12.*om3*(om2 + om3)*(om1 + om2 + om3))
            ) * ff[-4]
        )

    return u_new

def adaptive_estimate_Leapfrog4(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = uu[-4] + h * (
            (
                (21 - 12*np.sqrt(3) + 8*(9 - 5*np.sqrt(3))*om2 + 4*(9 - 5*np.sqrt(3))*om3 + 12*(9*(om1*om1*om1*om1) + 6*(om1*om1*om1)*(3 - np.sqrt(3) + 4*om2 + 2*om3) + 3*(om1*om1)*(6 - 3*np.sqrt(3) - 2*(-3 + np.sqrt(3))*om3 + 2*om2*(6 - 2*np.sqrt(3) + 3*om2 + 3*om3)) + om1*(9 - 5*np.sqrt(3) - 6*(-2 + np.sqrt(3))*om3 - 6*om2*(2*(-2 + np.sqrt(3)) + (-3 + np.sqrt(3))*om2 + (-3 + np.sqrt(3))*om3)) - 3*(om2 + om3)*(om2*om2*om2 + om2*om2*om3 - om3*om3*om3 + om2*(-2 + np.sqrt(3) - om3*om3))))/(432.*om1*(om1 + om2)*(om1 + om2 + om3))
            ) * ff[-1] + (
                (3*(-7 + 4*np.sqrt(3)) + 8*(-9 + 5*np.sqrt(3))*om2 + 4*(-9 + 5*np.sqrt(3))*om3 + 4*(9*(om1*om1*om1*om1) + 18*(om1*om1*om1)*(2*om2 + om3) + 9*(om1*om1)*(-2 + np.sqrt(3) + 6*om2*(om2 + om3)) + om1*(2*(-9 + 5*np.sqrt(3)) + 18*(-2 + np.sqrt(3))*om2 + 36*(om2*om2*om2) + 54*(om2*om2)*om3 + 9*om3*(-2 + np.sqrt(3) - 2*(om3*om3))) + 9*(om2 + om3)*(om2*om2*om2 + om2*om2*om3 - om3*om3*om3 + om2*(-2 + np.sqrt(3) - om3*om3))))/(432.*om1*om2*(om2 + om3))
            ) * ff[-2] + (
                (21 - 12*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om2 + 4*(9 - 5*np.sqrt(3))*om3 + 4*(-9*(-2 + np.sqrt(3))*(om1*om1) - 9*(om1*om1*om1*om1) - 18*(om1*om1*om1)*(om2 + om3) + 9*((om2 + om3)*(om2 + om3)*(om2 + om3)*(om2 + om3)) + om1*(18 - 10*np.sqrt(3) - 9*(-2 + np.sqrt(3))*om3 + 9*(2*(om2*om2*om2) + 6*(om2*om2)*om3 + 2*(om3*om3*om3) + om2*(2 - np.sqrt(3) + 6*(om3*om3))))))/(432.*om2*(om1 + om2)*om3)
            ) * ff[-3] + (
                (3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om2 + 4*(9*(-2 + np.sqrt(3))*(om1*om1) + 9*(om1*om1*om1*om1) + 18*(om1*om1*om1)*om2 - 9*(om2 - 3*om3)*((om2 + om3)*(om2 + om3)*(om2 + om3)) + om1*(-18*(om2*om2*om2) + 9*om2*(-2 + np.sqrt(3) + 6*(om3*om3)) + 2*(-9 + 5*np.sqrt(3) + 18*(om3*om3*om3)))))/(432.*om3*(om2 + om3)*(om1 + om2 + om3))
            ) * ff[-4]
        )
    u2int = uu[-4] + h * (
            (
                (3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 12*(9*(om1*om1*om1*om1) + 6*(om1*om1*om1)*(3 + np.sqrt(3) + 4*om2 + 2*om3) + 3*(om1*om1)*(3*(2 + np.sqrt(3)) + 2*(3 + np.sqrt(3))*om3 + 2*om2*(6 + 2*np.sqrt(3) + 3*om2 + 3*om3)) + om1*(9 + 5*np.sqrt(3) + 6*(2 + np.sqrt(3))*om3 + 6*om2*(2*(2 + np.sqrt(3)) + (3 + np.sqrt(3))*om2 + (3 + np.sqrt(3))*om3)) - 3*(om2 + om3)*(om2*om2*om2 + om2*om2*om3 - om3*om3*om3 - om2*(2 + np.sqrt(3) + om3*om3))))/(432.*om1*(om1 + om2)*(om1 + om2 + om3))
            ) * ff[-1] + (
                -(3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(-9*(om1*om1*om1*om1) - 18*(om1*om1*om1)*(2*om2 + om3) + 9*(om1*om1)*(2 + np.sqrt(3) - 6*om2*(om2 + om3)) - 9*(om2 + om3)*(om2*om2*om2 + om2*om2*om3 - om3*om3*om3 - om2*(2 + np.sqrt(3) + om3*om3)) + om1*(2*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om3 + 18*((2 + np.sqrt(3))*om2 - 2*(om2*om2*om2) - 3*(om2*om2)*om3 + om3*om3*om3))))/(432.*om1*om2*(om2 + om3))
            ) * ff[-2] + (
                (3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(9*(2 + np.sqrt(3))*(om1*om1) - 9*(om1*om1*om1*om1) - 18*(om1*om1*om1)*(om2 + om3) + 9*((om2 + om3)*(om2 + om3)*(om2 + om3)*(om2 + om3)) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(om2*om2*om2) + 54*(om2*om2)*om3 + 9*om3*(2 + np.sqrt(3) + 2*(om3*om3)) + 9*om2*(2 + np.sqrt(3) + 6*(om3*om3)))))/(432.*om2*(om1 + om2)*om3)
            ) * ff[-3] + (
                -(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om2 + 4*(9*(2 + np.sqrt(3))*(om1*om1) - 9*(om1*om1*om1*om1) - 18*(om1*om1*om1)*om2 + 9*(om2 - 3*om3)*((om2 + om3)*(om2 + om3)*(om2 + om3)) + om1*(18*(om2*om2*om2) + 9*om2*(2 + np.sqrt(3) - 6*(om3*om3)) + 2*(9 + 5*np.sqrt(3) - 18*(om3*om3*om3)))))/(432.*om3*(om2 + om3)*(om1 + om2 + om3))
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_Leapfrog4(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                           idx_u_old=-4,
                           **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_Leapfrog4, adaptive_step_Leapfrog4,
                            idx_u_old=idx_u_old,
                            **kwargs)

def cons_or_diss_Leapfrog4(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                           idx_u_old=-4,
                           **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_Leapfrog4, adaptive_step_Leapfrog4,
                            fixed_estimate_Leapfrog4, adaptive_estimate_Leapfrog4,
                            idx_u_old=idx_u_old,
                            **kwargs)

#NOTE: This method does not work well with relaxation
def conservative_Leapfrog4mod(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                              eta=etaL2, deta=detaL2,
                              return_gamma=False,
                              projection=False, relaxation=False,
                              adapt_dt=False, adapt_coefficients=False,
                              method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    uu = [u0, u1, u2, u3]
    ff = [f(u0), f(u1), f(u2), f(u3)]
    tt = [t0, t1, t2, t3]
    old_omega = [1.0, 1.0, 1.0, 1.0]
    old_gamma = [1.0, 1.0, 1.0, 1.0]
    h = t1 - t0
    np.testing.assert_approx_equal(h, t2 - t1)
    np.testing.assert_approx_equal(h, t3 - t2)

    if relaxation and projection:
        raise Exception("Use either relaxation or projection, not both.")

    if relaxation and method == None:
        method = "brentq"
    elif projection and method == None:
        method = "simplified Newton"

    t = t3
    gammas = [1.0, 1.0, 1.0, 1.0]
    step = 0
    while t < t_final and step < maxsteps:
        step += 1

        if relaxation and adapt_coefficients:
            om3 = old_omega[-3]
            om2 = old_omega[-2]
            om1 = old_omega[-1]
            du_new = 8 * (24 + om3 * (-16 + 3*om3) + om2 * (-8 + 3*om3)) / (3*om1 * (om1 + om2) * (om1 + om2 + om3)) * ff[-1]
            du_new = du_new - 8 * (3*om3 * (om2 + om3) - 8 * (-3 + om2 + 2*om3) + om1 * (-8 + 3*om3)) / (3 * om1 * om2 * (om2 + om3)) * ff[-2]
            du_new = du_new + 8 * (3 * (om2 + om3)*(om2 + om3) - 8 * (-3 + 2*om2 + 2*om3) + om1 * (-8 + 3*om2 + 3*om3)) / (3*om2 * (om1 + om2) * om3) * ff[-3]
            du_new = du_new + 4 * (16 * (-3 + 2*om2 + 3*om3) + 3 * (om2 + om3) * (om2 * (-2 + om3) + (-6 + om3) * om3) + om1 * (16 + 3*om2 * (-2 + om3) + 3 * (-4 + om3) * om3)) / (3 * om3 * (om2 + om3) * (om1 + om2 + om3)) * ff[-4]
            u_new = uu[-4] + h * du_new
        else:
            u_new = uu[-4] + h * ((8.0/3.0) * ff[-1] - (4.0/3.0) * ff[-2] + (8.0/3.0) * ff[-3])

        u_old = uu[-1]
        eta_old = eta(u_old)
        if projection:
            gamma, u_new = conservative_projection_solve(eta, deta, u_old, eta_old, u_new, method, tol, maxiter)
        elif relaxation:
            gamma = conservative_relaxation_solve(eta, deta, u_old, eta_old, u_new, old_gamma[-1], method, tol, maxiter)
            u_new = u_old + gamma * (u_new - u_old)
            old_gamma[-4] = old_gamma[-3]
            old_gamma[-3] = old_gamma[-2]
            old_gamma[-2] = old_gamma[-1]
            old_gamma[-1] = gamma
        else:
            gamma = 1.0

        if return_gamma:
            gammas.append(gamma)

        uu.append(u_new)
        if relaxation and adapt_dt:
            t = tt[-1] + gamma * h
            old_omega[-4] = old_omega[-3]
            old_omega[-3] = old_omega[-2]
            old_omega[-2] = old_omega[-1]
            old_omega[-1] = gamma
            if gamma < 1.0e-14:
                raise Exception("gamma = %.2e is too small in step %d!" % (gamma, step))
        else:
            t = tt[-1] + h
        tt.append(t)

        ff[-4] = ff[-3]
        ff[-3] = ff[-2]
        ff[-2] = ff[-1]
        ff[-1] = f(u_new)

    if return_gamma:
        return np.array(tt), uu, np.array(gammas)
    else:
        return np.array(tt), uu


# SSP LMMs with variable step size by Hadjimichael, Ketcheson, Lóczi, and Németh (2016)
def fixed_step_SSP32(uu, ff, h):
    u_new = 0.25 * uu[-3] + 0.75 * (uu[-1] + 2 * h * ff[-1])
    return u_new

def fixed_estimate_SSP32(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    eta_est = 0.25 * old_eta[-3] + 0.75 * (old_eta[-1] + 2 * h * old_deta_f[-1])
    return eta_est

def adaptive_step_SSP32(uu, ff, h, old_omega):
    Omega = old_omega[-1] + old_omega[-2]
    Omega2 = Omega * Omega
    u_new = (1.0/Omega2) * uu[-3] + (Omega2-1)/Omega2 * (uu[-1] + Omega/(Omega-1) * h * ff[-1])
    return u_new

def adaptive_estimate_SSP32(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    Omega = old_omega[-1] + old_omega[-2]
    Omega2 = Omega * Omega
    eta_est = (1.0/Omega2) * old_eta[-3] + (Omega2-1)/Omega2 * (old_eta[-1] + Omega/(Omega-1) * h * old_deta_f[-1])
    return eta_est

def adaptive_u_old_SSP32(old_omega):
    Omega = old_omega[-1] + old_omega[-2]
    Omega2 = Omega * Omega
    return [1.0/Omega2, 0.0, (Omega2-1)/Omega2]

def conservative_SSP32(f, t_final, t0, u0, t1, u1, t2, u2,
                       **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_SSP32, adaptive_step_SSP32,
                            **kwargs)

def cons_or_diss_SSP32(f, t_final, t0, u0, t1, u1, t2, u2,
                       **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_SSP32, adaptive_step_SSP32,
                            fixed_estimate_SSP32, adaptive_estimate_SSP32,
                            **kwargs)


def fixed_step_SSP43(uu, ff, h):
    u_new = (16./27.) * (uu[-1] + 3 * h * ff[-1]) + (11./27.) * (uu[-4] + (12./11.) * h * ff[-4])
    return u_new

def fixed_estimate_SSP43(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    eta_est = (16./27.) * (old_eta[-1] + 3 * h * old_deta_f[-1]) + (11./27.) * (old_eta[-4] + (12./11.) * h * old_deta_f[-4])
    return eta_est

def adaptive_step_SSP43(uu, ff, h, old_omega):
    Omega = old_omega[-1] + old_omega[-2] + old_omega[-3]
    Omega3 = Omega * Omega * Omega
    u_new = (
            (Omega+1) * (Omega+1) * (Omega-2) / Omega3
        ) * ( uu[-1] + Omega/(Omega-2) * h * ff[-1] ) + (
            (3*Omega + 2) / Omega3
        ) * ( uu[-4] + Omega*(Omega+1)/(3*Omega+2) * h * ff[-4] )
    return u_new

def adaptive_estimate_SSP43(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    Omega = old_omega[-1] + old_omega[-2] + old_omega[-3]
    Omega3 = Omega * Omega * Omega
    eta_est = (
            (Omega+1) * (Omega+1) * (Omega-2) / Omega3
        ) * ( old_eta[-1] + Omega/(Omega-2) * h * old_deta_f[-1] ) + (
            (3*Omega + 2) / Omega3
        ) * ( old_eta[-4] + Omega*(Omega+1)/(3*Omega+2) * h * old_deta_f[-4] )
    return eta_est

def adaptive_u_old_SSP43(old_omega):
    Omega = old_omega[-1] + old_omega[-2] + old_omega[-3]
    Omega3 = Omega * Omega * Omega
    return [(3*Omega + 2) / Omega3, 0.0, 0.0, (Omega+1) * (Omega+1) * (Omega-2) / Omega3]

def conservative_SSP43(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                       **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_SSP43, adaptive_step_SSP43,
                            **kwargs)

def cons_or_diss_SSP43(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                       **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_SSP43, adaptive_step_SSP43,
                            fixed_estimate_SSP43, adaptive_estimate_SSP43,
                            **kwargs)


# SSP LMMs with variable step size by Mohammadi, Arévalo, and Führer (2019), based on Arévalo & Söderlind (2017)
def fixed_step_SSP32AS(uu, ff, h):
    u_new = (
            0.75
        ) * uu[-1] + (
            0.25
        ) * uu[-3] + h * (
            (
                1.5
            ) * ff[-1]
        )
    return u_new

def fixed_estimate_SSP32AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            0.9375
        ) * uu[-1] + (
            0.0625
        ) * uu[-3] + h * (
            (
                0.625
            ) * ff[-1]
        )

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def adaptive_step_SSP32AS(uu, ff, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            1 - 1/((om1 + om2)*(om1 + om2))
        ) * uu[-1] + (
            1/((om1 + om2)*(om1 + om2))
        ) * uu[-3] + h * (
            (
                1 + 1/(om1 + om2)
            ) * ff[-1]
        )

    return u_new

def adaptive_estimate_SSP32AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            1 - 1/(4.*((om1 + om2)*(om1 + om2)))
        ) * uu[-1] + (
            1/(4.*((om1 + om2)*(om1 + om2)))
        ) * uu[-3] + h * (
            (
                (2 + 1/(om1 + om2))/4.
            ) * ff[-1]
        )

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def conservative_SSP32AS(f, t_final, t0, u0, t1, u1, t2, u2,
                         **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_SSP32AS, adaptive_step_SSP32AS,
                            **kwargs)

def cons_or_diss_SSP32AS(f, t_final, t0, u0, t1, u1, t2, u2,
                         **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_SSP32AS, adaptive_step_SSP32AS,
                            fixed_estimate_SSP32AS, adaptive_estimate_SSP32AS,
                            **kwargs)


def fixed_step_SSP43AS(uu, ff, h):
    u_new = (
            0.5925925925925926
        ) * uu[-1] + (
            0.4074074074074074
        ) * uu[-4] + h * (
            (
                1.7777777777777777
            ) * ff[-1] + (
                0.4444444444444444
            ) * ff[-4]
        )
    return u_new

def fixed_estimate_SSP43AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            0.9844148679880743
        ) * uu[-1] + (
            0.015585132011925783
        ) * uu[-4] + h * (
            (
                0.2421455965461624
            ) * ff[-1] + (
                0.01593466489480193
            ) * ff[-4]
        )
    u2int = (
            0.7563258727526666
        ) * uu[-1] + (
            0.2436741272473335
        ) * uu[-4] + h * (
            (
                1.2578544034538375
            ) * ff[-1] + (
                0.26184311288297585
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_SSP43AS(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            ((-2 + om1 + om2 + om3)*((1 + om1 + om2 + om3)*(1 + om1 + om2 + om3)))/((om1 + om2 + om3)*(om1 + om2 + om3)*(om1 + om2 + om3))
        ) * uu[-1] + (
            (2 + 3*(om1 + om2 + om3))/((om1 + om2 + om3)*(om1 + om2 + om3)*(om1 + om2 + om3))
        ) * uu[-4] + h * (
            (
                ((1 + om1 + om2 + om3)*(1 + om1 + om2 + om3))/((om1 + om2 + om3)*(om1 + om2 + om3))
            ) * ff[-1] + (
                (1 + om1 + om2 + om3)/((om1 + om2 + om3)*(om1 + om2 + om3))
            ) * ff[-4]
        )

    return u_new

def adaptive_estimate_SSP43AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            1 + (-9 + 5*np.sqrt(3))/(18.*((om1 + om2 + om3)*(om1 + om2 + om3)*(om1 + om2 + om3))) + (-2 + np.sqrt(3))/(2.*((om1 + om2 + om3)*(om1 + om2 + om3)))
        ) * uu[-1] + (
            (9 - 5*np.sqrt(3) + 18*(om1 + om2 + om3) - 9*np.sqrt(3)*(om1 + om2 + om3))/(18.*((om1 + om2 + om3)*(om1 + om2 + om3)*(om1 + om2 + om3)))
        ) * uu[-4] + h * (
            (
                (-6*(-3 + np.sqrt(3)) + (9 - 5*np.sqrt(3))/((om1 + om2 + om3)*(om1 + om2 + om3)) - (12*(-2 + np.sqrt(3)))/(om1 + om2 + om3))/36.
            ) * ff[-1] + (
                (9 - 5*np.sqrt(3) - 6*(-2 + np.sqrt(3))*(om1 + om2 + om3))/(36.*((om1 + om2 + om3)*(om1 + om2 + om3)))
            ) * ff[-4]
        )
    u2int = (
            -(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*(om1 + om2 + om3) - 18*((om1 + om2 + om3)*(om1 + om2 + om3)*(om1 + om2 + om3)))/(18.*((om1 + om2 + om3)*(om1 + om2 + om3)*(om1 + om2 + om3)))
        ) * uu[-1] + (
            (9 + 5*np.sqrt(3) + 18*(om1 + om2 + om3) + 9*np.sqrt(3)*(om1 + om2 + om3))/(18.*((om1 + om2 + om3)*(om1 + om2 + om3)*(om1 + om2 + om3)))
        ) * uu[-4] + h * (
            (
                (6*(3 + np.sqrt(3)) + (9 + 5*np.sqrt(3))/((om1 + om2 + om3)*(om1 + om2 + om3)) + (12*(2 + np.sqrt(3)))/(om1 + om2 + om3))/36.
            ) * ff[-1] + (
                (9 + 5*np.sqrt(3) + 6*(2 + np.sqrt(3))*(om1 + om2 + om3))/(36.*((om1 + om2 + om3)*(om1 + om2 + om3)))
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_SSP43AS(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                         **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_SSP43AS, adaptive_step_SSP43AS,
                            **kwargs)

def cons_or_diss_SSP43AS(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                         **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_SSP43AS, adaptive_step_SSP43AS,
                            fixed_estimate_SSP43AS, adaptive_estimate_SSP43AS,
                            **kwargs)


# extrapolated BDF (eBDF) methods with variable step size
def fixed_step_eBDF2(uu, ff, h):
    u_new = (
            1.3333333333333333
        ) * uu[-1] + (
            -0.3333333333333333
        ) * uu[-2] + h * (
            (
                1.3333333333333333
            ) * ff[-1] + (
                -0.6666666666666666
            ) * ff[-2]
        )
    return u_new

def adaptive_step_eBDF2(uu, ff, h, old_omega):
    om1 = old_omega[-1]
    u_new = (
            ((1 + om1)*(1 + om1))/(om1*(2 + om1))
        ) * uu[-1] + (
            -(1/(2*om1 + om1*om1))
        ) * uu[-2] + h * (
            (
                ((1 + om1)*(1 + om1))/(om1*(2 + om1))
            ) * ff[-1] + (
                -((1 + om1)/(2*om1 + om1*om1))
            ) * ff[-2]
        )

    return u_new

def conservative_eBDF2(f, t_final, t0, u0, t1, u1,
                     idx_u_old=-1,
                     eta=etaL2, deta=detaL2,
                     return_gamma=False,
                     projection=False, relaxation=False,
                     adapt_dt=False, adapt_coefficients=False,
                     method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    return conservative_LMM(f, t_final, [t0, t1], [u0, u1],
                     fixed_step_eBDF2, adaptive_step_eBDF2,
                     idx_u_old,
                     eta, deta,
                     return_gamma,
                     projection, relaxation,
                     adapt_dt, adapt_coefficients,
                     method, tol, maxiter, maxsteps)


def fixed_step_eBDF3(uu, ff, h):
    u_new = (
            1.6363636363636365
        ) * uu[-1] + (
            -0.8181818181818182
        ) * uu[-2] + (
            0.18181818181818182
        ) * uu[-3] + h * (
            (
                1.6363636363636365
            ) * ff[-1] + (
                -1.6363636363636365
            ) * ff[-2] + (
                0.5454545454545454
            ) * ff[-3]
        )
    return u_new

def adaptive_step_eBDF3(uu, ff, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            ((1 + om1)*(1 + om1)*((1 + om1 + om2)*(1 + om1 + om2)))/(om1*(om1 + om2)*(3 + 2*om2 + om1*(4 + om1 + om2)))
        ) * uu[-1] + (
            -(((1 + om1 + om2)*(1 + om1 + om2))/(om1*om2*(3 + 2*om2 + om1*(4 + om1 + om2))))
        ) * uu[-2] + (
            ((1 + om1)*(1 + om1))/(om2*(om1 + om2)*(3 + 2*om2 + om1*(4 + om1 + om2)))
        ) * uu[-3] + h * (
            (
                ((1 + om1)*(1 + om1)*((1 + om1 + om2)*(1 + om1 + om2)))/(om1*(om1 + om2)*(3 + 2*om2 + om1*(4 + om1 + om2)))
            ) * ff[-1] + (
                -(((1 + om1)*((1 + om1 + om2)*(1 + om1 + om2)))/(om1*om2*(3 + 2*om2 + om1*(4 + om1 + om2))))
            ) * ff[-2] + (
                ((1 + om1)*(1 + om1)*(1 + om1 + om2))/(om2*(om1 + om2)*(3 + 2*om2 + om1*(4 + om1 + om2)))
            ) * ff[-3]
        )

    return u_new

def conservative_eBDF3(f, t_final, t0, u0, t1, u1, t2, u2,
                     idx_u_old=-1,
                     eta=etaL2, deta=detaL2,
                     return_gamma=False,
                     projection=False, relaxation=False,
                     adapt_dt=False, adapt_coefficients=False,
                     method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    return conservative_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                     fixed_step_eBDF3, adaptive_step_eBDF3,
                     idx_u_old,
                     eta, deta,
                     return_gamma,
                     projection, relaxation,
                     adapt_dt, adapt_coefficients,
                     method, tol, maxiter, maxsteps)


def fixed_step_eBDF4(uu, ff, h):
    u_new = (
            1.92
        ) * uu[-1] + (
            -1.44
        ) * uu[-2] + (
            0.64
        ) * uu[-3] + (
            -0.12
        ) * uu[-4] + h * (
            (
                1.92
            ) * ff[-1] + (
                -2.88
            ) * ff[-2] + (
                1.92
            ) * ff[-3] + (
                -0.48
            ) * ff[-4]
        )
    return u_new

def adaptive_step_eBDF4(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            ((1 + om1)*(1 + om1)*((1 + om1 + om2)*(1 + om1 + om2))*((1 + om1 + om2 + om3)*(1 + om1 + om2 + om3)))/(om1*(om1 + om2)*(om1 + om2 + om3)*(4 + om1*om1*om1 + 3*om3 + 2*om2*(3 + om2 + om3) + om1*om1*(6 + 2*om2 + om3) + om1*(9 + 4*om3 + om2*(8 + om2 + om3))))
        ) * uu[-1] + (
            -(((1 + om1 + om2)*(1 + om1 + om2)*((1 + om1 + om2 + om3)*(1 + om1 + om2 + om3)))/(om1*om2*(om2 + om3)*(4 + om1*om1*om1 + 3*om3 + 2*om2*(3 + om2 + om3) + om1*om1*(6 + 2*om2 + om3) + om1*(9 + 4*om3 + om2*(8 + om2 + om3)))))
        ) * uu[-2] + (
            ((1 + om1)*(1 + om1)*((1 + om1 + om2 + om3)*(1 + om1 + om2 + om3)))/(om2*(om1 + om2)*om3*(4 + om1*om1*om1 + 3*om3 + 2*om2*(3 + om2 + om3) + om1*om1*(6 + 2*om2 + om3) + om1*(9 + 4*om3 + om2*(8 + om2 + om3))))
        ) * uu[-3] + (
            -(((1 + om1)*(1 + om1)*((1 + om1 + om2)*(1 + om1 + om2)))/(om3*(om2 + om3)*(om1 + om2 + om3)*(4 + om1*om1*om1 + 3*om3 + 2*om2*(3 + om2 + om3) + om1*om1*(6 + 2*om2 + om3) + om1*(9 + 4*om3 + om2*(8 + om2 + om3)))))
        ) * uu[-4] + h * (
            (
                ((1 + om1)*(1 + om1)*((1 + om1 + om2)*(1 + om1 + om2))*((1 + om1 + om2 + om3)*(1 + om1 + om2 + om3)))/(om1*(om1 + om2)*(om1 + om2 + om3)*(4 + om1*om1*om1 + 3*om3 + 2*om2*(3 + om2 + om3) + om1*om1*(6 + 2*om2 + om3) + om1*(9 + 4*om3 + om2*(8 + om2 + om3))))
            ) * ff[-1] + (
                -(((1 + om1)*((1 + om1 + om2)*(1 + om1 + om2))*((1 + om1 + om2 + om3)*(1 + om1 + om2 + om3)))/(om1*om2*(om2 + om3)*(4 + om1*om1*om1 + 3*om3 + 2*om2*(3 + om2 + om3) + om1*om1*(6 + 2*om2 + om3) + om1*(9 + 4*om3 + om2*(8 + om2 + om3)))))
            ) * ff[-2] + (
                ((1 + om1)*(1 + om1)*(1 + om1 + om2)*((1 + om1 + om2 + om3)*(1 + om1 + om2 + om3)))/(om2*(om1 + om2)*om3*(4 + om1*om1*om1 + 3*om3 + 2*om2*(3 + om2 + om3) + om1*om1*(6 + 2*om2 + om3) + om1*(9 + 4*om3 + om2*(8 + om2 + om3))))
            ) * ff[-3] + (
                -(((1 + om1)*(1 + om1)*((1 + om1 + om2)*(1 + om1 + om2))*(1 + om1 + om2 + om3))/(om3*(om2 + om3)*(om1 + om2 + om3)*(4 + om1*om1*om1 + 3*om3 + 2*om2*(3 + om2 + om3) + om1*om1*(6 + 2*om2 + om3) + om1*(9 + 4*om3 + om2*(8 + om2 + om3)))))
            ) * ff[-4]
        )

    return u_new

def conservative_eBDF4(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                       **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_eBDF4, adaptive_step_eBDF4,
                            **kwargs)


# extrapolated BDF (eBDF) methods with variable step size version of Arévalo & Söderlind (2017)
def fixed_step_eBDF2AS(uu, ff, h):
    u_new = (
            1.3333333333333333
        ) * uu[-1] + (
            -0.3333333333333333
        ) * uu[-2] + h * (
            (
                1.3333333333333333
            ) * ff[-1] + (
                -0.6666666666666666
            ) * ff[-2]
        )
    return u_new

def fixed_estimate_eBDF2AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            1.0833333333333333
        ) * uu[-1] + (
            -0.08333333333333333
        ) * uu[-2] + h * (
            (
                0.5833333333333334
            ) * ff[-1] + (
                -0.16666666666666666
            ) * ff[-2]
        )

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def adaptive_step_eBDF2AS(uu, ff, h, old_omega):
    om1 = old_omega[-1]
    u_new = (
            1 + 1/(3.*(om1*om1))
        ) * uu[-1] + (
            -1/(3.*(om1*om1))
        ) * uu[-2] + h * (
            (
                1 + 1/(3.*om1)
            ) * ff[-1] + (
                -2/(3.*om1)
            ) * ff[-2]
        )

    return u_new

def adaptive_estimate_eBDF2AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om1 = old_omega[-1]
    u1int = (
            1 + 1/(12.*(om1*om1))
        ) * uu[-1] + (
            -1/(12.*(om1*om1))
        ) * uu[-2] + h * (
            (
                (6 + 1/om1)/12.
            ) * ff[-1] + (
                -1/(6.*om1)
            ) * ff[-2]
        )

    eta_est = old_eta[-1] + h * ( 1 * np.dot(deta(u1int), f(u1int)) )
    return eta_est

def conservative_eBDF2AS(f, t_final, t0, u0, t1, u1,
                         **kwargs):
    return conservative_LMM(f, t_final, [t0, t1], [u0, u1],
                            fixed_step_eBDF2AS, adaptive_step_eBDF2AS,
                            **kwargs)

def cons_or_diss_eBDF2AS(f, t_final, t0, u0, t1, u1,
                         **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1], [u0, u1],
                            fixed_step_eBDF2AS, adaptive_step_eBDF2AS,
                            fixed_estimate_eBDF2AS, adaptive_estimate_eBDF2AS,
                            **kwargs)


def fixed_step_eBDF3AS(uu, ff, h):
    u_new = (
            1.6363636363636365
        ) * uu[-1] + (
            -0.8181818181818182
        ) * uu[-2] + (
            0.18181818181818182
        ) * uu[-3] + h * (
            (
                1.6363636363636365
            ) * ff[-1] + (
                -1.6363636363636365
            ) * ff[-2] + (
                0.5454545454545454
            ) * ff[-3]
        )
    return u_new

def fixed_estimate_eBDF3AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            1.0244164888675966
        ) * uu[-1] + (
            -0.030134742440450477
        ) * uu[-2] + (
            0.005718253572853871
        ) * uu[-3] + h * (
            (
                0.23574135427278373
            ) * ff[-1] + (
                -0.060269484880900955
            ) * ff[-2] + (
                0.01715476071856161
            ) * ff[-3]
        )
    u2int = (
            1.3808865414354339
        ) * uu[-1] + (
            -0.4850167727110647
        ) * uu[-2] + (
            0.104130231275631
        ) * uu[-3] + h * (
            (
                1.1695616760302463
            ) * ff[-1] + (
                -0.9700335454221294
            ) * ff[-2] + (
                0.31239069382689294
            ) * ff[-3]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_eBDF3AS(uu, ff, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            (2*(-2 + om1)*(om1*om1)*((1 + om1)*(1 + om1)) - 2*(-2 + om1)*om1*((1 + om1)*(1 + om1))*om2 + 5*(1 + 3*om1 + 4*(om1*om1*om1))*(om2*om2) + 8*(1 + 3*(om1*om1))*(om2*om2*om2))/(2.*(om1*om1)*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-1] + (
            (om1 + om1*om1 - 7*om1*om2 - om2*(5 + 8*om2))/(2.*(om1*om1)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-2] + (
            (3 + 5*om1)/(2.*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-3] + h * (
            (
                (2*((om1 + om1*om1)*(om1 + om1*om1)) - 2*om1*((1 + om1)*(1 + om1))*om2 + 5*(1 + om1*(3 + 4*om1))*(om2*om2) + 8*(1 + 3*om1)*(om2*om2*om2))/(2.*om1*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
            ) * ff[-1] + (
                (om1 + om1*om1 - 5*om2 - 7*om1*om2 - 8*(om2*om2))/(om1*om1*om1 - 2*(om1*om1)*om2 + 12*om1*(om2*om2))
            ) * ff[-2] + (
                (3*(3 + 5*om1)*om2)/(2.*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
            ) * ff[-3]
        )

    return u_new

def adaptive_estimate_eBDF3AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            (4*(om1*om1)*(-9 + 5*np.sqrt(3) + 9*om1*(-2 + np.sqrt(3) + 2*(om1*om1))) - 4*om1*(-9 + 5*np.sqrt(3) + 9*om1*(-2 + np.sqrt(3) + 2*(om1*om1)))*om2 + 5*(9 - 5*np.sqrt(3) + 18*om1*(2 - np.sqrt(3) + 8*(om1*om1)))*(om2*om2) - 48*(-2 + np.sqrt(3) - 18*(om1*om1))*(om2*om2*om2))/(72.*(om1*om1)*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-1] + (
            (-6*(-2 + np.sqrt(3))*(om1*om1) + om1*(9 - 5*np.sqrt(3) + 42*(-2 + np.sqrt(3))*om2) + om2*(5*(-9 + 5*np.sqrt(3)) + 48*(-2 + np.sqrt(3))*om2))/(72.*(om1*om1)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-2] + (
            (9 - 5*np.sqrt(3) - 10*(-2 + np.sqrt(3))*om1)/(24.*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-3] + h * (
            (
                -(20*(-2 + np.sqrt(3) + 3*(-3 + np.sqrt(3))*om1) + (3*(-9 + 5*np.sqrt(3) + 10*(-2 + np.sqrt(3))*om1))/(om1 + om2) + ((-9 + 5*np.sqrt(3) + 10*(-2 + np.sqrt(3))*om1)*(7*om1 - 11*om2))/(om1*om1 - 2*om1*om2 + 12*(om2*om2)))/(360.*om1)
            ) * ff[-1] + (
                (-6*(-2 + np.sqrt(3))*(om1*om1) + om1*(9 - 5*np.sqrt(3) + 42*(-2 + np.sqrt(3))*om2) + om2*(5*(-9 + 5*np.sqrt(3)) + 48*(-2 + np.sqrt(3))*om2))/(36.*om1*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
            ) * ff[-2] + (
                ((9 - 5*np.sqrt(3) - 10*(-2 + np.sqrt(3))*om1)*om2)/(8.*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
            ) * ff[-3]
        )
    u2int = (
            (4*(om1*om1)*(-9 - 5*np.sqrt(3) - 9*(2 + np.sqrt(3))*om1 + 18*(om1*om1*om1)) + 4*om1*(9 + 5*np.sqrt(3) + 9*om1*(2 + np.sqrt(3) - 2*(om1*om1)))*om2 + 5*(9 + 5*np.sqrt(3) + 18*om1*(2 + np.sqrt(3) + 8*(om1*om1)))*(om2*om2) + 48*(2 + np.sqrt(3) + 18*(om1*om1))*(om2*om2*om2))/(72.*(om1*om1)*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-1] + (
            (6*(2 + np.sqrt(3))*(om1*om1) + om1*(9 + 5*np.sqrt(3) - 42*(2 + np.sqrt(3))*om2) - om2*(5*(9 + 5*np.sqrt(3)) + 48*(2 + np.sqrt(3))*om2))/(72.*(om1*om1)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-2] + (
            (9 + 5*np.sqrt(3) + 10*(2 + np.sqrt(3))*om1)/(24.*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
        ) * uu[-3] + h * (
            (
                (20*(2 + np.sqrt(3) + 3*(3 + np.sqrt(3))*om1) + (3*(9 + 5*np.sqrt(3) + 10*(2 + np.sqrt(3))*om1))/(om1 + om2) + ((9 + 5*np.sqrt(3) + 10*(2 + np.sqrt(3))*om1)*(7*om1 - 11*om2))/(om1*om1 - 2*om1*om2 + 12*(om2*om2)))/(360.*om1)
            ) * ff[-1] + (
                (6*(2 + np.sqrt(3))*(om1*om1) + om1*(9 + 5*np.sqrt(3) - 42*(2 + np.sqrt(3))*om2) - om2*(5*(9 + 5*np.sqrt(3)) + 48*(2 + np.sqrt(3))*om2))/(36.*om1*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
            ) * ff[-2] + (
                ((9 + 5*np.sqrt(3) + 10*(2 + np.sqrt(3))*om1)*om2)/(8.*(om1 + om2)*(om1*om1 - 2*om1*om2 + 12*(om2*om2)))
            ) * ff[-3]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_eBDF3AS(f, t_final, t0, u0, t1, u1, t2, u2,
                         **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_eBDF3AS, adaptive_step_eBDF3AS,
                            **kwargs)

def cons_or_diss_eBDF3AS(f, t_final, t0, u0, t1, u1, t2, u2,
                         **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_eBDF3AS, adaptive_step_eBDF3AS,
                            fixed_estimate_eBDF3AS, adaptive_estimate_eBDF3AS,
                            **kwargs)


def fixed_step_eBDF4AS(uu, ff, h):
    u_new = (
            1.92
        ) * uu[-1] + (
            -1.44
        ) * uu[-2] + (
            0.64
        ) * uu[-3] + (
            -0.12
        ) * uu[-4] + h * (
            (
                1.92
            ) * ff[-1] + (
                -2.88
            ) * ff[-2] + (
                1.92
            ) * ff[-3] + (
                -0.48
            ) * ff[-4]
        )
    return u_new

def fixed_estimate_eBDF4AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            1.031595708345322
        ) * uu[-1] + (
            -0.045873800526233105
        ) * uu[-2] + (
            0.01731545426764106
        ) * uu[-3] + (
            -0.003037362086729975
        ) * uu[-4] + h * (
            (
                0.2429205737505091
            ) * ff[-1] + (
                -0.09174760105246621
            ) * ff[-2] + (
                0.05194636280292318
            ) * ff[-3] + (
                -0.0121494483469199
            ) * ff[-4]
        )
    u2int = (
            1.5346388595559126
        ) * uu[-1] + (
            -0.8220891624367299
        ) * uu[-2] + (
            0.35249936054717373
        ) * uu[-3] + (
            -0.06504905766635645
        ) * uu[-4] + h * (
            (
                1.3233139941507255
            ) * ff[-1] + (
                -1.6441783248734598
            ) * ff[-2] + (
                1.0574980816415214
            ) * ff[-3] + (
                -0.2601962306654258
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_eBDF4AS(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            -((3*(om2*om2)*(om1 + om2)*((1 + om1 + om2)*(1 + om1 + om2))*(-4*(om1*om1*om1*om1) - om1*om1*om1*(-8 + om2) + 3*om1*om2 + om2*om2 + 3*(om1*om1)*(4 + (-2 + om2)*om2)) - 3*om2*(om1 + om2)*((1 + om1 + om2)*(1 + om1 + om2))*(-4*(om1*om1*om1*om1) - om1*om1*om1*(-8 + om2) + 3*om1*om2 + om2*om2 + 3*(om1*om1)*(4 + (-2 + om2)*om2))*om3 + 7*(2*(-3 + om1)*(om1*om1*om1)*((1 + om1)*(1 + om1)*(1 + om1)) + 2*(om1*om1)*((1 + om1)*(1 + om1))*(-3 + 5*(-2 + om1)*om1)*om2 + 12*om1*((1 + om1)*(1 + om1))*(om2*om2) + (7 + om1*(52 + 78*om1 + 23*(om1*om1*om1)))*(om2*om2*om2) + 19*(1 + 3*om1 + 4*(om1*om1*om1))*(om2*om2*om2*om2) + 15*(1 + 3*(om1*om1))*(om2*om2*om2*om2*om2))*(om3*om3) + 11*(4*(om1*om1)*((1 + om1)*(1 + om1))*(-1 + (-2 + om1)*om1) + 4*om1*((-1 + om1*om1)*(-1 + om1*om1))*om2 + (5 + om1*(36 + 54*om1 + 17*(om1*om1*om1)))*(om2*om2) + 20*(1 + 3*om1 + 4*(om1*om1*om1))*(om2*om2*om2) + 21*(1 + 3*(om1*om1))*(om2*om2*om2*om2))*(om3*om3*om3) + 15*(2*(-2 + om1)*(om1*om1)*((1 + om1)*(1 + om1)) - 2*(-2 + om1)*om1*((1 + om1)*(1 + om1))*om2 + 5*(1 + 3*om1 + 4*(om1*om1*om1))*(om2*om2) + 8*(1 + 3*(om1*om1))*(om2*om2*om2))*(om3*om3*om3*om3))/(om1*om1*(om1 + om2)*(om1 + om2 + om3)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3))))
        ) * uu[-1] + (
            (3*om2*((om1 + om2)*(om1 + om2))*((1 + om1 + om2)*(1 + om1 + om2)) - 3*(om1 + om2)*((1 + om1 + om2)*(1 + om1 + om2))*(om1 + 2*om2)*om3 + (-(om1*(1 + om1)*(11 + 18*om1)) + (55 + om1*(116 + 75*om1))*om2 + (145 + 204*om1)*(om2*om2) + 111*(om2*om2*om2))*(om3*om3) - 15*(om1 + om1*om1 - 7*om1*om2 - om2*(5 + 8*om2))*(om3*om3*om3))/(om1*om1*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
        ) * uu[-2] + (
            (-2*(om1*om1*om1*om1) + om1*om1*om1*(-4 + om2 + 5*om3) + 2*(om1*om1)*(-1 + om2 + 4*(om2*om2) - 30*om2*om3 + (5 - 34*om3)*om3) + om1*(om2 + om2*om2*(9 + 5*om2) + 5*om3 - 13*om2*(6 + 5*om2)*om3 - 29*(3 + 5*om2)*(om3*om3) - 75*(om3*om3*om3)) + 3*(om2 + om3)*(om2 + om2*om2 - 14*om2*om3 - om3*(11 + 15*om3)))/((om1 + om2)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
        ) * uu[-3] + (
            (2*(om1*om1)*((1 + om1)*(1 + om1)) - 4*om1*((1 + om1)*(1 + om1))*om2 + (24 + 7*om1*(9 + 7*om1))*(om2*om2) + 11*(3 + 5*om1)*(om2*om2*om2))/(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*((om1 + om2)*(om1 + om2)*(om1 + om2))*om3 - 7*(om1 + om2)*(2*(om1*om1*om1) + 6*(om1*om1)*om2 - 14*om1*(om2*om2) + 45*(om2*om2*om2))*(om3*om3) - 11*(4*(om1*om1*om1) + 17*om1*(om2*om2) + 63*(om2*om2*om2))*(om3*om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3*om3))
        ) * uu[-4] + h * (
            (
                -((3*(om2*om2)*((om1 + om2)*(om1 + om2))*((1 + om1 + om2)*(1 + om1 + om2))*(-4*om1*(1 + om1) + om2 + 3*om1*om2) - 3*om2*((om1 + om2)*(om1 + om2))*((1 + om1 + om2)*(1 + om1 + om2))*(-4*om1*(1 + om1) + om2 + 3*om1*om2)*om3 + 7*(2*(om1*om1*om1)*((1 + om1)*(1 + om1)*(1 + om1)) + 2*(om1*om1)*((1 + om1)*(1 + om1))*(2 + 5*om1)*om2 - 6*om1*((1 + om1)*(1 + om1))*(om2*om2) + (7 + om1*(16 + om1*(18 + 23*om1)))*(om2*om2*om2) + 19*(1 + om1*(3 + 4*om1))*(om2*om2*om2*om2) + 15*(1 + 3*om1)*(om2*om2*om2*om2*om2))*(om3*om3) + 11*(2*(om1*om1)*((1 + om1)*(1 + om1))*(1 + 2*om1) + 2*om1*((1 + om1)*(1 + om1))*(-1 + 2*om1)*om2 + (5 + om1*(12 + om1*(14 + 17*om1)))*(om2*om2) + 20*(1 + om1*(3 + 4*om1))*(om2*om2*om2) + 21*(1 + 3*om1)*(om2*om2*om2*om2))*(om3*om3*om3) + 15*(2*(om1*om1)*((1 + om1)*(1 + om1)) - 2*om1*((1 + om1)*(1 + om1))*om2 + 5*(1 + om1*(3 + 4*om1))*(om2*om2) + 8*(1 + 3*om1)*(om2*om2*om2))*(om3*om3*om3*om3))/(om1*(om1 + om2)*(om1 + om2 + om3)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3))))
            ) * ff[-1] + (
                (2*(3*om2*((om1 + om2)*(om1 + om2))*((1 + om1 + om2)*(1 + om1 + om2)) - 3*(om1 + om2)*((1 + om1 + om2)*(1 + om1 + om2))*(om1 + 2*om2)*om3 + (-(om1*(1 + om1)*(11 + 18*om1)) + (55 + om1*(116 + 75*om1))*om2 + (145 + 204*om1)*(om2*om2) + 111*(om2*om2*om2))*(om3*om3) - 15*(om1 + om1*om1 - 7*om1*om2 - om2*(5 + 8*om2))*(om3*om3*om3)))/(om1*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
            ) * ff[-2] + (
                (3*om2*(-2*(om1*om1*om1*om1) + om1*om1*om1*(-4 + om2 + 5*om3) + 2*(om1*om1)*(-1 + om2 + 4*(om2*om2) - 30*om2*om3 + (5 - 34*om3)*om3) + om1*(om2 + om2*om2*(9 + 5*om2) + 5*om3 - 13*om2*(6 + 5*om2)*om3 - 29*(3 + 5*om2)*(om3*om3) - 75*(om3*om3*om3)) + 3*(om2 + om3)*(om2 + om2*om2 - 14*om2*om3 - om3*(11 + 15*om3))))/((om1 + om2)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
            ) * ff[-3] + (
                (4*(2*(om1*om1)*((1 + om1)*(1 + om1)) - 4*om1*((1 + om1)*(1 + om1))*om2 + (24 + 7*om1*(9 + 7*om1))*(om2*om2) + 11*(3 + 5*om1)*(om2*om2*om2))*om3)/(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*((om1 + om2)*(om1 + om2)*(om1 + om2))*om3 - 7*(om1 + om2)*(2*(om1*om1*om1) + 6*(om1*om1)*om2 - 14*om1*(om2*om2) + 45*(om2*om2*om2))*(om3*om3) - 11*(4*(om1*om1*om1) + 17*om1*(om2*om2) + 63*(om2*om2*om2))*(om3*om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3*om3))
            ) * ff[-4]
        )

    return u_new

def adaptive_estimate_eBDF4AS(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            (18*(-2 + np.sqrt(3) - 18*(om1*om1))*(om2*om2*om2*om2*om2*om2*om2) + 2*(om1*om1)*(om3*om3)*(-7*om1*(3*(-7 + 4*np.sqrt(3)) + 4*om1*(2*(-9 + 5*np.sqrt(3)) + 9*om1*(-2 + np.sqrt(3) + om1*om1))) - 22*(-7 + 4*np.sqrt(3) + 4*om1*(-9 + 5*np.sqrt(3) + 6*(-2 + np.sqrt(3))*om1 + 9*(om1*om1*om1)))*om3 - 30*(-9 + 5*np.sqrt(3) + 9*om1*(-2 + np.sqrt(3) + 2*(om1*om1)))*(om3*om3)) + om2*om2*om2*om2*(12*om1*(-7 + 4*np.sqrt(3) + 8*(-9 + 5*np.sqrt(3))*om1 + 60*(-2 + np.sqrt(3))*(om1*om1) + 108*(om1*om1*om1*om1)) + 3*(7 - 4*np.sqrt(3))*om3 + 6*om1*(45 - 25*np.sqrt(3) - 45*(-2 + np.sqrt(3))*om1 + 36*(om1*om1*om1))*om3 + 133*(-9 + 5*np.sqrt(3) + 18*om1*(-2 + np.sqrt(3) - 8*(om1*om1)))*(om3*om3) + 1386*(-2 + np.sqrt(3) - 18*(om1*om1))*(om3*om3*om3)) + 2*om1*om2*om3*(-6*(om1*om1)*(3*(-7 + 4*np.sqrt(3)) + 4*om1*(2*(-9 + 5*np.sqrt(3)) + 9*om1*(-2 + np.sqrt(3) + om1*om1))) - 7*om1*(3*(-7 + 4*np.sqrt(3)) + 4*om1*(4*(-9 + 5*np.sqrt(3)) + 27*(-2 + np.sqrt(3))*om1 + 45*(om1*om1*om1)))*om3 - 22*(7 - 4*np.sqrt(3) + 12*(om1*om1)*(-2 + np.sqrt(3) + 3*(om1*om1)))*(om3*om3) + 30*(-9 + 5*np.sqrt(3) + 9*om1*(-2 + np.sqrt(3) + 2*(om1*om1)))*(om3*om3*om3)) + om2*om2*om2*(3*(om1*om1)*(15*(-7 + 4*np.sqrt(3)) + 56*(-9 + 5*np.sqrt(3))*om1 + 324*(-2 + np.sqrt(3))*(om1*om1) + 468*(om1*om1*om1*om1)) - 12*om1*(-7 + 4*np.sqrt(3) + 8*(-9 + 5*np.sqrt(3))*om1 + 60*(-2 + np.sqrt(3))*(om1*om1) + 108*(om1*om1*om1*om1))*om3 + 7*(7*(-7 + 4*np.sqrt(3)) + 52*(-9 + 5*np.sqrt(3))*om1 + 468*(-2 + np.sqrt(3))*(om1*om1) - 828*(om1*om1*om1*om1))*(om3*om3) + 220*(-9 + 5*np.sqrt(3) + 18*(-2 + np.sqrt(3))*om1 - 144*(om1*om1*om1))*(om3*om3*om3) + 720*(-2 + np.sqrt(3) - 18*(om1*om1))*(om3*om3*om3*om3)) + 6*(om2*om2*om2*om2*om2*om2)*(-9 + 5*np.sqrt(3) - 3*(-2 + np.sqrt(3))*om3 + 18*om1*(-2 + np.sqrt(3) - 8*(om1*om1) + 3*om1*om3)) + 3*(om2*om2*om2*om2*om2)*(-7 + 4*np.sqrt(3) - 72*(om1*om1*om1*om1) + 288*(om1*om1*om1)*om3 + om1*(-90 + 50*np.sqrt(3) - 36*(-2 + np.sqrt(3))*om3) + 2*om3*(9 - 5*np.sqrt(3) + 105*(-2 + np.sqrt(3))*om3) + 90*(om1*om1)*(-2 + np.sqrt(3) - 42*(om3*om3))) + om2*om2*(432*(-2 + np.sqrt(3))*(om1*om1*om1*om1*om1) + 432*(om1*om1*om1*om1*om1*om1*om1) - 1404*(om1*om1*om1*om1*om1*om1)*om3 + 5*(om3*om3*om3)*(-77 + 44*np.sqrt(3) + 15*(-9 + 5*np.sqrt(3))*om3) + 6*om1*(om3*om3)*(-98 + 56*np.sqrt(3) + 66*(-9 + 5*np.sqrt(3))*om3 + 225*(-2 + np.sqrt(3))*(om3*om3)) + 3*(om1*om1)*om3*(105 - 60*np.sqrt(3) + 56*(-9 + 5*np.sqrt(3))*om3 + 1188*(-2 + np.sqrt(3))*(om3*om3)) - 12*(om1*om1*om1*om1)*(72 - 40*np.sqrt(3) + 81*(-2 + np.sqrt(3))*om3 + 561*(om3*om3*om3)) + 12*(om1*om1*om1)*(3*(-7 + 4*np.sqrt(3)) + 14*(9 - 5*np.sqrt(3))*om3 + 42*(-2 + np.sqrt(3))*(om3*om3) - 900*(om3*om3*om3*om3))))/(36.*(om1*om1)*(om1 + om2)*(om1 + om2 + om3)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
        ) * uu[-1] + (
            (-18*(-2 + np.sqrt(3))*(om1*om1*om1*om1)*(om2 - om3) + 6*(om1*om1*om1)*(-12*(-2 + np.sqrt(3))*(om2*om2) + om2*(9 - 5*np.sqrt(3) + 15*(-2 + np.sqrt(3))*om3) + om3*(-9 + 5*np.sqrt(3) + 18*(-2 + np.sqrt(3))*om3)) + om2*(-18*(-2 + np.sqrt(3))*(om2*om2*om2*om2) + 5*(om3*om3)*(77 - 44*np.sqrt(3) + 15*(9 - 5*np.sqrt(3))*om3) + 6*(om2*om2*om2)*(9 - 5*np.sqrt(3) + 6*(-2 + np.sqrt(3))*om3) + om2*om3*(6*(-7 + 4*np.sqrt(3)) + 145*(9 - 5*np.sqrt(3))*om3 - 720*(-2 + np.sqrt(3))*(om3*om3)) - 3*(om2*om2)*(-7 + 4*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om3 + 222*(-2 + np.sqrt(3))*(om3*om3))) + om1*om1*(-108*(-2 + np.sqrt(3))*(om2*om2*om2) + 18*(om2*om2)*(9 - 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om3) + om3*(3*(-7 + 4*np.sqrt(3)) + 29*(-9 + 5*np.sqrt(3))*om3 + 90*(-2 + np.sqrt(3))*(om3*om3)) - 3*om2*(-7 + 4*np.sqrt(3) + 2*om3*(36 - 20*np.sqrt(3) + 75*(-2 + np.sqrt(3))*om3))) + om1*(-72*(-2 + np.sqrt(3))*(om2*om2*om2*om2) + 18*(om2*om2*om2)*(9 - 5*np.sqrt(3) + 7*(-2 + np.sqrt(3))*om3) + om3*om3*(-77 + 44*np.sqrt(3) + 15*(-9 + 5*np.sqrt(3))*om3) + om2*om3*(9*(-7 + 4*np.sqrt(3)) + 116*(9 - 5*np.sqrt(3))*om3 - 630*(-2 + np.sqrt(3))*(om3*om3)) - 6*(om2*om2)*(-7 + 4*np.sqrt(3) + om3*(45 - 25*np.sqrt(3) + 204*(-2 + np.sqrt(3))*om3))))/(36.*(om1*om1)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
        ) * uu[-2] + (
            (12*(-2 + np.sqrt(3))*(om1*om1*om1*om1) + om1*om1*om1*(-36 + 20*np.sqrt(3) - 6*(-2 + np.sqrt(3))*om2 + 60*om3 - 30*np.sqrt(3)*om3) - 3*(om2 + om3)*((-9 + 5*np.sqrt(3))*(om2*om2) + om2*(-7 + 4*np.sqrt(3) + 14*(9 - 5*np.sqrt(3))*om3) + om3*(77 - 44*np.sqrt(3) + 15*(9 - 5*np.sqrt(3))*om3)) + 2*(om1*om1)*(-7 + 4*np.sqrt(3) - 24*(-2 + np.sqrt(3))*(om2*om2) + om2*(9 - 5*np.sqrt(3) + 180*(-2 + np.sqrt(3))*om3) + om3*(45 - 25*np.sqrt(3) + 204*(-2 + np.sqrt(3))*om3)) + om1*(-30*(-2 + np.sqrt(3))*(om2*om2*om2) + om2*om2*(81 - 45*np.sqrt(3) + 390*(-2 + np.sqrt(3))*om3) + om3*(35 - 20*np.sqrt(3) + 87*(-9 + 5*np.sqrt(3))*om3 + 450*(-2 + np.sqrt(3))*(om3*om3)) + om2*(7 - 4*np.sqrt(3) + 78*(-9 + 5*np.sqrt(3))*om3 + 870*(-2 + np.sqrt(3))*(om3*om3))))/(36.*(om1 + om2)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
        ) * uu[-3] + (
            (-12*(-2 + np.sqrt(3))*(om1*om1*om1*om1) + 3*(om2*om2)*(56 - 32*np.sqrt(3) + (99 - 55*np.sqrt(3))*om2) + 4*(om1*om1*om1)*(9 - 5*np.sqrt(3) + 6*(-2 + np.sqrt(3))*om2) + om1*om2*(4*(-7 + 4*np.sqrt(3)) + 63*(9 - 5*np.sqrt(3))*om2 - 330*(-2 + np.sqrt(3))*(om2*om2)) + om1*om1*(14 - 8*np.sqrt(3) + 8*(-9 + 5*np.sqrt(3))*om2 - 294*(-2 + np.sqrt(3))*(om2*om2)))/(36.*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*((om1 + om2)*(om1 + om2)*(om1 + om2))*om3 - 7*(om1 + om2)*(2*(om1*om1*om1) + 6*(om1*om1)*om2 - 14*om1*(om2*om2) + 45*(om2*om2*om2))*(om3*om3) - 11*(4*(om1*om1*om1) + 17*om1*(om2*om2) + 63*(om2*om2*om2))*(om3*om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3*om3)))
        ) * uu[-4] + h * (
            (
                (-12*(-3 + np.sqrt(3))*(om1*om1*om1*om1*om1*om1)*(6*(om2*om2) - 6*om2*om3 - 7*(om3*om3)) + 6*(om1*om1*om1*om1*om1)*(-39*(-3 + np.sqrt(3))*(om2*om2*om2) + 42*(-2 + np.sqrt(3))*(om3*om3) + 44*(-3 + np.sqrt(3))*(om3*om3*om3) + 3*(om2*om2)*(-12*(-2 + np.sqrt(3)) + 13*(-3 + np.sqrt(3))*om3) + 2*om2*om3*(18*(-2 + np.sqrt(3)) + 35*(-3 + np.sqrt(3))*om3)) + 6*(om1*om1*om1*om1)*(-36*(-3 + np.sqrt(3))*(om2*om2*om2*om2) + 9*(om2*om2*om2)*(-11*(-2 + np.sqrt(3)) + 4*(-3 + np.sqrt(3))*om3) + 3*(om2*om2)*(18 - 10*np.sqrt(3) + 33*(-2 + np.sqrt(3))*om3) + 2*om2*om3*(3*(-9 + 5*np.sqrt(3)) + 84*(-2 + np.sqrt(3))*om3 + 22*(-3 + np.sqrt(3))*(om3*om3)) + om3*om3*(7*(-9 + 5*np.sqrt(3)) + 110*(-2 + np.sqrt(3))*om3 + 30*(-3 + np.sqrt(3))*(om3*om3))) + om2*om2*(om2 + om3)*(18*(-2 + np.sqrt(3))*(om2*om2*om2*om2) + 6*(om2*om2*om2)*(-9 + 5*np.sqrt(3) - 6*(-2 + np.sqrt(3))*om3) + 5*(om3*om3)*(-77 + 44*np.sqrt(3) + 15*(-9 + 5*np.sqrt(3))*om3) + 3*(om2*om2)*(-7 + 4*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om3 + 222*(-2 + np.sqrt(3))*(om3*om3)) + om2*om3*(42 - 24*np.sqrt(3) + 145*(-9 + 5*np.sqrt(3))*om3 + 720*(-2 + np.sqrt(3))*(om3*om3))) + om1*om1*om1*(36*(-3 + np.sqrt(3))*(om2*om2*om2*om2*om2) - 36*(om2*om2*om2*om2)*(13*(-2 + np.sqrt(3)) + (-3 + np.sqrt(3))*om3) + 3*(om2*om2*om2)*(27*(9 - 5*np.sqrt(3)) + 156*(-2 + np.sqrt(3))*om3 + 322*(-3 + np.sqrt(3))*(om3*om3)) + 6*om2*om3*(-14 + 8*np.sqrt(3) + 21*(-9 + 5*np.sqrt(3))*om3 + 66*(-2 + np.sqrt(3))*(om3*om3) - 30*(-3 + np.sqrt(3))*(om3*om3*om3)) + 3*(om2*om2)*(28 - 16*np.sqrt(3) + 27*(-9 + 5*np.sqrt(3))*om3 - 84*(-2 + np.sqrt(3))*(om3*om3) + 374*(-3 + np.sqrt(3))*(om3*om3*om3)) + 2*(om3*om3)*(7*(-7 + 4*np.sqrt(3)) + 4*om3*(-99 + 55*np.sqrt(3) + 45*(-2 + np.sqrt(3))*om3))) + om1*om1*(144*(-3 + np.sqrt(3))*(om2*om2*om2*om2*om2*om2) - 144*(-3 + np.sqrt(3))*(om2*om2*om2*om2*om2)*om3 + 2*(om3*om3*om3)*(-77 + 44*np.sqrt(3) + 15*(-9 + 5*np.sqrt(3))*om3) + 24*(om2*om2*om2*om2)*(18 - 10*np.sqrt(3) + 133*(-3 + np.sqrt(3))*(om3*om3)) + 4*om2*(om3*om3)*(7*(-7 + 4*np.sqrt(3)) - 90*(-2 + np.sqrt(3))*(om3*om3)) + 3*(om2*om2)*om3*(7*(-7 + 4*np.sqrt(3)) + 4*om3*(63 - 35*np.sqrt(3) + 77*(-2 + np.sqrt(3))*om3 + 150*(-3 + np.sqrt(3))*(om3*om3))) + 3*(om2*om2*om2)*(49 - 28*np.sqrt(3) + 4*om3*(4*(-9 + 5*np.sqrt(3)) + 63*(-2 + np.sqrt(3))*om3 + 440*(-3 + np.sqrt(3))*(om3*om3)))) + om1*om2*(54*(-3 + np.sqrt(3))*(om2*om2*om2*om2*om2*om2) + 2*(om3*om3*om3)*(77 - 44*np.sqrt(3) + 15*(9 - 5*np.sqrt(3))*om3) - 54*(om2*om2*om2*om2*om2)*(4 - 2*np.sqrt(3) + (-3 + np.sqrt(3))*om3) + 6*om2*(om3*om3)*(49 - 28*np.sqrt(3) + 22*(-9 + 5*np.sqrt(3))*om3 + 225*(-2 + np.sqrt(3))*(om3*om3)) + 3*(om2*om2*om2*om2)*(-9 + 5*np.sqrt(3) + 18*om3*(4 - 2*np.sqrt(3) + 35*(-3 + np.sqrt(3))*om3)) + 2*(om2*om2)*om3*(3*(-7 + 4*np.sqrt(3)) + 4*om3*(14*(-9 + 5*np.sqrt(3)) + 45*om3*(11*(-2 + np.sqrt(3)) + 6*(-3 + np.sqrt(3))*om3))) + 3*(om2*om2*om2)*(14 - 8*np.sqrt(3) + om3*(9 - 5*np.sqrt(3) + 42*om3*(19*(-2 + np.sqrt(3)) + 33*(-3 + np.sqrt(3))*om3)))))/(36.*om1*(om1 + om2)*(om1 + om2 + om3)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
            ) * ff[-1] + (
                (-18*(-2 + np.sqrt(3))*(om1*om1*om1*om1)*(om2 - om3) + 6*(om1*om1*om1)*(-12*(-2 + np.sqrt(3))*(om2*om2) + om2*(9 - 5*np.sqrt(3) + 15*(-2 + np.sqrt(3))*om3) + om3*(-9 + 5*np.sqrt(3) + 18*(-2 + np.sqrt(3))*om3)) + om2*(-18*(-2 + np.sqrt(3))*(om2*om2*om2*om2) + 5*(om3*om3)*(77 - 44*np.sqrt(3) + 15*(9 - 5*np.sqrt(3))*om3) + 6*(om2*om2*om2)*(9 - 5*np.sqrt(3) + 6*(-2 + np.sqrt(3))*om3) + om2*om3*(6*(-7 + 4*np.sqrt(3)) + 145*(9 - 5*np.sqrt(3))*om3 - 720*(-2 + np.sqrt(3))*(om3*om3)) - 3*(om2*om2)*(-7 + 4*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om3 + 222*(-2 + np.sqrt(3))*(om3*om3))) + om1*om1*(-108*(-2 + np.sqrt(3))*(om2*om2*om2) + 18*(om2*om2)*(9 - 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om3) + om3*(3*(-7 + 4*np.sqrt(3)) + 29*(-9 + 5*np.sqrt(3))*om3 + 90*(-2 + np.sqrt(3))*(om3*om3)) - 3*om2*(-7 + 4*np.sqrt(3) + 2*om3*(36 - 20*np.sqrt(3) + 75*(-2 + np.sqrt(3))*om3))) + om1*(-72*(-2 + np.sqrt(3))*(om2*om2*om2*om2) + 18*(om2*om2*om2)*(9 - 5*np.sqrt(3) + 7*(-2 + np.sqrt(3))*om3) + om3*om3*(-77 + 44*np.sqrt(3) + 15*(-9 + 5*np.sqrt(3))*om3) + om2*om3*(9*(-7 + 4*np.sqrt(3)) + 116*(9 - 5*np.sqrt(3))*om3 - 630*(-2 + np.sqrt(3))*(om3*om3)) - 6*(om2*om2)*(-7 + 4*np.sqrt(3) + om3*(45 - 25*np.sqrt(3) + 204*(-2 + np.sqrt(3))*om3))))/(18.*om1*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
            ) * ff[-2] + (
                (om2*(-12*(-2 + np.sqrt(3))*(om1*om1*om1*om1) + 2*(om1*om1*om1)*(18 - 10*np.sqrt(3) + 3*(-2 + np.sqrt(3))*om2 + 15*(-2 + np.sqrt(3))*om3) + 3*(om2 + om3)*((-9 + 5*np.sqrt(3))*(om2*om2) + om2*(-7 + 4*np.sqrt(3) + 14*(9 - 5*np.sqrt(3))*om3) + om3*(77 - 44*np.sqrt(3) + 15*(9 - 5*np.sqrt(3))*om3)) + 2*(om1*om1)*(7 - 4*np.sqrt(3) + 24*(-2 + np.sqrt(3))*(om2*om2) + 5*(-9 + 5*np.sqrt(3))*om3 - 204*(-2 + np.sqrt(3))*(om3*om3) + om2*(-9 + 5*np.sqrt(3) - 180*(-2 + np.sqrt(3))*om3)) + om1*(30*(-2 + np.sqrt(3))*(om2*om2*om2) + om2*om2*(9*(-9 + 5*np.sqrt(3)) - 390*(-2 + np.sqrt(3))*om3) + om2*(-7 + 4*np.sqrt(3) + 78*(9 - 5*np.sqrt(3))*om3 - 870*(-2 + np.sqrt(3))*(om3*om3)) + om3*(5*(-7 + 4*np.sqrt(3)) + 87*(9 - 5*np.sqrt(3))*om3 - 450*(-2 + np.sqrt(3))*(om3*om3)))))/(12.*(om1 + om2)*(-3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) + 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 + (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) + 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
            ) * ff[-3] + (
                ((-12*(-2 + np.sqrt(3))*(om1*om1*om1*om1) + 3*(om2*om2)*(56 - 32*np.sqrt(3) + (99 - 55*np.sqrt(3))*om2) + 4*(om1*om1*om1)*(9 - 5*np.sqrt(3) + 6*(-2 + np.sqrt(3))*om2) + om1*om2*(4*(-7 + 4*np.sqrt(3)) + 63*(9 - 5*np.sqrt(3))*om2 - 330*(-2 + np.sqrt(3))*(om2*om2)) + om1*om1*(14 - 8*np.sqrt(3) + 8*(-9 + 5*np.sqrt(3))*om2 - 294*(-2 + np.sqrt(3))*(om2*om2)))*om3)/(9.*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*((om1 + om2)*(om1 + om2)*(om1 + om2))*om3 - 7*(om1 + om2)*(2*(om1*om1*om1) + 6*(om1*om1)*om2 - 14*om1*(om2*om2) + 45*(om2*om2*om2))*(om3*om3) - 11*(4*(om1*om1*om1) + 17*om1*(om2*om2) + 63*(om2*om2*om2))*(om3*om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3*om3)))
            ) * ff[-4]
        )
    u2int = (
            (-18*(2 + np.sqrt(3) + 18*(om1*om1))*(om2*om2*om2*om2*om2*om2*om2) - 3*(om2*om2*om2*om2*om2)*(7 + 4*np.sqrt(3) + 2*om1*(5*(9 + 5*np.sqrt(3)) + 45*(2 + np.sqrt(3))*om1 + 36*(om1*om1*om1)) - 2*(9 + 5*np.sqrt(3))*om3 - 36*om1*(2 + np.sqrt(3) + 8*(om1*om1))*om3 + 210*(2 + np.sqrt(3) + 18*(om1*om1))*(om3*om3)) + 2*(om1*om1)*(om3*om3)*(7*om1*(3*(7 + 4*np.sqrt(3)) + 4*om1*(2*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om1 - 9*(om1*om1*om1))) + 22*(7 + 4*np.sqrt(3) + 4*om1*(9 + 5*np.sqrt(3) + 6*(2 + np.sqrt(3))*om1 - 9*(om1*om1*om1)))*om3 + 30*(9 + 5*np.sqrt(3) + 9*om1*(2 + np.sqrt(3) - 2*(om1*om1)))*(om3*om3)) + om2*om2*om2*om2*(12*om1*(-7 - 4*np.sqrt(3) - 8*(9 + 5*np.sqrt(3))*om1 - 60*(2 + np.sqrt(3))*(om1*om1) + 108*(om1*om1*om1*om1)) + 3*(7 + 4*np.sqrt(3) + 10*(9 + 5*np.sqrt(3))*om1 + 90*(2 + np.sqrt(3))*(om1*om1) + 72*(om1*om1*om1*om1))*om3 - 133*(9 + 5*np.sqrt(3) + 18*om1*(2 + np.sqrt(3) + 8*(om1*om1)))*(om3*om3) - 1386*(2 + np.sqrt(3) + 18*(om1*om1))*(om3*om3*om3)) + om2*om2*om2*(3*(om1*om1)*(-15*(7 + 4*np.sqrt(3)) - 56*(9 + 5*np.sqrt(3))*om1 - 324*(2 + np.sqrt(3))*(om1*om1) + 468*(om1*om1*om1*om1)) + 12*om1*(7 + 4*np.sqrt(3) + 8*(9 + 5*np.sqrt(3))*om1 + 60*(2 + np.sqrt(3))*(om1*om1) - 108*(om1*om1*om1*om1))*om3 - 7*(7*(7 + 4*np.sqrt(3)) + 52*(9 + 5*np.sqrt(3))*om1 + 468*(2 + np.sqrt(3))*(om1*om1) + 828*(om1*om1*om1*om1))*(om3*om3) - 220*(9 + 5*np.sqrt(3) + 18*om1*(2 + np.sqrt(3) + 8*(om1*om1)))*(om3*om3*om3) - 720*(2 + np.sqrt(3) + 18*(om1*om1))*(om3*om3*om3*om3)) - 6*(om2*om2*om2*om2*om2*om2)*(9 + 5*np.sqrt(3) - 3*(2 + np.sqrt(3))*om3 + 18*om1*(2 + np.sqrt(3) + 8*(om1*om1) - 3*om1*om3)) + 2*om1*om2*om3*(-216*(om1*om1*om1*om1*om1*om1) - 1260*(om1*om1*om1*om1*om1)*om3 - 2*(om3*om3)*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3) + 72*(om1*om1*om1*om1)*(6 + 3*np.sqrt(3) - 11*(om3*om3)) + 3*om1*om3*(7*(7 + 4*np.sqrt(3)) - 90*(2 + np.sqrt(3))*(om3*om3)) + 2*(om1*om1)*(9*(7 + 4*np.sqrt(3)) + 56*(9 + 5*np.sqrt(3))*om3 + 132*(2 + np.sqrt(3))*(om3*om3)) + 12*(om1*om1*om1)*(4*(9 + 5*np.sqrt(3)) + 63*(2 + np.sqrt(3))*om3 + 45*(om3*om3*om3))) + om2*om2*(-432*(2 + np.sqrt(3))*(om1*om1*om1*om1*om1) + 432*(om1*om1*om1*om1*om1*om1*om1) - 1404*(om1*om1*om1*om1*om1*om1)*om3 - 5*(om3*om3*om3)*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3) + 3*(om1*om1)*om3*(15*(7 + 4*np.sqrt(3)) - 56*(9 + 5*np.sqrt(3))*om3 - 1188*(2 + np.sqrt(3))*(om3*om3)) - 6*om1*(om3*om3)*(98 + 56*np.sqrt(3) + 66*(9 + 5*np.sqrt(3))*om3 + 225*(2 + np.sqrt(3))*(om3*om3)) + 12*(om1*om1*om1*om1)*(-8*(9 + 5*np.sqrt(3)) + 81*(2 + np.sqrt(3))*om3 - 561*(om3*om3*om3)) - 12*(om1*om1*om1)*(3*(7 + 4*np.sqrt(3)) - 14*(9 + 5*np.sqrt(3))*om3 + 42*(2 + np.sqrt(3))*(om3*om3) + 900*(om3*om3*om3*om3))))/(36.*(om1*om1)*(om1 + om2)*(om1 + om2 + om3)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
        ) * uu[-1] + (
            (18*(2 + np.sqrt(3))*(om2*om2*om2*om2*om2) + 6*(om2*om2*om2*om2)*(9 + 5*np.sqrt(3) + 12*(2 + np.sqrt(3))*om1 - 6*(2 + np.sqrt(3))*om3) + om1*om3*(-3*om1*(7 + 4*np.sqrt(3) + 2*om1*(9 + 5*np.sqrt(3) + 3*(2 + np.sqrt(3))*om1)) + (-11*(7 + 4*np.sqrt(3)) - 29*(9 + 5*np.sqrt(3))*om1 - 108*(2 + np.sqrt(3))*(om1*om1))*om3 - 15*(9 + 5*np.sqrt(3) + 6*(2 + np.sqrt(3))*om1)*(om3*om3)) + 3*(om2*om2*om2)*(7 + 4*np.sqrt(3) + 36*(2 + np.sqrt(3))*(om1*om1) - 4*(9 + 5*np.sqrt(3))*om3 + 222*(2 + np.sqrt(3))*(om3*om3) + 6*om1*(9 + 5*np.sqrt(3) - 7*(2 + np.sqrt(3))*om3)) + om2*(18*(2 + np.sqrt(3))*(om1*om1*om1*om1) + 6*(om1*om1*om1)*(9 + 5*np.sqrt(3) - 15*(2 + np.sqrt(3))*om3) + 5*(om3*om3)*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3) + 3*(om1*om1)*(7 + 4*np.sqrt(3) - 8*(9 + 5*np.sqrt(3))*om3 + 150*(2 + np.sqrt(3))*(om3*om3)) + om1*om3*(-9*(7 + 4*np.sqrt(3)) + 116*(9 + 5*np.sqrt(3))*om3 + 630*(2 + np.sqrt(3))*(om3*om3))) + om2*om2*(72*(2 + np.sqrt(3))*(om1*om1*om1) - 18*(om1*om1)*(-9 - 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om3) + 6*om1*(7 + 4*np.sqrt(3) - 5*(9 + 5*np.sqrt(3))*om3 + 204*(2 + np.sqrt(3))*(om3*om3)) + om3*(-6*(7 + 4*np.sqrt(3)) + 145*(9 + 5*np.sqrt(3))*om3 + 720*(2 + np.sqrt(3))*(om3*om3))))/(36.*(om1*om1)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
        ) * uu[-2] + (
            (-12*(2 + np.sqrt(3))*(om1*om1*om1*om1) + 2*(om1*om1*om1)*(-2*(9 + 5*np.sqrt(3)) + 3*(2 + np.sqrt(3))*om2 + 15*(2 + np.sqrt(3))*om3) + 2*(om1*om1)*(-7 - 4*np.sqrt(3) + 24*(2 + np.sqrt(3))*(om2*om2) + 5*(9 + 5*np.sqrt(3))*om3 - 204*(2 + np.sqrt(3))*(om3*om3) + om2*(9 + 5*np.sqrt(3) - 180*(2 + np.sqrt(3))*om3)) + 3*(om2 + om3)*((9 + 5*np.sqrt(3))*(om2*om2) + om2*(7 + 4*np.sqrt(3) - 14*(9 + 5*np.sqrt(3))*om3) - om3*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3)) + om1*(30*(2 + np.sqrt(3))*(om2*om2*om2) + om2*om2*(9*(9 + 5*np.sqrt(3)) - 390*(2 + np.sqrt(3))*om3) + om3*(5*(7 + 4*np.sqrt(3)) - 87*(9 + 5*np.sqrt(3))*om3 - 450*(2 + np.sqrt(3))*(om3*om3)) - om2*(-7 - 4*np.sqrt(3) + 78*(9 + 5*np.sqrt(3))*om3 + 870*(2 + np.sqrt(3))*(om3*om3))))/(36.*(om1 + om2)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
        ) * uu[-3] + (
            (12*(2 + np.sqrt(3))*(om1*om1*om1*om1) + 3*(om2*om2)*(56 + 32*np.sqrt(3) + 99*om2 + 55*np.sqrt(3)*om2) + 4*(om1*om1*om1)*(9 + 5*np.sqrt(3) - 6*(2 + np.sqrt(3))*om2) + 2*(om1*om1)*(7 + 4*np.sqrt(3) - 4*(9 + 5*np.sqrt(3))*om2 + 147*(2 + np.sqrt(3))*(om2*om2)) + om1*om2*(-4*(7 + 4*np.sqrt(3)) + 63*(9 + 5*np.sqrt(3))*om2 + 330*(2 + np.sqrt(3))*(om2*om2)))/(36.*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*((om1 + om2)*(om1 + om2)*(om1 + om2))*om3 - 7*(om1 + om2)*(2*(om1*om1*om1) + 6*(om1*om1)*om2 - 14*om1*(om2*om2) + 45*(om2*om2*om2))*(om3*om3) - 11*(4*(om1*om1*om1) + 17*om1*(om2*om2) + 63*(om2*om2*om2))*(om3*om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3*om3)))
        ) * uu[-4] + h * (
            (
                -(-12*(3 + np.sqrt(3))*(om1*om1*om1*om1*om1*om1)*(6*(om2*om2) - 6*om2*om3 - 7*(om3*om3)) - 6*(om1*om1*om1*om1*om1)*(39*(3 + np.sqrt(3))*(om2*om2*om2) + om2*om2*(36*(2 + np.sqrt(3)) - 39*(3 + np.sqrt(3))*om3) - 2*(om3*om3)*(21*(2 + np.sqrt(3)) + 22*(3 + np.sqrt(3))*om3) - 2*om2*om3*(18*(2 + np.sqrt(3)) + 35*(3 + np.sqrt(3))*om3)) + om2*om2*(om2 + om3)*(18*(2 + np.sqrt(3))*(om2*om2*om2*om2) + 6*(om2*om2*om2)*(9 + 5*np.sqrt(3) - 6*(2 + np.sqrt(3))*om3) + 5*(om3*om3)*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3) + 3*(om2*om2)*(7 + 4*np.sqrt(3) - 4*(9 + 5*np.sqrt(3))*om3 + 222*(2 + np.sqrt(3))*(om3*om3)) + om2*om3*(-6*(7 + 4*np.sqrt(3)) + 145*(9 + 5*np.sqrt(3))*om3 + 720*(2 + np.sqrt(3))*(om3*om3))) - 6*(om1*om1*om1*om1)*(36*(3 + np.sqrt(3))*(om2*om2*om2*om2) + om2*om2*(6*(9 + 5*np.sqrt(3)) - 99*(2 + np.sqrt(3))*om3) - 9*(om2*om2*om2)*(-11*(2 + np.sqrt(3)) + 4*(3 + np.sqrt(3))*om3) - 2*om2*om3*(3*(9 + 5*np.sqrt(3)) + 84*(2 + np.sqrt(3))*om3 + 22*(3 + np.sqrt(3))*(om3*om3)) - om3*om3*(7*(9 + 5*np.sqrt(3)) + 110*(2 + np.sqrt(3))*om3 + 30*(3 + np.sqrt(3))*(om3*om3))) + om1*om1*om1*(36*(3 + np.sqrt(3))*(om2*om2*om2*om2*om2) - 36*(om2*om2*om2*om2)*(13*(2 + np.sqrt(3)) + (3 + np.sqrt(3))*om3) + 3*(om2*om2*om2)*(-27*(9 + 5*np.sqrt(3)) + 156*(2 + np.sqrt(3))*om3 + 322*(3 + np.sqrt(3))*(om3*om3)) + 6*om2*om3*(14 + 8*np.sqrt(3) + 21*(9 + 5*np.sqrt(3))*om3 + 66*(2 + np.sqrt(3))*(om3*om3) - 30*(3 + np.sqrt(3))*(om3*om3*om3)) + 3*(om2*om2)*(-4*(7 + 4*np.sqrt(3)) + 27*(9 + 5*np.sqrt(3))*om3 - 84*(2 + np.sqrt(3))*(om3*om3) + 374*(3 + np.sqrt(3))*(om3*om3*om3)) + 2*(om3*om3)*(7*(7 + 4*np.sqrt(3)) + 4*om3*(99 + 55*np.sqrt(3) + 45*(2 + np.sqrt(3))*om3))) + om1*om1*(144*(3 + np.sqrt(3))*(om2*om2*om2*om2*om2*om2) - 144*(3 + np.sqrt(3))*(om2*om2*om2*om2*om2)*om3 + 2*(om3*om3*om3)*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3) + 4*om2*(om3*om3)*(7*(7 + 4*np.sqrt(3)) - 90*(2 + np.sqrt(3))*(om3*om3)) + 24*(om2*om2*om2*om2)*(-2*(9 + 5*np.sqrt(3)) + 133*(3 + np.sqrt(3))*(om3*om3)) + 3*(om2*om2)*om3*(7*(7 + 4*np.sqrt(3)) + 4*om3*(-7*(9 + 5*np.sqrt(3)) + 77*(2 + np.sqrt(3))*om3 + 150*(3 + np.sqrt(3))*(om3*om3))) + 3*(om2*om2*om2)*(-7*(7 + 4*np.sqrt(3)) + 4*om3*(4*(9 + 5*np.sqrt(3)) + 63*(2 + np.sqrt(3))*om3 + 440*(3 + np.sqrt(3))*(om3*om3)))) + om1*om2*(54*(3 + np.sqrt(3))*(om2*om2*om2*om2*om2*om2) - 54*(om2*om2*om2*om2*om2)*(-2*(2 + np.sqrt(3)) + (3 + np.sqrt(3))*om3) - 2*(om3*om3*om3)*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3) + 6*om2*(om3*om3)*(-7*(7 + 4*np.sqrt(3)) + 22*(9 + 5*np.sqrt(3))*om3 + 225*(2 + np.sqrt(3))*(om3*om3)) + 3*(om2*om2*om2*om2)*(9 + 5*np.sqrt(3) - 36*(2 + np.sqrt(3))*om3 + 630*(3 + np.sqrt(3))*(om3*om3)) + 2*(om2*om2)*om3*(3*(7 + 4*np.sqrt(3)) + 4*om3*(14*(9 + 5*np.sqrt(3)) + 45*om3*(11*(2 + np.sqrt(3)) + 6*(3 + np.sqrt(3))*om3))) + 3*(om2*om2*om2)*(-2*(7 + 4*np.sqrt(3)) + om3*(-9 - 5*np.sqrt(3) + 42*om3*(19*(2 + np.sqrt(3)) + 33*(3 + np.sqrt(3))*om3)))))/(36.*om1*(om1 + om2)*(om1 + om2 + om3)*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
            ) * ff[-1] + (
                (18*(2 + np.sqrt(3))*(om2*om2*om2*om2*om2) + 6*(om2*om2*om2*om2)*(9 + 5*np.sqrt(3) + 12*(2 + np.sqrt(3))*om1 - 6*(2 + np.sqrt(3))*om3) + om1*om3*(-3*om1*(7 + 4*np.sqrt(3) + 2*om1*(9 + 5*np.sqrt(3) + 3*(2 + np.sqrt(3))*om1)) + (-11*(7 + 4*np.sqrt(3)) - 29*(9 + 5*np.sqrt(3))*om1 - 108*(2 + np.sqrt(3))*(om1*om1))*om3 - 15*(9 + 5*np.sqrt(3) + 6*(2 + np.sqrt(3))*om1)*(om3*om3)) + 3*(om2*om2*om2)*(7 + 4*np.sqrt(3) + 36*(2 + np.sqrt(3))*(om1*om1) - 4*(9 + 5*np.sqrt(3))*om3 + 222*(2 + np.sqrt(3))*(om3*om3) + 6*om1*(9 + 5*np.sqrt(3) - 7*(2 + np.sqrt(3))*om3)) + om2*(18*(2 + np.sqrt(3))*(om1*om1*om1*om1) + 6*(om1*om1*om1)*(9 + 5*np.sqrt(3) - 15*(2 + np.sqrt(3))*om3) + 5*(om3*om3)*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3) + 3*(om1*om1)*(7 + 4*np.sqrt(3) - 8*(9 + 5*np.sqrt(3))*om3 + 150*(2 + np.sqrt(3))*(om3*om3)) + om1*om3*(-9*(7 + 4*np.sqrt(3)) + 116*(9 + 5*np.sqrt(3))*om3 + 630*(2 + np.sqrt(3))*(om3*om3))) + om2*om2*(72*(2 + np.sqrt(3))*(om1*om1*om1) - 18*(om1*om1)*(-9 - 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om3) + 6*om1*(7 + 4*np.sqrt(3) - 5*(9 + 5*np.sqrt(3))*om3 + 204*(2 + np.sqrt(3))*(om3*om3)) + om3*(-6*(7 + 4*np.sqrt(3)) + 145*(9 + 5*np.sqrt(3))*om3 + 720*(2 + np.sqrt(3))*(om3*om3))))/(18.*om1*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 - (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
            ) * ff[-2] + (
                (om2*(12*(2 + np.sqrt(3))*(om1*om1*om1*om1) - 2*(om1*om1*om1)*(-2*(9 + 5*np.sqrt(3)) + 3*(2 + np.sqrt(3))*om2 + 15*(2 + np.sqrt(3))*om3) + 2*(om1*om1)*(7 + 4*np.sqrt(3) - 24*(2 + np.sqrt(3))*(om2*om2) - 5*(9 + 5*np.sqrt(3))*om3 + 204*(2 + np.sqrt(3))*(om3*om3) + om2*(-9 - 5*np.sqrt(3) + 180*(2 + np.sqrt(3))*om3)) - 3*(om2 + om3)*((9 + 5*np.sqrt(3))*(om2*om2) + om2*(7 + 4*np.sqrt(3) - 14*(9 + 5*np.sqrt(3))*om3) - om3*(77 + 44*np.sqrt(3) + 15*(9 + 5*np.sqrt(3))*om3)) + om1*(-30*(2 + np.sqrt(3))*(om2*om2*om2) + om2*om2*(-9*(9 + 5*np.sqrt(3)) + 390*(2 + np.sqrt(3))*om3) + om3*(-5*(7 + 4*np.sqrt(3)) + 87*(9 + 5*np.sqrt(3))*om3 + 450*(2 + np.sqrt(3))*(om3*om3)) + om2*(-7 - 4*np.sqrt(3) + 78*(9 + 5*np.sqrt(3))*om3 + 870*(2 + np.sqrt(3))*(om3*om3)))))/(12.*(om1 + om2)*(-3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)) + 3*(4*om1 - 3*om2)*om2*(om1 + om2)*(om1 + 2*om2)*om3 + (14*(om1*om1*om1) + 30*(om1*om1)*om2 - 113*om1*(om2*om2) + 333*(om2*om2*om2))*(om3*om3) + 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3)))
            ) * ff[-3] + (
                ((12*(2 + np.sqrt(3))*(om1*om1*om1*om1) + 3*(om2*om2)*(56 + 32*np.sqrt(3) + 99*om2 + 55*np.sqrt(3)*om2) + 4*(om1*om1*om1)*(9 + 5*np.sqrt(3) - 6*(2 + np.sqrt(3))*om2) + 2*(om1*om1)*(7 + 4*np.sqrt(3) - 4*(9 + 5*np.sqrt(3))*om2 + 147*(2 + np.sqrt(3))*(om2*om2)) + om1*om2*(-4*(7 + 4*np.sqrt(3)) + 63*(9 + 5*np.sqrt(3))*om2 + 330*(2 + np.sqrt(3))*(om2*om2)))*om3)/(9.*(3*(4*om1 - 3*om2)*(om2*om2)*((om1 + om2)*(om1 + om2)*(om1 + om2)) - 3*(4*om1 - 3*om2)*om2*((om1 + om2)*(om1 + om2)*(om1 + om2))*om3 - 7*(om1 + om2)*(2*(om1*om1*om1) + 6*(om1*om1)*om2 - 14*om1*(om2*om2) + 45*(om2*om2*om2))*(om3*om3) - 11*(4*(om1*om1*om1) + 17*om1*(om2*om2) + 63*(om2*om2*om2))*(om3*om3*om3) - 30*(om1*om1 - 2*om1*om2 + 12*(om2*om2))*(om3*om3*om3*om3)))
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_eBDF4AS(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                         **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_eBDF4AS, adaptive_step_eBDF4AS,
                            **kwargs)

def cons_or_diss_eBDF4AS(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                         **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_eBDF4AS, adaptive_step_eBDF4AS,
                            fixed_estimate_eBDF4AS, adaptive_estimate_eBDF4AS,
                            **kwargs)


# explicit difference correction methods of Arévalo, Claus & Söderlind (2000), with variable step size version of Arévalo & Söderlind (2017)
def fixed_step_EDC22(uu, ff, h):
    u_new = (
            1.3333333333333333
        ) * uu[-1] + (
            -0.3333333333333333
        ) * uu[-2] + h * (
            (
                1.7777777777777777
            ) * ff[-1] + (
                -1.5555555555555556
            ) * ff[-2] + (
                0.4444444444444444
            ) * ff[-3]
        )
    return u_new

def fixed_estimate_EDC22(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            1.0119509986711015
        ) * uu[-1] + (
            -0.011950998671101448
        ) * uu[-2] + h * (
            (
                0.24141182965239458
            ) * ff[-1] + (
                -0.055771327131806755
            ) * ff[-2] + (
                0.013733364213497898
            ) * ff[-3]
        )
    u2int = (
            1.196382334662232
        ) * uu[-1] + (
            -0.19638233466223187
        ) * uu[-2] + h * (
            (
                1.255115948125383
            ) * ff[-1] + (
                -0.9164508950904154
            ) * ff[-2] + (
                0.2536277468976132
            ) * ff[-3]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_EDC22(uu, ff, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            (-2 - 3*om2 + om1*(-3 + om1*om1 - 25*om1*om2))/(om1*om1*(om1 - 25*om2))
        ) * uu[-1] + (
            (2 + 3*om1 + 3*om2)/(om1*om1*(om1 - 25*om2))
        ) * uu[-2] + h * (
            (
                (3*om1*((1 + om1)*(1 + om1)) - 2*(11 + 33*om1 + 36*(om1*om1))*om2 - 3*(11 + 25*om1)*(om2*om2))/(3.*om1*(om1 - 25*om2)*(om1 + om2))
            ) * ff[-1] + (
                (28 + 42*om1 + 42*om2)/(3*(om1*om1) - 75*om1*om2)
            ) * ff[-2] + (
                -(25 + 39*om1)/(3.*(om1 - 25*om2)*(om1 + om2))
            ) * ff[-3]
        )

    return u_new

def adaptive_estimate_EDC22(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            (-9 + 5*np.sqrt(3) + 9*om1*(-2 + np.sqrt(3) + 2*om1*(om1 - 25*om2)) + 9*(-2 + np.sqrt(3))*om2)/(18.*(om1*om1)*(om1 - 25*om2))
        ) * uu[-1] + (
            -(-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om1 + 9*(-2 + np.sqrt(3))*om2)/(18.*(om1*om1)*(om1 - 25*om2))
        ) * uu[-2] + h * (
            (
                (-18*(-3 + np.sqrt(3))*(om1*om1*om1) + 36*(om1*om1)*(2 - np.sqrt(3) + 12*(-3 + np.sqrt(3))*om2) + 22*om2*(-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om2) + 3*om1*(9 - 5*np.sqrt(3) + 6*om2*(22*(-2 + np.sqrt(3)) + 25*(-3 + np.sqrt(3))*om2)))/(108.*om1*(om1 - 25*om2)*(om1 + om2))
            ) * ff[-1] + (
                (-7*(-9 + 5*np.sqrt(3) + 9*(-2 + np.sqrt(3))*om1 + 9*(-2 + np.sqrt(3))*om2))/(27.*om1*(om1 - 25*om2))
            ) * ff[-2] + (
                (25*(-9 + 5*np.sqrt(3)) + 234*(-2 + np.sqrt(3))*om1)/(108.*(om1 - 25*om2)*(om1 + om2))
            ) * ff[-3]
        )
    u2int = (
            -(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om2 + 9*om1*(2 + np.sqrt(3) - 2*(om1*om1) + 50*om1*om2))/(18.*(om1*om1)*(om1 - 25*om2))
        ) * uu[-1] + (
            (9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om1 + 9*(2 + np.sqrt(3))*om2)/(18.*(om1*om1)*(om1 - 25*om2))
        ) * uu[-2] + h * (
            (
                -(-18*(3 + np.sqrt(3))*(om1*om1*om1) + 22*om2*(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om2) + 36*(om1*om1)*(-2 - np.sqrt(3) + 12*(3 + np.sqrt(3))*om2) + 3*om1*(-9 - 5*np.sqrt(3) + 6*om2*(22*(2 + np.sqrt(3)) + 25*(3 + np.sqrt(3))*om2)))/(108.*om1*(om1 - 25*om2)*(om1 + om2))
            ) * ff[-1] + (
                (7*(9 + 5*np.sqrt(3) + 9*(2 + np.sqrt(3))*om1 + 9*(2 + np.sqrt(3))*om2))/(27.*om1*(om1 - 25*om2))
            ) * ff[-2] + (
                -(25*(9 + 5*np.sqrt(3)) + 234*(2 + np.sqrt(3))*om1)/(108.*(om1 - 25*om2)*(om1 + om2))
            ) * ff[-3]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_EDC22(f, t_final, t0, u0, t1, u1, t2, u2,
                       **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_EDC22, adaptive_step_EDC22,
                            **kwargs)

def cons_or_diss_EDC22(f, t_final, t0, u0, t1, u1, t2, u2,
                       **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2], [u0, u1, u2],
                            fixed_step_EDC22, adaptive_step_EDC22,
                            fixed_estimate_EDC22, adaptive_estimate_EDC22,
                            **kwargs)


def fixed_step_EDC23(uu, ff, h):
    u_new = (
            1.3333333333333333
        ) * uu[-1] + (
            -0.3333333333333333
        ) * uu[-2] + h * (
            (
                2.1666666666666665
            ) * ff[-1] + (
                -2.7222222222222223
            ) * ff[-2] + (
                1.6111111111111112
            ) * ff[-3] + (
                -0.3888888888888889
            ) * ff[-4]
        )
    return u_new

def fixed_estimate_EDC23(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            1.010183197601602
        ) * uu[-1] + (
            -0.010183197601602053
        ) * uu[-2] + h * (
            (
                0.25167174254055585
            ) * ff[-1] + (
                -0.08316278041308342
            ) * ff[-2] + (
                0.04215603478531588
            ) * ff[-3] + (
                -0.009523329109203206
            ) * ff[-4]
        )
    u2int = (
            1.1884985349784358
        ) * uu[-1] + (
            -0.1884985349784356
        ) * uu[-2] + h * (
            (
                1.4678040892234117
            ) * ff[-1] + (
                -1.5394047023238908
            ) * ff[-2] + (
                0.8811804372799698
            ) * ff[-3] + (
                -0.2094032245631132
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_EDC23(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            (-3 + om1*om1*om1*om1 - 8*om2 - 4*om3 - 6*om2*(om2 + om3) + 2*(om1*om1*om1)*(2*om2 + om3) - 2*om1*(4 + 6*om2 + 3*om3) - 2*(om1*om1)*(3 + 46*om2*(om2 + om3)))/(om1*om1*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-1] + (
            (3 + 6*(om1*om1) + 8*om2 + 4*om3 + 6*om2*(om2 + om3) + 2*om1*(4 + 6*om2 + 3*om3))/(om1*om1*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-2] + h * (
            (
                (6*(om1*om1*om1*om1*om1) + 18*(om1*om1*om1*om1)*(1 + 2*om2 + om3) - 6*(om1*om1)*(-1 + 3*om2*(-4 + 39*om2 + 60*(om2*om2)) - 6*om3 + 9*om2*(13 + 30*om2)*om3 + 2*(-2 + 45*om2)*(om3*om3)) + 3*om1*(-2*om2*(-3 + 4*om2*(1 + om2)*(20 + 23*om2)) + 3*om3 - 4*om2*(40 + om2*(129 + 92*om2))*om3 - 4*(-1 + om2*(43 + 46*om2))*(om3*om3)) - 43*om2*(om2 + om3)*(3 + 4*om3 + 2*om2*(4 + 3*om2 + 3*om3)) + 3*(om1*om1*om1)*(6 - 166*(om2*om2) + om2*(30 - 166*om3) + om3*(15 + 4*om3)))/(6.*om1*(om1 + om2)*(om1 + om2 + om3)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-1] + (
                (49*(3 + 6*(om1*om1) + 8*om2 + 4*om3 + 6*om2*(om2 + om3) + 2*om1*(4 + 6*om2 + 3*om3)))/(6.*om1*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-2] + (
                -(-3*(om1*om1*om1) + 46*(om2 + om3)*(3 + 4*om2 + 4*om3) + 3*(om1*om1)*(-2 + 93*om2 + 93*om3) + om1*(-3 + 282*(om2*om2) + 368*om3 + 282*(om3*om3) + 4*om2*(92 + 141*om3)))/(6.*(om1 + om2)*om3*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-3] + (
                (-3*om1*((1 + om1)*(1 + om1)) + (138 + om1*(368 + 279*om1))*om2 + 2*(92 + 141*om1)*(om2*om2))/(6.*om3*(om1 + om2 + om3)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-4]
        )

    return u_new

def adaptive_estimate_EDC23(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            (3*(-7 + 4*np.sqrt(3)) + 8*(-9 + 5*np.sqrt(3))*om2 + 4*(-9 + 5*np.sqrt(3))*om3 + 4*(9*(om1*om1*om1*om1) + 9*(-2 + np.sqrt(3))*om2*(om2 + om3) + 18*(om1*om1*om1)*(2*om2 + om3) + om1*(2*(-9 + 5*np.sqrt(3)) + 18*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3) + 9*(om1*om1)*(-2 + np.sqrt(3) - 92*om2*(om2 + om3))))/(36.*(om1*om1)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-1] + (
            (21 - 12*np.sqrt(3) + 8*(9 - 5*np.sqrt(3))*om2 + 4*(9 - 5*np.sqrt(3))*om3 + 4*(-9*(-2 + np.sqrt(3))*(om1*om1) - 9*(-2 + np.sqrt(3))*om2*(om2 + om3) + om1*(18 - 10*np.sqrt(3) - 18*(-2 + np.sqrt(3))*om2 - 9*(-2 + np.sqrt(3))*om3)))/(36.*(om1*om1)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-2] + h * (
            (
                (-36*(-3 + np.sqrt(3))*(om1*om1*om1*om1*om1) - 108*(om1*om1*om1*om1)*(-2 + np.sqrt(3) + 2*(-3 + np.sqrt(3))*om2 + (-3 + np.sqrt(3))*om3) + 18*(om1*om1*om1)*(9 - 5*np.sqrt(3) - 30*(-2 + np.sqrt(3))*om2 + 166*(-3 + np.sqrt(3))*(om2*om2) - 15*(-2 + np.sqrt(3))*om3 + 166*(-3 + np.sqrt(3))*om2*om3 - 4*(-3 + np.sqrt(3))*(om3*om3)) + 43*om2*(om2 + om3)*(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om3 + 4*om2*(2*(-9 + 5*np.sqrt(3)) + 9*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3)) + 6*(om1*om1)*(7 - 4*np.sqrt(3) + 1080*(-3 + np.sqrt(3))*(om2*om2*om2) + 54*(om2*om2)*(13*(-2 + np.sqrt(3)) + 30*(-3 + np.sqrt(3))*om3) - 6*om3*(-9 + 5*np.sqrt(3) + 4*(-2 + np.sqrt(3))*om3) + 6*om2*(18 - 10*np.sqrt(3) + 117*(-2 + np.sqrt(3))*om3 + 90*(-3 + np.sqrt(3))*(om3*om3))) + 3*om1*(1104*(-3 + np.sqrt(3))*(om2*om2*om2*om2) + om3*(21 - 12*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om3) + 48*(om2*om2*om2)*(43*(-2 + np.sqrt(3)) + 46*(-3 + np.sqrt(3))*om3) + 8*(om2*om2)*(20*(-9 + 5*np.sqrt(3)) + 387*(-2 + np.sqrt(3))*om3 + 138*(-3 + np.sqrt(3))*(om3*om3)) + 2*om2*(21 - 12*np.sqrt(3) + 80*(-9 + 5*np.sqrt(3))*om3 + 516*(-2 + np.sqrt(3))*(om3*om3))))/(216.*om1*(om1 + om2)*(om1 + om2 + om3)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-1] + (
                (-49*(3*(-7 + 4*np.sqrt(3)) + 8*(-9 + 5*np.sqrt(3))*om2 + 4*(-9 + 5*np.sqrt(3))*om3 + 4*(9*(-2 + np.sqrt(3))*(om1*om1) + 9*(-2 + np.sqrt(3))*om2*(om2 + om3) + om1*(2*(-9 + 5*np.sqrt(3)) + 18*(-2 + np.sqrt(3))*om2 + 9*(-2 + np.sqrt(3))*om3))))/(216.*om1*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-2] + (
                (-18*(-2 + np.sqrt(3))*(om1*om1*om1) + 138*(-7 + 4*np.sqrt(3))*(om2 + om3) + 184*(-9 + 5*np.sqrt(3))*((om2 + om3)*(om2 + om3)) + 6*(om1*om1)*(9 - 5*np.sqrt(3) + 279*(-2 + np.sqrt(3))*om2 + 279*(-2 + np.sqrt(3))*om3) + om1*(21 - 12*np.sqrt(3) + 1692*(-2 + np.sqrt(3))*(om2*om2) + 8*om2*(46*(-9 + 5*np.sqrt(3)) + 423*(-2 + np.sqrt(3))*om3) + 4*om3*(92*(-9 + 5*np.sqrt(3)) + 423*(-2 + np.sqrt(3))*om3)))/(216.*(om1 + om2)*om3*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-3] + (
                (18*(-2 + np.sqrt(3))*(om1*om1*om1) - 6*(om1*om1)*(9 - 5*np.sqrt(3) + 279*(-2 + np.sqrt(3))*om2) - 46*om2*(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om2) + om1*(3*(-7 + 4*np.sqrt(3)) - 4*om2*(92*(-9 + 5*np.sqrt(3)) + 423*(-2 + np.sqrt(3))*om2)))/(216.*om3*(om1 + om2 + om3)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-4]
        )
    u2int = (
            -(3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(-9*(om1*om1*om1*om1) + 9*(2 + np.sqrt(3))*om2*(om2 + om3) - 18*(om1*om1*om1)*(2*om2 + om3) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3) + 9*(om1*om1)*(2 + np.sqrt(3) + 92*om2*(om2 + om3))))/(36.*(om1*om1)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-1] + (
            (3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(9*(2 + np.sqrt(3))*(om1*om1) + 9*(2 + np.sqrt(3))*om2*(om2 + om3) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3)))/(36.*(om1*om1)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
        ) * uu[-2] + h * (
            (
                (36*(3 + np.sqrt(3))*(om1*om1*om1*om1*om1) + 108*(om1*om1*om1*om1)*(2 + np.sqrt(3) + 2*(3 + np.sqrt(3))*om2 + (3 + np.sqrt(3))*om3) - 18*(om1*om1*om1)*(-9 - 5*np.sqrt(3) - 30*(2 + np.sqrt(3))*om2 + 166*(3 + np.sqrt(3))*(om2*om2) - 15*(2 + np.sqrt(3))*om3 + 166*(3 + np.sqrt(3))*om2*om3 - 4*(3 + np.sqrt(3))*(om3*om3)) - 43*om2*(om2 + om3)*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3 + 4*om2*(2*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3)) - 6*(om1*om1)*(-7 - 4*np.sqrt(3) + 1080*(3 + np.sqrt(3))*(om2*om2*om2) - 6*om3*(9 + 5*np.sqrt(3) + 4*(2 + np.sqrt(3))*om3) + 54*(om2*om2)*(13*(2 + np.sqrt(3)) + 30*(3 + np.sqrt(3))*om3) + 6*om2*(-2*(9 + 5*np.sqrt(3)) + 117*(2 + np.sqrt(3))*om3 + 90*(3 + np.sqrt(3))*(om3*om3))) + 3*om1*(-1104*(3 + np.sqrt(3))*(om2*om2*om2*om2) - 48*(om2*om2*om2)*(43*(2 + np.sqrt(3)) + 46*(3 + np.sqrt(3))*om3) + om3*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3) - 2*om2*(-3*(7 + 4*np.sqrt(3)) + 80*(9 + 5*np.sqrt(3))*om3 + 516*(2 + np.sqrt(3))*(om3*om3)) - 8*(om2*om2)*(20*(9 + 5*np.sqrt(3)) + 387*(2 + np.sqrt(3))*om3 + 138*(3 + np.sqrt(3))*(om3*om3))))/(216.*om1*(om1 + om2)*(om1 + om2 + om3)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-1] + (
                (49*(3*(7 + 4*np.sqrt(3)) + 8*(9 + 5*np.sqrt(3))*om2 + 4*(9 + 5*np.sqrt(3))*om3 + 4*(9*(2 + np.sqrt(3))*(om1*om1) + 9*(2 + np.sqrt(3))*om2*(om2 + om3) + om1*(2*(9 + 5*np.sqrt(3)) + 18*(2 + np.sqrt(3))*om2 + 9*(2 + np.sqrt(3))*om3))))/(216.*om1*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-2] + (
                -(-18*(2 + np.sqrt(3))*(om1*om1*om1) + 138*(7 + 4*np.sqrt(3))*(om2 + om3) + 184*(9 + 5*np.sqrt(3))*((om2 + om3)*(om2 + om3)) + 6*(om1*om1)*(-9 - 5*np.sqrt(3) + 279*(2 + np.sqrt(3))*om2 + 279*(2 + np.sqrt(3))*om3) + om1*(-3*(7 + 4*np.sqrt(3)) + 1692*(2 + np.sqrt(3))*(om2*om2) + 8*om2*(46*(9 + 5*np.sqrt(3)) + 423*(2 + np.sqrt(3))*om3) + 4*om3*(92*(9 + 5*np.sqrt(3)) + 423*(2 + np.sqrt(3))*om3)))/(216.*(om1 + om2)*om3*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-3] + (
                (-18*(2 + np.sqrt(3))*(om1*om1*om1) + 6*(om1*om1)*(-9 - 5*np.sqrt(3) + 279*(2 + np.sqrt(3))*om2) + 46*om2*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om2) + om1*(-3*(7 + 4*np.sqrt(3)) + 4*om2*(92*(9 + 5*np.sqrt(3)) + 423*(2 + np.sqrt(3))*om2)))/(216.*om3*(om1 + om2 + om3)*(om1*om1 - 92*om2*(om2 + om3) + 2*om1*(2*om2 + om3)))
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_EDC23(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                       **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_EDC23, adaptive_step_EDC23,
                            **kwargs)

def cons_or_diss_EDC23(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                       **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_EDC23, adaptive_step_EDC23,
                            fixed_estimate_EDC23, adaptive_estimate_EDC23,
                            **kwargs)


def fixed_step_EDC33(uu, ff, h):
    u_new = (
            1.6363636363636365
        ) * uu[-1] + (
            -0.8181818181818182
        ) * uu[-2] + (
            0.18181818181818182
        ) * uu[-3] + h * (
            (
                2.0454545454545454
            ) * ff[-1] + (
                -2.8636363636363638
            ) * ff[-2] + (
                1.7727272727272727
            ) * ff[-3] + (
                -0.4090909090909091
            ) * ff[-4]
        )
    return u_new

def fixed_estimate_EDC33(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h):
    u1int = (
            1.0205478614436652
        ) * uu[-1] + (
            -0.025360387027637658
        ) * uu[-2] + (
            0.004812525583972387
        ) * uu[-3] + h * (
            (
                0.2475844717004499
            ) * ff[-1] + (
                -0.08876135459673179
            ) * ff[-2] + (
                0.04692212444373077
            ) * ff[-3] + (
                -0.010155712001954682
            ) * ff[-4]
        )
    u2int = (
            1.3645475369850666
        ) * uu[-1] + (
            -0.46422434922376526
        ) * uu[-2] + (
            0.09967681223869877
        ) * uu[-3] + h * (
            (
                1.3976325129609792
            ) * ff[-1] + (
                -1.6247852222831785
            ) * ff[-2] + (
                0.971848919327313
            ) * ff[-3] + (
                -0.2208918001566686
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def adaptive_step_EDC33(uu, ff, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u_new = (
            (-2*om2*((1 + om1 + om2)*(1 + om1 + om2))*(-7*(om1*om1*om1*om1) - om1*om1*om1*(-14 + om2) + 3*om1*om2 + om2*om2 + 3*(om1*om1)*(7 + 2*(-2 + om2)*om2)) + (14*(-3 + om1)*(om1*om1)*((1 + om1)*(1 + om1)*(1 + om1)) + 21*om1*(5 + om1*(8 + 2*om1 + om1*om1*om1))*om2 - 3*((1 + om1)*(1 + om1))*(-37 + 59*(-2 + om1)*om1)*(om2*om2) + 2*(148 + 331*om1 + 580*(om1*om1*om1))*(om2*om2*om2) + 224*(1 + 6*(om1*om1))*(om2*om2*om2*om2))*om3 + 2*(14*(-2 + om1)*(om1*om1)*((1 + om1)*(1 + om1)) - 35*(-2 + om1)*om1*((1 + om1)*(1 + om1))*om2 + 37*(2 + 6*om1 + 17*(om1*om1*om1))*(om2*om2) + 113*(1 + 6*(om1*om1))*(om2*om2*om2))*(om3*om3))/(om1*om1*(om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-1] + (
            (2*((om1 + om2)*(om1 + om2))*((1 + om1 + om2)*(1 + om1 + om2)) + (2*om1*(1 + om1)*(3 + 5*om1) - (111 + 4*om1*(70 + 51*om1))*om2 - 2*(148 + 219*om1)*(om2*om2) - 224*(om2*om2*om2))*om3 + 2*(4*om1*(1 + om1) - (74 + 109*om1)*om2 - 113*(om2*om2))*(om3*om3))/(om1*om1*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-2] + (
            (-2*(om1*om1*om1) + 12*(om2 + om3)*(3 + 4*om2 + 4*om3) + om1*om1*(-4 + 74*om2 + 74*om3) + 2*om1*(-1 + 38*(om2*om2) + 48*om3 + 38*(om3*om3) + om2*(48 + 76*om3)))/((om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-3] + h * (
            (
                (-2*om2*((om1 + om2)*(om1 + om2))*((1 + om1 + om2)*(1 + om1 + om2))*(-14*om1*(1 + om1) + (5 + 12*om1)*om2) + (28*(om1*om1*om1)*((1 + om1)*(1 + om1)*(1 + om1)) + 14*(om1*om1)*((1 + om1)*(1 + om1))*(1 + 7*om1)*om2 - 42*om1*((1 + om1)*(1 + om1))*(5 + 6*om1)*(om2*om2) + (545 + 2*om1*(880 + om1*(1215 + 989*om1)))*(om2*om2*om2) + 292*(5 + om1*(15 + 17*om1))*(om2*om2*om2*om2) + 222*(5 + 12*om1)*(om2*om2*om2*om2*om2))*om3 + 3*(14*(om1*om1)*((1 + om1)*(1 + om1))*(1 + 2*om1) - 7*om1*((1 + om1)*(1 + om1))*(5 + 2*om1)*om2 + (185 + 2*om1*(300 + om1*(415 + 337*om1)))*(om2*om2) + 148*(5 + om1*(15 + 17*om1))*(om2*om2*om2) + 150*(5 + 12*om1)*(om2*om2*om2*om2))*(om3*om3) + 2*(28*(om1*om1)*((1 + om1)*(1 + om1)) - 70*om1*((1 + om1)*(1 + om1))*om2 + 74*(5 + om1*(15 + 17*om1))*(om2*om2) + 113*(5 + 12*om1)*(om2*om2*om2))*(om3*om3*om3))/(2.*om1*(om1 + om2)*(om1 + om2 + om3)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-1] + (
                (7*(2*((om1 + om2)*(om1 + om2))*((1 + om1 + om2)*(1 + om1 + om2)) + (2*om1*(1 + om1)*(3 + 5*om1) - (111 + 4*om1*(70 + 51*om1))*om2 - 2*(148 + 219*om1)*(om2*om2) - 224*(om2*om2*om2))*om3 + 2*(4*om1*(1 + om1) - (74 + 109*om1)*om2 - 113*(om2*om2))*(om3*om3)))/(2.*om1*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-2] + (
                (39*om2*(-(om1*om1*om1) + 6*(om2 + om3)*(3 + 4*om2 + 4*om3) + om1*om1*(-2 + 37*om2 + 37*om3) + om1*(-1 + 38*(om2*om2) + 48*om3 + 38*(om3*om3) + om2*(48 + 76*om3))))/(2.*(om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-3] + (
                -(14*((om1 + om1*om1)*(om1 + om1*om1)) - 49*om1*((1 + om1)*(1 + om1))*om2 + (678 + om1*(1800 + 1381*om1))*(om2*om2) + 76*(12 + 19*om1)*(om2*om2*om2))/(2.*(om1 + om2 + om3)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-4]
        )

    return u_new

def adaptive_estimate_EDC33(eta, deta, f, uu, ff, old_eta, old_deta_f, idx_u_old, h, old_omega):
    om3 = old_omega[-3]
    om2 = old_omega[-2]
    om1 = old_omega[-1]
    u1int = (
            (12*(-2 + np.sqrt(3) - 36*(om1*om1))*(om2*om2*om2*om2*om2) + 2*(om2*om2*om2)*(-7 + 4*np.sqrt(3) + 4*om1*(2*(-9 + 5*np.sqrt(3)) + 15*(-2 + np.sqrt(3))*om1 + 27*(om1*om1*om1)) + 148*(9 - 5*np.sqrt(3))*om3 + 6*om1*(-331*(-2 + np.sqrt(3)) + 3480*(om1*om1))*om3 - 678*(-2 + np.sqrt(3) - 36*(om1*om1))*(om3*om3)) + om2*om2*(6*om1*(-7 + 4*np.sqrt(3) + 12*om1*(-9 + 5*np.sqrt(3) + 8*(-2 + np.sqrt(3))*om1 + 15*(om1*om1*om1))) + 111*(7 - 4*np.sqrt(3))*om3 - 36*om1*(16*(-9 + 5*np.sqrt(3)) + 107*(-2 + np.sqrt(3))*om1 + 177*(om1*om1*om1))*om3 - 148*(-9 + 5*np.sqrt(3) + 18*om1*(-2 + np.sqrt(3) - 17*(om1*om1)))*(om3*om3)) + 7*om1*om2*(2*om1*(3*(-7 + 4*np.sqrt(3)) + 4*om1*(2*(-9 + 5*np.sqrt(3)) + 9*om1*(-2 + np.sqrt(3) + om1*om1))) + 15*(7 - 4*np.sqrt(3))*om3 + 12*om1*(18 - 10*np.sqrt(3) - 3*(-2 + np.sqrt(3))*om1 + 9*(om1*om1*om1))*om3 - 20*(-9 + 5*np.sqrt(3) + 9*om1*(-2 + np.sqrt(3) + 2*(om1*om1)))*(om3*om3)) + 4*(om2*om2*om2*om2)*(-9 + 5*np.sqrt(3) - 336*(-2 + np.sqrt(3))*om3 + 3*om1*(5*(-2 + np.sqrt(3)) - 66*(om1*om1) + 4032*om1*om3)) + 14*(om1*om1)*om3*(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om3 + 4*om1*(2*(-9 + 5*np.sqrt(3)) + 9*(-2 + np.sqrt(3))*om3 + 9*om1*(-2 + np.sqrt(3) + om1*om1 + 2*om1*om3))))/(36.*(om1*om1)*(om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-1] + (
            (-12*(-2 + np.sqrt(3))*(om1*om1*om1*om1) + 4*(om1*om1*om1)*(9 - 5*np.sqrt(3) - 12*(-2 + np.sqrt(3))*om2 - 15*(-2 + np.sqrt(3))*om3) - 2*(om1*om1)*(-7 + 4*np.sqrt(3) + 36*(-2 + np.sqrt(3))*(om2*om2) + 6*om2*(-9 + 5*np.sqrt(3) - 102*(-2 + np.sqrt(3))*om3) + 8*om3*(-9 + 5*np.sqrt(3) + 3*(-2 + np.sqrt(3))*om3)) + 2*om1*(-24*(-2 + np.sqrt(3))*(om2*om2*om2) + om3*(21 - 12*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om3) + 6*(om2*om2)*(9 - 5*np.sqrt(3) + 219*(-2 + np.sqrt(3))*om3) + 2*om2*(7 - 4*np.sqrt(3) + 70*(-9 + 5*np.sqrt(3))*om3 + 327*(-2 + np.sqrt(3))*(om3*om3))) + om2*(-12*(-2 + np.sqrt(3))*(om2*om2*om2) + 37*om3*(-21 + 12*np.sqrt(3) - 36*om3 + 20*np.sqrt(3)*om3) + 4*(om2*om2)*(9 - 5*np.sqrt(3) + 336*(-2 + np.sqrt(3))*om3) + 2*om2*(7 - 4*np.sqrt(3) + 148*(-9 + 5*np.sqrt(3))*om3 + 678*(-2 + np.sqrt(3))*(om3*om3))))/(36.*(om1*om1)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-2] + (
            (6*(-2 + np.sqrt(3))*(om1*om1*om1) - 18*(-7 + 4*np.sqrt(3))*(om2 + om3) - 24*(-9 + 5*np.sqrt(3))*((om2 + om3)*(om2 + om3)) + 2*(om1*om1)*(-9 + 5*np.sqrt(3) - 111*(-2 + np.sqrt(3))*om2 - 111*(-2 + np.sqrt(3))*om3) + om1*(-7 + 4*np.sqrt(3) - 228*(-2 + np.sqrt(3))*(om2*om2) + 12*om3*(36 - 20*np.sqrt(3) - 19*(-2 + np.sqrt(3))*om3) + 24*om2*(18 - 10*np.sqrt(3) - 19*(-2 + np.sqrt(3))*om3)))/(18.*(om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-3] + h * (
            (
                (-168*(-3 + np.sqrt(3))*(om1*om1*om1*om1*om1*om1)*(om2 + om3) + 12*(om1*om1*om1*om1*om1)*(-44*(-3 + np.sqrt(3))*(om2*om2) - 42*om3*(-2 + np.sqrt(3) + (-3 + np.sqrt(3))*om3) - 7*om2*(6*(-2 + np.sqrt(3)) + 7*(-3 + np.sqrt(3))*om3)) + 5*(om2*om2)*(om2 + om3)*(12*(-2 + np.sqrt(3))*(om2*om2*om2) - 4*(om2*om2)*(9 - 5*np.sqrt(3) + 336*(-2 + np.sqrt(3))*om3) - 37*om3*(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om3) - 2*om2*(7 - 4*np.sqrt(3) + 148*(-9 + 5*np.sqrt(3))*om3 + 678*(-2 + np.sqrt(3))*(om3*om3))) + 12*(om1*om1*om1*om1)*(-36*(-3 + np.sqrt(3))*(om2*om2*om2) + 3*(om2*om2)*(-37*(-2 + np.sqrt(3)) + 42*(-3 + np.sqrt(3))*om3) - 7*om3*(-9 + 5*np.sqrt(3) + 15*(-2 + np.sqrt(3))*om3 + 4*(-3 + np.sqrt(3))*(om3*om3)) + 7*om2*(9 - 5*np.sqrt(3) + 3*om3*(-5*(-2 + np.sqrt(3)) + (-3 + np.sqrt(3))*om3))) + 2*(om1*om1*om1)*(96*(-3 + np.sqrt(3))*(om2*om2*om2*om2) + 6*(om2*om2*om2)*(-76*(-2 + np.sqrt(3)) - 989*(-3 + np.sqrt(3))*om3) - 18*(om2*om2)*(5*(-9 + 5*np.sqrt(3)) - 119*(-2 + np.sqrt(3))*om3 + 337*(-3 + np.sqrt(3))*(om3*om3)) + 7*om2*(14 - 8*np.sqrt(3) + 9*(9 - 5*np.sqrt(3))*om3 + 81*(-2 + np.sqrt(3))*(om3*om3) + 60*(-3 + np.sqrt(3))*(om3*om3*om3)) + 14*om3*(7 - 4*np.sqrt(3) - 6*om3*(-9 + 5*np.sqrt(3) + 4*(-2 + np.sqrt(3))*om3))) + 2*(om1*om1)*(204*(-3 + np.sqrt(3))*(om2*om2*om2*om2*om2) + 7*(om3*om3)*(21 - 12*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om3) - 12*(om2*om2*om2*om2)*(-9*(-2 + np.sqrt(3)) + 1241*(-3 + np.sqrt(3))*om3) + om2*om2*om2*(44*(9 - 5*np.sqrt(3)) - 7290*(-2 + np.sqrt(3))*om3 - 22644*(-3 + np.sqrt(3))*(om3*om3)) + om2*om2*(161 - 92*np.sqrt(3) + 336*(-9 + 5*np.sqrt(3))*om3 - 7470*(-2 + np.sqrt(3))*(om3*om3) - 7548*(-3 + np.sqrt(3))*(om3*om3*om3)) + 7*om2*om3*(7 - 4*np.sqrt(3) + 6*om3*(3*(-9 + 5*np.sqrt(3)) + 20*(-2 + np.sqrt(3))*om3))) + om1*om2*(144*(-3 + np.sqrt(3))*(om2*om2*om2*om2*om2) - 72*(om2*om2*om2*om2)*(-5*(-2 + np.sqrt(3)) + 222*(-3 + np.sqrt(3))*om3) + 35*(om3*om3)*(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om3) - 30*om2*om3*(49 - 28*np.sqrt(3) + 60*(-9 + 5*np.sqrt(3))*om3 + 444*(-2 + np.sqrt(3))*(om3*om3)) - 4*(om2*om2*om2)*(63 - 35*np.sqrt(3) + 90*om3*(73*(-2 + np.sqrt(3)) + 90*(-3 + np.sqrt(3))*om3)) - 8*(om2*om2)*(-7 + 4*np.sqrt(3) + om3*(220*(-9 + 5*np.sqrt(3)) + 9*om3*(555*(-2 + np.sqrt(3)) + 226*(-3 + np.sqrt(3))*om3)))))/(72.*om1*(om1 + om2)*(om1 + om2 + om3)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-1] + (
                (7*(-12*(-2 + np.sqrt(3))*(om1*om1*om1*om1) + 4*(om1*om1*om1)*(9 - 5*np.sqrt(3) - 12*(-2 + np.sqrt(3))*om2 - 15*(-2 + np.sqrt(3))*om3) - 2*(om1*om1)*(-7 + 4*np.sqrt(3) + 36*(-2 + np.sqrt(3))*(om2*om2) + 6*om2*(-9 + 5*np.sqrt(3) - 102*(-2 + np.sqrt(3))*om3) + 8*om3*(-9 + 5*np.sqrt(3) + 3*(-2 + np.sqrt(3))*om3)) + 2*om1*(-24*(-2 + np.sqrt(3))*(om2*om2*om2) + om3*(21 - 12*np.sqrt(3) + 4*(9 - 5*np.sqrt(3))*om3) + 6*(om2*om2)*(9 - 5*np.sqrt(3) + 219*(-2 + np.sqrt(3))*om3) + 2*om2*(7 - 4*np.sqrt(3) + 70*(-9 + 5*np.sqrt(3))*om3 + 327*(-2 + np.sqrt(3))*(om3*om3))) + om2*(-12*(-2 + np.sqrt(3))*(om2*om2*om2) + 4*(om2*om2)*(9 - 5*np.sqrt(3) + 336*(-2 + np.sqrt(3))*om3) + 37*om3*(3*(-7 + 4*np.sqrt(3)) + 4*(-9 + 5*np.sqrt(3))*om3) + 2*om2*(7 - 4*np.sqrt(3) + 148*(-9 + 5*np.sqrt(3))*om3 + 678*(-2 + np.sqrt(3))*(om3*om3)))))/(72.*om1*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-2] + (
                (-13*om2*(-6*(-2 + np.sqrt(3))*(om1*om1*om1) + 18*(-7 + 4*np.sqrt(3))*(om2 + om3) + 24*(-9 + 5*np.sqrt(3))*((om2 + om3)*(om2 + om3)) + 2*(om1*om1)*(9 - 5*np.sqrt(3) + 111*(-2 + np.sqrt(3))*om2 + 111*(-2 + np.sqrt(3))*om3) + om1*(7 - 4*np.sqrt(3) + 48*(-9 + 5*np.sqrt(3))*om2 + 228*(-2 + np.sqrt(3))*(om2*om2) + 48*(-9 + 5*np.sqrt(3))*om3 + 456*(-2 + np.sqrt(3))*om2*om3 + 228*(-2 + np.sqrt(3))*(om3*om3))))/(24.*(om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-3] + (
                (84*(-2 + np.sqrt(3))*(om1*om1*om1*om1) + 678*(-7 + 4*np.sqrt(3))*(om2*om2) + 912*(-9 + 5*np.sqrt(3))*(om2*om2*om2) + 14*(om1*om1*om1)*(2*(-9 + 5*np.sqrt(3)) - 21*(-2 + np.sqrt(3))*om2) + 2*(om1*om1)*(7*(-7 + 4*np.sqrt(3)) + 49*(9 - 5*np.sqrt(3))*om2 + 4143*(-2 + np.sqrt(3))*(om2*om2)) + om1*om2*(49*(7 - 4*np.sqrt(3)) + 24*om2*(75*(-9 + 5*np.sqrt(3)) + 361*(-2 + np.sqrt(3))*om2)))/(72.*(om1 + om2 + om3)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-4]
        )
    u2int = (
            (-12*(2 + np.sqrt(3) + 36*(om1*om1))*(om2*om2*om2*om2*om2) - 4*(om2*om2*om2*om2)*(9 + 5*np.sqrt(3) + 30*om1 + 15*np.sqrt(3)*om1 + 198*(om1*om1*om1) - 336*(2 + np.sqrt(3) + 36*(om1*om1))*om3) + 2*(om2*om2*om2)*(-7 - 4*np.sqrt(3) + 4*om1*(-2*(9 + 5*np.sqrt(3)) - 15*(2 + np.sqrt(3))*om1 + 27*(om1*om1*om1)) + 148*(9 + 5*np.sqrt(3))*om3 + 6*om1*(331*(2 + np.sqrt(3)) + 3480*(om1*om1))*om3 + 678*(2 + np.sqrt(3) + 36*(om1*om1))*(om3*om3)) + om2*om2*(6*om1*(-7 - 4*np.sqrt(3) + 12*om1*(-9 - 5*np.sqrt(3) - 8*(2 + np.sqrt(3))*om1 + 15*(om1*om1*om1))) + 111*(7 + 4*np.sqrt(3))*om3 + 36*om1*(16*(9 + 5*np.sqrt(3)) + 107*(2 + np.sqrt(3))*om1 - 177*(om1*om1*om1))*om3 + 148*(9 + 5*np.sqrt(3) + 18*om1*(2 + np.sqrt(3) + 17*(om1*om1)))*(om3*om3)) + 7*om1*om2*(72*(om1*om1*om1*om1*om1) + 108*(om1*om1*om1*om1)*om3 + 4*(om1*om1)*(-4*(9 + 5*np.sqrt(3)) + 9*(2 + np.sqrt(3))*om3) + 5*om3*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3) - 72*(om1*om1*om1)*(2 + np.sqrt(3) + 5*(om3*om3)) + 6*om1*(-7 - 4*np.sqrt(3) + 4*(9 + 5*np.sqrt(3))*om3 + 30*(2 + np.sqrt(3))*(om3*om3))) + 14*(om1*om1)*om3*(-3*(7 + 4*np.sqrt(3)) - 4*(9 + 5*np.sqrt(3))*om3 + 4*om1*(-2*(9 + 5*np.sqrt(3)) - 9*(2 + np.sqrt(3))*om3 + 9*om1*(-2 - np.sqrt(3) + om1*om1 + 2*om1*om3))))/(36.*(om1*om1)*(om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-1] + (
            (12*(2 + np.sqrt(3))*(om1*om1*om1*om1) + 4*(om1*om1*om1)*(9 + 5*np.sqrt(3) + 12*(2 + np.sqrt(3))*om2 + 15*(2 + np.sqrt(3))*om3) + 2*(om1*om1)*(7 + 4*np.sqrt(3) + 36*(2 + np.sqrt(3))*(om2*om2) + 6*om2*(9 + 5*np.sqrt(3) - 102*(2 + np.sqrt(3))*om3) + 8*om3*(9 + 5*np.sqrt(3) + 3*(2 + np.sqrt(3))*om3)) + 2*om1*(24*(2 + np.sqrt(3))*(om2*om2*om2) + om3*(21 + 12*np.sqrt(3) + 36*om3 + 20*np.sqrt(3)*om3) - 6*(om2*om2)*(-9 - 5*np.sqrt(3) + 219*(2 + np.sqrt(3))*om3) - 2*om2*(-7 - 4*np.sqrt(3) + 70*(9 + 5*np.sqrt(3))*om3 + 327*(2 + np.sqrt(3))*(om3*om3))) + om2*(12*(2 + np.sqrt(3))*(om2*om2*om2) - 37*om3*(21 + 12*np.sqrt(3) + 36*om3 + 20*np.sqrt(3)*om3) - 4*(om2*om2)*(-9 - 5*np.sqrt(3) + 336*(2 + np.sqrt(3))*om3) - 2*om2*(-7 - 4*np.sqrt(3) + 148*(9 + 5*np.sqrt(3))*om3 + 678*(2 + np.sqrt(3))*(om3*om3))))/(36.*(om1*om1)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-2] + (
            (-6*(2 + np.sqrt(3))*(om1*om1*om1) + 18*(7 + 4*np.sqrt(3))*(om2 + om3) + 24*(9 + 5*np.sqrt(3))*((om2 + om3)*(om2 + om3)) + 2*(om1*om1)*(-9 - 5*np.sqrt(3) + 111*(2 + np.sqrt(3))*om2 + 111*(2 + np.sqrt(3))*om3) + om1*(-7 - 4*np.sqrt(3) + 48*(9 + 5*np.sqrt(3))*om2 + 228*(2 + np.sqrt(3))*(om2*om2) + 48*(9 + 5*np.sqrt(3))*om3 + 456*(2 + np.sqrt(3))*om2*om3 + 228*(2 + np.sqrt(3))*(om3*om3)))/(18.*(om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
        ) * uu[-3] + h * (
            (
                (168*(3 + np.sqrt(3))*(om1*om1*om1*om1*om1*om1)*(om2 + om3) + 12*(om1*om1*om1*om1*om1)*(44*(3 + np.sqrt(3))*(om2*om2) + 42*om3*(2 + np.sqrt(3) + (3 + np.sqrt(3))*om3) + 7*om2*(6*(2 + np.sqrt(3)) + 7*(3 + np.sqrt(3))*om3)) - 5*(om2*om2)*(om2 + om3)*(12*(2 + np.sqrt(3))*(om2*om2*om2) - 4*(om2*om2)*(-9 - 5*np.sqrt(3) + 336*(2 + np.sqrt(3))*om3) - 37*om3*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3) - 2*om2*(-7 - 4*np.sqrt(3) + 148*(9 + 5*np.sqrt(3))*om3 + 678*(2 + np.sqrt(3))*(om3*om3))) + 2*(om1*om1*om1)*(-96*(3 + np.sqrt(3))*(om2*om2*om2*om2) + 6*(om2*om2*om2)*(76*(2 + np.sqrt(3)) + 989*(3 + np.sqrt(3))*om3) + 18*(om2*om2)*(5*(9 + 5*np.sqrt(3)) - 119*(2 + np.sqrt(3))*om3 + 337*(3 + np.sqrt(3))*(om3*om3)) - 7*om2*(-2*(7 + 4*np.sqrt(3)) - 9*(9 + 5*np.sqrt(3))*om3 + 81*(2 + np.sqrt(3))*(om3*om3) + 60*(3 + np.sqrt(3))*(om3*om3*om3)) + 14*om3*(7 + 4*np.sqrt(3) + 6*om3*(9 + 5*np.sqrt(3) + 4*(2 + np.sqrt(3))*om3))) + 12*(om1*om1*om1*om1)*(36*(3 + np.sqrt(3))*(om2*om2*om2) + 3*(om2*om2)*(37*(2 + np.sqrt(3)) - 42*(3 + np.sqrt(3))*om3) + 7*om3*(9 + 5*np.sqrt(3) + 15*(2 + np.sqrt(3))*om3 + 4*(3 + np.sqrt(3))*(om3*om3)) - 7*om2*(-9 - 5*np.sqrt(3) + 3*om3*(-5*(2 + np.sqrt(3)) + (3 + np.sqrt(3))*om3))) + 2*(om1*om1)*(-204*(3 + np.sqrt(3))*(om2*om2*om2*om2*om2) + 12*(om2*om2*om2*om2)*(-9*(2 + np.sqrt(3)) + 1241*(3 + np.sqrt(3))*om3) + 7*(om3*om3)*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3) + om2*om2*(23*(7 + 4*np.sqrt(3)) - 336*(9 + 5*np.sqrt(3))*om3 + 7470*(2 + np.sqrt(3))*(om3*om3) + 7548*(3 + np.sqrt(3))*(om3*om3*om3)) + 7*om2*om3*(7 + 4*np.sqrt(3) - 6*om3*(3*(9 + 5*np.sqrt(3)) + 20*(2 + np.sqrt(3))*om3)) + 2*(om2*om2*om2)*(22*(9 + 5*np.sqrt(3)) + 9*om3*(405*(2 + np.sqrt(3)) + 1258*(3 + np.sqrt(3))*om3))) + om1*om2*(-144*(3 + np.sqrt(3))*(om2*om2*om2*om2*om2) + 72*(om2*om2*om2*om2)*(-5*(2 + np.sqrt(3)) + 222*(3 + np.sqrt(3))*om3) - 35*(om3*om3)*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3) + 30*om2*om3*(-7*(7 + 4*np.sqrt(3)) + 60*(9 + 5*np.sqrt(3))*om3 + 444*(2 + np.sqrt(3))*(om3*om3)) + 4*(om2*om2*om2)*(-7*(9 + 5*np.sqrt(3)) + 90*om3*(73*(2 + np.sqrt(3)) + 90*(3 + np.sqrt(3))*om3)) + 8*(om2*om2)*(7 + 4*np.sqrt(3) + om3*(220*(9 + 5*np.sqrt(3)) + 9*om3*(555*(2 + np.sqrt(3)) + 226*(3 + np.sqrt(3))*om3)))))/(72.*om1*(om1 + om2)*(om1 + om2 + om3)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-1] + (
                (7*(12*(2 + np.sqrt(3))*(om1*om1*om1*om1) + 4*(om1*om1*om1)*(9 + 5*np.sqrt(3) + 12*(2 + np.sqrt(3))*om2 + 15*(2 + np.sqrt(3))*om3) + 2*(om1*om1)*(7 + 4*np.sqrt(3) + 36*(2 + np.sqrt(3))*(om2*om2) + 6*om2*(9 + 5*np.sqrt(3) - 102*(2 + np.sqrt(3))*om3) + 8*om3*(9 + 5*np.sqrt(3) + 3*(2 + np.sqrt(3))*om3)) + 2*om1*(24*(2 + np.sqrt(3))*(om2*om2*om2) - 6*(om2*om2)*(-9 - 5*np.sqrt(3) + 219*(2 + np.sqrt(3))*om3) + om3*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3) - 2*om2*(-7 - 4*np.sqrt(3) + 70*(9 + 5*np.sqrt(3))*om3 + 327*(2 + np.sqrt(3))*(om3*om3))) + om2*(12*(2 + np.sqrt(3))*(om2*om2*om2) - 4*(om2*om2)*(-9 - 5*np.sqrt(3) + 336*(2 + np.sqrt(3))*om3) - 37*om3*(3*(7 + 4*np.sqrt(3)) + 4*(9 + 5*np.sqrt(3))*om3) - 2*om2*(-7 - 4*np.sqrt(3) + 148*(9 + 5*np.sqrt(3))*om3 + 678*(2 + np.sqrt(3))*(om3*om3)))))/(72.*om1*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-2] + (
                (-13*om2*(6*(2 + np.sqrt(3))*(om1*om1*om1) - 18*(7 + 4*np.sqrt(3))*(om2 + om3) - 24*(9 + 5*np.sqrt(3))*((om2 + om3)*(om2 + om3)) - 2*(om1*om1)*(-9 - 5*np.sqrt(3) + 111*(2 + np.sqrt(3))*om2 + 111*(2 + np.sqrt(3))*om3) - om1*(-7 - 4*np.sqrt(3) + 48*(9 + 5*np.sqrt(3))*om2 + 228*(2 + np.sqrt(3))*(om2*om2) + 48*(9 + 5*np.sqrt(3))*om3 + 456*(2 + np.sqrt(3))*om2*om3 + 228*(2 + np.sqrt(3))*(om3*om3))))/(24.*(om1 + om2)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-3] + (
                (-84*(2 + np.sqrt(3))*(om1*om1*om1*om1) + 14*(om1*om1*om1)*(-2*(9 + 5*np.sqrt(3)) + 21*(2 + np.sqrt(3))*om2) - 6*(om2*om2)*(113*(7 + 4*np.sqrt(3)) + 152*(9 + 5*np.sqrt(3))*om2) - 2*(om1*om1)*(7*(7 + 4*np.sqrt(3)) - 49*(9 + 5*np.sqrt(3))*om2 + 4143*(2 + np.sqrt(3))*(om2*om2)) + om1*om2*(49*(7 + 4*np.sqrt(3)) + 24*om2*(-75*(9 + 5*np.sqrt(3)) - 361*(2 + np.sqrt(3))*om2)))/(72.*(om1 + om2 + om3)*(14*(om1*om1*om1)*(om2 + om3) - 12*(om2*om2)*(om2 - 113*om3)*(om2 + om3) + om1*om1*(16*(om2*om2) + 7*om2*om3 + 28*(om3*om3)) - 2*om1*om2*(5*(om2*om2) + 92*om2*om3 + 49*(om3*om3))))
            ) * ff[-4]
        )

    eta_est = old_eta[-1] + h * ( 0.5 * np.dot(deta(u1int), f(u1int)) + 0.5 * np.dot(deta(u2int), f(u2int)) )
    return eta_est

def conservative_EDC33(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                       **kwargs):
    return conservative_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_EDC33, adaptive_step_EDC33,
                            **kwargs)

def cons_or_diss_EDC33(f, t_final, t0, u0, t1, u1, t2, u2, t3, u3,
                       **kwargs):
    return cons_or_diss_LMM(f, t_final, [t0, t1, t2, t3], [u0, u1, u2, u3],
                            fixed_step_EDC33, adaptive_step_EDC33,
                            fixed_estimate_EDC33, adaptive_estimate_EDC33,
                            **kwargs)




def relaxation_ERK(rkm, dt, f, eta, deta, w0, num_steps,
                   relaxed=True, method="brentq", tol=1.e-14, maxiter=10000, jac=False, newdt=True,
                   debug=False, print_gamma=False):
    """Relaxed explicit Runge-Kutta method for general functionals."""

    rkm = rkm.__num__()

    w = np.array(w0) # current value of the unknown function
    t = 0 # current time
    ww = np.zeros([np.size(w0), 1]) # values at each time step
    ww[:,0] = w.copy()
    tt = np.zeros(1) # time points for ww
    gg = np.ones(1)  # values of gamma
    tt[0] = t
    b = rkm.b
    s = len(rkm)
    y = np.zeros((s, np.size(w0))) # stage values
    F = np.zeros((s, np.size(w0))) # stage derivatives
    max_gammam1 = 0.  # max(gamma-1) over all timesteps
    old_gamma = 1.0


    step = 0
    while step < num_steps:
        step = step + 1

        for i in range(s):
            y[i,:] = w.copy()
            for j in range(i):
                y[i,:] += rkm.A[i,j]*dt*F[j,:]
            F[i,:] = f(y[i,:])

        if relaxed:
            direction = dt * sum([b[i]*F[i,:] for i in range(s)])
            estimate = dt * sum([b[i]*np.dot(deta(y[i,:]),F[i,:]) for i in range(s)])

            r = lambda gamma: eta(w+gamma*direction) - eta(w) - gamma*estimate
            if debug:
                print('r(1): ', r(1))
            rjac= lambda gamma: np.array([np.dot(deta(w+gamma*direction), direction) - estimate])

            if rjac == False:
                use_jac = False
            else:
                use_jac = rjac

            if method == "newton":
                gam = newton(r, old_gamma, fprime=rjac, tol=tol, maxiter=maxiter)
                success = True
                msg = "Newton method did not converge"
            elif method == "brentq" or method == "bisect":
                left = 0.9 * old_gamma
                right = 1.1 * old_gamma
                left_right_iter = 0
                while r(left) * r(right) > 0:
                    left *= 0.9
                    right *= 1.1
                    left_right_iter += 1
                    if left_right_iter > 100:
                        raise SolveForGammaException(
                            "No suitable bounds found after %d iterations.\nLeft = %e; r(left) = %e\nRight = %e; r(right) = %e\n"%(
                                left_right_iter, left, r(left), right, r(right)),
                            w)

                if method == "brentq":
                    gam = brentq(r, left, right, xtol=tol, maxiter=maxiter)
                else:
                    gam = bisect(r, left, right, xtol=tol, maxiter=maxiter)
                success = True
                msg = "%s method did not converge"%method
            else:
                sol = root(r, old_gamma, jac=use_jac, method=method, tol=tol,
                           options={'xtol': tol, 'maxiter': maxiter})
                gam = sol.x; success = sol.success; msg = sol.message

            if success == False:
                print('Warning: fsolve did not converge.')
                print(gam)
                print(msg)

            if gam <= 0:
                print('Warning: gamma is negative.')

        else:
            gam = 1.

        old_gamma = gam

        if debug:
            gm1 = np.abs(1.-gam)
            max_gammam1 = max(max_gammam1,gm1)
            if gm1 > 0.5:
                print(gam)
                raise Exception("The time step is probably too large.")

        w = w + dt*gam*sum([b[i]*F[i] for i in range(s)])
        if newdt == True:
            t += gam*dt
        else:
            t += dt

        tt = np.append(tt, t)
        ww = np.append(ww, np.reshape(w.copy(), (len(w), 1)), axis=1)
        gg = np.append(gg, gam)

    if debug:
        if print_gamma:
            print(max_gammam1)
        return tt, ww, gg
    else:
        return tt, ww

def relaxation_DIRK(rkm, dt, f, eta, deta, w0, num_steps,
                    relaxed=True, method="brentq", tol=1.e-14, maxiter=10000, jac=False, newdt=True,
                    debug=False, print_gamma=False):
    """Relaxed diagonally implicit Runge-Kutta method for general functionals."""

    rkm = rkm.__num__()

    w = np.array(w0) # current value of the unknown function
    t = 0 # current time
    ww = np.zeros([np.size(w0), 1]) # values at each time step
    ww[:,0] = w.copy()
    tt = np.zeros(1) # time points for ww
    gg = np.ones(1)  # values of gamma
    tt[0] = t
    b = rkm.b
    s = len(rkm)
    y = np.zeros((s, np.size(w0))) # stage values
    F = np.zeros((s, np.size(w0))) # stage derivatives
    max_gammam1 = 0.  # max(gamma-1) over all timesteps
    old_gamma = 1.0


    step = 0
    while step < num_steps:
        step = step + 1

        for i in range(s):
            stageeq = lambda Y: (Y - w - dt*sum([rkm.A[i,j]*F[j,:] for j in range(i)]) \
                                 - dt*rkm.A[i,i]*f(Y)).squeeze()
            nexty, info, ier, mesg = fsolve(stageeq,w,full_output=1)
            if ier != 1:
                print(mesg)
                # print(info)
                # raise Exception("System couldn't be solved.")
            y[i,:] = nexty.copy()
            F[i,:] = f(y[i,:])

        if relaxed:
            direction = dt * sum([b[i]*F[i,:] for i in range(s)])
            estimate = dt * sum([b[i]*np.dot(deta(y[i,:]),F[i,:]) for i in range(s)])

            r = lambda gamma: eta(w+gamma*direction) - eta(w) - gamma*estimate
            if debug:
                print('r(1): ', r(1))
            rjac= lambda gamma: np.array([np.dot(deta(w+gamma*direction), direction) - estimate])

            if rjac == False:
                use_jac = False
            else:
                use_jac = rjac

            if method == "newton":
                gam = newton(r, old_gamma, fprime=rjac, tol=tol, maxiter=maxiter)
                success = True
                msg = "Newton method did not converge"
            elif method == "brentq" or method == "bisect":
                left = 0.9 * old_gamma
                right = 1.1 * old_gamma
                left_right_iter = 0
                while r(left) * r(right) > 0:
                    left *= 0.9
                    right *= 1.1
                    left_right_iter += 1
                    if left_right_iter > 100:
                        raise SolveForGammaException(
                            "No suitable bounds found after %d iterations.\nLeft = %e; r(left) = %e\nRight = %e; r(right) = %e\n"%(
                                left_right_iter, left, r(left), right, r(right)),
                            w)

                if method == "brentq":
                    gam = brentq(r, left, right, xtol=tol, maxiter=maxiter)
                else:
                    gam = bisect(r, left, right, xtol=tol, maxiter=maxiter)
                success = True
                msg = "%s method did not converge"%method
            else:
                sol = root(r, old_gamma, jac=use_jac, method=method, tol=tol,
                           options={'xtol': tol, 'maxiter': maxiter})
                gam = sol.x; success = sol.success; msg = sol.message

            if success == False:
                print('Warning: fsolve did not converge.')
                print(gam)
                print(msg)

            if gam <= 0:
                print('Warning: gamma is negative.')

        else:
            gam = 1.

        old_gamma = gam

        if debug:
            gm1 = np.abs(1.-gam)
            max_gammam1 = max(max_gammam1,gm1)
            if gm1 > 0.5:
                print(gam)
                raise Exception("The time step is probably too large.")

        w = w + dt*gam*sum([b[i]*F[i] for i in range(s)])
        if newdt == True:
            t += gam*dt
        else:
            t += dt

        tt = np.append(tt, t)
        ww = np.append(ww, np.reshape(w.copy(), (len(w), 1)), axis=1)
        gg = np.append(gg, gam)

    if debug:
        if print_gamma:
            print(max_gammam1)
        return tt, ww, gg
    else:
        return tt, ww


def conservative_BDF2(f, t_final, t0, u0, t1, u1,
                      idx_u_old=-1,
                      eta=etaL2, deta=detaL2,
                      return_gamma=False,
                      projection=False, relaxation=False,
                      adapt_dt=False, adapt_coefficients=False,
                      method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    uu = [u0, u1]
    ff = [f(u) for u in uu]
    tt = [t0, t1]

    h = tt[1] - tt[0]
    old_omega = [(tt[i+1] - tt[i]) / h for i in np.arange(len(tt)-1)]
    old_gamma = [1.0 for i in np.arange(len(tt)-1)]

    if relaxation and projection:
        raise Exception("Use either relaxation or projection, not both.")

    if relaxation and method == None:
        method = "brentq"
    elif projection and method == None:
        method = "simplified Newton"

    t = tt[-1]
    gammas = [1.0 for t in tt]
    step = 0
    while t < t_final and step < maxsteps:
        step += 1

        if relaxation and adapt_coefficients:
            om1 = old_omega[-1]
            uval = (
                    ((1 + om1)*(1 + om1))/(om1*(2 + om1))
                ) * uu[-1] + (
                    -(1/(2*om1 + om1*om1))
                ) * uu[-2]
            fcoeff = (1 + om1)/(2 + om1)
            stageeq = lambda Y: (Y - uval - h * fcoeff * f(Y)).squeeze()
            nexty, info, ier, mesg = fsolve(stageeq, uu[-1], full_output=1)
            if ier != 1:
                print(mesg)
                # print(info)
                # raise Exception("System couldn't be solved.")
            u_new = nexty.copy()
        else:
            uval = (
                    1.3333333333333333
                ) * uu[-1] + (
                    -0.3333333333333333
                ) * uu[-2]
            fcoeff = 0.6666666666666666
            stageeq = lambda Y: (Y - uval - h * fcoeff * f(Y)).squeeze()
            nexty, info, ier, mesg = fsolve(stageeq, uu[-1], full_output=1)
            if ier != 1:
                print(mesg)
                # print(info)
                # raise Exception("System couldn't be solved.")
            u_new = nexty.copy()

        u_old = uu[idx_u_old]
        eta_old = eta(u_old)
        if projection:
            gamma, u_new = conservative_projection_solve(eta, deta, u_old, eta_old, u_new, method, tol, maxiter)
        elif relaxation:
            gamma = conservative_relaxation_solve(eta, deta, u_old, eta_old, u_new, old_gamma[-1], method, tol, maxiter)
            u_new = u_old + gamma * (u_new - u_old)
            for i in np.arange(-len(old_gamma), -1):
                old_gamma[i] = old_gamma[i+1]
            old_gamma[-1] = gamma
        else:
            gamma = 1.0

        if return_gamma:
            gammas.append(gamma)

        uu.append(u_new)
        if relaxation and adapt_dt:
            t = tt[idx_u_old] - gamma * idx_u_old * h
            new_omega = -idx_u_old*gamma - np.sum([old_omega[i] for i in np.arange(-1, idx_u_old, -1)])
            for i in np.arange(-len(old_omega), -1):
                old_omega[i] = old_omega[i+1]
            old_omega[-1] = new_omega
            if gamma < 1.0e-14:
                raise Exception("gamma = %.2e is too small in step %d!" % (gamma, step))
        else:
            t += h
        tt.append(t)

        for i in np.arange(-len(ff), -1):
            ff[i] = ff[i+1]
        ff[-1] = f(u_new)

    if return_gamma:
        return np.array(tt), uu, np.array(gammas)
    else:
        return np.array(tt), uu

def conservative_BDF3(f, t_final, t0, u0, t1, u1, t2, u2,
                      idx_u_old=-1,
                      eta=etaL2, deta=detaL2,
                      return_gamma=False,
                      projection=False, relaxation=False,
                      adapt_dt=False, adapt_coefficients=False,
                      method=None, tol=1.e-14, maxiter=10000, maxsteps=10**12):
    uu = [u0, u1, u2]
    ff = [f(u) for u in uu]
    tt = [t0, t1, t2]

    h = tt[1] - tt[0]
    old_omega = [(tt[i+1] - tt[i]) / h for i in np.arange(len(tt)-1)]
    old_gamma = [1.0 for i in np.arange(len(tt)-1)]

    if relaxation and projection:
        raise Exception("Use either relaxation or projection, not both.")

    if relaxation and method == None:
        method = "brentq"
    elif projection and method == None:
        method = "simplified Newton"

    t = tt[-1]
    gammas = [1.0 for t in tt]
    step = 0
    while t < t_final and step < maxsteps:
        step += 1

        if relaxation and adapt_coefficients:
            om2 = old_omega[-2]
            om1 = old_omega[-1]
            uval = (
                    ((1 + om1)*(1 + om1)*((1 + om1 + om2)*(1 + om1 + om2)))/(om1*(om1 + om2)*(3 + 2*om2 + om1*(4 + om1 + om2)))
                ) * uu[-1] + (
                    -(((1 + om1 + om2)*(1 + om1 + om2))/(om1*om2*(3 + 2*om2 + om1*(4 + om1 + om2))))
                ) * uu[-2] + (
                    ((1 + om1)*(1 + om1))/(om2*(om1 + om2)*(3 + 2*om2 + om1*(4 + om1 + om2)))
                ) * uu[-3]
            fcoeff = 1/(1 + 1/(1 + om1) + 1/(1 + om1 + om2))
            stageeq = lambda Y: (Y - uval - h * fcoeff * f(Y)).squeeze()
            nexty, info, ier, mesg = fsolve(stageeq, uu[-1], full_output=1)
            if ier != 1:
                print(mesg)
                # print(info)
                # raise Exception("System couldn't be solved.")
            u_new = nexty.copy()
        else:
            uval = (
                    1.6363636363636365
                ) * uu[-1] + (
                    -0.8181818181818182
                ) * uu[-2] + (
                    0.18181818181818182
                ) * uu[-3]
            fcoeff = 0.5454545454545454
            stageeq = lambda Y: (Y - uval - h * fcoeff * f(Y)).squeeze()
            nexty, info, ier, mesg = fsolve(stageeq, uu[-1], full_output=1)
            if ier != 1:
                print(mesg)
                # print(info)
                # raise Exception("System couldn't be solved.")
            u_new = nexty.copy()

        u_old = uu[idx_u_old]
        eta_old = eta(u_old)
        if projection:
            gamma, u_new = conservative_projection_solve(eta, deta, u_old, eta_old, u_new, method, tol, maxiter)
        elif relaxation:
            gamma = conservative_relaxation_solve(eta, deta, u_old, eta_old, u_new, old_gamma[-1], method, tol, maxiter)
            u_new = u_old + gamma * (u_new - u_old)
            for i in np.arange(-len(old_gamma), -1):
                old_gamma[i] = old_gamma[i+1]
            old_gamma[-1] = gamma
        else:
            gamma = 1.0

        if return_gamma:
            gammas.append(gamma)

        uu.append(u_new)
        if relaxation and adapt_dt:
            t = tt[idx_u_old] - gamma * idx_u_old * h
            new_omega = -idx_u_old*gamma - np.sum([old_omega[i] for i in np.arange(-1, idx_u_old, -1)])
            for i in np.arange(-len(old_omega), -1):
                old_omega[i] = old_omega[i+1]
            old_omega[-1] = new_omega
            if gamma < 1.0e-14:
                raise Exception("gamma = %.2e is too small in step %d!" % (gamma, step))
        else:
            t += h
        tt.append(t)

        for i in np.arange(-len(ff), -1):
            ff[i] = ff[i+1]
        ff[-1] = f(u_new)

    if return_gamma:
        return np.array(tt), uu, np.array(gammas)
    else:
        return np.array(tt), uu
