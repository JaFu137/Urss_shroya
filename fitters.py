import numpy as np
from lmfit import Minimizer, Parameters, report_fit
import emcee


def least_square_fit(func, data_x, data_y, data_y_err, label, initals, max_, min_, report=None, min_func=None, args=None):
    """
    #########################
    # Least Squares Fitting #
    #########################
    
    Arguments:
        func {[Callable Function]} -- [Model function take free parameter in list form as argument]
        data_x {[Array]} -- [Input data]
        data_y {[Array]} -- [Data to fit model to]
        data_y_err {[Array]} -- [Uncertainty in the data]
        label {[List]} -- [List of string with name of free parameters]
        initial, max_, min_: {[List]} -- [Lists of parameter initial values, maximum and minimum bounds]
    
    Keyword Arguments:
        report {[String]} -- ["report_fit" or "pretty_print"] (default: {None})
        min_func {[Callable Function]} -- [Custom residual function to minimise] (default: {None})
        args {[Tuple]} -- [Any extra arguments required for min_func] (default: {None})
        Returns -- Two lists containing fitted parameters and uncertainty as a tuple
    """

    func_args = (data_x, data_y, data_y_err)    # Required arguments for res function

    if args is not None:
        func_args = (data_x, data_y, data_y_err) + args     # Any additional arguments for custom res function

    def res(param, x, y, y_err):
        # Default res function
        theta = []
        for k, name in enumerate(param):    # Adding parameter values to a list
            theta.append(param[name].value)
        model_y = func(theta, x)
        return (y - model_y) / y_err

    params = Parameters()   # Initialise Parameters class for
    for i in range(len(label)):
        params.add(label[i], value=initals[i], max=max_[i], min=min_[i])

    print("Running Least-Square Fitting...")
    if min_func is None:
        fit = Minimizer(res, params, fcn_args=func_args)

    else:
        fit = Minimizer(min_func, params, fcn_args=func_args)

    result = fit.leastsq()
    if report == "report_fit":
        report_fit(result)

    if report == "pretty_print":
        result.params.pretty_print()

    return_theta = []
    return_theta_err = []

    for i, name in enumerate(result.params):
        return_theta.append(result.params[name].value)
        return_theta_err.append(result.params[name].stderr)
        # print(result[name].name, result[name].value)

    return return_theta, return_theta_err


def mcmc_fit(func, initial, initial_err, nwalkers, ndim, nburns, niter, *extra_args, lnlikefunc=None, args=None):
    """
    ########
    # MCMC #
    ########
    
    Arguments:
        func {[Callable Function]} -- [Model function take free parameter in list form as argument]
        initial, initial_err {[List]} -- [Lists of parameter initial values, and uncertainty in initial value. If initial_err has None
                                          type elements uncertainty is taken to be +/- 3*initial]
        arr {[Array]} -- [Array with 5 columns. 3nd is x_data, 4th is y_data, 5th is y_data_error]
        nwalkers {[int]} -- [Number of walkers to use for MCMC]
        ndim {[int]} -- [Number of free parameters in the func]
        nburns {[int]} -- [Number of iterations at the start to burn]
        niter {[int]} -- [Number of iterations to run mcmc for]
    
    Keyword Arguments:
        lnlikefunc {[Callable Function]} -- [Custom likely-hood function] (default: {None})
        args {[tuple]} -- [Arguments to pass to the posterior probability function must be given as a tuple] (default: {None})
        *extra_args{[tuple]} --  Any extra arguments to pass into custom likely-hood function
        Returns -- Sampler EnsembleSampler class
    """

    def lnlike(theta, x, y, y_err):

        #######################
        # Likelihood function #
        #######################

        model_y = func(theta, x)
        return -0.5 * (np.sum((y - model_y) ** 2 / y_err ** 2))

    def lnprior(theta):

        ##################
        # Prior function #
        ##################

        initial_max = [None]*ndim
        initial_min = [None]*ndim
        for i in range(ndim):
            if initial_err[i] is None:
                initial_max[i] = initial[i] + 3*abs(initial[i])
                initial_min[i] = initial[i] - 3*abs(initial[i])
            else:
                initial_max[i] = initial[i] + 3 * initial_err[i]
                initial_min[i] = initial[i] - 3 * initial_err[i]
                # initial_max[4] = 2.2*np.pi
                # initial_max[8] = 2.2*np.pi
                # initial_min[4] = -0.5*np.pi
                # initial_min[8] = -0.5*np.pi

        if all(initial_min[i] < theta[i] < initial_max[i] for i in range(ndim)):
            return 0.0
        else:
            return -np.inf

    def lnprob(theta, x, y, y_err):

        ###########################
        # Probability to maximise #
        ###########################

        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        if lnlikefunc is None:
            return lp + lnlike(theta, x, y, y_err)
        else:
            return lp + lnlikefunc(theta, x, y, y_err, *extra_args)

    p0 = [initial + (initial * np.random.uniform(-1, 1, ndim)) / 10 for i in range(nwalkers)]

    if args is None:
        print("Arguments for likely-hood function not given")
        exit(1)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

    if nburns != 0:
        print("Running Burn-in...")
        try:
            p0, _, _ = sampler.run_mcmc(p0, nburns)
            sampler.reset()
        except TypeError:
            raise

    print("Running Production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler


def sample_walkers(func, x, nsamples, flattened_chain, theta_func=None, **theta_func_kwargs):

    ###########################
    # 1-sigma spread of model #
    ###########################

    models = []
    draw = np.floor(np.random.uniform(0, len(flattened_chain), size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        if theta_func is not None:
            mod = func(theta_func(i, **theta_func_kwargs), x)
        else:
            mod = func(i, x)

        models.append(mod)
    spread = np.std(models, axis=0)
    med_model = np.mean(models, axis=0)
    return med_model, spread
