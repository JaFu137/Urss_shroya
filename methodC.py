import csv
import numpy as np 
import corner
import fitters
import matplotlib.pyplot as plt 

def import_data(filename):

    ################
    # Reading Data #
    ################

    data = np.empty([0, 4])
    copy = [0] * 4
    try:
        with open(filename, 'r') as file:
            read = csv.reader(file)
            for row in read:
                copy[0] = float(row[2])
                copy[1] = float(row[3])
                copy[2] = float(row[4])
                copy[3] = float(row[5])

                data = np.append(data, [copy], axis=0)
        return data
    except NameError:
        exit(10)

def poly(poly_theta, array):

    ########################
    # 4th order Polynomial #
    ########################

    a00, a01, a02, a03, a04, \
    a10, a11, a12, a13, a14, \
    a20, a21, a22, a23, a24, \
    a30, a31, a32, a33, a34 = poly_theta

    poly = np.empty([0])

    for row in array:
        n = row[0]
        l = row[1]
    
        if l == 0:
            poly = np.append(poly, a00 + a01*n + a02*n*n + a03*n*n*n + a04*n*n*n*n)
            
        if l == 1:
            poly = np.append(poly, a10 + a11*n + a12*n*n + a13*n*n*n + a14*n*n*n*n)
        
        if l == 2:
            poly = np.append(poly, a20 + a21*n + a22*n*n + a23*n*n*n + a24*n*n*n*n)
        
        if l == 3:
            poly = np.append(poly, a30 + a31*n + a32*n*n + a33*n*n*n + a34*n*n*n*n)

    return poly

def poly_diff(poly_theta, array):   

    ################################
    # 2nd Derivative of Polynomial #
    ################################

    a00, a01, a02, a03, a04, \
    a10, a11, a12, a13, a14, \
    a20, a21, a22, a23, a24, \
    a30, a31, a32, a33, a34 = poly_theta

    poly_diff = np.empty([0])

    for row in array:
        n = row[0]
        l = row[1]

        if l == 0:
            poly_diff = np.append(poly_diff, a02 + a03*n + a04*n*n)
            
        if l == 1:
            poly_diff = np.append(poly_diff, a12 + a13*n + a14*n*n)
        
        if l == 2:
            poly_diff = np.append(poly_diff, a22 + a23*n + a24*n*n)
        
        if l == 3:
            poly_diff = np.append(poly_diff, a32 + a33*n + a34*n*n)

    return poly_diff

def poly_res(poly_param, flat_array, f, ferr, lam):

    ###############################################
    # Residual to be minimised for Polynomial fit #
    ###############################################

    array = np.reshape(flat_array, (len(f), 4))

    f = array[:, 2]
    ferr = array[:, 3]
    theta_ = []

    # Adding parameter values to a list
    for i, name in enumerate(poly_param):
        theta_.append(poly_param[name].value)
    
    poly_diff_ = poly_diff(theta_, array)
    poly_ = poly(theta_, array)
    res = np.sqrt((f-poly_)**2 + lam*poly_diff_**2)

    return res

def model_func(theta, f, mode=None):

    ##################
    # Model Function #
    ##################
    
    a_cz, tau_cz, psi_cz, a_he, c_he, tau_he, psi_he = theta

    convec = (a_cz/f**2)*np.sin(4*np.pi*tau_cz*f + psi_cz)   # Convection zone signal
    helium = a_he*f*np.exp(-c_he*f**2)*np.sin(4*np.pi*tau_he*f + psi_he)   # He ionisation zone

    if mode == "convec":
        return convec
    elif mode == "helium":
        return helium
    elif mode is None:
        return convec + helium

def main(filename, nwalkers, nburns, niter, lam, plot_steps=False, report=None, all_theta=False, spread=False):
    """    
    Arguments:
        filename {[String]} -- [CSV file to run the script on, (do not include .csv extention)]
        nwalkers {[int]} -- [Number of walkers to use for MCMC]
        nburns {[int]} -- [Number of iterations to burn for MCMC]
        niter {[type]} -- [Number of iterations to run for MCMC]
        lam {[float]} -- [Second order smoothing factor for polynomail fit]
    
    Keyword Arguments:
        plot_steps {bool} -- [Plot the steps taken by walkers at each steps] (default: {False})
        report {[type]} -- ["report_fit" or "pretty_print"] (default: {None})
        all_theta {bool} -- [Plot model for all values taken by walkers] (default: {False})
        spread {bool} -- [Plot 1-sigma spread in median model] (default: {False})
    """

    print(filename)

    ################
    # Reading data #
    ################

    data = import_data("input_data/" + filename + ".csv")

    ndim = 7    # Number of free parameters in model

    ######################
    # Polynomial Fitting #
    ######################

    label = ["a11", "a12", "a13", "a14", "a10",
             "a21", "a22", "a23", "a24", "a20",
             "a31", "a32", "a33", "a34", "a30",
             "a41", "a42", "a43", "a44", "a40"]
    
    poly_initial = [0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0]  # Initial values for polynomail fit
    
    poly_max = [None]*20    # Maximum and minimum value polynomial parameters can take

    poly_min = [None]*20

    flat = data.flatten()

    poly_leastsq = fitters.least_square_fit(poly, flat, data[:, 2], data[:, 3], label, poly_initial, poly_max, poly_min, report,
                                            min_func=poly_res, args=(lam,))

    #################
    # Model Fitting #
    #################

    filtered = data[:, 2] - poly(poly_leastsq[0], data)     # Extracting signal

    label = ["a_cz", "tau_cz", "psi_cz", "a_he", "c_he", "tau_he", "psi_he"]

    # Changes these to optimmise fit
    initial = [400000, 3000e-6, np.pi, -0.08, 0, 900e-6, np.pi]     # Initial values for least-square fitting
    min_ = [None, 2000e-6, 0, None, None, 500e-6, 0]    # Maximum and minimum value parameters can take
    max_ = [None, 4000e-6, 2*np.pi, None, None, 1500e-6, 2*np.pi]

    results = fitters.least_square_fit(model_func, data[:, 2], filtered, data[:, 3], label, initial, max_, min_, report)
    sampler = fitters.mcmc_fit(model_func, results[0], results[1], nwalkers, ndim, nburns, niter, args=(data[:, 2], filtered, data[:, 3]))
    
    samples = sampler.flatchain
    sample = sampler.chain

    ############
    # Plotting #
    ############

    print("Plotting...")

    # Plots value of walkers at each step
    if plot_steps is True:
        for i in range(ndim):
            plt.title(label[i])
            plt.figure(label[i])
            for j in range(nwalkers):
                plt.plot(sample[j, :, i], color="r", alpha=0.1)
            plt.savefig(label[i] + "_" + filename + ".png")

    theta_max = samples[np.argmax(sampler.flatlnprobability)]  # Most likely parameters from mcmc

    x = np.linspace(data[0, 2], np.max(data[:, 2]), 1000)

    # Least square model data
    leastsq_model = model_func(results[0], x)
    leastsq_convec = model_func(results[0], x, mode="convec")
    leastsq_he = model_func(results[0], x, mode="helium")

    # MCMC most likely fit model
    emcee_data = model_func(theta_max, x)
    emcee_convec = model_func(theta_max, x, mode="convec")
    emcee_he = model_func(theta_max, x, mode="helium")

    # Corner plot
    try:
        fig = corner.corner(samples, show_titles=True, labels=label, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=theta_max)
        fig.savefig("output/Method_C/" + filename + "_triangle.png")
        Corner = True
    except ValueError:
        Corner = False
        pass

    plt.figure(filename)
    plt.ylim(-3, 3)
    plt.xlabel(r'$\nu$ Hz')
    plt.ylabel(r'$\delta^2\nu$ Hz')

    plt.errorbar(data[:, 2], filtered, yerr=data[:, 3], fmt='x')

    """
    # Plotting Least square fit model
    plt.plot(x, leastsq_model)
    # plt.plot(x, leastsq_convec, '--', label="convec", linewidth=1)
    # plt.plot(x, leastsq_he, '-', label="helium", linewidth=1)
    """

    # Plotting model for all values taken by walkers
    if all_theta is True:
        for theta in samples[np.random.randint(len(samples), size=500)]:
            plt.plot(x, model_func(theta, x), color="b", alpha=0.007)
    
    # Plotting MCMC fit model
    plt.plot(x, emcee_data, label='MCMC Highest Likelihood Model')
    # plt.plot(x, emcee_convec,'--', linewidth=1)
    # plt.plot(x, emcee_he, '-', label="helium", linewidth=1)

    # Plotting 1-sigma spread in model  
    if spread is True:
        med_model, spread = fitters.sample_walkers(model_func, x, nwalkers, samples)
        plt.fill_between(x, med_model - spread, med_model + spread, color='grey', alpha=0.5,
                         label=r'$1\sigma$ Posterior Spread')

    plt.legend(loc='upper left')
    plt.savefig("output/Method_C/" + filename + ".png")
    # plt.show(block=False)
    # plt.pause(3)
    plt.close()
    if Corner:
        plt.close(fig)

    samples[:, 2] = np.exp(samples[:, 2])

    # Gives 16th, 50th, 84th percentile values of each parameter
    a_cz, tau_cz, psi_cz, a_he, c_he, tau_he, psi_he = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                            zip(*np.percentile(samples, [16, 50, 84],
                                                            axis=0)))
    final = np.array([a_cz, tau_cz, psi_cz, a_he, c_he, tau_he, psi_he])

    print("Wrinting...")
    with open('output/method_C/' + filename + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['Parameter', 'Most-likely Model', '16th', '50th', '84th'])
        for i in range(ndim):
            filewriter.writerow([label[i], theta_max[i], final[i, 0], final[i, 1], final[i, 2]])

    print("Finished")

if __name__ == "__main__":
    main("data_cyg_a", 1000, 500, 1000, 58, report="report_fit")