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
    except:
        exit(10)


def diff(f_n, f_m, f_p, f_n_error, f_m_error, f_p_error):
    del_f = f_m + f_p - 2*f_n    # Calculating second difference 
    del_f_error = np.sqrt(f_m_error**2 + f_p_error**2 + (2*f_n_error)**2)   # Calculating error
    return del_f, del_f_error


def model_func(theta, f, mode=None):

    ##################
    # Model Function #
    ##################

    a_0, a_1, a_cz, tau_cz, psi_cz, a_he, c_he, tau_he, psi_he = theta
    
    linear = a_0 + a_1*f
    convec = (a_cz/f**2)*np.sin(4*np.pi*tau_cz*f + psi_cz)   # Convection zone signal
    helium = a_he*f*np.exp(-c_he*f**2)*np.sin(4*np.pi*tau_he*f + psi_he)   # He ionisation zone

    if mode == "linear":
        return linear
    elif mode == "convec":
        return convec
    elif mode == "helium":
        return helium
    elif mode is None:
        return linear + convec + helium


def main(filename, nwalkers, nburns, niter, plot_steps=False, report=None, all_theta=False, spread=False):
    """    
    Arguments:
        filename {[String]} -- [CSV file to run the script on, (do not include .csv extention)]
        nwalkers {[int]} -- [Number of walkers to use for MCMC]
        nburns {[int]} -- [Number of iterations to burn for MCMC]
        niter {[type]} -- [Number of iterations to run for MCMC]
    
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

    n = np.min(data[:, 0])      # Radial number
    n_max = np.max(data[:, 0])
    arr = np.empty([0, 5])
    arr_0 = np.empty([0, 5])
    arr_1 = np.empty([0, 5])
    arr_2 = np.empty([0, 5])
    arr_3 = np.empty([0, 5])

    l_max = int(np.max(data[:, 1]))     # Angular number
    l_min = int(np.min(data[:, 1]))

    ndim = 9    # Number of free parameters

    #####################
    # Calculating delf2 #
    #####################

    while n < n_max:
        l = l_min
        n_p = n + 1
        n_m = n - 1
        while l < l_max:
            
            loc = np.where((data[:, 0] == n) & (data[:, 1] == l))   # Finds where in the array n and l are located
            loc_m = np.where((data[:, 0] == n_m) & (data[:, 1] == l))
            loc_p = np.where((data[:, 0] == n_p) & (data[:, 1] == l))

            if len(loc[0]) == 0 or len(loc_m[0]) == 0 or len(loc_p[0]) == 0:
                l = l + 1
                continue
            else:
                f_n = data[loc[0][0], 2]
                f_n_error = data[loc[0][0], 3]

                f_m = data[loc_m[0][0], 2]
                f_m_error = data[loc_m[0][0], 3]

                f_p = data[loc_p[0][0], 2]
                f_p_error = data[loc_p[0][0], 3]
                delf = diff(f_n, f_m, f_p, f_n_error, f_m_error, f_p_error)

            if l == 0:
                arr_0 = np.append(arr_0, np.array([[n, l, f_n, delf[0], delf[1]]]), axis=0)
            if l == 1:
                arr_1 = np.append(arr_1, np.array([[n, l, f_n, delf[0], delf[1]]]), axis=0)
            if l == 2:
                arr_2 = np.append(arr_2, np.array([[n, l, f_n, delf[0], delf[1]]]), axis=0)
            if l == 3:
                arr_3 = np.append(arr_3, np.array([[n, l, f_n, delf[0], delf[1]]]), axis=0)

            l = l + 1
        n = n + 1
    
    arr = np.append(arr, arr_0, axis=0)
    arr = np.append(arr, arr_1, axis=0)
    arr = np.append(arr, arr_2, axis=0)
    arr = np.append(arr, arr_3, axis=0)

    arr = arr[arr[:, 2].argsort()]

    ###########
    # Fitting #
    ###########

    # Changes these to optimmise fit
    label = ["a_0", "a_1", "a_cz", "tau_cz", "psi_cz", "a_he", "c_he", "tau_he", "psi_he"]  
    initials = [0, 0, 0, 3000e-6, np.pi, 0, 0, 900e-6, np.pi]   # Initial values for least-square fitting
    max_ = [None]*9     # Maximum and minimum value parameters can take
    min_ = [None]*9
    results = fitters.least_square_fit(model_func, arr[:, 2], arr[:, 3], arr[:, 4], label, initials, max_, min_, report)
    
    sampler = fitters.mcmc_fit(model_func, results[0], results[1], nwalkers, ndim, nburns, niter, args=(arr[:, 2], arr[:, 3], arr[:, 4]))

    flat_samples = sampler.flatchain
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

    x = np.linspace(arr[0, 2], np.max(arr[:, 2]), 1000)
    
    
    # Least square model data
    leastsq_model = model_func(results[0], x)
    leastsq_convec = model_func(results[0], x, mode="convec")
    leastsq_he = model_func(results[0], x, mode="helium")
    leastsq_linear = model_func(results[0], x, mode="linear")

    theta_max = flat_samples[np.argmax(sampler.flatlnprobability)]  # Most likely parameters from mcmc
    
    print(theta_max)
    # MCMC most likely fit model
    emcee_data = model_func(theta_max, x)
    emcee_convec = model_func(theta_max, x, mode="convec")
    emcee_he = model_func(theta_max, x, mode="helium")
    emcee_linear = model_func(theta_max, x, mode="linear")

    # Corner plot
    try:
        fig = corner.corner(flat_samples, show_titles=True, labels=label, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=theta_max)
        fig.savefig("output/Method_A/" + filename + "_triangle.png")
        Corner = True
    except ValueError:
        Corner = False
        pass

    plt.figure(filename)
    plt.ylim(-3, 3)     # Limit Y values on figure
    plt.xlabel(r'$\nu$ Hz')
    plt.ylabel(r'$\delta^2\nu$ Hz')

    """
    # Plot l modes seperatly
    plt.errorbar(arr_0[:, 2], arr_0[:, 3], yerr=arr_0[:, 4], fmt='o', label = 'l=0')
    plt.errorbar(arr_1[:, 2], arr_1[:, 3], yerr=arr_1[:, 4], fmt='o', label = 'l=1')
    plt.errorbar(arr_2[:, 2], arr_2[:, 3], yerr=arr_2[:, 4], fmt='o', label = 'l=2')
    plt.errorbar(arr_3[:, 2], arr_3[:, 3], yerr=arr_3[:, 4], fmt='o', label = 'l=3')
    """

    plt.errorbar(arr[:, 2], arr[:, 3], yerr=arr[:, 4], fmt='x')

    """
    # Plotting Least square fit model
    plt.plot(x, leastsq_model)
    plt.plot(x, leastsq_convec, '--', label="convec", linewidth=1)
    plt.plot(x, leastsq_he, '-', label="helium", linewidth=1)
    plt.plot(x, leastsq_linear, label="linear", linewidth=1)
    """

    # Plotting model for all values taken by walkers
    if all_theta is True:
        for theta in flat_samples[np.random.randint(len(flat_samples), size=500)]:
            plt.plot(x, model_func(theta, x), color="b", alpha=0.007)

    # Plotting MCMC fit model
    plt.plot(x, emcee_data, label='MCMC Highest Likelihood Model')
    # plt.plot(x, emcee_convec,'--', linewidth=1)
    # plt.plot(x, emcee_he, '-', label="helium", linewidth=1)
    # plt.plot(x, emcee_linear, label="linear", linewidth=1)

    # Plotting 1-sigma spread in model  
    if spread is True:
        med_model, spread = fitters.sample_walkers(model_func, x, nwalkers, flat_samples)
        plt.fill_between(x, med_model - spread, med_model + spread, color='grey', alpha=0.5,
                         label=r'$1\sigma$ Posterior Spread')

    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    plt.legend(loc='upper left')
    plt.savefig("output/Method_A/" + filename + ".png")
    # plt.show(block=False)
    # plt.pause(3)
    plt.close()
    if Corner:
        plt.close(fig)

    # samples[:, 2] = np.exp(samples[:, 2])

    print("Wrinting...")

    
    # Gives 16th, 50th, 84th percentile values of each parameter
    mcmc = np.empty([ndim, 3])
    q = np.empty([ndim, 2])
    for i in range(ndim):
        mcmc[i, :] = np.percentile(flat_samples[:, i], [16, 50, 84])
        q[i, :] = np.diff(mcmc[i, :])


    with open('output/method_A/' + filename + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['Parameter', 'Most-likely Model', '16th', '50th', '84th'])
        for i in range(ndim):
            filewriter.writerow([label[i], theta_max[i], q[i, 0], mcmc[i, 1], q[i, 1]])
    
    print("Finished")

if __name__ == "__main__":
    main("data_cyg_A", 1000, 500, 1000, spread=True, report="pretty_print")
