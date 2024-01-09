import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def main():
    weighted()


## 2.2 General Least Squares Fitting
def damped():
    readin = np.loadtxt("DampedData1.txt", skiprows=1)
    
    # Adding 0.001 Standard Uncertainty to Position
    errors = np.ones(shape = readin.shape[0]) * 0.001
    data = np.insert(readin, 2, errors, axis=1)

    # Model Fitting
    def f(t, A, B, gamma, omega, alpha) -> float:
        return A + B * np.exp(-gamma * t) * np.cos(omega * t + alpha)


    A0 = 0.282
    B0 = 0.028
    gamma0 = 0.25
    omega0 = 21.5
    alpha0 = 0

    p0 = [A0, B0, gamma0, omega0, alpha0]
    names = ["A", "B", "gamma", "omega", "alpha"]

    # Plotting to judge initial guess parameters
    tmodel = np.linspace(0, 5, 1000)
    ystart = f(tmodel, *p0)

    # Optimization
    popt, pcov = curve_fit(f, data[:,0], data[:,1], p0, sigma=data[:,2], absolute_sigma=True)

    # Evaluation 1
    dymin = (data[:,1] - f(data[:,0], *popt)) / data[:,2]
    min_chisq = sum(dymin * dymin)
    dof = len(data[:,0]) - len(popt)

    print(f"Chi-Squared: {min_chisq:.3f}\nDegrees of Freedom: {dof}\nChi-Square per DoF: {min_chisq / dof}\n")

    print("Fitted parameters with 68% C.I.:")
    for i, pmin in enumerate(popt):
        print("%2i %-10s %12f +/- %10f"%(i, names[i], pmin, np.sqrt(pcov[i,i]) * np.sqrt(min_chisq / dof))) # All errors scaled by X^2/dof
    print()
    
    perr = np.sqrt(np.diag(pcov))
    print(f"{perr}\n")

    print("Correlation Matrix")
    print("       ", end=" ")
    for i in range(len(popt)):
        print("%10s"%(names[i]), end=" ")
    print()

    for i in range(len(popt)):
        print("%10s"%(names[i]), end=" ")
        for j in range(i + 1):
            print("%10f"%(pcov[i,j] / np.sqrt(pcov[i,i] * pcov[j,j])), end=" ")
        print()

    # Evaluation 2
    yfit = f(tmodel, *popt)
    
    # Plotting
    fig, ax =plt.subplots(1)
    ax.errorbar(data[:,0], data[:,1], yerr=data[:,2], ecolor="b", fmt="_", color="b", capsize=2, alpha=0.5, label="Data")
    #ax.plot(tmodel, ystart, ".g", alpha=0.1, label="Initial Guess Parameters")
    ax.plot(tmodel, yfit, "-r", label="$Best Fit: \chi^2$/ dof=1.032")
    ax.set_title("The Result of The Model Fit to the Data of Oscillator 1")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Position y (m)")
    ax.legend()
    fig.show()

## 2.3 Fitting to Models Linear in Parameters Unweighted Linear Least-Squares
def unweighted():
    data = np.loadtxt("LinearNoErrors.txt", skiprows=1)

    # linear function definition
    def f(x, m, c) -> float : 
        return m*x + c

    # Guess parameters
    m0 = 7/10
    c0 = 1.5
    p0 = [m0, c0]

    u_y = np.ones(np.shape(data)[0]) * 1e-22  # very small 'y' uncertainty

    #optimization
    popt, pcov = curve_fit(f, data[:, 0], data[:, 1], p0, sigma=None, absolute_sigma=True)
    print(popt)
    print(pcov)
    
    # Plots 
    t = np.linspace(0, 13, 50) 
    g = f(t, *popt)
    
    m, c = popt
    fig, ax = plt.subplots(1)
    ax.errorbar(data[:,0], data[:,1], fmt='s', yerr=u_y, color="red", ecolor="black", label="PHY2004W Data")
    ax.plot(t, g, color="b", label=f"Best Fit\nm={popt[0]:.2f}\nc={popt[1]:.2f}")
    ax.set_title("Unweighted Linear Regression")
    ax.legend()
    fig.show()


def weighted():
    data = np.loadtxt("LinearwithErrors.txt", skiprows=1)
    def f(x, m, c) -> float: 
        return m*x + c
    
    # Optimization
    m0 = 0.59
    c0 = 1
    p0 = [m0, c0]
    names = ["m", "c"]
    popt, pcov = curve_fit(f, data[:,0], data[:,1], p0, sigma=data[:,2] ,absolute_sigma=True)

    # best fit
    t = np.linspace(0, 13, 50)
    g = f(t, *popt)

    # Evaluation 1
    dymin = (data[:,1] - f(data[:,0], *popt)) / data[:,2]
    min_chisq = sum(dymin * dymin)
    dof = len(data[:,0]) - len(popt)

    print(f"Chi-Squared: {min_chisq:.3f}\nDegrees of Freedom: {dof}\nChi-Square per DoF: {min_chisq / dof}\n")

    print("Fitted parameters with 68% C.I.:")
    for i, pmin in enumerate(popt):
        print("%2i %-10s %12f +/- %10f"%(i, names[i], pmin, np.sqrt(pcov[i,i]) * np.sqrt(min_chisq / dof))) # All errors scaled by X^2/dof
    print()
    
    perr = np.sqrt(np.diag(pcov))
    print(f"{perr}\n")

    print("Correlation Matrix")
    print("       ", end=" ")
    for i in range(len(popt)):
        print("%10s"%(names[i]), end=" ")
    print()

    for i in range(len(popt)):
        print("%10s"%(names[i]), end=" ")
        for j in range(i + 1):
            print("%10f"%(pcov[i,j] / np.sqrt(pcov[i,i] * pcov[j,j])), end=" ")
        print()

    # Contours
    Npts = 10000
    mscan = np.zeros(Npts)
    cscan = np.zeros(Npts)
    chi_dof = np.zeros(Npts)

    i = 0

    for mpar in np.linspace(0.5, 0.7, 100, True):
        for cpar in np.linspace(0.5, 1.7, 100, True):
            mscan[i] = mpar
            cscan[i] = cpar
            dymin1 = (data[:,1] - f(data[:,0], mpar, cpar)) / data[:,2]
            chi_dof[i] = sum(dymin1*dymin1) / dof
            i += 1
    ncols = 100
    
    # Plots
    fig1, ax1 = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    ax1.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt="s", color="red", ecolor="black", label="PHY2004W Data")
    ax1.plot(t, g, color="blue", label=f"Best Fit: $\chi^2$/dof = 0.61")
    ax1.set_title("weighted Linear Regression")
    im = ax2.tricontourf(mscan, cscan, chi_dof, ncols)
    ax2.set_title("$\chi^2$/ dof Goodness of Fit vs Parameter Space Contour Plot")
    ax2.set_xlabel("parameter1: m")
    ax2.set_ylabel("parameter2: c")
    fig2.colorbar(im, ax=ax2, label="$\chi^2_i$/ dof")
    ax1.legend()
    fig1.show()
    fig2.show()


def visualization():
    data = np.loadtxt("LinearwithErrors.txt", skiprows=1)
    xdata = data[:,0]
    ydata = data[:,1]
    udata = data[:,2]

## 2.4 Visualizing Uncertainties


if __name__ == "__main__":
    main()























    
