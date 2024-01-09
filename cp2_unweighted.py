import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def main():
    unweighted()


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
    names = ["m", "c"]
    u_y = np.ones(data.shape[0])

    #optimization
    popt, pcov = curve_fit(f, data[:, 0], data[:, 1], p0, sigma=None)
    print(popt)
    
    
    # Evaluation 1
    dymin = (data[:,1] - f(data[:,0], *popt)) / u_y
    min_chisq = sum(dymin * dymin)
    dof = len(data[:,0]) - len(popt)

    print(np.sqrt(np.diag(pcov)) * np.sqrt(min_chisq / dof))
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
            dymin1 = (data[:,1] - f(data[:,0], mpar, cpar)) / u_y
            chi_dof[i] = sum(dymin1*dymin1) / dof
            i += 1
    ncols = 100

    # Plots 
    t = np.linspace(0, 13, 50) 
    g = f(t, *popt)
    
    fig, ax = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    ax.errorbar(data[:,0], data[:,1], fmt='s', color="red", ecolor="black", label="PHY2004W Data")
    ax.plot(t, g, color="b", label=f"Best Fit: $\chi^2$/dof = 0.1 ")
    ax.set_title("Unweighted Linear Regression on 'LinearNoErrors.txt' Data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    im = ax2.tricontourf(mscan, cscan, chi_dof, ncols)
    ax2.set_title("$\chi^2$/ dof Goodness of Fit vs Parameter Space Contour Plot")
    ax2.set_xlabel("parameter1: m")
    ax2.set_ylabel("parameter2: c")
    fig2.colorbar(im, ax=ax2, label="$\chi^2_i$/ dof")
    ax.legend()
    fig.savefig("unweighted.png")
    fig2.savefig("chi_contour_unweighted.png")
    fig.show()
    fig2.show()


if __name__ == "__main__":
    main()