# coding: utf-8

# In[70]:


import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# exponential function
def f(x):
    return np.exp(x**3)

# trapezoidal rule
def trapz(f, a, b, n):
    # n evenly spaced
    g = 0
    if b > a:
        h = (b-a)/float(n)
    else:
        h = (a-b)/float(n)
    for i in range (0, n):
        k = 0.5 * h * ( f(a + i*h) + f(a + (i+1)*h) )
        g = g + k
    return g

# Gaussian quadrature
def gauss(f,a,b):
    return integrate.quad(f,a,b)


# main function
if __name__ == "__main__":
    # if the user includes the flag -h or --help print the options
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s -N [number of intervals for integration] -a [lower bound] -b [upper bound]" % sys.argv[0])
        print
        sys.exit(1)
    
    # define maximum interval (default)
    a = -1
    b = 1
    N = 40

    if '-N' in sys.argv:
        p = sys.argv.index('-N')
        N = int(sys.argv[p + 1])
        if N > 0:
            Nmax = N
               
    if '-a' in sys.argv:
        p = sys.argv.index('-a')
        a0 = int(sys.argv[p+1])
        if a0 > -100:
            a = a0
    
    if '-b' in sys.argv:
        p = sys.argv.index('-b')
        b0 = int(sys.argv[p+1])
        if b0 > a:
            b = b0
    
    
    # do the integration for every n and plot
    tz = []
    tz_err = []
    gs = []
    gs_err = []
    true = []
    for n in range(1,N):
        x = np.linspace(a,b,n)
    
        # true value of the integral
        true_v = f(b) - f(a)
        # integration using trapezoidal rule
        tz_v = trapz(f, a, b, n)
        # calculate the error in the trapez approximation
        t_err = tz_v - true_v
        # integration using Gaussian quadrature
        gs_v, g_err = gauss(f, a, b)
        
        
        #append into list for plotting
        true.append(true_v)
        tz.append(tz_v)
        gs.append(gs_v)
        gs_err.append(g_err)
        tz_err.append(t_err)
        
    tz = np.array(tz)
    gs = np.array(gs)
    gs_err = np.array(gs_err)
    tz_err = np.array(tz_err)
    # define x -axis as number of integration 0~n
    xn = np.arange(1,n+1)
    plt.figure(figsize=[12,7])

    # Plot +/- 1 sigma filled in uncertainty range
    plt.plot(xn, tz_err, 'r^', label='Using Trapezoidal rule Method ')
    plt.plot(xn, gs_err, 'g^', label='Using Gaussian quadrature Method ')
    plt.plot(xn, tz-gs, 'bo-', label='Difference of two integration methods ')
    plt.ylabel("Difference of True value and numerical integrals (True value - numerical estimation)")
    plt.xlabel("Number of intervals for integration (N)")
    plt.legend()
    plt.grid()
    plt.savefig("DiffvsN_integration.pdf")
    # Plot sigma filled in uncertainty range
    #plt.figure(figsize=[12,7])
    #plt.plot(xn, tz, 'r^', label='Using Trapezoidal rule Method ')
    #plt.plot(xn, gs, 'g^', label='Using Gaussian quadrature Method ')
    #plt.plot(xn, true, 'ko-', label='True integration Value ')
    #plt.fill_between(xn, tz-tz_err, tz+tz_err, color='grey',alpha=0.5,label=r'Trapezoidal rule error Spread')
    #plt.fill_between(xn, gs-gs_err, gs+gs_err, color='yellow',alpha=0.5,label=r'Gaussian quadrature error Spread')
    #plt.ylabel("Estimate of numerical integrals")
    #plt.xlabel("Number of intervals for integration (N)")
    #plt.legend()
    #plt.grid()
    #plt.savefig("IntegralvsN.pdf")
    plt.show()
