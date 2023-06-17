####################################################################################################
# Coded Random Access 2023 - Homework
# Authors: Marco Skocaj and Giulia Torcolacci
# Date: 2023-06-17
####################################################################################################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set()

############ Fixed parameters ############
# User distributions
Lambda_dict = {'CRDSA2': [1, 2],  # prob, num of retransmissions
               'CRDSA3': [1, 3],  # prob, num of retransmissions
               'CRDSA4': [1, 4],  # prob, num of retransmissions
               'CRDSA5': [1, 5],  # prob, num of retransmissions
               'CRDSA6': [1, 6],  # prob, num of retransmissions
               'IRSA': [np.array([0.5, 0.28, 0.22]), np.array([2, 3, 8])]}  # prob, num of retransmissions


############ Utility functions ############
def compute_lamb(Lamb, p_l, eta):
    return np.sum(np.multiply(Lamb[1], Lamb[0]) * eta * p_l ** (Lamb[1] - 1))

# Initialize density evolution (1st p_l)
def initialize_pl(G, eta):
    return 1 - np.exp(-G / eta)

# Density evolution function
def compute_pl(G, Lamb, p_l_1, eta):
    return 1 - np.exp(-G * compute_lamb(Lamb, p_l_1, eta) / eta)

# Compute pl in a recursive manner, for n iterations
def compute_pl_n(G, Lamb, eta, n):
    p_l = initialize_pl(G, eta)
    for i in range(n):
        p_l = compute_pl(G, Lamb, p_l, eta)
    return p_l

# Numerical evaluation of the theoretical bound
def theoretical_bound(eta_values):
    def evaluate_G(eta):
        G = 0.5  # Initial guess for G
        epsilon = 1e-6  # Desired precision

        while True:
            new_G = 1 - np.exp(-G / eta)
            if abs(new_G - G) < epsilon:
                break
            G = new_G
        return G

    results = [evaluate_G(eta) for eta in eta_values]
    return(results)

# Check if DE converges
def check_convergence(pl):
    return pl <= 10 ** -10


############ Simulation ############
results = []
etas = []

# Test all algorithms
for alg in Lambda_dict.keys():
    # Set inferior and superior bounds for G
    g_inf = .1
    g_sup = .99
    G = (g_inf + g_sup) / 2
    # Compute eta given alg and Lambda
    if alg[:-1] == 'CRDSA':
        Lamb = Lambda_dict[alg]
        eta = 1 / Lamb[1]
    else:
        Lamb = Lambda_dict[alg]
        eta = 1 / np.sum(np.multiply(Lamb[0], Lamb[1]))
    etas.append(eta)

    # Search optimal G*. Stop when G* is found with a precision of 0.005
    count_iterations = 0
    while g_sup - g_inf > .005:
        count_iterations += 1
        # If current G is not a feasible solution, update inferior or superior bound
        G = (g_inf + g_sup) / 2
        if check_convergence(compute_pl_n(G, Lamb, eta, 10 ** 4)):
            g_inf = g_inf + .005
            print("G: {:.3f}".format(G), end='\r')
        else:
            g_sup = g_sup - .005
            print("G: {:.3f}".format(G), end='\r')

    # Further check if current G* converges, if not decrease g_sup by one step
    if not check_convergence(compute_pl_n(G, Lamb, eta, 10 ** 4)):
        g_sup = g_sup - .005
        G = (g_inf + g_sup) / 2

    results.append([G])

    # Print results
    print("Algorithm: ", str(alg), ", Optimal threshold G*: {:.3f}".format(G))
    print("Convergence: ", str(check_convergence(compute_pl_n(G, Lamb, eta, 10 ** 4))))
    print("Iterations to convergence: ", str(count_iterations))
    print("\n")

# Plot results
plt.plot(np.linspace(0.01, 1, 100), theoretical_bound(np.linspace(0.01, 1, 100)), label='Theoretical bound', c='k', linewidth=1)
plt.scatter(etas[0:-1], results[0:-1], label='CRDSA', c='r', edgecolor='k', linewidths=.3)
plt.scatter(etas[-1:], results[-1:], label='IRSA', c='b', edgecolor='k', linewidths=.3)
plt.xlabel(r'$\eta$')
plt.ylabel('G*')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
