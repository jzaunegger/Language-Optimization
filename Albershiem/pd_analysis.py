import numpy as np
import matplotlib.pyplot as plt
import os

# Compute the Probability of Detection
def compute_PD_from_Albershiems(N, SNR, PFA):

    # Compute A
    A = np.log(0.62 / PFA)

    # Compute Z
    numer = SNR + (5 * np.log10(N))
    denom = 6.2 + (4.54 / np.sqrt(N + 0.44))
    Z = numer / denom

    # Compute B 
    numer = (10 ** Z) - A
    denom = 1.7 + (0.12 * A)
    B = numer / denom

    PD = 1 / (1 + (np.e ** -B))
    return PD
PFA_values = [
    (10 ** -3),
    (10 ** -4),
    (10 ** -5),
    (10 ** -6),
    (10 ** -7)
]

for j in PFA_values:

    snr_vals, prob_detection = [], []

    for i in range(-15, 15):
        pd = compute_PD_from_Albershiems(N=10, SNR=i, PFA=j)
        snr_vals.append(i)
        prob_detection.append(pd)


    # Plot the probability of detection over time
    plt.plot(snr_vals, prob_detection)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Probability of Detection')
    plt.title('Prob. Detection over Time for {}'.format(j))
    plt.savefig(os.path.join('graphs', 'Prob-Detection-Chart-{}.png'.format(j)))
    plt.close()