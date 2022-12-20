import numpy as np
import matplotlib.pyplot as plt

def compute_SNR_from_Albershiems(N, prob_detection, prob_false_alarm):

    # Compute A
    A = np.log(0.62 / prob_false_alarm)

    # Compute B
    B = prob_detection / (1 - prob_detection)

    print("A: {}".format(A))
    print("B: {}".format(B))


    part_1 = -5 * np.log10(N)
    part_2 = 6.2 + (4.54 / np.sqrt(N + 0.44))
    part_3 = np.log10(A + (0.12 * A * B) + (1.7 * B))
    SNR = part_1 + (part_2 * part_3)


    print("P1: {}".format(part_1))
    print("P2: {}".format(part_2))
    print("P3: {}".format(part_3))



    return SNR


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



if __name__ == '__main__':

    prob_of_detection = []
    snr_values = []

    N_Values = [5, 10, 15, 20, 2000]
    for j in N_Values:
        prob_of_detection = []
        snr_values = []

        for i in range(-15, 15, 1):

            pd = compute_PD_from_Albershiems(N=j, SNR=i, PFA=(10 ** -4))
            
            prob_of_detection.append(pd)
            snr_values.append(i)


        plt.plot(snr_values, prob_of_detection, label='N={}'.format(j))
    plt.xlabel('SNR (dB)')
    plt.ylabel('Probability of Detection')
    plt.title('Prob. of Detection for PFA=10^-4')
    plt.legend()
    plt.show()