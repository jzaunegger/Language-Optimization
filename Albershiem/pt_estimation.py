import matplotlib.pyplot as plt
import os


PRF = 1000
NP = 10
PAVG = 50

print(NP / PRF)

bandwidths, peak_power = [], []
for i in range(1, 900):
    bandwidth = i  * (10 ** 6)

    pt = PAVG * (NP / PRF) * (bandwidth / NP)

    bandwidths.append(i)
    peak_power.append(pt)


    plt.plot(bandwidths, peak_power)
    plt.xlabel('Bandwidth (MHz)')
    plt.ylabel('Peak Transmission Power (W)')
    plt.title('Peak Transmission Power over Time')
    plt.savefig(os.path.join('graphs', 'Peak-Power-Chart.png'))
    plt.close()