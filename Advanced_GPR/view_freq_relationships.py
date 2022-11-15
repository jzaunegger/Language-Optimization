'''
    This script is intended to illustrate the relationship between frequency, sampling frequency, and sampling interval.
'''

import matplotlib.pyplot as plt


freq, sample_freqs, sample_intervals = [], [], []
for i in range(10_000_000, 2_000_000_000, 1_000):

    sample_freq = i * 6
    sample_interval = 1 / sample_freq

    freq.append(i / (10 ** 6))
    sample_freqs.append(sample_freq / (10 ** 9))
    sample_intervals.append(sample_interval / (10 ** -9))



figure, axis = plt.subplots(1, 2)

axis[0].plot(freq, sample_freqs)
axis[0].set_xlabel('Center Frequency (Hz)')
axis[0].set_ylabel('Sampling Frequency (GHz)')
axis[0].set_title('Center vs. Sampling Frequency')

axis[1].plot(freq, sample_intervals)
axis[1].set_xlabel('Center Frequency (MHz)')
axis[1].set_ylabel('Sampling Interval (ns)')
axis[1].set_title('Center Frequency vs. Sampling Interval')
plt.show()