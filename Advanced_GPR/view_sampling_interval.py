import math, os, sys
import matplotlib.pyplot as plt

def compute_sampling_interval(frequency):
    return 1 / (6 * frequency)


if __name__ == "__main__":

    
    output_path = os.path.join('./images')

    frequency_labels, sampling_interval = [], []

        # check the output path exists
    if os.path.exists(output_path) == False: os.mkdir(output_path)

    # Iterate over freequency
    for i in range(10_000_000, 1_000_000_000, 1_000_000):
        current_sample_rate = compute_sampling_interval(i)

        frequency_labels.append(i/1_000_000)
        sampling_interval.append(current_sample_rate * (10**6))

    plt.plot(frequency_labels, sampling_interval)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Sampling Rate (µs)')
    plt.title('Frequency vs Sampling Rate')
    plt.savefig(os.path.join(output_path, 'Sampling-Interval-Chart.png'))
    plt.close()