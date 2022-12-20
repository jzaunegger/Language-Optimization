import matplotlib.pyplot as plt


def compute_peak_power(power_avg, dwell_time, bandwidth, num_pulses):
    return power_avg * dwell_time * (bandwidth / num_pulses)


def compute_avg_power(peak_power, dwell_time, bandwidth, num_pulses):
    return (num_pulses * peak_power) / (bandwidth * dwell_time)

power_avg = 500             # Watts
num_pulses = 10             # Unitless
PRF = 10_000                # Hz
central_dwell_time = num_pulses / PRF

mid_bandwidth = 450 * (10 ** 6)

bandwidth = [1, 900]

# Vary the bandwidth
output_pt, band_vals = [], []
for i in range(bandwidth[0], bandwidth[1]+1):
    current_band = i * (10 ** 6)
    current_pt = compute_peak_power(power_avg, central_dwell_time, current_band, num_pulses)

    band_vals.append(i)
    output_pt.append(current_pt)


plot_1_title = "Pt as a f(B), Pa:{}, NP: {}, Td:{}".format(power_avg, num_pulses, central_dwell_time)


# Vary the number of pulses, this effects both the number of pulses and the dwell time
num_pulse_arr, output_pt_2 = [], []
for i in range(1, 100):
    dwell_time = i / PRF

    current_pt = compute_peak_power(power_avg, dwell_time, mid_bandwidth, i)

    num_pulse_arr.append(i)
    output_pt_2.append(current_pt)

# Vary the peak power from 1 to 900 watts.
peak_powers, avg_powers = [], []
for i in range(1, 900):
    peak_powers.append(i)
    avg_powers.append(compute_avg_power(i, central_dwell_time, 450*(10**6), 10))


plt.subplot(2, 1, 1)
plt.plot(band_vals, output_pt)
plt.xlabel('Bandwidth (MHz)')
plt.ylabel('Peak Power (W)')
plt.title(plot_1_title)

plt.subplot(2, 1, 2)
plt.plot(num_pulse_arr, output_pt_2)
plt.xlabel('Number of Pulses')
plt.ylabel('Peak Power (W)')

plt.show()
plt.close()

plt.plot(peak_powers, avg_powers)
plt.xlabel("Peak Power (W)")
plt.ylabel("Average Power")
plt.show()