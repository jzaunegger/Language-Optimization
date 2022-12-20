import math, os
import matplotlib.pyplot as plt
import numpy as np

# Function to convert the prf and num pulses to a dwell time
# ====================================================================
def prf_to_dwell_time(prf, num_pulses):
    return num_pulses / prf

# Function to compute the peak power from the average power, dwell 
# time, bandwidth, and number of pulses.
# ====================================================================
def compute_peak_power(power_avg, dwell_time, bandwidth, num_pulses):
    return power_avg * dwell_time * (bandwidth / num_pulses)

# Function to compute the average power from the peak power, dwell 
# time, bandwidth, and number of pulses
# ====================================================================
def compute_avg_power(peak_power, dwell_time, bandwidth, num_pulses):
    return (num_pulses * peak_power) / (bandwidth * dwell_time)

# Function to convert a dB value to a linear value
# ====================================================================
def decibelToLinear(db_value):
    return 10 ** (db_value / 10)

# Function to convert a linear value to a dB value
# ====================================================================
def linearToDecibel(linear_value):
    return 10 * math.log10(linear_value)

# Function to compute the range resolution from bandwidth (hz).
# Returns a value in meters.
# ====================================================================
def compute_range_resolution(bandwidth):
    return speed_of_light / (2 * bandwidth)

# Function to convert the frequency (hz) to wavelength (m).
# ====================================================================
def frequency_to_wavelength(frequency):
    return speed_of_light / frequency

# Compute the Probability of Detection from the number of samples,
# current snr, and prob of false alarm.
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

# Function to convert the signal to noise ratio. 
# ====================================================================
def compute_snr(peak_power, gain, wavelength, rcs, num_pulses, range, noise, bandwidth, loss):

    four_pi_cubed = ((4 * math.pi) ** 3)
    numer = peak_power * (gain**2) * (wavelength**2) * rcs * num_pulses
    denom = four_pi_cubed * (range ** 4) * boltzmanns_constant * standard_temp * noise * bandwidth * loss

    result_linear = numer / denom
    result_db = linearToDecibel(result_linear)

    return result_linear, result_db

# Function to plot the bandwidth vs the snr
# ====================================================================
def plot_b_vs_snr_chart(peak_power, gain, wavelength, rcs, num_pulses, range, noise, bandwidth_range, loss_val, title, path):

    snr_vals, band_vals = [], []
    for val in bandwidth_range:
        current_b = val * (10 ** 6)
        current_snr = compute_snr(peak_power, gain, wavelength, rcs, num_pulses, range, noise, current_b, loss_val)

        band_vals.append(val)
        snr_vals.append(current_snr[1])

    plt.plot(band_vals, snr_vals)
    plt.xlabel("Bandwidth (MHz)")
    plt.ylabel("SNR (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
# Function to plot the bandwidth vs the range resolution
# ====================================================================
def plot_range_res_chart(bandwidth_range, title, path):

    bandwidth_values, range_res_vals = [], []
    for band_val in bandwidth_range:
        band_hz = band_val * (10 ** 6)

        range_res_vals.append(compute_range_resolution(band_hz))
        bandwidth_values.append(band_val)

    plt.plot(bandwidth_values, range_res_vals)
    plt.xlabel("Bandwidth (MHz)")
    plt.ylabel("Range Resolution (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Function to plot the peak power vs the average power
# ====================================================================
def plot_power_a_v_power_t(power_span,  dwell_time, bandwidth, num_pulses, title, path):

    peak_powers, avg_powers = [], []
    for i in range(power_span[0], power_span[1]):
        avg_power = compute_avg_power(i, dwell_time, bandwidth, num_pulses)

        avg_powers.append(avg_power*1000)
        peak_powers.append(i)

    plt.plot(peak_powers, avg_powers)
    plt.xlabel("Peak Trans. Power (W)")
    plt.ylabel("Avg Power (mW)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Function to plot the bandwidth vs the average power
# ====================================================================
def plot_power_a_v_bandwidth(peak_power, dwell_time, bandwidth_range, num_pulses, title, path):

    avg_powers, bandwith_vals = [], []
    for i in range(bandwidth_range[0], bandwidth_range[1]):
        current_band = i * (10 ** 6)
        avg_power = compute_avg_power(peak_power, dwell_time, current_band, num_pulses )

        avg_powers.append(avg_power*1000)
        bandwith_vals.append(i)
        
    plt.plot(bandwith_vals, avg_powers)
    plt.xlabel("Bandwidth (MHz)")
    plt.ylabel("Avg Power (mW)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Function to plot the dwell time vs the average power
# ====================================================================
def plot_power_a_v_dwell_time(peak_power, dwell_range, bandwidth, num_pulses, title, path):
    
    dwells, avg_powers = [], []
    for i in range(dwell_range[0], dwell_range[1], -1):

        current_dwell = (i / 100)
        
        avg_power = compute_avg_power(peak_power, current_dwell, bandwidth, num_pulses)
        avg_powers.append(avg_power)
        dwells.append(current_dwell)

    plt.plot(dwells, avg_powers)
    plt.xlabel("Dwell Time (S)")
    plt.ylabel("Avg Power (mW)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# Function to plot the prf vs the dwell time
# ====================================================================

def plot_prf_V_dwell_time(prf_range, num_pulses, title, path):
    prf_vals, dwell_times = [], []
    for i in range(prf_range[0], prf_range[1], prf_range[2]):
        current_prf = i

        dwell_time = prf_to_dwell_time(current_prf, num_pulses)
        prf_vals.append(i)
        dwell_times.append(dwell_time)

    plt.plot(prf_vals, dwell_times)
    plt.xlabel("PRF (Hz)")
    plt.ylabel("Dwell Time (S)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Initalize Global Variables
boltzmanns_constant = 1.38 * math.pow(10, -23)  # Joules per Kelvin
standard_temp = 290                             # Kelvin
speed_of_light = 3 * (10 ** 8)

if __name__ == '__main__':

    # Set Configuration Parameters
    frequency = 1 * (10**9)                             # Given in Hertz
    wavelength = frequency_to_wavelength(frequency)     # Given in meters
    gain = 1.258                                        # Unitless
    rcs = 1                                             # Given in meters squared
    target_range = 120.15                                      # Given in meters
    num_pulses = 10                                     # Unitless
    noise = 1.778
    losses = [0.5912, 0.0732, 6.0301, 5.1841, 0.0732]
    prf = 10_000                                        # Hz
    dwell_time = prf_to_dwell_time(prf, num_pulses)

    bandwidth = 600 * (10 ** 6)
    peak_power = 500
    pfa = 10 ** -4

    output_folder = os.path.join('./optimization_3', 'details')

    bandwidth_range = [1, 900] # In MHz

    # Plot Range Resolution Charts
    plot_range_res_chart([1, 900], 'B vs RR (Full)', os.path.join(output_folder, 'BvRR', 'BvRR-Full.png'))
    plot_range_res_chart([300, 600], 'B vs RR (Middle)', os.path.join(output_folder, 'BvRR', 'BvRR-Mid.png'))
    plot_range_res_chart([1, 450], 'B vs RR (First Half)', os.path.join(output_folder, 'BvRR', 'BvRR-FH.png'))
    plot_range_res_chart([450, 900], 'B vs RR (Last Half)', os.path.join(output_folder, 'BvRR', 'BvRR-LH.png'))

    # Plot SNR Charts
    plot_b_vs_snr_chart(peak_power, gain, wavelength, rcs, num_pulses, target_range, noise, [1, 900], losses[0], 'BvSNR (Full)', os.path.join(output_folder, 'BvSNR', 'BvSNR-Full.png'))
    plot_b_vs_snr_chart(peak_power, gain, wavelength, rcs, num_pulses, target_range, noise, [300, 600], losses[0], 'BvSNR (Mid)', os.path.join(output_folder, 'BvSNR', 'BvSNR-Mid.png'))
    plot_b_vs_snr_chart(peak_power, gain, wavelength, rcs, num_pulses, target_range, noise, [1, 450], losses[0], 'BvSNR (First Half)', os.path.join(output_folder, 'BvSNR', 'BvSNR-FH.png'))
    plot_b_vs_snr_chart(peak_power, gain, wavelength, rcs, num_pulses, target_range, noise, [450, 900], losses[0], 'BvSNR (Last Half)', os.path.join(output_folder, 'BvSNR', 'BvSNR-LH.png'))

    # Plot the Avg Power as a function of dwell time
    plot_power_a_v_dwell_time(peak_power, [1000, 1], bandwidth, num_pulses, 'PavTd (Full)', os.path.join(output_folder, 'PavTd', 'PavTd-Full.png'))
    plot_power_a_v_dwell_time(peak_power, [750, 250], bandwidth, num_pulses, 'PavTd (Mid)', os.path.join(output_folder, 'PavTd', 'PavTd-Mid.png'))
    plot_power_a_v_dwell_time(peak_power, [500, 1], bandwidth, num_pulses, 'PavTd (FH)', os.path.join(output_folder, 'PavTd', 'PavTd-FH.png'))
    plot_power_a_v_dwell_time(peak_power, [1000, 500], bandwidth, num_pulses, 'PavTd (LH)', os.path.join(output_folder, 'PavTd', 'PavTd-LH.png'))

    # Plot the Avg Power as a function of bandwidth
    plot_power_a_v_bandwidth(peak_power, dwell_time, [1, 900], num_pulses, 'PavB (Full)', os.path.join(output_folder, 'PavB', 'PavB-Full.png'))
    plot_power_a_v_bandwidth(peak_power, dwell_time, [300, 600], num_pulses, 'PavB (Mid)', os.path.join(output_folder, 'PavB', 'PavB-Mid.png'))
    plot_power_a_v_bandwidth(peak_power, dwell_time, [1, 450], num_pulses, 'PavB (FH)', os.path.join(output_folder, 'PavB', 'PavB-FH.png'))
    plot_power_a_v_bandwidth(peak_power, dwell_time, [450, 900], num_pulses, 'PavB (LH)', os.path.join(output_folder, 'PavB', 'PavB-LH.png'))


    # Plot the avg power as a function of peak trans power
    plot_power_a_v_power_t([1, 1000], dwell_time, bandwidth, num_pulses, 'PavPt (Full)', os.path.join(output_folder, 'PavPt', 'PavPt-Full.png'))
    plot_power_a_v_power_t([250, 750], dwell_time, bandwidth, num_pulses, 'PavPt (Mid)', os.path.join(output_folder, 'PavPt', 'PavPt-Mid.png'))
    plot_power_a_v_power_t([1, 500], dwell_time, bandwidth, num_pulses, 'PavPt (FH)', os.path.join(output_folder, 'PavPt', 'PavPt-FH.png'))
    plot_power_a_v_power_t([500, 1000], dwell_time, bandwidth, num_pulses, 'PavPt (LH)', os.path.join(output_folder, 'PavPt', 'PavPt-LH.png'))

    # Plot the dwell time as a function of the prf
    plot_prf_V_dwell_time([1, 10000, 100], num_pulses, 'PRFvTd (Full)', os.path.join(output_folder, 'PRFvTd', 'PRFvTd-Full.png'))
    plot_prf_V_dwell_time([2500, 7500, 100], num_pulses, 'PRFvTd (Mid)', os.path.join(output_folder, 'PRFvTd', 'PRFvTd-Mid.png'))
    plot_prf_V_dwell_time([1, 5000, 100], num_pulses, 'PRFvTd (FH)', os.path.join(output_folder, 'PRFvTd', 'PRFvTd-FH.png'))
    plot_prf_V_dwell_time([5000, 10000, 100], num_pulses, 'PRFvTd (LH)', os.path.join(output_folder, 'PRFvTd', 'PRFvTd-LH.png'))

    def compute_values(current_bandwidth, current_peak_transmission_power):

        for loss_val in losses:
            current_snr = compute_snr(current_peak_transmission_power, gain, wavelength, rcs, num_pulses, target_range, noise, current_bandwidth, loss_val)[1]
            range_res = compute_range_resolution(current_bandwidth)
            avg_power = compute_avg_power(current_peak_transmission_power, dwell_time, bandwidth, num_pulses)
            pd = compute_PD_from_Albershiems(num_pulses, current_snr, pfa)
            print('| Loss: {} | SNR: {:.4f} (dB) | Range Res: {} (m) | Avg Power: {:.4} (mW) | Prob Detection: {:.4f} |'.format(loss_val, current_snr, range_res, avg_power*1000, pd))

    compute_values(800*(10**6), 900)
    print(" ")

    compute_values(800*(10**6), 400)
    print(" ")
