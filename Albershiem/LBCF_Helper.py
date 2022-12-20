import math
import numpy as np

boltzmanns_constant = 1.38 * math.pow(10, -23)  # Joules per Kelvin
standard_temp = 290                             # Kelvin
speed_of_light = 3 * (10 ** 8)                  # Meters per second

# Statistical Functions
# ==============================================================================================


# Plotting Functions
# ==============================================================================================



# Radar Computation Functions
# ==============================================================================================

'''
    Function to compute the wavelength from a frequency.

    Inputs:
        frequency - The frequency in hertz.

    Returns the wavelength in meters.
'''
def frequency_to_wavelength(frequency):
    return speed_of_light / frequency

'''
    Function to compute the frequency from a wavelength.

    Inputs:
        wavelength - The wavelength in meters.

    Returns the frequency in hz.
'''
def wavelength_to_frequency(wavelength):
    return speed_of_light / wavelength

'''
    Function to compute the linear equivalent of a 
    decibel value.

    Inputs:
        db_value - the value in decibel units
    
    Returns the value in linear format.
'''
def decibel_to_linear(db_value):
    return 10 ** (db_value / 10)

'''
    Function to compute the decibel equivalent of a 
    linear value.

    Inputs:
        linear_value - the value in linear units
    
    Returns the value in decibel format.
'''
def linear_to_decibel(linear_value):
    return 10 * math.log10(linear_value)

'''
    Function to compute the the dwell time from the prf
    and the number of pulses.

    Inputs:
       prf        - the pulse repetition frequency (hz)
       num_pulses - the number of pulses (unitless)

    Returns the dwell time in seconds.
'''
def prf_to_dwell_time(prf, num_pulses):
    return num_pulses / prf

'''
    Function to compute the average power from the peak power, dwell 
    time, bandwidth, and number of pulses.

    Inputs:
       peak_power - the peak transmission power in watts
       dwell_time - the dwell time in seconds
       bandwidth  - the bandwidth in hz
       num_pulses - the number of pulses (unitless)

    Returns the average power in watts.
'''
def compute_avg_power(peak_power, dwell_time, bandwidth, num_pulses):
    return (num_pulses * peak_power) / (bandwidth * dwell_time)


'''
    Compute the range resolution from the bandwidth.

    Inputs:
       bandwidth  - the bandwidth in hz

    Returns the the range resolution in meters
'''
def compute_range_resolution(bandwidth):
    speed_of_light = 3 * (10 ** 8)

    return speed_of_light / (2 * bandwidth)

'''
    Compute the probability of detection using the number
    of pulses, the signal to noise ratio, and the probability
    of false alarm. This is a implementation of Albersheims 
    formula.

    Inputs:
        num_pulses  - the number of pulses (unitless)
        snr         - the signal to noise ratio (db)
        pfa         - the probability of false alarm (unitless)

    Returns the the probability of detection
'''
def compute_PD_from_Albershiems(num_pulses, snr, pfa):

    # Compute A
    A = np.log(0.62 / pfa)

    # Compute Z
    numer = snr + (5 * np.log10(num_pulses))
    denom = 6.2 + (4.54 / np.sqrt(num_pulses + 0.44))
    Z = numer / denom

    # Compute B 
    numer = (10 ** Z) - A
    denom = 1.7 + (0.12 * A)
    B = numer / denom


    denom3 = (1 + (np.e ** -B))
    if denom3 == 0:
        return 0

    else:
        return 1 / denom3

'''
    Compute the attenuation coefficient of a given material from 
    the conductivity, permittivity, angular frequency, and material 
    thickness.

    Inputs:
        conductivity        - 
        permittivity        - 
        angular_frequency   - 
        distance            -

    Returns the the attenuation coefficient in decibels.
'''
def compute_attenuation_coefficient(conductivity, permittivity, angular_frequency, distance):

    cond_seimens = conductivity * math.pow(10, -3)
    check_value = cond_seimens / (angular_frequency * permittivity)
    
    # Check if a good dielectric
    if check_value < 0.000000001:

        part_a = cond_seimens / 2
        part_b = (120 * math.pi) / math.sqrt(permittivity)
        attenuation = part_a * part_b

    elif check_value > 1000000000:
        numer = angular_frequency * (120 * math.pi) * cond_seimens
        attenuation = math.sqrt(numer / 2)
    
    else:
        part_a = angular_frequency * math.sqrt( (120 * math.pi) * permittivity)
        part_b = math.pow( math.sqrt(1 + check_value)-1, 0.5)
        attenuation = part_a * part_b

    result = math.pow(math.e, -4 * attenuation * distance)

    result_db = 10 * math.log10(result)
    return result_db

'''
    Compute the total system loss from a dict of losses.

    Inputs:
        losses        - The losses dict. Is assumed to have the following keys:
                            transmitter_loss
                            reciever_loss
                            signal_processing_loss
                            ground_loss

    Returns the the total system loss in decibels
'''
def compute_total_loss(losses):
    total_loss_db = losses['transmitter_loss'] + losses['reciever_loss'] + losses['signal_processing_loss'] + losses['ground_loss']
    total_loss_linear = decibel_to_linear(total_loss_db)
    return total_loss_db, total_loss_linear



'''
    Compute the signal to noise ratio.

    Inputs:
        peak_power  - The peak transmision power (W)
        gain        - The gain of the antenna (unitless)
        wavelength  - The wavelength (m)
        rcs         - The radar cross section in (m^2)
        num_pulses  - The number of pulses (unitless)
        range       - The total range to the target (m)
        noise       - The system noise figure ()
        bandwidth   - The bandwidth (hz)
        loss        - The total system loss ()

    Returns the snr in linear and decibel units.
'''
def compute_snr(peak_power, gain, wavelength, rcs, num_pulses, range, noise, bandwidth, loss):

    #print(peak_power)
    #print(gain)
    #print(wavelength)
    #print(rcs)
    #print(num_pulses)
    #print(range)
    #print(noise)
    #print(bandwidth)
    #print(loss)

    four_pi_cubed = math.pow((4 * math.pi), 3)
    numer = peak_power * (gain**2) * (wavelength**2) * rcs * num_pulses
    denom = four_pi_cubed * (range ** 4) * boltzmanns_constant * standard_temp * noise * bandwidth * loss

    #print(numer, denom)

    result_linear = numer / denom
    result_db = linear_to_decibel(result_linear)

    #print(result_linear, result_db)
    #input('Conitnue?')

    return result_linear, result_db