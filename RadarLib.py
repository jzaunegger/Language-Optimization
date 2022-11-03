# Import Dependencies
import math

from torch import result_type

# Define Physical Constants
# ====================================================================
boltzmanns_constant = 1.38 * math.pow(10, -23)  # Joules per Kelvin
standard_temp = 290                             # Kelvin
speed_of_light = 2.99792458 * math.pow(10, 8)   # Meters per Second


# Function to compute the maximum detectable range
# ====================================================================
def compute_detectable_range(parameters):

    numer = parameters['peak_transmisison_power'] * (parameters['gain'] ** 2) * (parameters['wavelength'] ** 2) * parameters['radar_cross_section'] * parameters['number_of_pulses']
    denom = math.pow((4 * math.pi), 3) * parameters['signal_to_noise'] * boltzmanns_constant * standard_temp * parameters['noise_figure'] * parameters['bandwidth'] * parameters['total_loss']
    result = math.pow(numer / denom, 0.25)
    return result


# Function to compute the maximum detectable range
# ====================================================================
def compute_snr(parameters):
    numer = parameters['peak_transmisison_power'] * (parameters['gain'] ** 2) * (parameters['wavelength'] ** 2) * parameters['radar_cross_section'] * parameters['number_of_pulses']
    denom = math.pow((4 * math.pi), 3) * math.pow(parameters['target_range'], 4) * boltzmanns_constant * standard_temp * parameters['noise_figure'] * parameters['bandwidth'] * parameters['total_loss']
    
    #print(numer, denom)
    #print(parameters)

    result_linear = numer / denom
    result_db = linearToDecibel(result_linear)

    return result_linear, result_db


# Function to convert a frequency (Hz) to wavelength (m)
# ====================================================================
def frequencyToWavelength(frequency):
    return speed_of_light / frequency

# Function to convert a wavelength (m) to frequency (Hz)
# ====================================================================
def wavelengthToFrequency(wavelength):
    return speed_of_light / wavelength

# Function to convert a dB value to a linear value
# ====================================================================
def decibelToLinear(db_value):
    return 10 ** (db_value / 10)

# Function to convert a linear value to a dB value
# ====================================================================
def linearToDecibel(linear_value):
    return 10 * math.log10(linear_value)


# Compute the loss for 
def compute_total_loss(losses, collection_elevation_in_meters, ground_depth):
    part_a_db = losses['transitter_loss'] + losses['reciever_loss'] + losses['signal_processing_loss']
    part_a_linear = decibelToLinear(part_a_db)

    atmo_linear = decibelToLinear(losses['atmospheric_loss'])
    ground_linear = decibelToLinear(losses['ground_loss'])
    
    atmo_loss = ( (2 * atmo_linear) * collection_elevation_in_meters )      
    ground_loss = (ground_linear * ground_depth)

    total_loss_linear = part_a_linear * atmo_loss * ground_loss
    total_loss_db = linearToDecibel(total_loss_linear)

    return total_loss_linear, total_loss_db