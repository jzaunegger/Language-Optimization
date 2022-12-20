# Import Dependencies
import os, sys, math, json
sys.path.append('../')
import RadarLib
import matplotlib.pyplot as plt
import torch
import LanguageHelper
from scipy.optimize import minimize
import Optimization
import numpy as np
'''
    Create a script to optimize the bandwidth, sampling rate, transmission power, and probability of detection.

    Going to use a static value for the probability of false alarm.
'''

def compute_range_resolution(bandwidth):
    speed_of_light = 3 * (10 ** 8)

    return speed_of_light / (2 * bandwidth)


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

# Create a function to process 
#################################################
def analyze_sentences(sentences):
    
    formatted_dict = {}

    # Process the sentences
    for sentence in sentences:

        # Get the padded index and convert to a tensor
        tokens, padded_ind = LanguageHelper.preprocess_sentence(sentence, token2index, 14)
        input_tensor = torch.LongTensor([padded_ind]).to(device)

        # Analyze sentiment, directivity, and named entity recognitions
        sent_analysis = LanguageHelper.analyze_sentiment(input_tensor, sent_model, sentiment_labels)
        gp_analysis = LanguageHelper.analyze_goal_priority(input_tensor, dir_model, directivity_labels)
        ner_analysis = LanguageHelper.analyze_ner(input_tensor, ner_model, idx2nertype, tokens, padded_ind, index2token)
        dir_analysis = LanguageHelper.analyze_direction(input_tensor, direction_model, direction_labels)


        # Check determine the current parameter
        if len(ner_analysis['parameters']) == 1:
            param = ner_analysis['parameters'][0]
        elif len(ner_analysis['parameters']) > 1:
            param = "_".join(ner_analysis['parameters'])

        # Initalize the parameter in the dictionary
        if param not in formatted_dict:
            formatted_dict[param] = {
                'values': [],
                'direction_label': [],
                'direction_num': [],
                'goal_weights': [],
            }

        # Format the value data and direction label
        current_value = ner_analysis['values'][0]
        if current_value.isdigit():
            value = int(current_value)
        else:
            value = float(current_value)

        direction_label = dir_analysis['direction_label']

        if direction_label == 'Higher': direction_num = 1
        if direction_label == 'Default': direction_num = 0
        if direction_label == 'Lower': direction_num = -1 

        # Determine the goal weight
        gp_prob = gp_analysis['gp_probs'][0]
        gp_weight = int(gp_prob[0] * 100)

        if gp_weight == 0: gp_weight = 1

        # Record data
        formatted_dict[param]['values'].append(value)
        formatted_dict[param]['direction_label'].append(direction_label)
        formatted_dict[param]['direction_num'].append(direction_num)
        formatted_dict[param]['goal_weights'].append(gp_weight)

    return formatted_dict

# Function to compute the ground loss for some depth
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

# Function to compute the total loss
def compute_total_loss(losses):
    total_loss_db = losses['transmitter_loss'] + losses['reciever_loss'] + losses['signal_processing_loss'] + losses['ground_loss']
    total_loss_linear = RadarLib.decibelToLinear(total_loss_db)
    return total_loss_db, total_loss_linear

# Define the cost function
#################################################################
def compute_cost(x, *args):

    # Destructure optimization values
    bandwidth = x[0]
    peak_trans_power = x[1]

    # Destrcuture the parameters, and update with optimization values
    parameters = args[0]
    parameters['bandwidth'] = bandwidth
    parameters['peak_transmission_power'] = peak_trans_power

    # Compute the maximum detectable range
    snr_linear, snr_db = RadarLib.compute_snr(parameters)

    # Get the probability estimate
    prob_detection = compute_PD_from_Albershiems(
        N = parameters["number_of_pulses"],
        SNR = snr_db,
        PFA=(10 ** -4)
    )

    # Get the range resolution
    range_res = compute_range_resolution(bandwidth)

    # Destrcuture optimization data
    language_data = args[1]
    total_cost = 0

    for param in language_data:

        if param == 'snr':
            current_cost = Optimization.analyze_cost(
                performance_metric=snr_db,
                goal_values = language_data[param]['values'],
                direction_values = language_data[param]['direction_num'],
                weights = language_data[param]['goal_weights'],
                r_value = 10000
            )
            total_cost += current_cost

        if param == 'prob_detection':
            current_cost = Optimization.analyze_cost(
                performance_metric = prob_detection,
                goal_values = language_data[param]['values'],
                direction_values = language_data[param]['direction_num'],
                weights = language_data[param]['goal_weights'],
                r_value = 10000
            )
            total_cost += current_cost

        if param == 'range_resolution':
            current_cost = Optimization.analyze_cost(
                performance_metric = range_res,
                goal_values = language_data[param]['values'],
                direction_values = language_data[param]['direction_num'],
                weights = language_data[param]['goal_weights'],
                r_value = 10000
            )
            total_cost += current_cost

    return total_cost


# Run the main loop
#################################################################
if __name__ == '__main__':

    # Check GPU Accesibility
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print('Using Device: {}'.format(device))

    # Define program paths
    DATA_ROOT = '../data'
    TOKEN_DATA_PATH = os.path.join(DATA_ROOT, 'system', 'CombinedTokenData.pickle')
    TC_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Sentiment-Model.pt')
    DC_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Directivity-Model.pt')
    DIR_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Direction-Model.pt')
    NER_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'NER2-Model.pt')
    output_path = os.path.join('./optimization_2')

    # Load in the system token data
    token2index, index2token = LanguageHelper.load_token_data(TOKEN_DATA_PATH)
    directivity_labels = {0: "Hard", 1: "Soft"}
    sentiment_labels = {0: "Negative", 1: "Positive"}
    idx2nertype = {0: 'PAD', 1: 'Parameter', 2: 'Value', 3: 'Units', 4: 'Word'}
    direction_labels = {0:'Lower', 1:'Default', 2:'Higher'}
    seq_len = 14
    verbose = True

    # Load in the lstm sentiment text classifier
    sent_model = torch.load(TC_MODEL_PATH, map_location=device)
    dir_model = torch.load(DC_MODEL_PATH, map_location=device)
    ner_model = torch.load(NER_MODEL_PATH, map_location=device)
    direction_model = torch.load(DIR_MODEL_PATH, map_location=device)

    # Create a series of test sentences
    test_sentences = [
        "The bandwidth must be less than 800000000 hertz.",
        "The bandwidth should be less than 600000000 hertz.",
        "The peak transmission power must be less than 900 watts.",
        "The peak transmission power should be less than 500 watts.",
        "The range resolution must be less than 1 meters.",
        "The range resolution should be lower than 0.1 meters."
    ]

    # Analyze the language data
    language_data = analyze_sentences(test_sentences)

    language_data['snr'] = {
        'values' : [50, 70],
        'direction_label' :['Higher', 'Higher'],
        'direction_num' : [1, 1],
        'goal_weights' : [100, 1]
    }

    language_data['prob_detection'] = {
        'values' : [0.8, 0.9],
        'direction_label' :['Higher', 'Higher'],
        'direction_num' : [1, 1],
        'goal_weights' : [100, 1]
    }

    # Log the extracted data
    if verbose == True:
        for key in language_data:
            current = language_data[key]
            print(key)
            for sub in current:
                print('    {}: {}'.format(sub, current[sub]))
            print(' ')


    # Load the materials data
    materials_file = '../materials.json'
    with open(materials_file, 'r') as f:
        materials_data = json.load(f)


    # Define the simulation parameters, some of these parameters will be updated
    # as the simulation runs, such as the ground loss and tubable parameters
    parameters = {
        "collection_elevation": 120,            # Given in meters
        "ground_depth": .15,                    # Given in meters
        "peak_transmisison_power": 500,         # Given in watts
        "gain": 1.258,                          # Given in linear units
        "frequency": 1.0 * math.pow(10, 9),     # Given in hertz
        "wavelength": 0.29979246,               # Given in meters
        "radar_cross_section": 1,               # Given in square meters
        "target_range": 120.15,                 # Given in meters
        "number_of_pulses": 10,                 # Unitless
        "noise_figure": 1.778,                  # Given in linear units
        "bandwidth": 0,                         # Given in hertz
        "total_loss": 0,                        # Given in linear units

        "losses": {
            "transmitter_loss": 3.1,            # 3.1 decibels
            "reciever_loss": 2.4,               # 2.4 decibels
            "signal_processing_loss": 3.2,      # 3.2 decibels
            "ground_loss": 0.0,                 # 
            "atmospheric_loss": 0.0,

            "ground_materials": {
                "names": ['clay_dry', 'soil_sandy_dry', 'sand_wet', 'concrete_dry', 'soil_clayey_dry'],
            }
        }
    }

    # Resolve some parameters
    parameters['wavelength'] = RadarLib.frequencyToWavelength(parameters['frequency'])

    # Define the intial guess, and physically allowed bounds
    initial_guess = [90_000_000, 850]
    bounds = [
        [1_000_000, 900_000_000],     # Bandwidth
        [1, 900]           # Peak Transmission Power
    ]

    # Define structures to hold results
    optimal_bandwidths, peak_trans_powers, snr_values = [], [], []
    time_steps, loss_values, system_loss = [], [], []
    ground_losses = []
    prob_detect, range_res = [], []

    time_regions = [ [0, 200], [200, 400], [400, 600], [600, 800], [800, 1000] ]

    # Run the simulation over time, altering the ground loss characteristics
    for t in range(0, 1000):
        
        # Check to update the loss for a given ground material
        for r in range(0, len(time_regions)):
            if t >= time_regions[r][0] and t <= time_regions[r][1]:
                current_mat_name = parameters["losses"]["ground_materials"]["names"][r]
                current_mat = materials_data[current_mat_name]

        # Compute the ground loss
        parameters['losses']['ground_loss'] = compute_attenuation_coefficient(
            conductivity=current_mat['conductivity'][1], 
            permittivity=current_mat['permittivity'][1], 
            angular_frequency= 2 * math.pi * parameters['frequency'], 
            distance=parameters['ground_depth']
        )

        # Compute the total loss
        total_loss_db, total_loss_linear = compute_total_loss(parameters['losses'])
        parameters['total_loss'] = total_loss_linear
        system_loss.append(total_loss_db)

        # Compute the optimal result for the given parameters
        res = minimize(
            compute_cost, 
            x0=(initial_guess), 
            args = (parameters, language_data),
            method="Nelder-Mead", 
            bounds=bounds
        )

        parameters['bandwidth'] = res.x[0]
        parameters['peak_transmisison_power'] = res.x[1]

        current_snr_linear, current_snr_db = RadarLib.compute_snr(parameters)
        prob_detection = compute_PD_from_Albershiems(N=parameters['number_of_pulses'], SNR=current_snr_db, PFA=(10 ** -4))

        current_range_resolution = compute_range_resolution(res.x[0])

        # Record data
        optimal_bandwidths.append(res.x[0] / 1_000_000)
        peak_trans_powers.append(res.x[1])
        snr_values.append( current_snr_db )
        time_steps.append(t)
        loss_values.append( abs(res.fun) )
        prob_detect.append(prob_detection)
        range_res.append(current_range_resolution)

        print('Current Time: {:3} | Current Bandwidth: {:8.2f} (MHz) | Current Power: {:8.2f} (W) | Current Range Res: {:8.2f} (m) | Current SNR: {:.2f} (dB) | Total Loss: {:.4f} |'.format(
            t, res.x[0]/1_000_000, res.x[1], current_range_resolution, current_snr_db, total_loss_linear
        ))

        #print(parameters['losses']['ground_loss'])
        ground_losses.append(parameters['losses']['ground_loss'])


    print('Average Bandwidth    : {} (kHz)'.format( np.mean( np.array(optimal_bandwidths)) ))
    print('Average Trans. Power : {} (W)'.format( np.mean( np.array(peak_trans_powers)) ))
    print('Average Range Res. : {} (m)'.format( np.mean( np.array(range_res)) ))
    print('Average SNR. : {} (dB)'.format( np.mean( np.array(snr_values)) ))
    print(' ')

    # Plot the optimal bandwidth over time
    plt.plot(time_steps, optimal_bandwidths)
    plt.xlabel('Time (t)')
    plt.ylabel('Bandwidth (kHz)')
    plt.title('Optimal Bandwidth over Time')
    plt.savefig(os.path.join(output_path, 'Bandwidth-Chart.png'))
    plt.close()

    # Plot the peak transmision power over time
    plt.plot(time_steps, peak_trans_powers)
    plt.xlabel('Time (t)')
    plt.ylabel('Peak Transmission Power (W)')
    plt.title('Peak Transmission Power over Time')
    plt.savefig(os.path.join(output_path, 'Power-Chart.png'))
    plt.close()

    # Plot the SNR over time
    plt.plot(time_steps, snr_values)
    plt.xlabel('Time (t)')
    plt.ylabel('Signal to Noise Ratio (dB)')
    plt.title('SNR over Time')
    plt.savefig(os.path.join(output_path, 'SNR-Chart.png'))
    plt.close()

    # Plot the loss over time
    plt.plot(time_steps, loss_values)
    plt.xlabel('Time (t)')
    plt.ylabel('Optimization Loss')
    plt.title('Optimization Loss over Time')
    plt.savefig(os.path.join(output_path, 'Loss-Chart.png'))
    plt.close()

    # Plot the ground loss over time
    plt.plot(time_steps, ground_losses)
    plt.xlabel('Time (t)')
    plt.ylabel('Attenuation Coefficients')
    plt.title('Attenuation Loss over Time')
    plt.savefig(os.path.join(output_path, 'Attenuation-Chart.png'))
    plt.close()

    # Plot the probability of detection over time
    plt.plot(time_steps, prob_detect)
    plt.xlabel('Time (t)')
    plt.ylabel('Probability of Detection')
    plt.title('Prob. Detection over Time')
    plt.savefig(os.path.join(output_path, 'Prob-Detection-Chart.png'))
    plt.close()

    plt.plot(time_steps, range_res)
    plt.xlabel('Time (t)')
    plt.ylabel('Range Resolution')
    plt.title('Range Resolution over Time')
    plt.savefig(os.path.join(output_path, 'Range-Resolution-Chart.png'))
    plt.close()