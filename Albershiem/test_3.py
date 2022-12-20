# Import Dependencies
import os, sys, math, json
sys.path.append('../')

import matplotlib.pyplot as plt
import torch
import LanguageHelper
from scipy.optimize import minimize
import Optimization
import numpy as np


from LBCF_Helper import *
'''
    Create a script to optimize the values for bandwidth, and peak transmission power. 
    This script puts constraints on the range resolution (bandwidth), average power (peak power, bandwidth),
    and the SNR (peak power, bandwidth). 

'''

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

# Define the cost function
#################################################################
def compute_cost(x, *args):

    # Destrcuture the parameters, and update with optimization values
    parameters = args[0]
    parameters['bandwidth'] = x[0]
    parameters['peak_transmission_power'] = x[1]

    # Compute the signal to noise ratio
    snr_linear, snr_db = compute_snr(
        peak_power=parameters['peak_transmission_power'],
        gain = parameters['gain'],
        wavelength = parameters['wavelength'],
        rcs = parameters['radar_cross_section'],
        num_pulses = parameters['number_of_pulses'],
        range = parameters['target_range'],
        noise = parameters['noise_figure'],
        bandwidth =  parameters['bandwidth'],
        loss = parameters['total_loss'] 
    )

    # Get the probability estimate, range resolution and average power
    prob_detection = compute_PD_from_Albershiems( parameters["number_of_pulses"], snr_db,  parameters['Pfa'] )
    range_res = compute_range_resolution(parameters['bandwidth'])
    avg_power = compute_avg_power(parameters['peak_transmission_power'], parameters['dwell_time'], parameters['bandwidth'], parameters['number_of_pulses'])

    # Destrcuture optimization data
    language_data = args[1]
    total_cost = 0

    for param in language_data:

        if param == 'average_power':
            current_cost = Optimization.analyze_cost(
                performance_metric=avg_power,
                goal_values = language_data[param]['values'],
                direction_values = language_data[param]['direction_num'],
                weights = language_data[param]['goal_weights'],
                r_value = 10000
            )

            total_cost += current_cost

        if param == 'bandwidth':
            current_cost = Optimization.analyze_cost(
                performance_metric=parameters['bandwidth'],
                goal_values = language_data[param]['values'],
                direction_values = language_data[param]['direction_num'],
                weights = language_data[param]['goal_weights'],
                r_value = 10000
            )

        if param == 'peak_transmission_power':
            current_cost = Optimization.analyze_cost(
                performance_metric=parameters['peak_transmission_power'],
                goal_values = language_data[param]['values'],
                direction_values = language_data[param]['direction_num'],
                weights = language_data[param]['goal_weights'],
                r_value = 10000
            )
        
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

        if param == 'snr':
            current_cost = Optimization.analyze_cost(
                performance_metric=snr_db,
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
    output_path = os.path.join('./optimization_3')

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
        "The range resolution should be lower than 0.5 meters.",
        "The average power must be lower than 3 watts.",
        "The average power should be lower than 1 watts.",
    ]

    # Analyze the language data
    language_data = analyze_sentences(test_sentences)

    language_data['snr'] = {
        'values' : [15, 30],
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
        "prf": 10000,                           # Given in hertz
        "Pfa": (10 ** -4),  
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
    parameters['wavelength'] = frequency_to_wavelength(parameters['frequency'])
    parameters["dwell_time"] = prf_to_dwell_time(parameters['prf'], parameters['number_of_pulses'])

    # Define the intial guess, and physically allowed bounds
    initial_guess = [90_000_000, 850]
    bounds = [
        [1_000_000, 900_000_000],     # Bandwidth
        [1, 900]           # Peak Transmission Power
    ]

    # Define structures to hold results
    optimal_bandwidths, peak_trans_powers, snr_values = [], [], []
    time_steps, loss_values, system_loss, ground_losses = [], [], [], []
    prob_detect, range_res = [], []
    avg_powers = []
    time_regions = [ [0, 200], [200, 400], [400, 600], [600, 800], [800, 1000] ]
    verbose_level = 0

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

        # Update the bandwidth and peak transmission power
        parameters['bandwidth'] = res.x[0]
        parameters['peak_transmisison_power'] = res.x[1]

        # Compute the signal to noise ratio
        current_snr_linear, current_snr_db = compute_snr(
            peak_power=parameters['peak_transmission_power'],
            gain = parameters['gain'],
            wavelength = parameters['wavelength'],
            rcs = parameters['radar_cross_section'],
            num_pulses = parameters['number_of_pulses'],
            range = parameters['target_range'],
            noise = parameters['noise_figure'],
            bandwidth =  parameters['bandwidth'],
            loss = parameters['total_loss'] 
        )

        # Compute the probability of detection, range resolution, and average power
        prob_detection = compute_PD_from_Albershiems( parameters["number_of_pulses"], current_snr_db,  parameters['Pfa'] )
        current_range_resolution = compute_range_resolution(res.x[0])
        current_avg_power = compute_avg_power(res.x[1], parameters['dwell_time'], res.x[0], parameters['number_of_pulses'])

        # Record data
        optimal_bandwidths.append(res.x[0] / 1_000_000)
        peak_trans_powers.append(res.x[1])
        snr_values.append( current_snr_db )
        time_steps.append(t)
        loss_values.append( abs(res.fun) )
        prob_detect.append(prob_detection)
        range_res.append(current_range_resolution)
        avg_powers.append(current_avg_power)

        # Log Information
        if verbose_level > 0:   
            print('Current Time: {:3} | Current Bandwidth: {:8.2f} (MHz) | Current Power: {:8.2f} (W) | Current Range Res: {:8.2f} (m) | Current SNR: {:.2f} (dB) | Total Loss: {:.4f} |'.format(
                t, res.x[0]/1_000_000, res.x[1], current_range_resolution, current_snr_db, total_loss_linear
            ))

        #print(parameters['losses']['ground_loss'])
        ground_losses.append(parameters['losses']['ground_loss'])
        

    def compute_stats(data, name, label):
    
        print('| {:20s} | Min: {:8.3f} {:5s} | Mean: {:8.3f} {:5s} | Max: {:8.3f} {:5s} | Std. Dev: {:8.3f} {:5s} | Var. : {:14.3f} {:5s} |'.format(
            name, np.min(data), label, np.mean(data), label, np.max(data), label, np.std(data),  label, np.var(data), label
        ))
         

    def log_stats(region):
        current_bands       = np.array( optimal_bandwidths[region[0]:region[1]] )
        current_power       = np.array( peak_trans_powers[region[0]:region[1]] )
        current_range_res   = np.array( range_res[region[0]:region[1]] )
        current_snr_vals    = np.array( snr_values[region[0]:region[1]] ) 
        current_prob        = np.array( prob_detect[region[0]:region[1]])
        current_avg_power   = np.array( avg_powers[region[0]:region[1]])

        compute_stats(current_bands, 'Bandwidth', '(MHz)')
        compute_stats(current_power, 'Peak Trans. Power', '(W)')
        compute_stats(current_avg_power, 'Avg Pow.', '(W)')
        compute_stats(current_range_res, 'Range Resolution', '(m)')
        compute_stats(current_snr_vals, 'SNR Val', '(dB)')
        compute_stats(current_prob, 'Prob Detect.', '(N/A)')
        
    # Loop through the data
    for i in range(0, len(time_regions)):
        region = time_regions[i]
        bar = '-' * 148

        res = parameters['losses']['ground_materials']['names'][i]
        print(bar)
        print('Time Region: {:4}-{:4} Material = {}'.format(region[0], region[1], res))
        print(bar)
        log_stats(region)
        print(bar)
        print(' ')


    def plot_chart(x_data, y_data, x_label, y_label, title, path):
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    # Plot Charts
    plot_chart(time_steps, optimal_bandwidths, 'Time (t)', 'Bandwidth (kHz)', 'Bandwidth over Time', os.path.join(output_path, 'Bandwidth-Chart.png'))
    plot_chart(time_steps, peak_trans_powers, 'Time (t)', 'Peak Trans. Power (W)', 'Peak Power over Time', os.path.join(output_path, 'Peak-Power-Chart.png'))
    plot_chart(time_steps, avg_powers, 'Time (t)', 'Avg. Power (W)', 'Avg. Power over Time', os.path.join(output_path, 'Avg-Power-Chart.png'))
    plot_chart(time_steps, snr_values, 'Time (t)', 'SNR (dB)', 'SNR over Time', os.path.join(output_path, 'SNR-Chart.png'))
    plot_chart(time_steps, loss_values, 'Time (t)', 'Loss', 'Optimization Loss over Time', os.path.join(output_path, 'Opt-Loss-Chart.png'))
    plot_chart(time_steps, ground_losses, 'Time (t)', 'Attenuation', 'Ground Loss over Time', os.path.join(output_path, 'Attenuation-Loss-Chart.png'))
    plot_chart(time_steps, prob_detect, 'Time (t)', 'Prob. of Detection', 'Prob. of Detection over Time', os.path.join(output_path, 'prob-Detection-Chart.png'))
    plot_chart(time_steps, range_res, 'Time (t)', 'Range Res. (m)', 'Range Resolution over Time', os.path.join(output_path, 'Range-Resolution-Chart.png'))