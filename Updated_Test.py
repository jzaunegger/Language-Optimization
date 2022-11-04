# Import Dependencies
import os, sys, math, json

from click import launch
import RadarLib
import matplotlib.pyplot as plt
import torch
import LanguageHelper
from scipy.optimize import minimize
import Optimization
import numpy as np

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
        value = int(ner_analysis['values'][0])
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
    bandwidth = x[0] * 1000
    number_of_pulses = int(x[1])
    peak_trans_power = x[2]

    # Destrcuture the parameters, and update with optimization values
    parameters = args[0]
    parameters['bandwidth'] = bandwidth
    parameters['number_of_pulses'] = number_of_pulses
    parameters['peak_transmission_power'] = peak_trans_power

    # Compute the maximum detectable range
    snr_linear, snr_db = RadarLib.compute_snr(parameters)

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

    return total_cost


    ## Loop through the linguistic constraints
    #for param in language_data:
#
    #    if param == 'bandwidth': temp_param = x[0]
    #    if param == 'number_of_pulses': temp_param = number_of_pulses
    #    if param == 'peak_transmission_power': x[2]
    #    
    #    current_cost = Optimization.analyze_cost(
    #        performance_metric = temp_param,
    #        goal_values = language_data[param]['values'],
    #        direction_values = language_data[param]['direction_num'],
    #        weights = language_data[param]['goal_weights'],
    #        r_value = 10000
    #    )
#
    #    total_cost += current_cost
    #
    #return total_cost


def compute_stats(time_stamps, opt_values, time_region):

    if len(time_stamps) != len(opt_values):
        print('Error: The number of provided time stamps and number of values are not equal.')
        return False

    if time_region[0] < 0 or time_region[1] > len(time_stamps):
        print('Error: Invalid time range provided')

    selected_vals = opt_values[time_region[0]:time_region[1]]
    np_select_vals = np.array(selected_vals)

    stats = {
        "length": len(selected_vals),
        "min": np.min(np_select_vals),
        "mean": np.mean(np_select_vals),
        "max": np.max(np_select_vals),
        "std": np.std(np_select_vals),
        "var": np.var(np_select_vals)
    }
    return stats


def plot_stats(time_region, ba_stats, np_stats, pp_stats, sn_stats, region_name, region_num):
    print('---------------------------------------------------------------------------------')    
    print('Region {:2} : {:20} for t = {} to {}'.format(region_num, region_name, time_region[0], time_region[1]))
    print('---------------------------------------------------------------------------------')
    print('| {:12s} | {:^10s} | {:^10s} |  {:^13} | {:^13s} |'.format('Average', "Bandwidth (kHz)", "Num. Pulses", "Power (W)", "SNR (dB)"))
    print('---------------------------------------------------------------------------------')
    print('| {:12s} | {:9.3f} (kHz) | {:^11.3f} |  {:9.3f} (W) | {:8.3f} (dB) |'.format('Minimum', ba_stats['min'], np_stats['min'], pp_stats['min'], sn_stats['min']))
    print('| {:12s} | {:9.3f} (kHz) | {:^11.3f} |  {:9.3f} (W) | {:8.3f} (dB) |'.format('Average', ba_stats['mean'], np_stats['mean'], pp_stats['mean'], sn_stats['mean']))
    print('| {:12s} | {:9.3f} (kHz) | {:^11.3f} |  {:9.3f} (W) | {:8.3f} (dB) |'.format('Maximum', ba_stats['max'], np_stats['max'], pp_stats['max'], sn_stats['max']))
    print('| {:12s} | {:9.3f} (kHz) | {:^11.3f} |  {:9.3f} (W) | {:8.3f} (dB) |'.format('Std. Dev.', ba_stats['std'], np_stats['std'], pp_stats['std'], sn_stats['std']))
    #print('| {:12s} | {:8.3f} (kHz) | {:5.3f} |  {:5.3f} |  {:5.3f} |'.format('Variance', ba_stats['var'], np_stats['var'], pp_stats['var'], sn_stats['var']))
    print('---------------------------------------------------------------------------------')
    print(' ')


# Run the main loop
#################################################################
if __name__ == '__main__':

    # Check GPU Accesibility
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print('Using Device: {}'.format(device))

    # Define program paths
    DATA_ROOT = './data'
    TOKEN_DATA_PATH = os.path.join(DATA_ROOT, 'system', 'CombinedTokenData.pickle')
    TC_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Sentiment-Model.pt')
    DC_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Directivity-Model.pt')
    DIR_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Direction-Model.pt')
    NER_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'NER2-Model.pt')
    output_path = os.path.join('./optimization')

    # Load in the system token data
    token2index, index2token = LanguageHelper.load_token_data(TOKEN_DATA_PATH)
    directivity_labels = {0: "Hard", 1: "Soft"}
    sentiment_labels = {0: "Negative", 1: "Positive"}
    idx2nertype = {0: 'PAD', 1: 'Parameter', 2: 'Value', 3: 'Units', 4: 'Word'}
    direction_labels = {0:'Lower', 1:'Default', 2:'Higher'}
    seq_len = 14
    verbose = False

    # Load in the lstm sentiment text classifier
    sent_model = torch.load(TC_MODEL_PATH, map_location=device)
    dir_model = torch.load(DC_MODEL_PATH, map_location=device)
    ner_model = torch.load(NER_MODEL_PATH, map_location=device)
    direction_model = torch.load(DIR_MODEL_PATH, map_location=device)

    # Create a series of test sentences
    test_sentences = [
        "The bandwidth must be less than 1000000 hertz.",
        "The bandwidth should be less than 800000 hertz.",
        "The number of pulses should be greater than 50.",
        "The number of pulses must be greater than 20.",
        "The peak transmission power must be less than 900 watts.",
        "The peak transmission power should be less than 500 watts.",
    ]

    # Analyze the language data
    language_data = analyze_sentences(test_sentences)

    language_data['snr'] = {
        'values' : [50, 70],
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
    materials_file = './materials.json'
    with open(materials_file, 'r') as f:
        materials_data = json.load(f)


    # Define the simulation parameters, some of these parameters will be updated
    # as the simulation runs, such as the ground loss and tubable parameters
    parameters = {
        "collection_elevation": 120,            # Given in meters
        "ground_depth": .15,                    # Given in meters
        "peak_transmisison_power": 500,         # Given in watts
        "gain": 1.258,                        # Given in linear units
        "frequency": 1.0 * math.pow(10, 9),     # Given in hertz
        "wavelength": 0.031889,                 # Given in meters
        "radar_cross_section": 1,               # Given in square meters
        "target_range": 120.15,                 # Given in meters
        "number_of_pulses": 0,                  # Unitless
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
    initial_guess = [50000, 10, 300]
    bounds = [
        [1000, 900000],     # Bandwidth
        [10, 100],          # Number of Pulses
        [1, 900]           # Peak Transmission Power
    ]

    # Define structures to hold results
    optimal_bandwidths, optimal_num_pulses, peak_trans_powers, snr_values = [], [], [], []
    time_steps, loss_values, system_loss = [], [], []
    ground_losses = []

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
        parameters['number_of_pulses'] = res.x[1]
        parameters['peak_transmisison_power'] = res.x[2]


        current_snr_linear, current_snr_db = RadarLib.compute_snr(parameters)

        # Record data
        optimal_bandwidths.append(res.x[0] / 1000)
        optimal_num_pulses.append(res.x[1])
        peak_trans_powers.append(res.x[2])
        snr_values.append( current_snr_db )
        time_steps.append(t)
        loss_values.append( abs(res.fun) )

        #print(parameters['losses']['ground_loss'])
        ground_losses.append(parameters['losses']['ground_loss'])


    print('Optimal Bandwidth    : {} (kHz)'.format( np.mean( np.array(optimal_bandwidths)) ))
    print('Optimal Num Pulses   : {} '.format( np.mean( np.array(optimal_num_pulses)) ))
    print('Optimal Trans. Power : {} (W)'.format( np.mean( np.array(peak_trans_powers)) ))
    print(' ')

    #"length": len(selected_vals),
    #    "min": np.min(np_select_vals),
    #    "mean": np.mean(np_select_vals),
    #    "max": np.max(np_select_vals),
    #    "std": np.std(np_select_vals),
    #    "var": np.var(np_select_vals)

    # Compute stats
    for r in range(0, len(time_regions)):

        ba_stats = compute_stats(time_steps, optimal_bandwidths, time_regions[r])
        np_stats = compute_stats(time_steps, optimal_num_pulses, time_regions[r])
        pp_stats = compute_stats(time_steps, peak_trans_powers, time_regions[r])
        sn_stats = compute_stats(time_steps, snr_values, time_regions[r])

        print('---------------------------------------------------------------------------------')    
        print('Region {:2} : {:20} for t = {} to {}'.format(r, parameters["losses"]["ground_materials"]["names"][r], time_regions[r][0], time_regions[r][1]))
        print('---------------------------------------------------------------------------------')
        print('| {:12s} | {:^10s} | {:^10s} |  {:^13} | {:^13s} |'.format('Average', "Bandwidth (kHz)", "Num. Pulses", "Power (W)", "SNR (dB)"))
        print('---------------------------------------------------------------------------------')
        print('| {:12s} | {:9.3f} (kHz) | {:^11.3f} |  {:9.3f} (W) | {:8.3f} (dB) |'.format('Minimum', ba_stats['min'], np_stats['min'], pp_stats['min'], sn_stats['min']))
        print('| {:12s} | {:9.3f} (kHz) | {:^11.3f} |  {:9.3f} (W) | {:8.3f} (dB) |'.format('Average', ba_stats['mean'], np_stats['mean'], pp_stats['mean'], sn_stats['mean']))
        print('| {:12s} | {:9.3f} (kHz) | {:^11.3f} |  {:9.3f} (W) | {:8.3f} (dB) |'.format('Maximum', ba_stats['max'], np_stats['max'], pp_stats['max'], sn_stats['max']))
        print('| {:12s} | {:9.3f} (kHz) | {:^11.3f} |  {:9.3f} (W) | {:8.3f} (dB) |'.format('Std. Dev.', ba_stats['std'], np_stats['std'], pp_stats['std'], sn_stats['std']))
        #print('| {:12s} | {:8.3f} (kHz) | {:5.3f} |  {:5.3f} |  {:5.3f} |'.format('Variance', ba_stats['var'], np_stats['var'], pp_stats['var'], sn_stats['var']))
        print('---------------------------------------------------------------------------------')
        print(' ')

    total_opt_band = compute_stats(time_steps, optimal_bandwidths, [0, 1000])
    total_opt_nump = compute_stats(time_steps, optimal_num_pulses, [0, 1000])
    total_opt_powe = compute_stats(time_steps, peak_trans_powers, [0, 1000])
    total_opt_snr = compute_stats(time_steps, snr_values, [0, 1000])


    plot_stats([0, 1000], total_opt_band, total_opt_nump, total_opt_powe, total_opt_snr, 'all', 0)


    # Plot the optimal bandwidth over time
    plt.plot(time_steps, optimal_bandwidths)
    plt.xlabel('Time (t)')
    plt.ylabel('Bandwidth (kHz)')
    plt.title('Optimal Bandwidth over Time')
    plt.savefig(os.path.join(output_path, 'Bandwidth-Chart.png'))
    plt.close()

    # Plot the optimal number of pulses over time
    plt.plot(time_steps, optimal_num_pulses)
    plt.xlabel('Time (t)')
    plt.ylabel('Number of Pulses')
    plt.title('Optimal Number of Pulses over Time')
    plt.savefig(os.path.join(output_path, 'Pulse-Chart.png'))
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


