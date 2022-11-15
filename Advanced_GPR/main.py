# Import Dependencies
import os, sys, json, math
sys.path.append('../')
import torch
import LanguageHelper
import Optimization
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np


# Compute the spatial resolution from POMR V2 - 15.2
# Frequency is being given in Hz, so we need to convert to MHz for the
# equation.
def compute_spatial_res(frequency, rel_perm):
    return 150 / (math.sqrt(rel_perm) * (frequency / (10**6)) )

# Compute the maximum depth from POMR V2 - 15.3
def compute_max_depth(time_window, rel_perm):
    return time_window / (8.7 * math.sqrt(rel_perm))

# Compute the sampling frequency from POMR V2 - 15.4
def compute_sampling_frequency(frequency):
    return 6 * frequency

# Compute the sampling interval from POMR V2 - 15.5
def compute_sampling_interval(sampling_frequency):
    return 1 / sampling_frequency

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
        if ner_analysis['values'][0].isdigit(): value = int(ner_analysis['values'][0])
        else: value = float(ner_analysis['values'][0])

        #value = int(ner_analysis['values'][0])
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

def compute_cost(x, *args):

    frequency = x[0]
    time_window = x[1]

    language_data = args[0]
    rel_perm = args[1]

    sampling_freq = compute_sampling_frequency(frequency)
    sampling_interval = compute_sampling_interval(sampling_freq)


    total_cost = 0

    # Loop through commands
    for param in language_data:

        # Compute the spatial res cost
        if param == 'spatial_resolution': 
            temp_cost = Optimization.analyze_cost(
                performance_metric = compute_spatial_res(frequency, rel_perm),
                goal_values = language_data['spatial_resolution']['values'],
                direction_values = language_data['spatial_resolution']['direction_num'],
                weights = language_data['spatial_resolution']['goal_weights'],
                r_value = 10000
            )
        
        # Compute the max depth cost
        if param == 'maximum_depth': 
            temp_cost = Optimization.analyze_cost(
                performance_metric = compute_max_depth(time_window, rel_perm),
                goal_values = language_data['maximum_depth']['values'],
                direction_values = language_data['maximum_depth']['direction_num'],
                weights = language_data['maximum_depth']['goal_weights'],
                r_value = 10000
            )

        # Compute the time window cost
        if param == 'time_window':
            temp_cost = Optimization.analyze_cost(
                performance_metric = time_window,
                goal_values = language_data['time_window']['values'],
                direction_values = language_data['time_window']['direction_num'],
                weights = language_data['time_window']['goal_weights'],
                r_value = 10000
            )

        # Compute the frequency cost
        if param == 'frequency':
            temp_cost = Optimization.analyze_cost(
                performance_metric = frequency,
                goal_values = language_data['frequency']['values'],
                direction_values = language_data['frequency']['direction_num'],
                weights = language_data['frequency']['goal_weights'],
                r_value = 10000
            )

        # Compute the sampling frequency cost
        if param == 'sampling_frequency':
            temp_cost = Optimization.analyze_cost(
                performance_metric = sampling_freq,
                goal_values = language_data['sampling_frequency']['values'],
                direction_values = language_data['sampling_frequency']['direction_num'],
                weights = language_data['sampling_frequency']['goal_weights'],
                r_value = 10000
            )

        # Compute the sampling interval cost
        if param == 'sampling_interval':
            temp_cost = Optimization.analyze_cost(
                performance_metric = sampling_interval,
                goal_values = language_data['sampling_interval']['values'],
                direction_values = language_data['sampling_interval']['direction_num'],
                weights = language_data['sampling_interval']['goal_weights'],
                r_value = 10000
            )

        total_cost += temp_cost

    return total_cost

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

# Run the main body of the script
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
    output_path = os.path.join('./costs')

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
        "The frequency must be larger than 600000000 hertz.",
        "The frequency should be larger than 500000000 hertz.",
        "The sampling frequency must be less than 1_000_000_000 hertz.",
        "The sampling frequency should be lower than 800_000_000 hertz.",
        "The sampling interval must be lower than 0.000_000_000_30 nanoseconds.",
        "The sampling interval should be lower than 0.000_000_000_15 nanoseconds.",
        "The time window must be less than 5 seconds.",
        "The time window should be less than 2 second.",
        "The spatial resolution must be less than 1.",
        "The spatial resolution should be less than 0.5.",
    ]

    # Analyze the language data
    language_data = analyze_sentences(test_sentences)

    language_data['maximum_depth'] = {
        'values': [1, 5],
        'direction_label': ['Higher', 'Higher'],
        'direction_num': [1, 1],
        'goal_weights': [100, 1]
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

    # Define the intial guess, and physically allowed bounds
    initial_guess = [100_000_000, 1]
    bounds = [
        [10_000_000, 2_000_000_000],    # Frequency
        [1e-6, 6],                      # Time Window
    ]

    parameters = {
        "ground_materials": {
            "names": ['clay_dry', 'soil_sandy_dry', 'sand_wet', 'concrete_dry', 'soil_clayey_dry'],
        }
    }


    time_regions = [ [0, 200], [200, 400], [400, 600], [600, 800], [800, 1000] ]
    optimal_frequencies, optimal_time_window = [], []
    time_steps, dielectric_constants = [], []
    max_depths, spatial_resolutions = [], []
    sampling_frequencies, sampling_intervals = [], []

    # Run the simulation over time, altering the ground loss characteristics
    for t in range(0, 1000):
    
        # Check to update the loss for a given ground material
        for r in range(0, len(time_regions)):
            if t >= time_regions[r][0] and t <= time_regions[r][1]:
                current_mat_name = parameters["ground_materials"]["names"][r]
                current_mat = materials_data[current_mat_name]
                avg_perm = (current_mat['permittivity'][0] + current_mat['permittivity'][1]) / 2

        # Optimize to find the ideal variables
        res = minimize(
            compute_cost,
            x0 = initial_guess,
            args = (language_data, avg_perm),
            method = 'Nelder-Mead',
            bounds = bounds
        )

        # Update guess
        initial_guess[0] = res.x[0]
        initial_guess[1] = res.x[1]

        # Record Data
        current_sample_freq = compute_sampling_frequency(res.x[0])
        current_sample_interval = compute_sampling_interval(current_sample_freq)

        optimal_frequencies.append(res.x[0] / (10 ** 6))
        optimal_time_window.append(res.x[1])
        spatial_resolutions.append( compute_spatial_res(res.x[0], avg_perm))
        max_depths.append( compute_max_depth(res.x[1], avg_perm))
        sampling_frequencies.append(current_sample_freq / (10 ** 6))
        sampling_intervals.append(current_sample_interval / (10 ** -9))


        dielectric_constants.append(avg_perm)
        time_steps.append(t)


    # Plot frequency results
    plt.plot(time_steps, optimal_frequencies, color='b', label='Optimal Frequency')
    plt.axhline(y=language_data['frequency']['values'][0] / (10 ** 6) , color='r', label='Hard Constraint' )
    plt.axhline(y=language_data['frequency']['values'][1] / (10 ** 6) , color='g', label='Soft Constraint' )
    plt.xlabel('Time (t)')
    plt.ylabel('Frequency (MHz)')
    plt.title('Optimal Frequency Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Optimal-Frequency.png'))
    plt.close()

    # Plot sampling frequency results
    plt.plot(time_steps, sampling_frequencies, color='b', label='Optimal Sampling Frequency')
    plt.axhline(y=language_data['sampling_frequency']['values'][0]/ (10 ** 6), color='r', label='Hard Constraint' )
    plt.axhline(y=language_data['sampling_frequency']['values'][1]/ (10 ** 6), color='g', label='Soft Constraint' )
    plt.xlabel('Time (t)')
    plt.ylabel('Sampling Frequency (MHz)')
    plt.title('Optimal Sampling Frequency Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Optimal-Sampling-Frequency.png'))
    plt.close()

    # Plot sampling interval results
    plt.plot(time_steps, sampling_intervals, color='b', label='Optimal Time Window')
    plt.axhline(y=language_data['sampling_interval']['values'][0] / (10 ** -9), color='r', label='Hard Constraint' )
    plt.axhline(y=language_data['sampling_interval']['values'][1] / (10 ** -9), color='g', label='Soft Constraint' )
    plt.xlabel('Time (t)')
    plt.ylabel('Sampling Interval (ns)')
    plt.title('Optimal Sampling Interval Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Optimal-Sampling-Interval.png'))
    plt.close()

    # Plot time window results
    plt.plot(time_steps, optimal_time_window, color='b', label='Optimal Time Window')
    plt.axhline(y=language_data['time_window']['values'][0], color='r', label='Hard Constraint' )
    plt.axhline(y=language_data['time_window']['values'][1], color='g', label='Soft Constraint' )
    plt.xlabel('Time (t)')
    plt.ylabel('Time Window (s)')
    plt.title('Optimal Time Window Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Optimal-Time-Window.png'))
    plt.close()

    # Plot max depth results
    plt.plot(time_steps, max_depths, color='b', label='Optimal Max Depth')
    plt.axhline(y=language_data['maximum_depth']['values'][0], color='r', label='Hard Constraint' )
    plt.axhline(y=language_data['maximum_depth']['values'][1], color='g', label='Soft Constraint' )
    plt.xlabel('Time (t)')
    plt.ylabel('Max Depth (m)')
    plt.title('Max Depth Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Max-Depth.png'))
    plt.close()

    # Plot spatial resolution results
    plt.plot(time_steps, spatial_resolutions, color='b', label='Optimal Spatial Resolution')
    plt.axhline(y=language_data['spatial_resolution']['values'][0], color='r', label='Hard Constraint' )
    plt.axhline(y=language_data['spatial_resolution']['values'][1], color='g', label='Soft Constraint' )
    plt.xlabel('Time (t)')
    plt.ylabel('Spatial Resolution')
    plt.legend()
    plt.title('Spatial Resolution Over Time')
    plt.savefig(os.path.join(output_path, 'Spatial-Resolution.png'))
    plt.close()

    # Plot dielectric constants
    plt.plot(time_steps, dielectric_constants)
    plt.xlabel('Time (t)')
    plt.ylabel('Dielectric Constant')
    plt.title('Dielectric Constant Over Time')
    plt.savefig(os.path.join(output_path, 'Dielectric-Constants.png'))
    plt.close()


    # Log Statistics
    for r in range(0, len(time_regions)):
        freq_stats = compute_stats(time_steps, optimal_frequencies, time_regions[r])
        sfreq_stats = compute_stats(time_steps, sampling_frequencies, time_regions[r])
        sint_stats = compute_stats(time_steps, sampling_intervals, time_regions[r])
        res_stats = compute_stats(time_steps, spatial_resolutions, time_regions[r])
        tw_stats = compute_stats(time_steps, optimal_time_window, time_regions[r])
        md_stats = compute_stats(time_steps, max_depths, time_regions[r])
        die_stats = compute_stats(time_steps, dielectric_constants, time_regions[r])


        print('Time Region: {} - {}'.format(time_regions[r][0], time_regions[r][1]))
        print('-----------------------------------------------------')
        print('{:20s}: {:10.3f} ({})'.format('Die. Const.', die_stats['mean'], 'N/A'))
        print('{:20s}: {:10.3f} ({})'.format('Avg Freq', freq_stats['mean'], 'MHz'))
        print('{:20s}: {:10.3f} ({})'.format('Avg Sample Freq.', sfreq_stats['mean'], 'MHz'))
        print('{:20s}: {:10.3f} ({})'.format('Avg Sample Int.', sint_stats['mean'], 'ns'))
        print('{:20s}: {:10.3f} ({})'.format('Avg Spat. Res.', res_stats['mean'], 'm'))
        print('{:20s}: {:10.3f} ({})'.format('Avg Time Window.', tw_stats['mean'], 's'))
        print('{:20s}: {:10.3f} ({})'.format('Avg Max Depth.', md_stats['mean'], 'm'))
        print(' ')