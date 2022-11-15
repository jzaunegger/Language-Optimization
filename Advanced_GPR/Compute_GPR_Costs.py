# Import Dependencies
import os, sys, json, math
sys.path.append('../')
import torch
import LanguageHelper
import Optimization
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Compute the spatial resolution from POMR V2 - 15.2
# Frequency is being given in Hz, so we need to convert to MHz for the
# equation.
def compute_spatial_res(frequency, rel_perm):
    return 150 / (math.sqrt(rel_perm) * (frequency / (10**6)) )

# Compute the maximum depth from POMR V2 - 15.3
def compute_max_depth(time_window, rel_perm):
    return time_window / (8.7 * math.sqrt(rel_perm))

def compute_sampling_frequency(frequency):
    return 6 * frequency

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
    output_path = os.path.join('./costs', 'sim_costs')

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
        "The frequency must be larger than 200000000 hertz.",
        "The frequency should be larger than 350000000 hertz.",
        "The sampling frequency must be less than 1_000_000_000",
        "The sampling frequency should be lower than 800_000_000",
        "The sampling interval must be lower than 0.000_000_000_30 nanoseconds.",
        "The sampling interval should be lower than 0.000_000_000_15 nanoseconds.",
        "The time window must be less than 5 seconds.",
        "The time window should be less than 2 second.",
        "The spatial resolution must be less than 1.",
        "The spatial resolution should be less than 0.7.",
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

    # Compute the cost for frequency
    ############################################################
    frequency_costs, frequency_values = [], []
    for i in range(10_000_000, 2_000_000_000, 1_000):
        temp_cost = Optimization.analyze_cost(
                performance_metric = i,
                goal_values = language_data['frequency']['values'],
                direction_values = language_data['frequency']['direction_num'],
                weights = language_data['frequency']['goal_weights'],
                r_value = 10000
            )

        frequency_costs.append(temp_cost)
        frequency_values.append(i / (10 ** 6))

    # Plot Frequency Costs
    plt.plot(frequency_values, frequency_costs)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Cost')
    plt.title('Frequency Costs')
    plt.savefig(os.path.join(output_path, 'Frequency-Costs.png'))
    plt.close()

    print("Computed Frequency Cost")

    # Compute the cost for spatial resolution
    ############################################################
    spatial_res_costs, spatial_res_values = [], []
    spatial_res = 0.001
    while spatial_res <= 3:
        temp_cost = Optimization.analyze_cost(
            performance_metric = spatial_res,
            goal_values = language_data['spatial_resolution']['values'],
            direction_values = language_data['spatial_resolution']['direction_num'],
            weights = language_data['spatial_resolution']['goal_weights'],
            r_value = 10000
        )
        spatial_res_costs.append(temp_cost)
        spatial_res_values.append(spatial_res)
        spatial_res += 0.01

    # Plot Spatial Res Costs
    plt.plot(spatial_res_values, spatial_res_costs)
    plt.xlabel('Spatial Resolution (m)')
    plt.ylabel('Cost')
    plt.title('Spatial Resolution Costs')
    plt.savefig(os.path.join(output_path, 'Spatial-Resolution-Costs.png'))
    plt.close()

    print("Computed Spatial Res Cost")

    # Compute the cost for sampling frequency
    ############################################################
    sample_freq_cost, sample_freq_values = [], []
    for i in range(10_000_000, 2_000_000_000, 1_000):
        temp_cost = Optimization.analyze_cost(
                performance_metric = compute_sampling_frequency(i),
                goal_values = language_data['sampling_frequency']['values'],
                direction_values = language_data['sampling_frequency']['direction_num'],
                weights = language_data['sampling_frequency']['goal_weights'],
                r_value = 10000
            )

        sample_freq_cost.append(temp_cost)
        sample_freq_values.append(i / (10 ** 6))

    # Plot the sampling frequency
    plt.plot(sample_freq_values, sample_freq_cost)
    plt.xlabel('Sampling Frequency (MHz)')
    plt.ylabel('Cost')
    plt.title('Sampling Frequency Costs')
    plt.savefig(os.path.join(output_path, 'Sampling-Frequency-Costs.png'))
    plt.close()

    print('Computed Sampling Frequency')

    # Compute the cost for sampling interval
    ############################################################
    sample_interval_cost, sample_interval_values = [], []
    for i in range(10_000_000, 2_000_000_000, 1_000):

        sample_freq = compute_sampling_frequency(i)

        temp_cost = Optimization.analyze_cost(
                performance_metric = compute_sampling_interval(sample_freq),
                goal_values = language_data['sampling_interval']['values'],
                direction_values = language_data['sampling_interval']['direction_num'],
                weights = language_data['sampling_interval']['goal_weights'],
                r_value = 10000
            )

        sample_interval_cost.append(temp_cost)
        sample_interval_values.append(i / (10 ** -9))

    # Plot the sampling interval
    plt.plot(sample_freq_values, sample_freq_cost)
    plt.xlabel('Sampling Interval (ns)')
    plt.ylabel('Cost')
    plt.title('Sampling Interval Costs')
    plt.savefig(os.path.join(output_path, 'Sampling-Interval-Costs.png'))
    plt.close()

    print('Computed Sampling Interval')

    # Compute the cost for time window
    ############################################################
    time_window_costs, time_window_values = [], []
    current_time = 0.0001
    while current_time <= 7:
        temp_cost = Optimization.analyze_cost(
            performance_metric = current_time,
            goal_values = language_data['time_window']['values'],
            direction_values = language_data['time_window']['direction_num'],
            weights = language_data['time_window']['goal_weights'],
            r_value = 10000
        )
        time_window_costs.append(temp_cost)
        time_window_values.append(current_time)
        current_time += 0.01


    # Plot Time Window Costs
    plt.plot(time_window_values, time_window_costs)
    plt.xlabel('Time Window (S)')
    plt.ylabel('Cost')
    plt.title('Time Window Costs')
    plt.savefig(os.path.join(output_path, 'Time-Window-Costs.png'))
    plt.close()

    print("Computed Time Window Cost")

    # Compute the cost for maximum depth
    ############################################################
    max_depth_costs, max_depth_values = [], []
    max_depth = 0.01
    while max_depth <= 7:
        temp_cost = Optimization.analyze_cost(
            performance_metric = max_depth,
            goal_values = language_data['maximum_depth']['values'],
            direction_values = language_data['maximum_depth']['direction_num'],
            weights = language_data['maximum_depth']['goal_weights'],
            r_value = 10000
        )
        max_depth_costs.append(temp_cost)
        max_depth_values.append(max_depth)
        max_depth += 0.01

    # Plot Max Depth Costs
    plt.plot(max_depth_values, max_depth_costs)
    plt.xlabel('Maximum Depth (m)')
    plt.ylabel('Cost')
    plt.title('Maximum Depth Costs')
    plt.savefig(os.path.join(output_path, 'Maximum-Depth-Costs.png'))
    plt.close()

    print("Computed Max Depth Cost")