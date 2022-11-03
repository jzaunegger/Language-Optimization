# Import Dependencies
import os
import torch
import LanguageHelper
import RadarLib
import Optimization
import matplotlib.pyplot as plt

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


def get_parameter_cost(language_data, R, bounds, eval_range, scale):
    costs, labels = [], []

    # Loop through the limits of the bounds
    for i in range(eval_range[0], eval_range[1]):
                
        current_cost = Optimization.analyze_cost(
            performance_metric = i,
            goal_values = language_data['values'],
            direction_values = language_data['direction_num'],
            weights = language_data['goal_weights'],
            r_value = R
        )
        
        costs.append(current_cost)
        labels.append(i / scale)

    return costs, labels

# Run the main body of the script
if __name__ == '__main__':

    # Define program paths
    DATA_ROOT = './data'
    TOKEN_DATA_PATH = os.path.join(DATA_ROOT, 'system', 'CombinedTokenData.pickle')
    TC_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Sentiment-Model.pt')
    DC_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Directivity-Model.pt')
    DIR_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'Direction-Model.pt')
    NER_MODEL_PATH = os.path.join(DATA_ROOT, 'trained_models', 'NER2-Model.pt')
    output_path = os.path.join('./costs', 'power')

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
        "The bandwidth must be less than 1000000 hertz.",
        "The bandwidth should be less than 800000 hertz.",
        "The number of pulses should be greater than 50.",
        "The number of pulses must be greater than 20.",
        "The peak transmission power must be less than 1000 watts.",
        "The peak transmission power should be less than 500 watts."
    ]

    # Analyze the language data
    language_data = analyze_sentences(test_sentences)

    # Get the cost data for each parameter set
    band_costs, band_labels   = get_parameter_cost(language_data=language_data['bandwidth'], R=10000, bounds=[1000, 900000], eval_range=[1000, 900000], scale=1000)
    pulse_costs, pulse_labels = get_parameter_cost(language_data=language_data['number_of_pulses'], R=10000, bounds=[10, 100], eval_range=[20, 100], scale=1)
    power_costs, power_labels = get_parameter_cost(language_data=language_data['peak_transmission_power'], R=10000, bounds=[1, 1000], eval_range=[1, 1000], scale=1)


    plt.plot(band_labels, band_costs, label='Bandwidth Cost')
    plt.plot(pulse_labels, pulse_costs, label='# Pulses Cost')
    plt.plot(power_labels, power_costs, label='Power Cost')
    plt.xlabel('Parameter Value')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()
   # plt.xlabel('Transmisison Power (kW)')
    #plt.ylabel('Cost')
    #plt.title('Cost for Transmission Power when r={}'.format(r))
    #plt.savefig(os.path.join(output_path, 'Power-Cost-R-{}.png'.format(r)))
    #plt.close()