# Import Dependencies
import os, sys
sys.path.append('../')
import torch
import LanguageHelper
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
        "The spatial resolution must be less than 1.",
        "The spatial resolution should be less than 0.7.",
    ]

    # Analyze the language data
    language_data = analyze_sentences(test_sentences)

    # Loop through the language data
    for key in language_data:
        print(key)

        for sub_key in language_data[key]:
            print('  {} : {}'.format(sub_key, language_data[key][sub_key]))

    current = language_data['spatial_resolution']

    # Set the bounds (10-5000 Mhz)
    bounds = [0.0001, 2]
    r_values = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

    # check the output path exists
    current_output_folder = os.path.join(output_path, 'spatial_resolution')
    if os.path.exists(output_path) == False: os.mkdir(output_path)
    if os.path.exists(current_output_folder) == False: os.mkdir(current_output_folder)


    # loop through r values
    for r in r_values:
        costs, labels = [], []
        current_res = bounds[0]

        while current_res <= bounds[1]:

            current_cost = Optimization.analyze_cost(
                performance_metric = current_res,
                goal_values = current['values'],
                direction_values = current['direction_num'],
                weights = current['goal_weights'],
                r_value = r
            )

            labels.append(current_res)
            costs.append(current_cost)

            current_res += 0.0001

        plt.plot(labels, costs)
        plt.xlabel('Spatial Spatial Resolution')
        plt.ylabel('Cost')
        plt.title('Cost for Spatial Resolution when r={}'.format(r))
        plt.savefig(os.path.join(current_output_folder, 'Spatial-Resolution-Cost-R-{}.png'.format(r)))
        plt.close()