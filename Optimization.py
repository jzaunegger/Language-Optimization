import math, sys
import numpy as np

# Create a function to compute sigma for LBCF
##############################################################
def compute_sigma(metric, goal_value, direction_num, r):

    part_a = (metric / goal_value) - 1
    part_b = direction_num * r * part_a

    try:
        part_c = 1 + math.pow( math.e, part_b )
    except OverflowError:
        part_c = 1.7e308   

    return 1 / part_c

# This function computes the y value that is used in the 
# compute h function.
##############################################################
def compute_y(direction_label, goal_values):

    if direction_label == 'Higher':
        return np.max(goal_values)
    
    elif direction_label == 'Lower':
        return np.min(goal_values)

    elif direction_label == 'Default':
        return 0

    else:
        print('Error: The direction should only be "Lower" or "Higher".')
        print('This function recieved: {}'.format(direction_label))
        sys.exit()

# This function computes the h function which is used to develop the cost function
##############################################################
def compute_h(direction_label, metric, y):

    part_a = math.pow(math.e, -1)
    part_b = 1 + math.e

    if direction_label == 'Higher': 
        exponent = y / metric
        try:
            part_c = math.pow(part_b, exponent)
        except OverflowError:
            part_c = 1.7e308            # Largest possible value in python!

        return part_a * (part_c - 1)

    elif direction_label == 'Lower':      
        exponent = metric / y
        try:
            part_c = math.pow(part_b, exponent)
        except OverflowError:
            part_c = 1.7e308            # Largest possible value in python!
        
        return part_a * (part_c - 1)



    elif direction_label == 'Default':    return 0
    else:
        print('Error: The direction should be "Lower", "Higher", or "Default".')
        print('This function recieved: {}'.format(direction_label))
        sys.exit()

# Create a function to check if every value in a array is the 
# same
##############################################################
def checkArrayValues(input_arr, value):
    PASSED = True

    for x in input_arr:
        if x != value:
            PASSED = False

    return PASSED



def analyze_cost(performance_metric, goal_values, direction_values, weights, r_value):

    cost = 0

    # Compute the sigma values
    for i in range(0, len(goal_values)):

        sigma_value = compute_sigma(
            metric = performance_metric,
            goal_value = goal_values[i],
            direction_num = direction_values[i],
            r = r_value
        )
        cost += weights[i] * sigma_value
    
    # Determine the SADP value
    if checkArrayValues(direction_values, -1) == True:
        y_value = compute_y( direction_label = 'Lower', goal_values = goal_values )
        h_value = compute_h( direction_label = 'Lower',  metric = performance_metric, y = y_value )

    elif checkArrayValues(direction_values, 1) == True:
        y_value = compute_y( direction_label = 'Higher', goal_values = goal_values )
        h_value = compute_h( direction_label = 'Higher',  metric = performance_metric, y = y_value )

    else:
        y_value = compute_y( direction_label = 'Default', goal_values = goal_values )
        h_value = compute_h( direction_label = 'Default',  metric = performance_metric, y = y_value )

    cost += h_value
    return cost