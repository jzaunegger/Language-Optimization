import math


# Compute the spatial resolution from POMR V2 - 15.2
def compute_spatial_res(frequency, rel_perm):
    return 150 / (math.sqrt(rel_perm) * (frequency))

# Compute the maximum depth from POMR V2 - 15.3
def compute_max_depth(time_window, rel_perm):
    return time_window / (8.7 * math.sqrt(rel_perm))

def compute_sampling_frequency(frequency):
    return 6 * frequency

def compute_sampling_interval(sampling_frequency):
    return 1 / sampling_frequency

   # return 8.7 / (time_window * math.sqrt(rel_perm))




print(compute_spatial_res(1000, 4.59))

print(compute_max_depth(194.53, 5))


freq = 10 * (10 ** 6)
sample_freq = compute_sampling_frequency(freq)
sample_interval = compute_sampling_interval(sample_freq)

print(sample_freq)
print(sample_interval)

print("{:0.4f}".format(sample_interval / (10 ** -9)))