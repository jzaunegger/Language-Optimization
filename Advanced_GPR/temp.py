import math, os
import matplotlib.pyplot as plt

# Run the main script
if __name__ == "__main__":

    output_folder = "./images"
    frequencies, spatial_resolutions = [], []
    depths, time_window = [], []

    depths_alt, times_alt = [], []

    # Dry Sand ~4.5
    dielectric_constant = 4.5
    max_depth = [1, 50]


    # Evalaute the possible frequencies
    for i in range(100, 10000):

        spatial_resolution_in_m = i / 1000

        freq = 150 / (spatial_resolution_in_m * math.sqrt(dielectric_constant))

        frequencies.append(freq)
        spatial_resolutions.append(spatial_resolution_in_m)

    plt.plot(spatial_resolutions, frequencies)
    plt.xlabel("Spatial Resolution (m)")
    plt.ylabel("Frequency (MHz)")
    plt.savefig( os.path.join(output_folder, 'Frequency_Chart.png') )
    plt.close()


    # Evaluate time window
    for i in range( max_depth[0], max_depth[1]):
        time = 8.7 * i * math.sqrt(dielectric_constant)
        depths.append(i)
        time_window.append(time)

    plt.plot(depths, time_window)
    plt.xlabel("Max Depth (m)")
    plt.ylabel("Time Window (s)")
    plt.savefig( os.path.join(output_folder, 'Time_Chart.png') )
    plt.close()



    for i in range(1, 1000):
        current_depth = i / 1000
        time = 8.7 * current_depth * math.sqrt(dielectric_constant)
        depths_alt.append(current_depth)
        times_alt.append(time)

    plt.plot(depths_alt, times_alt)
    plt.xlabel("Max Depth (m)")
    plt.ylabel("Time Window (s)")
    plt.savefig( os.path.join(output_folder, 'Time_Chart_2.png') )
    plt.close()