import math, os, sys
import matplotlib.pyplot as plt

def compute_spatial_resolution(frequency, dielectric_constant):
    return 150 / ( math.sqrt(dielectric_constant) * frequency)


if __name__ == "__main__":

    material_data = {
        "clay_dry": 11,
        "clay_wet": 27.5,
        "limestone_dry": 6,
        "limestone_wet": 10.5,
        "sandstone_dry":  5.5,
        "sandstone_wet": 10,
        "shale_wet": 7.5,
        "sand_dry": 4.5,
        "sand_wet": 20,
        "soil_sandy_dry":  5,
        "soil_sandy_wet":  22.5,
        "soil_loamy_dry": 5,
        "soil_loamy_wet":  15,
        "soil_clayey_dry": 5,
        "soil_clayey_wet": 12.5
    }
    
    output_path = os.path.join('./spatial_frequencies')

    # Loop through each material
    for mat in material_data:
        current_dielectric = material_data[mat]
    
        frequency_labels, resolution_values = [], []

        # check the output path exists
        if os.path.exists(output_path) == False: os.mkdir(output_path)

        # Iterate over freequency
        for i in range(10_000_000, 1_000_000_000, 1_000_000):
            current_res = compute_spatial_resolution(i, current_dielectric)

            frequency_labels.append(i/1_000_000)
            resolution_values.append(current_res)

        plt.plot(frequency_labels, resolution_values)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Spatial Resolution')
        plt.title('Spatial Resolution for Material {}'.format(mat))
        plt.savefig(os.path.join(output_path, '{}-Chart.png'.format(mat)))
        plt.close()