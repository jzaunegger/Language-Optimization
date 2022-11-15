import math, os, sys
import matplotlib.pyplot as plt

def compute_max_depth(time_window, dielectric_constant):
    return 8.7 / (time_window * math.sqrt(dielectric_constant))

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
    
    output_path = os.path.join('./max_depth')

    # Loop through each material
    for mat in material_data:
        current_dielectric = material_data[mat]
    
        time_labels, depth_values = [], []

        # check the output path exists
        if os.path.exists(output_path) == False: os.mkdir(output_path)


        current_time_window = 0.1

        while current_time_window <= 3:

            current_depth = compute_max_depth(current_time_window, current_dielectric)

            time_labels.append(current_time_window)
            depth_values.append(current_depth)

            current_time_window += 0.1

        plt.plot(time_labels, depth_values)
        plt.xlabel('Time Window (s)')
        plt.ylabel('Maximum Depth (m)')
        plt.title('Maximum Depth for Material {}'.format(mat))
        plt.savefig(os.path.join(output_path, '{}-Chart.png'.format(mat)))
        plt.close()