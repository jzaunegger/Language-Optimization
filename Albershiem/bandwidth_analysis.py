import matplotlib.pyplot as plt
import os

speed_of_light = 3 * (10 ** 8)

bandwidths, range_resolutions = [], []
for i in range(10, 900):

    bandwidth = i * (10 ** 6)
    range_res = speed_of_light / (2 * bandwidth)
    bandwidths.append(i)
    range_resolutions.append(range_res)

plt.plot(bandwidths, range_resolutions)
plt.xlabel('Bandwidth (MHz)')
plt.ylabel('Range Resolution')
plt.title('Bandwidth vs. Range Resolution')
plt.savefig(os.path.join('graphs', 'Range-Resolution-Chart.png'))
plt.close()