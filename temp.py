import RadarLib


parameters =  {
    "peak_transmisison_power": 472.68,
    "gain": 41.94,    
    "wavelength": 0.29979,                 # Given in meters
    "radar_cross_section": 1,               # Given in square meters
    "target_range": 120.15,
    "number_of_pulses": 100,                  # Unitless
    "noise_figure": 1.778,                  # Given in linear units
    "bandwidth": 1000,
    "total_loss": 0.5911,   
}

res_linear, res_db = RadarLib.compute_snr(parameters)

print(res_db)