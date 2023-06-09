# Path to dataset, download from:
#   https://ieee-dataport.org/open-access/car-mine-driver-pattern-dataset-extracted-can-bus
path_dataset = '/home/gean/workspace/ThisCarIsMine/'

# Debug on screen
debug_on_screen = False

# Sample size for each driver (amount): -1 to all
sample_size = -1
# sample_size = 9700

# Size for time series to apply Entropy-Complexity measure (amount)
hc_size = 20

# Parameter for Entropy-Complexity measure.
#   see https://arthurpessa.github.io/ordpy/_build/html/index.html#ordpy.complexity_entropy
dx, dy, taux, tauy = [3,1,1,1]  # Decision Tree, SVM RBF
# dx, dy, taux, tauy = [2,1,5,1]
# dx, dy, taux, tauy = [3,1,1,3]  # kNN
# dx, dy, taux, tauy = [3,1,3,2]  # SVM
# dx, dy, taux, tauy = best_parameter(get_sample(SAMPLE_SIZE))
# dx, dy, taux, tauy = [4,3,1,1]
entropy_complexity_parameters = [dx, dy, taux, tauy]

# Number of repetitions to average measurements
num_repetitions = 3


# DON'T EDIT THIS LIST
ALL_FEATURES = [
    'fuel_usage',
    'accelerator_position',
    'throttle_position',
    'short_fuel_bank',
    'inhale_pressure',
    'accelerator_position_filtered',
    'throttle_position_abs',
    'engine_pressure_maintanance_time',
    'reduce_block_fuel',
    'block_fuel',
    'fuel_pressure',
    'long_fuel_bank',
    'engine_speed',
    'engine_torque_revised',
    'friction_torque',
    'flywheel_torque_revised',
    'current_fire_timing',
    'cooling_temperature',
    'engine_idle_slippage',
    'engine_torque',
    'calculation_overhead',
    'engine_torque_min',
    'engine_torque_max',
    'flywheel_torque',
    'torque_transform_coeff',
    'standard_torque_ratio',
    'fire_angle_delay_tcu',
    'engine_torque_limit_tcu',
    'engine_velocity_increase_tcu',
    'target_engine_velocity_lockup',
    'glow_plug_limit_request',
    'compressor_activation',
    'torque_converter_speed',
    'current_gear',
    'mission_oil_temp',
    'wheel_velo_frontleft',
    'wheel_velo_backright',
    'wheel_velo_frontright',
    'wheel_velo_backleft',
    'torque_converter_turbin_speed',
    'clutch_check',
    'converter_clutch',
    'gear_choice',
    'car_speed',
    'logitude_acceleration',
    'brake_switch',
    'brake_sylinder',
    'road_slope',
    'latitude_acceleration',
    'steering_wheel_acceleration',
    'steering_wheel_angle'
]

# features to use
features = [
    'accelerator_position',
    'inhale_pressure',
    'throttle_position_abs',
    'long_fuel_bank',
    'engine_speed',
    'friction_torque',
    'cooling_temperature',
    'engine_torque',
    'car_speed'
]
features = ALL_FEATURES