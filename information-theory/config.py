# Path to dataset, download from:
#   https://ieee-dataport.org/open-access/car-mine-driver-pattern-dataset-extracted-can-bus
path_dataset = '/home/gean/workspace/ThisCarIsMineInf'

# Debug on screen
debug_on_screen = True

# Sample size for each driver (amount): -1 to all
sample_size = 300
# sample_size = 9700

# Size for time series to apply Entropy-Complexity measure (amount)
# hc_size = 30

# Window for theory information
inf_window_size = 300
inf_window_shift = 60
inf_window_dx = 6

# Parameter for Entropy-Complexity measure.
#   see https://arthurpessa.github.io/ordpy/_build/html/index.html#ordpy.complexity_entropy
dx, dy, taux, tauy = [3, 1, 1, 1]  # Decision Tree, SVM RBF
# dx, dy, taux, tauy = [2,1,5,1]
# dx, dy, taux, tauy = [3,1,1,3]  # kNN
# dx, dy, taux, tauy = [3,1,3,2]  # SVM
# dx, dy, taux, tauy = best_parameter(get_sample(SAMPLE_SIZE))
# dx, dy, taux, tauy = [4,3,1,1]
entropy_complexity_parameters = [dx, dy, taux, tauy]

# Number of repetitions to average measurements
num_repetitions = 3
k_fold = 5

# Size to plot figure
default_figsize = (10, 6)

# Feature names that contain the class (used to identify one driver)
# and the driver ID (used to identify each driver)
label = 'class'
driver = 'driver'

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
feature_uvarov = features
feature_uvarov_inf = []
for f in feature_uvarov:
    feature_uvarov_inf.append(f'{f}_entropy')
    feature_uvarov_inf.append(f'{f}_complexity')
# features = ALL_FEATURES

# # correlation = 0.7
# feature_included = ['accelerator_position', 'friction_torque', 'cooling_temperature', 'throttle_position_abs', 'car_speed'] # 5
# feature_excluded = ['inhale_pressure', 'engine_torque', 'long_fuel_bank', 'engine_speed'] # 4

# correlation = 0.95
feature_included = ['fuel_usage', 'accelerator_position', 'throttle_position', 'inhale_pressure',
                    'throttle_position_abs', 'engine_pressure_maintanance_time', 'reduce_block_fuel', 'block_fuel',
                    'engine_speed', 'friction_torque', 'current_fire_timing', 'cooling_temperature',
                    'engine_idle_slippage', 'engine_torque', 'calculation_overhead', 'engine_torque_min',
                    'engine_torque_max', 'fire_angle_delay_tcu', 'engine_torque_limit_tcu',
                    'engine_velocity_increase_tcu', 'target_engine_velocity_lockup', 'compressor_activation',
                    'current_gear', 'mission_oil_temp', 'wheel_velo_backleft', 'torque_converter_turbin_speed',
                    'converter_clutch', 'gear_choice', 'car_speed', 'brake_switch', 'road_slope',
                    'steering_wheel_acceleration', 'steering_wheel_angle']  # 33
feature_excluded = ['short_fuel_bank', 'long_fuel_bank', 'engine_torque_revised', 'torque_transform_coeff',
                    'standard_torque_ratio', 'torque_converter_speed', 'wheel_velo_frontleft', 'wheel_velo_backright',
                    'wheel_velo_frontright']  # 9

###################################################
# Remove aspas, quebra de linha, traço embaixo
# x.replace("'", "").replace('\n', '').replace(' ', '').replace('_', ' ').replace(',', ', ')

feature_invariance = ['flywheel_torque', 'accelerator_position_filtered', 'fire_angle_delay_tcu',
                      'latitude_acceleration', 'engine_pressure_maintanance_time', 'target_engine_velocity_lockup',
                      'clutch_check', 'flywheel_torque_revised', 'torque_transform_coeff',
                      'glow_plug_limit_request', 'logitude_acceleration', 'fuel_pressure', 'compressor_activation',
                      'brake_sylinder']  # 14
indifference = ['current_fire_timing', 'calculation_overhead', 'standard_torque_ratio', 'engine_torque_min',
                'engine_torque_max', 'engine_torque_limit_tcu', 'torque_converter_turbin_speed', 'brake_switch',
                'steering_wheel_angle', 'fuel_usage', 'inhale_pressure', 'cooling_temperature',
                'torque_converter_speed', 'accelerator_position', 'wheel_velo_frontleft', 'steering_wheel_acceleration',
                'engine_torque', 'current_gear', 'throttle_position', 'converter_clutch', 'reduce_block_fuel',
                'car_speed', 'engine_idle_slippage', 'mission_oil_temp', 'wheel_velo_backleft', 'wheel_velo_frontright',
                'throttle_position_abs', 'engine_torque_revised', 'road_slope', 'long_fuel_bank', 'engine_speed',
                'friction_torque', 'block_fuel', 'engine_velocity_increase_tcu', 'short_fuel_bank',
                'wheel_velo_backright', 'gear_choice']

all_features = ['fuel_usage', 'accelerator_position', 'throttle_position', 'short_fuel_bank', 'inhale_pressure',
                'accelerator_position_filtered', 'throttle_position_abs', 'engine_pressure_maintanance_time',
                'reduce_block_fuel', 'block_fuel', 'fuel_pressure', 'long_fuel_bank', 'engine_speed',
                'engine_torque_revised', 'friction_torque', 'flywheel_torque_revised', 'current_fire_timing',
                'cooling_temperature', 'engine_idle_slippage', 'engine_torque', 'calculation_overhead',
                'engine_torque_min', 'engine_torque_max', 'flywheel_torque', 'torque_transform_coeff',
                'standard_torque_ratio', 'fire_angle_delay_tcu', 'engine_torque_limit_tcu',
                'engine_velocity_increase_tcu', 'target_engine_velocity_lockup', 'glow_plug_limit_request',
                'compressor_activation', 'torque_converter_speed', 'current_gear', 'mission_oil_temp',
                'wheel_velo_frontleft', 'wheel_velo_backright', 'wheel_velo_frontright', 'wheel_velo_backleft',
                'torque_converter_turbin_speed', 'clutch_check', 'converter_clutch', 'gear_choice', 'car_speed',
                'logitude_acceleration', 'brake_switch', 'brake_sylinder', 'road_slope', 'latitude_acceleration',
                'steering_wheel_acceleration', 'steering_wheel_angle']

# pos literature preprocessing
feature_lit = ['calculation_overhead', 'block_fuel', 'gear_choice', 'throttle_position_abs', 'current_fire_timing',
               'steering_wheel_angle', 'fuel_usage', 'accelerator_position', 'reduce_block_fuel',
               'cooling_temperature', 'engine_torque_limit_tcu', 'brake_switch', 'torque_converter_turbin_speed',
               'throttle_position', 'engine_speed', 'mission_oil_temp', 'friction_torque', 'engine_idle_slippage',
               'engine_torque', 'engine_torque_min', 'converter_clutch', 'road_slope',
               'engine_velocity_increase_tcu', 'inhale_pressure', 'car_speed', 'current_gear', 'engine_torque_max',
               'wheel_velo_frontright', 'long_fuel_bank', 'steering_wheel_acceleration']
feature_lit_paper = ['calculation_overhead', 'block_fuel', 'gear_choice', 'throttle_position_abs',
                     'current_fire_timing', 'steering_wheel_angle', 'fuel_usage', 'accelerator_position',
                     'reduce_block_fuel', 'cooling_temperature', 'engine_torque_limit_tcu', 'brake_switch',
                     'torque_converter_turbin_speed', 'throttle_position', 'engine_speed', 'mission_oil_temp',
                     'friction_torque', 'engine_idle_slippage', 'engine_torque', 'engine_torque_min',
                     'converter_clutch', 'road_slope', 'engine_velocity_increase_tcu', 'inhale_pressure', 'car_speed',
                     'current_gear', 'engine_torque_max', 'wheel_velo_frontright', 'long_fuel_bank',
                     'steering_wheel_acceleration']

# # As features restantes abaixo foram obtidas considerando os dados de todas as viagens do motorista
# feature_lit_remaining = ['engine_idle_slippage', 'steering_wheel_angle', 'gear_choice',
#                          'engine_pressure_maintanance_time', 'compressor_activation', 'short_fuel_bank',
#                          'inhale_pressure', 'cooling_temperature', 'block_fuel', 'steering_wheel_acceleration',
#                          'current_gear', 'long_fuel_bank', 'accelerator_position', 'friction_torque',
#                          'engine_torque_limit_tcu', 'brake_switch', 'fuel_usage', 'converter_clutch',
#                          'engine_torque_max', 'standard_torque_ratio', 'torque_converter_turbin_speed',
#                          'throttle_position', 'reduce_block_fuel', 'car_speed', 'target_engine_velocity_lockup',
#                          'engine_torque_min', 'current_fire_timing', 'engine_velocity_increase_tcu', 'road_slope',
#                          'engine_torque', 'mission_oil_temp']
# feature_inf_remaining = ['short_fuel_bank_fisher', 'calculation_overhead_entropy', 'engine_torque_limit_tcu_complexity', 'calculation_overhead_fisher', 'fuel_usage_complexity', 'engine_idle_slippage_entropy', 'inhale_pressure_entropy', 'cooling_temperature_fisher', 'throttle_position_abs_entropy', 'brake_switch_fisher', 'friction_torque_entropy', 'friction_torque_fisher', 'short_fuel_bank_complexity', 'engine_torque_min_entropy', 'throttle_position_abs_fisher', 'calculation_overhead_complexity', 'steering_wheel_angle_fisher', 'steering_wheel_angle_complexity', 'inhale_pressure_complexity', 'engine_torque_min_fisher', 'fuel_usage_entropy', 'current_gear_entropy', 'converter_clutch_fisher', 'long_fuel_bank_entropy', 'brake_switch_entropy', 'short_fuel_bank_entropy', 'friction_torque_complexity', 'wheel_velo_backleft_complexity', 'road_slope_fisher', 'engine_speed_complexity', 'steering_wheel_acceleration_complexity', 'long_fuel_bank_fisher', 'current_fire_timing_complexity', 'torque_converter_speed_complexity', 'engine_torque_limit_tcu_fisher', 'engine_idle_slippage_complexity', 'engine_speed_entropy', 'current_fire_timing_fisher', 'steering_wheel_angle_entropy', 'engine_idle_slippage_fisher', 'engine_torque_min_complexity', 'cooling_temperature_entropy', 'wheel_velo_backleft_entropy', 'steering_wheel_acceleration_entropy', 'engine_torque_limit_tcu_entropy', 'current_gear_complexity', 'road_slope_complexity', 'brake_switch_complexity', 'road_slope_entropy', 'current_fire_timing_entropy', 'cooling_temperature_complexity', 'long_fuel_bank_complexity', 'current_gear_fisher', 'inhale_pressure_fisher', 'fuel_usage_fisher', 'torque_converter_speed_fisher', 'wheel_velo_backleft_fisher', 'engine_speed_fisher', 'steering_wheel_acceleration_fisher', 'torque_converter_speed_entropy', 'throttle_position_abs_complexity', 'converter_clutch_entropy', 'converter_clutch_complexity']
# feature_inf_remaining_hc = ['calculation_overhead_entropy', 'engine_torque_limit_tcu_complexity', 'fuel_usage_complexity', 'engine_idle_slippage_entropy', 'inhale_pressure_entropy', 'throttle_position_abs_entropy', 'friction_torque_entropy', 'short_fuel_bank_complexity', 'engine_torque_min_entropy', 'calculation_overhead_complexity', 'steering_wheel_angle_complexity', 'inhale_pressure_complexity', 'fuel_usage_entropy', 'current_gear_entropy', 'long_fuel_bank_entropy', 'brake_switch_entropy', 'short_fuel_bank_entropy', 'friction_torque_complexity', 'wheel_velo_backleft_complexity', 'engine_speed_complexity', 'steering_wheel_acceleration_complexity', 'current_fire_timing_complexity', 'torque_converter_speed_complexity', 'engine_idle_slippage_complexity', 'engine_speed_entropy', 'steering_wheel_angle_entropy', 'engine_torque_min_complexity', 'cooling_temperature_entropy', 'wheel_velo_backleft_entropy', 'steering_wheel_acceleration_entropy', 'engine_torque_limit_tcu_entropy', 'current_gear_complexity', 'road_slope_complexity', 'brake_switch_complexity', 'road_slope_entropy', 'current_fire_timing_entropy', 'cooling_temperature_complexity', 'long_fuel_bank_complexity', 'torque_converter_speed_entropy', 'throttle_position_abs_complexity', 'converter_clutch_entropy', 'converter_clutch_complexity']
# feature_inf_remaining_fs = ['short_fuel_bank_fisher', 'calculation_overhead_entropy', 'calculation_overhead_fisher', 'engine_idle_slippage_entropy', 'inhale_pressure_entropy', 'cooling_temperature_fisher', 'throttle_position_abs_entropy', 'brake_switch_fisher', 'friction_torque_entropy', 'friction_torque_fisher', 'engine_torque_min_entropy', 'throttle_position_abs_fisher', 'steering_wheel_angle_fisher', 'engine_torque_min_fisher', 'fuel_usage_entropy', 'current_gear_entropy', 'converter_clutch_fisher', 'long_fuel_bank_entropy', 'brake_switch_entropy', 'short_fuel_bank_entropy', 'road_slope_fisher', 'long_fuel_bank_fisher', 'engine_torque_limit_tcu_fisher', 'engine_speed_entropy', 'current_fire_timing_fisher', 'steering_wheel_angle_entropy', 'engine_idle_slippage_fisher', 'cooling_temperature_entropy', 'wheel_velo_backleft_entropy', 'steering_wheel_acceleration_entropy', 'engine_torque_limit_tcu_entropy', 'road_slope_entropy', 'current_fire_timing_entropy', 'current_gear_fisher', 'inhale_pressure_fisher', 'fuel_usage_fisher', 'torque_converter_speed_fisher', 'wheel_velo_backleft_fisher', 'engine_speed_fisher', 'steering_wheel_acceleration_fisher', 'torque_converter_speed_entropy', 'converter_clutch_entropy']


# # As features restantes abaixo foram obtidas considerando os dados da primeira viagem de cada motorista
##Correlation 0.9
# feature_lit_remaining = ['short_fuel_bank', 'engine_torque_min', 'long_fuel_bank', 'standard_torque_ratio', 'current_gear', 'block_fuel', 'mission_oil_temp', 'road_slope', 'brake_switch', 'steering_wheel_acceleration', 'friction_torque', 'steering_wheel_angle', 'reduce_block_fuel', 'engine_velocity_increase_tcu', 'engine_speed', 'current_fire_timing', 'inhale_pressure', 'gear_choice', 'engine_idle_slippage', 'cooling_temperature', 'converter_clutch', 'fuel_usage', 'torque_converter_speed', 'engine_torque_limit_tcu', 'wheel_velo_backleft']
# feature_inf_remaining = ['engine_torque_min_complexity', 'current_gear_complexity', 'engine_idle_slippage_fisher', 'long_fuel_bank_fisher', 'long_fuel_bank_entropy', 'steering_wheel_angle_complexity', 'short_fuel_bank_entropy', 'cooling_temperature_entropy', 'friction_torque_entropy', 'engine_torque_limit_tcu_entropy', 'short_fuel_bank_fisher', 'engine_speed_entropy', 'short_fuel_bank_complexity', 'engine_torque_limit_tcu_complexity', 'engine_idle_slippage_entropy', 'engine_idle_slippage_complexity', 'engine_speed_complexity', 'torque_converter_speed_entropy', 'brake_switch_complexity', 'fuel_usage_entropy', 'long_fuel_bank_complexity', 'steering_wheel_acceleration_fisher', 'torque_converter_speed_fisher', 'friction_torque_complexity', 'road_slope_complexity', 'current_gear_fisher', 'brake_switch_fisher', 'steering_wheel_angle_entropy', 'engine_speed_fisher', 'inhale_pressure_entropy', 'current_gear_entropy', 'current_fire_timing_fisher', 'road_slope_entropy', 'cooling_temperature_fisher', 'converter_clutch_entropy', 'fuel_usage_fisher', 'wheel_velo_backleft_entropy', 'engine_torque_min_fisher', 'cooling_temperature_complexity', 'brake_switch_entropy', 'steering_wheel_angle_fisher', 'converter_clutch_complexity', 'converter_clutch_fisher', 'current_fire_timing_entropy', 'wheel_velo_backleft_fisher', 'steering_wheel_acceleration_entropy', 'steering_wheel_acceleration_complexity', 'fuel_usage_complexity', 'torque_converter_speed_complexity', 'engine_torque_limit_tcu_fisher', 'inhale_pressure_complexity', 'current_fire_timing_complexity', 'engine_torque_min_entropy', 'friction_torque_fisher', 'inhale_pressure_fisher', 'wheel_velo_backleft_complexity', 'road_slope_fisher']
#Correlation 0.9
feature_lit_remaining = ['block_fuel', 'steering_wheel_acceleration', 'engine_velocity_increase_tcu', 'road_slope', 'cooling_temperature', 'friction_torque', 'current_gear', 'engine_torque_limit_tcu', 'engine_torque_min', 'short_fuel_bank', 'reduce_block_fuel', 'converter_clutch', 'calculation_overhead', 'mission_oil_temp', 'long_fuel_bank', 'standard_torque_ratio', 'gear_choice', 'steering_wheel_angle', 'current_fire_timing', 'brake_switch', 'throttle_position_abs', 'engine_torque_max', 'engine_idle_slippage']
feature_inf_remaining = ['engine_torque_min_complexity', 'engine_torque_min_entropy', 'engine_torque_max_entropy', 'engine_torque_max_complexity', 'engine_idle_slippage_fisher', 'throttle_position_abs_entropy', 'engine_idle_slippage_entropy', 'long_fuel_bank_entropy', 'short_fuel_bank_entropy', 'short_fuel_bank_complexity', 'cooling_temperature_complexity', 'current_gear_entropy', 'cooling_temperature_fisher', 'steering_wheel_angle_fisher', 'engine_idle_slippage_complexity', 'throttle_position_abs_fisher', 'current_gear_fisher', 'engine_torque_limit_tcu_entropy', 'steering_wheel_acceleration_entropy', 'steering_wheel_angle_complexity', 'current_fire_timing_fisher', 'calculation_overhead_complexity', 'calculation_overhead_fisher', 'brake_switch_entropy', 'brake_switch_fisher', 'steering_wheel_acceleration_complexity', 'engine_torque_limit_tcu_complexity', 'engine_torque_limit_tcu_fisher', 'long_fuel_bank_fisher', 'engine_torque_min_fisher', 'converter_clutch_complexity', 'calculation_overhead_entropy', 'current_gear_complexity', 'long_fuel_bank_complexity', 'friction_torque_complexity', 'current_fire_timing_complexity', 'road_slope_fisher', 'road_slope_complexity', 'friction_torque_entropy', 'steering_wheel_acceleration_fisher', 'steering_wheel_angle_entropy', 'brake_switch_complexity', 'engine_torque_max_fisher', 'friction_torque_fisher', 'current_fire_timing_entropy', 'cooling_temperature_entropy', 'short_fuel_bank_fisher', 'road_slope_entropy', 'converter_clutch_entropy', 'converter_clutch_fisher', 'throttle_position_abs_complexity']
feature_inf_remaining_hc = []
feature_inf_remaining_fs = []
for i in feature_inf_remaining:
    if '_entropy' in i:
        feature_inf_remaining_hc.append(i)
        feature_inf_remaining_fs.append(i)
    else:
        if '_complexity' in i:
            feature_inf_remaining_hc.append(i)
        if '_fisher' in i:
            feature_inf_remaining_fs.append(i)

# pos inf measure

feature_inf_hc = ['calculation_overhead_entropy', 'calculation_overhead_complexity', 'throttle_position_abs_entropy',
                  'throttle_position_abs_complexity', 'current_fire_timing_entropy', 'current_fire_timing_complexity',
                  'steering_wheel_angle_entropy', 'steering_wheel_angle_complexity', 'fuel_usage_entropy',
                  'fuel_usage_complexity', 'accelerator_position_entropy', 'accelerator_position_complexity',
                  'cooling_temperature_entropy', 'cooling_temperature_complexity',
                  # 'engine_torque_limit_tcu_entropy',
                  'brake_switch_entropy', 'brake_switch_complexity', 'torque_converter_turbin_speed_entropy',
                  'torque_converter_turbin_speed_complexity', 'throttle_position_entropy',
                  'throttle_position_complexity', 'engine_speed_entropy', 'engine_speed_complexity',
                  'friction_torque_entropy', 'friction_torque_complexity', 'engine_idle_slippage_entropy',
                  'engine_idle_slippage_complexity', 'engine_torque_entropy', 'engine_torque_complexity',
                  'engine_torque_min_entropy', 'engine_torque_min_complexity', 'road_slope_entropy',
                  'road_slope_complexity', 'inhale_pressure_entropy', 'inhale_pressure_complexity',
                  'car_speed_entropy', 'car_speed_complexity', 'engine_torque_max_entropy',
                  'engine_torque_max_complexity', 'wheel_velo_frontright_entropy', 'wheel_velo_frontright_complexity',
                  'long_fuel_bank_entropy', 'long_fuel_bank_complexity', 'steering_wheel_acceleration_entropy',
                  'steering_wheel_acceleration_complexity']
feature_inf = feature_inf_hc

feature_inf_120 = ['calculation_overhead_entropy', 'calculation_overhead_complexity', 'throttle_position_abs_entropy',
                   'throttle_position_abs_complexity', 'current_fire_timing_entropy', 'current_fire_timing_complexity',
                   'steering_wheel_angle_entropy', 'steering_wheel_angle_complexity', 'fuel_usage_entropy',
                   'fuel_usage_complexity', 'accelerator_position_entropy', 'accelerator_position_complexity',
                   'cooling_temperature_entropy', 'cooling_temperature_complexity',
                   # 'engine_torque_limit_tcu_entropy',
                   'torque_converter_turbin_speed_entropy',
                   'torque_converter_turbin_speed_complexity', 'throttle_position_entropy',
                   'throttle_position_complexity', 'engine_speed_entropy', 'engine_speed_complexity',
                   'friction_torque_entropy', 'friction_torque_complexity', 'engine_idle_slippage_entropy',
                   'engine_idle_slippage_complexity', 'engine_torque_entropy', 'engine_torque_complexity',
                   'engine_torque_min_entropy', 'engine_torque_min_complexity', 'road_slope_entropy',
                   'road_slope_complexity', 'inhale_pressure_entropy', 'inhale_pressure_complexity',
                   'car_speed_entropy', 'car_speed_complexity', 'engine_torque_max_entropy',
                   'engine_torque_max_complexity', 'wheel_velo_frontright_entropy', 'wheel_velo_frontright_complexity',
                   'long_fuel_bank_entropy', 'long_fuel_bank_complexity', 'steering_wheel_acceleration_entropy',
                   'steering_wheel_acceleration_complexity']

feature_inf_fs = ['calculation_overhead_fisher', 'calculation_overhead_shannon', 'throttle_position_abs_fisher',
                  'throttle_position_abs_shannon', 'current_fire_timing_fisher', 'current_fire_timing_shannon',
                  'steering_wheel_angle_fisher', 'steering_wheel_angle_shannon', 'fuel_usage_fisher',
                  'fuel_usage_shannon', 'accelerator_position_fisher', 'accelerator_position_shannon',
                  'cooling_temperature_fisher', 'cooling_temperature_shannon',
                  # 'engine_torque_limit_tcu_fisher',
                  'brake_switch_fisher', 'brake_switch_shannon', 'torque_converter_turbin_speed_fisher',
                  'torque_converter_turbin_speed_shannon', 'throttle_position_fisher',
                  'throttle_position_shannon', 'engine_speed_fisher', 'engine_speed_shannon',
                  'friction_torque_fisher', 'friction_torque_shannon', 'engine_idle_slippage_fisher',
                  'engine_idle_slippage_shannon', 'engine_torque_fisher', 'engine_torque_shannon',
                  'engine_torque_min_fisher', 'engine_torque_min_shannon', 'road_slope_fisher',
                  'road_slope_shannon', 'inhale_pressure_fisher', 'inhale_pressure_shannon', 'car_speed_fisher',
                  'car_speed_shannon', 'engine_torque_max_fisher', 'engine_torque_max_shannon',
                  'wheel_velo_frontright_fisher', 'wheel_velo_frontright_shannon', 'long_fuel_bank_fisher',
                  'long_fuel_bank_shannon', 'steering_wheel_acceleration_fisher',
                  'steering_wheel_acceleration_shannon']

feature_inf_hcfs = ['throttle_position_abs_complexity', 'throttle_position_abs_entropy',
                    'throttle_position_abs_fisher', 'engine_torque_max_complexity', 'engine_torque_max_entropy',
                    'engine_torque_max_fisher', 'accelerator_position_complexity', 'accelerator_position_entropy',
                    'accelerator_position_fisher', 'calculation_overhead_complexity',
                    'calculation_overhead_entropy', 'calculation_overhead_fisher',
                    'torque_converter_turbin_speed_complexity', 'torque_converter_turbin_speed_entropy',
                    'torque_converter_turbin_speed_fisher', 'throttle_position_complexity',
                    'throttle_position_entropy', 'throttle_position_fisher', 'engine_idle_slippage_complexity',
                    'engine_idle_slippage_entropy', 'engine_idle_slippage_fisher', 'car_speed_complexity',
                    'car_speed_entropy', 'car_speed_fisher', 'long_fuel_bank_complexity', 'long_fuel_bank_entropy',
                    'long_fuel_bank_fisher', 'inhale_pressure_complexity', 'inhale_pressure_entropy',
                    'inhale_pressure_fisher', 'friction_torque_complexity', 'friction_torque_entropy',
                    'friction_torque_fisher', 'fuel_usage_complexity', 'fuel_usage_entropy', 'fuel_usage_fisher',
                    'brake_switch_complexity', 'brake_switch_entropy', 'brake_switch_fisher',
                    'engine_torque_complexity', 'engine_torque_entropy', 'engine_torque_fisher',
                    'wheel_velo_frontright_complexity', 'wheel_velo_frontright_entropy',
                    'wheel_velo_frontright_fisher', 'engine_speed_complexity', 'engine_speed_entropy',
                    'engine_speed_fisher', 'steering_wheel_angle_complexity', 'steering_wheel_angle_entropy',
                    'steering_wheel_angle_fisher', 'road_slope_complexity', 'road_slope_entropy',
                    'road_slope_fisher', 'steering_wheel_acceleration_complexity',
                    'steering_wheel_acceleration_entropy', 'steering_wheel_acceleration_fisher',
                    'current_fire_timing_complexity', 'current_fire_timing_entropy', 'current_fire_timing_fisher',
                    'engine_torque_min_complexity', 'engine_torque_min_entropy', 'engine_torque_min_fisher',
                    'cooling_temperature_complexity', 'cooling_temperature_entropy', 'cooling_temperature_fisher']
