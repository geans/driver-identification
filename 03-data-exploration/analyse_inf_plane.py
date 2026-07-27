import json
import logging
import math
import multiprocessing
import os

import config

# from datetime import timedelta
import ordpy
import utils
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from ordpy import maximum_complexity_entropy, minimum_complexity_entropy

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def hc_limits(dx):
    m = 10
    hc_max = maximum_complexity_entropy(dx=dx, m=m)
    size = (math.factorial(dx) - 1) * m
    hc_min = minimum_complexity_entropy(dx=dx, size=size)
    #
    h_max = [x[0] for x in hc_max]
    c_max = [x[1] for x in hc_max]
    h_min = [x[0] for x in hc_min]
    c_min = [x[1] for x in hc_min]
    #
    return (h_min, c_min), (h_max, c_max)


def plot_limits(hc_min, hc_max):
    plt.plot(hc_min[0], hc_min[1], color="black", linewidth=0.8)
    plt.plot(hc_max[0], hc_max[1], color="black", linewidth=0.8)


def analyse_inf_plane(
        feature, 
        dataset_path, 
        path_to_save, 
        plane, 
        dx, 
        split_series=False
    ):
    logger.info(f'Feature: {feature}. Analysing Information Plane')

    os.makedirs(path_to_save, exist_ok=True)
    data_dict = multiprocessing.Manager().dict(
        {
            'A': ([], [], [], []),  # Entropy, Complexity, Fisher, Shannon
            'B': ([], [], [], []),
            'C': ([], [], [], []),
            'D': ([], [], [], [])
        }
    )
    
    def information_measure(series, driver, feature):
        if split_series:
            series_list = utils.split_data_to_window(series, 120, 60)
        else:
            series_list = [series]
        h_list_local, c_list_local, f_list_local, s_list_local = [], [], [], []
        for _series in series_list:
            _series = utils.preprocessing_to_hc(_series)
            try:
                if 'hc' in plane:
                    h, c = ordpy.complexity_entropy(_series, dx=dx)
                    h_list_local.append(h)
                    c_list_local.append(c)
                if 'fs' in plane:
                    s, f = ordpy.fisher_shannon(_series, dx=dx)
                    f_list_local.append(f)
                    s_list_local.append(s)
            except Exception as e:
                logger.warning(
                    f'Values: len(_series)={len(_series)}, dx={dx}, feature={feature}. '
                    f'What happened: {e}.'
                )
        h_list, c_list, f_list, s_list = data_dict[driver]
        data_dict[driver] = (h_list+h_list_local, c_list+c_list_local, f_list+f_list_local, s_list+s_list_local)
    
    data_arr = utils.get_data_list(dataset_path)
    process = []
    for df, driver in data_arr:
        p = multiprocessing.Process(target=information_measure,
                                    args=(df[feature], driver, feature))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    with open(f'{path_to_save}/data_dict_{feature}.json', 'w') as file:
        json.dump(dict(data_dict), file)
    return data_dict



def plot_inf_plane(feature, path_to_save, plane, loc1, loc2, zoom_position, dx=4):
    logger.info(f'Feature: {feature}. Plot Information Plane')

    with open(f'{path_to_save}/data_dict_{feature}.json', 'r') as file:
        data_dict = json.load(file)
    hc_min, hc_max = hc_limits(dx)
    #
    font_size = 24
    legend_font_size = 18
    label_size = 26
    zm_label_size = 20
    # index for dictionary
    H, C, F, S = 0, 1, 2, 3  # Entropy, Complexity, Fisher, Shannon
    #
    indicate_line_color = 'red' # 0.5
    if 'hc' in plane:
        fig_name = f'hc_plan__{feature}'
        fig, ax = plt.subplots(figsize=config.default_figsize)
        zm = ax.inset_axes(zoom_position)
        # zm = ax.inset_axes(
        #     width="35%",
        #     height="35%",
        #     loc='best'
        # )
        x1, x2 = None, None
        y1, y2 = None, None
        for driver, marker in zip('ABCD', 'o^dv'):
            h_list = data_dict[driver][H]
            c_list = data_dict[driver][C]
            limit = min(len(h_list), len(c_list))
            h_list= h_list[:limit]
            c_list= c_list[:limit]
            ax.scatter(h_list, c_list, label=f'driver {driver}', s=100, marker=marker)
            zm.scatter(h_list, c_list, label=f'driver {driver}', s=100, marker=marker)

            if x1 is not None:
                h_list.extend([x1, x2])
                c_list.extend([y1, y2])
            x1, x2 = min(h_list), max(h_list)
            y1, y2 = min(c_list), max(c_list)

        margin_x = (x2 - x1) * 0.2
        margin_y = (y2 - y1) * 0.2
        x1 -= margin_x
        x2 += margin_x
        y1 -= margin_y
        y2 += margin_y
        zm.set_xlim((x1, x2))
        zm.set_ylim((y1, y2))
        zm.tick_params(labelsize=zm_label_size)
        mark_inset(
            ax, 
            zm, 
            loc1=loc1, 
            loc2=loc2, 
            fc="none", 
            ec=indicate_line_color
        )

        ax.plot(hc_min[0], hc_min[1], color="black", linewidth=0.8)
        ax.plot(hc_max[0], hc_max[1], color="black", linewidth=0.8)
        ax.legend(fontsize=legend_font_size)
        ax.set_xlabel('Permutation entropy, $H$', fontsize=font_size)
        ax.set_ylabel('Statistical complexity, $C$', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)

        plt.subplots_adjust(bottom=.2, left=.2)
        plt.savefig(f'{path_to_save}/{fig_name}.png')
    #
    if 'fs' in plane:
        fig_name = f'fs_plan__{feature}'
        fig, ax = plt.subplots(figsize=config.default_figsize)
        zm = ax.inset_axes(zoom_position)
        # zm = ax.inset_axes(
        #     width="35%",
        #     height="35%",
        #     loc='best'
        # )
        x1, x2 = None, None
        y1, y2 = None, None
        for driver, marker in zip('ABCD', 'o^dv'):
            f_list = data_dict[driver][F]
            s_list = data_dict[driver][S]
            limit = min(len(f_list), len(s_list))
            f_list = f_list[:limit]
            s_list = s_list[:limit]

            ax.scatter(s_list, f_list, label=f'driver {driver}', s=100, marker=marker)
            zm.scatter(s_list, f_list, label=f'driver {driver}', s=100, marker=marker)

            if x1 is not None:
                f_list.extend([y1, y2])
                s_list.extend([x1, x2])
            x1, x2 = min(s_list), max(s_list)
            y1, y2 = min(f_list), max(f_list)

        margin_x = (x2 - x1) * 0.2
        margin_y = (y2 - y1) * 0.2
        x1 -= margin_x
        x2 += margin_x
        y1 -= margin_y
        y2 += margin_y
        zm.set_xlim((x1, x2))
        zm.set_ylim((y1, y2))
        zm.tick_params(labelsize=zm_label_size)
        mark_inset(
            ax, 
            zm, 
            loc1=loc1, 
            loc2=loc2, 
            fc="none", 
            ec=indicate_line_color
        )

        ax.legend(fontsize=legend_font_size)
        ax.set_ylim((0, 1))
        ax.set_xlim((0, 1))
        ax.set_xlabel('Shannon entropy, $S$', fontsize=font_size)
        ax.set_ylabel('Fisher entropy, $F$', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        plt.subplots_adjust(bottom=.2, left=.2)
        plt.savefig(f'{path_to_save}/{fig_name}.png')


if __name__ == '__main__':
    # INF THEORY PLAN
    dx = 4
    path_to_save = 'results/analyse_inf_plane'
    for feature, loc1, loc2, zoom_position in (
        (
            'accelerator_position', 1, 3, (0.1, 0.4, 0.45, 0.45)
        ),
        (
            'steering_wheel_angle', 1, 3, (0.1, 0.6, 0.45, 0.45)
        ),
        (
            'car_speed', 2, 4, (0.1, 0.1, 0.45, 0.45)
        
        ),
        (
            'cooling_temperature', 2, 4, (0.1, 0.1, 0.45, 0.45)
        )
    ):
        analyse_inf_plane(
            feature=feature, 
            dataset_path="../02-transformation/02.1-dataset-processed/multiprocessing/720/ThisCarIsMine", 
            path_to_save=path_to_save,
            plane='hc', 
            dx=dx
        )
        plot_inf_plane(
            feature=feature, 
            path_to_save=path_to_save, 
            plane='hc', 
            dx=dx,
            loc1=loc1, 
            loc2=loc2, 
            zoom_position=zoom_position
        )
    for feature, loc1, loc2, zoom_position in (
        (
            'accelerator_position', 1, 3, (0.1, 0.5, 0.45, 0.45)
        ),
        (
            'steering_wheel_angle', 1, 3, (0.1, 0.5, 0.45, 0.45)
        ),
        (
            'car_speed', 1, 3, (0.1, 0.5, 0.45, 0.45)
        
        ),
        (
            'cooling_temperature', 2, 4, (0.1, 0.1, 0.45, 0.45)
        )
    ):
        analyse_inf_plane(
            feature=feature, 
            dataset_path="../02-transformation/02.1-dataset-processed/multiprocessing/720/ThisCarIsMine", 
            path_to_save=path_to_save,
            plane='fs', 
            dx=dx
        )
        plot_inf_plane(
            feature=feature, 
            path_to_save=path_to_save, 
            plane='fs', 
            dx=dx,
            loc1=loc1, 
            loc2=loc2, 
            zoom_position=zoom_position
        )