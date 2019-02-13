import os
import pickle
from collections import defaultdict

import numpy as np


def get_paths(root_folder):
    """
    Creating a path dictionary for the features in the dataset.
    """
    path_dict = defaultdict(list)

    folders = os.listdir(root_folder)

    for feature in folders:
        file_names = os.listdir(os.path.join(root_folder, feature))

        for datafile in file_names:
            path_dict[feature].append(datafile)

    return path_dict


def read_file(file_path):
    data = []

    with open(file_path, 'r') as datafile:
        for line in datafile.readlines():
            data.append([float(value) for value in line.strip().split(' ')])

    return data


def extract_raw_data(root_folder, output_folder, filename):
    dictionary = defaultdict(list)
    print('Extracting Raw Data ...')

    output_file = os.path.join(output_folder, filename)

    for feature, file_names in get_paths(root_folder).items():
        data = []

        for datafile in file_names:
            path = os.path.join(root_folder, feature, datafile)
            data.append(read_file(path))

        dictionary[feature] = data
        print(f'    Completed data extraction for feature: {feature}')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pickle.dump(dictionary, open(output_file, 'wb'))

    print(f'Pickle dumped to file: {output_file}\n')


def create_sliding_windows(data, freq, window, slider):
    """
    Dividing the data into sliding windows

    :param freq: The frequency of which the accelerometer collects data
    :param window: The size of the sliding window in seconds
    :param slider: A number indicating the shifting factor of the sliding window
    """
    num = freq * window

    points = []

    for index in range(0, len(data) - num, freq * slider):
        points.append(data[index:index + num])

    return points


def prepare_data(pickle_file, freq, window, prepared_data, slider=1, func=None):
    """
    Preparing the data to fit the specification of the Deep Learning algorithms
    """
    print('Preparing data for Neural Network ...')

    with open(pickle_file, 'rb') as bin_file:
        raw_data = pickle.load(bin_file)

    key_to_int = {v: k for k, v in enumerate(raw_data.keys())}

    x_values = []
    y_values = []

    for feature, data_points in raw_data.items():

        for data in data_points:

            if callable(func):
                data = func(data)

            sliding_windows = create_sliding_windows(data, freq, window, slider=slider)
            x_values.extend(sliding_windows)
            y_values.extend([key_to_int[feature]]*len(sliding_windows))

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    print(f'    Number of sliding windows: {len(x_values)}')
    print(f'    Number of true values:     {len(y_values)}')

    pickle.dump((x_values, y_values), open(prepared_data, 'wb'))

    print(f'Pickle dumped to file: {prepared_data}\n')


def extract_adl(adl_categories, root, output_folder, filename):
    """
    Extracting Activities-of-Daily-Life categories from the dataset
    """
    dictionary = defaultdict(list)
    print('Extracting ADL Categories from Raw Data ...')

    output_file = os.path.join(output_folder, filename)

    paths = get_paths(root)

    for adl, motions in adl_categories.items():

        data = []
        for motion in motions:
            for file_name in paths[motion]:
                path = os.path.join(root, motion, file_name)
                data.append(read_file(path))

            print(f'    Completed data extraction for specific_movement: {motion}')

        dictionary[adl] = data
        print(f'    Data extracted for feature: {adl}\n')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pickle.dump(dictionary, open(output_file, 'wb'))

    print(f'Pickle dumped to file: {output_file}\n')


def extract_adl_from_raw_data(root, output_folder, filename):
    """
    Creating dictionary for Activity-of-Daily-Life --> Motion Primitives relations
    """
    motion_to_adl = {
        'Personal_hygiene': ['Brush_teeth', 'Comb_hair'],
        'Mobility': ['Climb_stairs', 'Descend_stairs', 'Walk'],
        'Feeding': ['Drink_glass', 'Pour_water', 'Eat_meat', 'Eat_soup'],
        'Communication': ['Use_telephone'],
        'Functional Transfers': ['Getup_bed', 'Liedown_bed', 'Standup_chair', 'Sitdown_chair']
    }

    extract_adl(motion_to_adl, root, output_folder, filename)


if __name__ == '__main__':
    print('Running code from read_dataset.py ... \n')
    data_folder = 'data/ucl_dataset'
    movement_type = 'specific'

    new_data = False

    raw_pickle = 'raw_data.p'
    output = f'data/processed_data/{movement_type}_movement/'

    if new_data and movement_type == 'basic':
        extract_adl_from_raw_data(data_folder, output, raw_pickle)

    elif new_data and not movement_type == 'basic':
        extract_raw_data(data_folder, output, raw_pickle)

    freq = 32  # Hz
    windows = [3, 5, 10]  # sec

    raw_data = os.path.join(output, raw_pickle)

    with open(raw_data, 'rb') as bin_file:
        d = pickle.load(bin_file)

    for window in windows:
        prep_pickle = f'data/processed_data/{movement_type}_movement/sliding_window_{window}_sec.p'
        if window == 3:
            slider = window
        else:
            slider = 1

        prepare_data(raw_data, freq, window, prep_pickle, slider=slider)
