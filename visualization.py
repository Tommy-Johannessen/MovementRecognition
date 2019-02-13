import itertools
import os

from collections import defaultdict

import matplotlib.pyplot as plt

#plt.style.use('ggplot')

from matplotlib.ticker import FuncFormatter
import pickle
import os
import numpy as np


def calculate_cm(pred_vals, true_vals, classes):
    """
    This function calculates the confusion matrix.
    """
    if len(pred_vals) != len(true_vals):
        raise ValueError("Dimensions do not match")

    n_classes = len(classes)
    d = [[0 for _ in range(n_classes)] for _ in range(n_classes)]

    for guess, ground_truth in zip(pred_vals, true_vals):
        d[ground_truth][guess] += 1

    d = np.asarray(d)
    recall = []
    precison = []
    f1 = []
    for index, values in enumerate(d):
        recall.append(0 if sum(values) == 0 else values[index] / sum(values))

    for index, values in enumerate(d.transpose()):
        precison.append(0 if sum(values) == 0 else values[index] / sum(values))

    for r, p in zip(recall, precison):
        f1.append((r + p)/2)

    return recall, precison, f1, d


def plot_confusion_matrix(cm, classes, path, name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(12, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path + name)
    plt.clf()
    plt.close()


def plot_data_distribution(filename, move_type='basic', is_sliding_window=False):
    image_folder = os.path.join('figures', 'data_distribution')
    figure_name = f'{move_type}_{filename}.png'
    data_folder = 'data/processed_data'
    movement_type = f'{move_type}_movement'

    pickle_file = os.path.join(data_folder, movement_type, f'{filename}.p')

    with open(pickle_file, 'rb') as bin_file:
        data = pickle.load(bin_file)

    x_labels = []
    y_labels = []

    if is_sliding_window:
        sliding_windows, categories = data
        data = defaultdict(list)

        for category, sliding_window in zip(categories, sliding_windows):
            data[category].append([sliding_window.tolist()])

    for category, data_lists in data.items():
        data_points_count = 0

        for data_list in data_lists:
            data_points_count += len(data_list)

        x_labels.append(category)
        y_labels.append(data_points_count)

    x_labels = np.arange(len(x_labels))

    fig, ax = plt.subplots()
    formatter = FuncFormatter(lambda x, p: format(int(x), ','))
    ax.yaxis.set_major_formatter(formatter)
    plt.title(f'Data distribution for {move_type} {filename.split("_")[0]} {filename.split("_")[1]}')
    plt.ylabel('Number of data elements')
    plt.xlabel('Movement Categories')
    plt.bar(x_labels, y_labels)
    plt.xticks(x_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, figure_name))
    plt.clf()
    plt.close()


if __name__ == '__main__':

    search_folder = 'data/processed_data'

    for folder in os.listdir(search_folder):
        if folder == 'custom_movement':
            for file in os.listdir(os.path.join(search_folder, folder)):
                plot_data_distribution(file.split('.')[0],
                                       folder.split('_')[0],
                                       True if 'sliding_window' in file else False)
        else:
            print(f'Image created for {folder} at an earlier stage')
