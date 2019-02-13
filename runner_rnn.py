import os
import pickle
from collections import defaultdict

import numpy as np
import tensorflow as tf
import visualization
from data_batcher import DataBatcher
from models.rnn import RNNClassifier

# from dnn import DNNClassifer
np.set_printoptions(threshold=np.nan)


def get_data_types(custom=False):
    if custom:
        return ['custom']

    return ['specific', 'basic']


def get_optimizers(adagrad=True, adam=True, sgd=True):
    opts = defaultdict()

    if adagrad:
        opts['adagrad'] = tf.train.AdagradOptimizer

    if adam:
        opts['adam'] = tf.train.AdamOptimizer

    if sgd:
        opts['sgd'] = tf.train.GradientDescentOptimizer

    return opts


if __name__ == '__main__':

    data_types = get_data_types(custom=False)
    versions = [3, 5, 10]
    optimizers = get_optimizers(adagrad=True, adam=True, sgd=False)

    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]

    for data_type in data_types:
        for version in versions:
            print(f'######################### Running {version} sliding window #########################')
            for opt_name, optimizer in optimizers.items():
                print(f'========================= {opt_name} =========================')
                for learning_rate in learning_rates:
                    print(f'------------------------- {learning_rate} -------------------------')

                    batch_size = 100
                    iterations = 2000

                    metrics = defaultdict(list)

                    data_path = f'data/processed_data/{data_type}_movement/sliding_window_{version}_sec.p'

                    batcher = DataBatcher(data_path)
                    length, dimensions, n_labels = batcher.get_info()

                    for nc in (4, 8):
                        print(f'_________________________ {nc} Cells _________________________')

                        for cs in (16, 32):
                            print(f'......................... {cs} size .........................')

                            name = f'rnn/{opt_name}/{data_type}_movement/{str(learning_rate).replace(".", "-")}/{nc}_cells/{cs}_size/{version}_sec'

                            metric_path = os.path.join(*name.split('/')[:-1])
                            metric_name = name.split('/')[-1]

                            model = RNNClassifier(
                                batch_size=batch_size,
                                data_length=length,
                                dimensions=dimensions,
                                num_classes=n_labels,
                                num_cells=nc,
                                cell_size=cs,
                                optimizer=optimizer,
                                learning_rate=learning_rate
                            )

                            model.build_network()

                            with tf.Session() as sess:

                                writer = tf.summary.FileWriter('tensorboard/graph_rnn')
                                writer.add_graph(sess.graph)

                                training_writer = tf.summary.FileWriter(f'tensorboard/{name}/train/')
                                validation_writer = tf.summary.FileWriter(f'tensorboard/{name}/val/')

                                tf.global_variables_initializer().run()

                                for i in range(iterations + 1):
                                    training_features, training_labels = batcher.next_batch_train(batch_size)

                                    one_hot_training_labels = batcher.make_labels_one_hot(training_labels)

                                    feed_dict = {
                                        model.input_x: training_features,
                                        model.input_y: one_hot_training_labels
                                    }

                                    _, training_loss, training_accuracy, training_summary = sess.run(
                                        [model.training_step, model.loss, model.accuracy, model.merged],
                                        feed_dict
                                    )

                                    # Check validation accuracy and loss for each 100 iterations
                                    if not i % 100:
                                        test_features, test_labels = batcher.get_testing_data()

                                        one_hot_test_labels = batcher.make_labels_one_hot(test_labels)

                                        feed_dict = {
                                            model.input_x: test_features,
                                            model.input_y: one_hot_test_labels
                                        }

                                        validation_loss, validation_accuracy, validation_summary = sess.run(
                                            [model.loss, model.accuracy, model.merged],
                                            feed_dict
                                        )
                                        print(i, validation_accuracy)

                                        training_writer.add_summary(training_summary, i)
                                        validation_writer.add_summary(validation_summary, i)

                                    # Create a confusion matrix for each 500 iterations
                                    if not i % 500 and i > 0:
                                        test_features, test_labels = batcher.get_testing_data()

                                        one_hot_test_labels = batcher.make_labels_one_hot(test_labels)

                                        feed_dict = {
                                            model.input_x: test_features,
                                            model.input_y: one_hot_test_labels
                                        }

                                        predictions = sess.run(
                                            model.predictions,
                                            feed_dict
                                        )

                                        recall, precision, f1, confusion_matrix = visualization.calculate_cm(
                                            pred_vals=predictions,
                                            true_vals=test_labels,
                                            classes=list(range(n_labels))
                                        )

                                        visualization.plot_confusion_matrix(
                                            cm=confusion_matrix,
                                            classes=list(range(n_labels)),
                                            path=f'figures/{name}/',
                                            name=f'step-{i}.png',
                                            normalize=True
                                        )

                                        metrics[i] = [confusion_matrix, recall, precision, f1]

                                path = f'metrics/{metric_path}/'

                                if not os.path.exists(path):
                                    os.makedirs(path)

                                with open(f'{path}{metric_name}.p', 'wb') as bin_file:
                                    pickle.dump(metrics, bin_file)

                                print('Pickle dumped')

                            tf.reset_default_graph()
                            batcher.reset()
