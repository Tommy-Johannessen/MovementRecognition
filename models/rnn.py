import tensorflow as tf


class RNNClassifier:

    def __init__(self, batch_size, data_length, dimensions, num_classes, num_cells, cell_size, optimizer, learning_rate=0.01):
        self.data_length = data_length
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.cell_state_size = cell_size
        self.num_rnn_layers = num_cells
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimizer = optimizer

    def cell(self):
        return tf.nn.rnn_cell.GRUCell(self.cell_state_size, activation=tf.nn.relu, name='GRU-Cell')

    def build_network(self):
        self.input_x = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.data_length, self.dimensions],
            name='input_features'
        )

        self.input_y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.num_classes],
            name='input_labels'
        )

        with tf.name_scope('rnn'):
            cell = tf.nn.rnn_cell.GRUCell(self.cell_state_size, activation=tf.nn.relu, name='GRU-Cell')

            if self.num_rnn_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([self.cell() for _ in range(self.num_rnn_layers)])

            outputs, state = tf.nn.dynamic_rnn(cell, self.input_x, dtype=tf.float32, time_major=False)

            predict_on = outputs

            # Get the last prediction to use in the feed forward network
            output = tf.transpose(predict_on, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)

        with tf.name_scope('fully_connected'):
            w1 = tf.get_variable(
                name='W1',
                shape=[self.cell_state_size, self.num_classes],
                dtype=tf.float32,
                trainable=True,
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            b1 = tf.get_variable(
                name='b1',
                shape=[self.num_classes], dtype=tf.float32,
                trainable=True,
                initializer=tf.constant_initializer(0.1)
            )

            logits = tf.nn.xw_plus_b(last, w1, b1, name='logits')

        with tf.name_scope('metrics'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.input_y)
            ) * self.batch_size

            self.predictions = tf.argmax(logits, 1, name='predictions')
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.accuracy)

        self.merged = tf.summary.merge_all()

        with tf.name_scope('training'):
            self.training_step = self.optimizer(self.learning_rate).minimize(self.loss)


