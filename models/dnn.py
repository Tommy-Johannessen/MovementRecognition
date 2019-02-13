import tensorflow as tf


class DNNClassifier:
    def __init__(self, batch_size, data_length, num_classes, shape, optimizer, learning_rate=0.05):
        self.batch_size = batch_size
        self.data_length = data_length
        self.num_classes = num_classes
        self.shape = shape
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def build_network(self):
        self.input_x = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.data_length],
            name='input_features'
        )

        self.input_y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.num_classes],
            name='input_labels'
        )

        output = tf.contrib.layers.stack(
            inputs=self.input_x,
            layer=tf.contrib.layers.fully_connected,
            stack_args=self.shape,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
        )


        logits = tf.layers.dense(
            inputs=output,
            units=self.num_classes,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0.1),
            name='logits'
        )


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
