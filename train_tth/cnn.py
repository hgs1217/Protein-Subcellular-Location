# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from train_tth.utils import add_loss_summaries, conv_layer, avg_pool_layer, fc_layer, cal_loss
from train_tth.config import CKPT_PATH, CKPT_PREFIX, LOG_PATH


class CNN:
    def __init__(self, classifer_num, raws, labels, test_raws, test_labels, keep_pb=0.5, epoch_size=100,
                 learning_rate=0.001, start_step=0, detail_log=False,
                 open_summary=False, new_ckpt_internal=0, ):
        """
        Convolutional neural network. Before running, you should modify the configuration first,
        which is in 'config.py'.
        :param raws: list
                    Raw data in training set.
        :param labels: list
                    Labels in training set.
        :param test_raws: list
                    Raw data in test set.
        :param test_labels: list
                    Labels in test set.
        :param keep_pb: float
                    The keep probabilities used in dropout layers.
        :param epoch_size: int
                    Epoch size
        :param learning_rate: float
                    Learning rate
        :param start_step: int
                    Current start step, which is used in further training based on existed model to
                    prevent the disorder of current epoch.
        :param detail_log: boolean
                    Whether detailed log is outputed.
                    If detailed log mode is on, the infomation of each batch will also be printed.
        :param open_summary: boolean
                    Whether summary is open.
                    If summary mode is on, the summary graph and logs will be written into the log
                    file, which can be shown in tensorboard.
        :param new_ckpt_internal: int
                    The epoch internal for new checkpoint file generation.
                    If set -1, the checkpoint file will not be saved.
                    If set 0, there will only exist one checkpoint file through the whole training.
                    If set n (n>0), every n step will generate a new checkpoint file. For example,
                    when n=5, epoch size=100, then 20 checkpoint files will be created all together.
        :param network_mode: string
                    The network type in training.
                    If set None(default), use the default Lenet binary relevance classifiers.
                    If set 'chain', use the Lenet classifier chain.
        """
        self.c_n=classifer_num
        self._raws = raws
        self._labels = labels
        self._test_raws = test_raws
        self._test_labels = test_labels
        self._keep_pb = keep_pb
        self._epoch_size = epoch_size
        self._start_step = start_step
        self._learning_rate = learning_rate
        self._detail_log = detail_log
        self._open_summary = open_summary
        self._new_ckpt_internal = new_ckpt_internal
        # self._network_mode = network_mode
        # self._image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
        [_, self._input_width, self._input_height, self._input_channels] = raws[0].shape
        [_, self._classes] = labels[0].shape
        # self._loss_array = [[0.5, 0.5] for _ in range(self._labels)] if loss_array is None else loss_array

        self._x = tf.placeholder(tf.float32, shape=[None, self._input_width, self._input_height, self._input_channels],
                                 name="input_x")
        self._y = tf.placeholder(tf.float32, shape=[None, self._classes], name="input_y")
        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._global_step = tf.Variable(0, trainable=False)

    # def _get_network_measure(self, y, out, mode=None):
    #     """
    #         Get some measure metrics of the network
    #         :param y: tensor, [batch_size, label_num, class_num]
    #                 Labels given
    #         :param out: tensor, [batch_size, label_num, class_num]
    #                 Predictions given by network
    #         :param mode: string, default None
    #                 mode 'dtree': dtree mode, take round function
    #         :return: accu, precision, recall, f1
    #                 accu: tensor, [batch_size]
    #                     Strict accuracy. Only when all labels are correct ranks accurate.
    #                 precision: tensor, [batch_size]
    #                     True positive / (True positive + False negative)
    #                 recall: tensor, [batch_size]
    #                     True positive / (True positive + False positive)
    #                 f1: tensor, [batch_size]
    #                     Harmonic mean of precision and recall
    #     """
    #     if mode == 'dtree':
    #         accu = tf.reduce_mean(tf.cast(tf.equal(out, tf.argmax(y, 2)), tf.float32), axis=0)
    #         pn = tf.cast(2 * out + tf.argmax(y, 2), tf.int32)
    #     else:
    #         accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 2), tf.argmax(y, 2)), tf.float32), axis=0)
    #         pn = tf.cast(2 * tf.argmax(out, 2) + tf.argmax(y, 2), tf.int32)
    #
    #     counter = tf.cast(tf.concat(tf.map_fn(lambda t: tf.bincount(t, minlength=4), pn), axis=0), tf.float32)
    #     epsilon = 1e-20 * tf.ones(tf.shape(counter[:, 0]))
    #     precision = counter[:, 0] / (counter[:, 0] + counter[:, 2] + epsilon)
    #     recall = counter[:, 0] / (counter[:, 0] + counter[:, 1] + epsilon)
    #     f1 = 2 * precision * recall / (precision + recall + epsilon)
    #
    #     return accu, precision, recall, f1

    def _build_network_lenet_dtree(self, x, y, is_training):
        x_resh = tf.reshape(x, [-1, self._input_width, self._input_height, self._input_channels])
        label_name = "classifier"
        conv1 = conv_layer(x_resh, 5, 1, 32, is_training, name=label_name + "conv1")
        pool1 = avg_pool_layer(conv1, 2, 2, name=label_name + "pool1")

        conv2 = conv_layer(pool1, 5, 1, 64, is_training, name=label_name + "conv2")
        pool2 = avg_pool_layer(conv2, 2, 2, name=label_name + "pool2")

        fc_in = tf.reshape(pool2, [-1, 5 * 5 * 64])
        fc3 = fc_layer(fc_in, 1024, is_training, name=label_name + "fc3", relu_flag=True)
        dropout3 = tf.nn.dropout(fc3, self._keep_prob)

        fc4 = fc_layer(dropout3, self._classes, is_training, name=label_name + "fc4", relu_flag=True)
        loss = cal_loss(fc4, y)
        return loss, fc4

    def _train_set(self, loss, global_step):

        """
            Training operation settings, including optimizer and so on.
            :param total_loss: list, [label number]
            :param global_step:
            :return: train_op
        """

        # We train the six labels at the same time
        loss_averages_op = add_loss_summaries(loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(self._learning_rate)
            grads = opt.compute_gradients(loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            loss, prediction = self._build_network_lenet_dtree(self._x, self._y, self._is_training)
        train_op = self._train_set(loss, self._global_step)

        saver = tf.train.Saver()
        tf.add_to_collection('prediction', prediction)

        summary_op = tf.summary.merge_all()

        if self._start_step > 0:
            saver.restore(sess, CKPT_PATH)
        else:
            sess.run(tf.global_variables_initializer())

        if self._open_summary:
            summary_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
        loss_pl, test_loss_pl = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
        # f1_pl, test_f1_pl = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
        loss_summary = tf.summary.scalar("Train_Average_Loss", loss_pl)
        test_loss_summary = tf.summary.scalar("Test_Average_Loss", test_loss_pl)
        # f1_summary = tf.summary.scalar("Train_Prediction_F1", f1_pl)
        # test_f1_summary = tf.summary.scalar("Test_Prediction_F1", test_f1_pl)

        for step in range(self._start_step + 1, self._start_step + self._epoch_size + 1):
            print("Training epoch %d/%d" % (step, self._start_step + self._epoch_size))
            total_batch = len(self._raws)
            epoch_loss = np.zeros((total_batch,))
            # epoch_accu = np.zeros((total_batch,))
            epoch_total_data = np.zeros((total_batch, 1))  # [loss, accu, precision, recall, f1]
            for bat in range(total_batch):
                batch_xs = self._raws[bat]
                batch_ys = self._labels[bat]
                _, sum_str, pd, epoch_loss[bat], = sess.run(
                    [train_op, summary_op, prediction, loss],
                    feed_dict={self._x: batch_xs, self._y: batch_ys, self._keep_prob: self._keep_pb,
                               self._is_training: True})
                epoch_total_data[bat, 0] = np.mean(epoch_loss[bat])
                # epoch_total_data[bat, 1] = np.mean(np.prod(epoch_accu[bat], axis=0))
                # epoch_total_data[bat, 2:] = np.array(list(map(lambda x: np.mean(x), [prec, rec, f])))
                if self._detail_log:
                    print("Training epoch %d/%d, batch %d/%d, loss %g, " %
                          (step, self._start_step + self._epoch_size, bat + 1, total_batch,
                           epoch_total_data[bat, 0],))
                    # if bat % 10 == 9:
                    # self._print_class_accu(epoch_loss[bat], epoch_accu[bat])

            avg_data = np.mean(epoch_total_data, axis=0)
            print("Training epoch %d/%d finished, loss %g, " %
                  (step, self._start_step + self._epoch_size, avg_data[0],))
            # self._print_class_accu(np.mean(epoch_loss, axis=0), np.mean(epoch_accu, axis=0))
            print("==============================================================")
            if self._open_summary:
                loss_str, f1_str = sess.run([loss_summary],
                                            feed_dict={loss_pl: avg_data[0],})
                summary_writer.add_summary(loss_str, step)
                # summary_writer.add_summary(f1_str, step)

            if step % 1 == 0:
                print("Testing epoch %d/%d" % (step, self._start_step + self._epoch_size))
                test_batch = len(self._test_raws)
                test_loss = np.zeros((test_batch,))
                # test_accu = np.zeros((test_batch, ))
                test_total_data = np.zeros((test_batch, 1))

                for bat in range(test_batch):
                    batch_xs = self._test_raws[bat]
                    batch_ys = self._test_labels[bat]
                    pd, test_loss[bat,], = sess.run(
                        [prediction, loss, ],
                        feed_dict={self._x: batch_xs, self._y: batch_ys, self._keep_prob: 1.0,
                                   self._is_training: False})
                    test_total_data[bat, 0] = np.mean(test_loss[bat])
                    # test_total_data[bat, 1] = np.mean(np.prod(test_accu[bat], axis=0))
                    # test_total_data[bat, 2:] = np.array(list(map(lambda x: np.mean(x), [t_prec, t_rec, t_f])))

                    if self._detail_log:
                        print("Testing epoch %d/%d, batch %d/%d, loss %g, " %
                              (step, self._start_step + self._epoch_size, bat + 1, test_batch,
                               test_total_data[bat, 0],))
                        # if bat % 10 == 9:
                        # self._print_class_accu(test_loss[bat], test_accu[bat])

                    test_avg_data = np.mean(test_total_data, axis=0)
                    print("Testing epoch %d/%d finished, loss %g, " %
                          (step, self._start_step + self._epoch_size, test_avg_data[0],))
                    # self._print_class_accu(np.mean(test_loss, axis=0), np.mean(test_accu, axis=0))
                    print("==============================================================")

                    if self._open_summary:
                        test_loss_str, test_f1_str = sess.run(
                            [test_loss_summary],
                            feed_dict={test_loss_pl: test_avg_data[0],})
                        summary_writer.add_summary(test_loss_str, step)
                        # summary_writer.add_summary(test_f1_str, step)

            print("saving model.....")
            if self._new_ckpt_internal == 0:
                saver.save(sess, CKPT_PATH)
            elif self._new_ckpt_internal > 0:
                path = 'c_'+str(self.c_n)+"{0}{1}/model.ckpt".format(CKPT_PREFIX, int((step - 1) / self._new_ckpt_internal))
                saver.save(sess, path)
            print("end saving....\n")
