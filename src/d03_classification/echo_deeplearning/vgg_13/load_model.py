import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import random
import os, sys

import nn
from util import *
from scipy.misc import imresize, imsave
from load_data import load_data as ld


def load_model(config, sess):
    return NN(config, sess)


def load_data(config, sess):
    data_dir = os.path.join(Data, config.data)
    sys.path.append(data_dir)
    return ld(config, sess)


def train_print(i, j, loss, train_acc, batch, batch_total, time):
    """
    Formats print statements to update on same print line.
    
    @params are integers or floats
    """
    print(
        "Epoch {:1} |".format(i),
        "Iter {:1} |".format(j),
        "Loss: {:.4} |".format(loss),
        "Training Acc: {:.4} |".format(train_acc),
        "Data: {}/{} |".format(batch, batch_total),
        "Time {:1.2} ".format(time),
        "   ",
        end="\r",
    )


def val_print(i, j, loss, train_acc, val_acc, batch_total, time):
    """
    Formats print statements to update on same print line.
    
    @params are integers or floats
    """
    print(
        "Epoch {:1} |".format(i),
        "Iter {:1} |".format(j),
        "Loss: {:.4} |".format(loss),
        "Training Acc: {:.4} |".format(train_acc),
        "Val Acc: {:.4}|".format(val_acc),
        "Data: {}/{} |".format(batch_total, batch_total),
        "Time {:1.2} ".format(time),
        "   ",
        end="\r",
    )


def crop_data(img, crop_max):
    """
    Crops an image by some random integer amount between 0 and crop_max from each side of image
    Returns cropped image

    @params img: numpy array of an image
    @params crop_max: integer of maximum amount cropped from each side of image
    """
    ret_img = img.copy()

    if crop_max:
        x_min = random.randint(0, crop_max)
        x_max = img.shape[0] - x_min
        y_min = x_min
        y_max = img.shape[1] - x_min
        for i in range(ret_img.shape[2]):
            crop = ret_img[:, :, i]
            crop = imresize(
                crop[x_min:x_max, y_min:y_max], (img.shape[0], img.shape[1])
            )
            ret_img[:, :, i] = crop

    return ret_img


def data_augmentation(x_train, crop_max):
    """
    Applies data augmentation to training images
    Returns augmented/altered training images
    
    @params x_train: numpy array of training images
    @params crop_max: integer of maximum amount cropped from each side of image
    """
    x_train_copy = x_train.copy()
    if crop_max > 0:
        for i in range(x_train.shape[0]):
            x_train_copy[i] = crop_data(x_train_copy[i], crop_max)
    return x_train_copy


class NN(object):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

        self.x_train = tf.placeholder(
            tf.float32, [None, config.image_size, config.image_size, config.feature_dim]
        )
        self.y_train = tf.placeholder(tf.uint8, [None, config.label_dim])
        self.x_test = tf.placeholder(
            tf.float32, [None, config.image_size, config.image_size, config.feature_dim]
        )
        self.y_test = tf.placeholder(tf.uint8, [None, config.label_dim])

        self.global_step = tf.Variable(0, trainable=False)

        self.output = self.network(self.x_train, 0.5)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output, labels=self.y_train
            )
        ) + config.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
        )

        self.opt = tf.train.AdamOptimizer(config.learning_rate).minimize(
            self.loss, global_step=self.global_step
        )

        # TRANSFER LEARNING CODE HERE ###################
        # W_retrain = tf.trainable_variables()[-6:]
        # self.opt = tf.train.AdamOptimizer(config.learning_rate).minimize(
        #    self.loss, global_step=self.global_step, var_list=W_retrain
        # )
        ########################################

        self.train_pred = self.network(self.x_train, keep_prob=1.0, reuse=True)
        self.train_accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(self.train_pred, 1), tf.argmax(self.y_train, 1)),
                tf.float32,
            )
        )
        self.val_pred = self.network(self.x_test, keep_prob=1.0, reuse=True)
        self.val_accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(self.val_pred, 1), tf.argmax(self.y_test, 1)),
                tf.float32,
            )
        )

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.train_summary = tf.summary.scalar("training_accuracy", self.train_accuracy)

    def train(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        saver,
        summary_writer,
        checkpoint_path,
        val_output_dir,
    ):
        """
        Network training loop

        @params x_train: Numpy array of training images
        @params y_train: Numpy array of training labels
        @params x_test: Numpy array of validation images
        @params y_test: Numpy array of validation labels
        @params saver: Tensorflow class to save and restore models
        @params summary_writer: Tensorflow filewriter class to store loss and accuracies in tensorboard
        @params checkpoint_path: string of directory for storing models
        @params val_output_dir: string of directory for storing visualations on validation data
        """
        sess = self.sess
        config = self.config
        batch_size = config.batch_size

        losses = deque([])
        train_accs = deque([])
        step = tf.train.global_step(sess, self.global_step)
        for i in range(config.epochs):
            # Shuffle indicies
            indicies = list(range(x_train.shape[0]))
            np.random.shuffle(indicies)
            # Start timer
            start = timeit.default_timer()

            for j in range(int(x_train.shape[0] / batch_size)):
                temp_indicies = indicies[j * batch_size : (j + 1) * batch_size]

                if step % config.summary_interval == 0:
                    train_acc, train_summary = self.train_validate(
                        x_train[temp_indicies], y_train[temp_indicies]
                    )
                    summary_writer.add_summary(train_summary, step)
                    if len(train_accs) == config.loss_smoothing:
                        train_accs.popleft()
                    train_accs.append(train_acc)

                x_train_temp = data_augmentation(
                    x_train[temp_indicies], config.crop_max
                )
                loss, loss_summary = self.fit_batch(
                    x_train_temp, y_train[temp_indicies]
                )
                step = tf.train.global_step(sess, self.global_step)

                if step % config.summary_interval == 0:
                    summary_writer.add_summary(loss_summary, step)
                if len(losses) == config.loss_smoothing:
                    losses.popleft()
                losses.append(loss)

                stop = timeit.default_timer()
                train_print(
                    i,
                    j,
                    np.mean(losses),
                    np.mean(train_accs),
                    j * batch_size,
                    x_train.shape[0],
                    stop - start,
                )

            val_acc = self.validate(x_test, y_test, batch_size)
            summary = tf.Summary()
            summary.value.add(tag="validation_accuracy", simple_value=val_acc)
            if summary_writer:
                summary_writer.add_summary(summary, step)

            stop = timeit.default_timer()
            val_print(
                i,
                "xxx",  # j,
                np.mean(losses),
                np.mean(train_accs),
                val_acc,
                x_train.shape[0],
                stop - start,
            )

            if (i + 1) % config.epoch_save_interval == 0:
                saver.save(sess, checkpoint_path, global_step=step)
                # if "no_vis" not in config:
                # self.visualize(x_test, y_test, val_output_dir)
        if (i + 1) % config.epoch_save_interval != 0:
            saver.save(sess, checkpoint_path, global_step=step)
            # if "no_vis" not in config:
            # self.visualize(x_test, y_test, val_output_dir)

        return True

    def fit_batch(self, x_train, y_train):
        """
        Runs one step of gradient descent
        Returns loss value and loss_summary for displaying progress and tensorboard visualizations

        """
        _, loss, loss_summary = self.sess.run(
            (self.opt, self.loss, self.loss_summary),
            feed_dict={self.x_train: x_train, self.y_train: y_train},
        )
        return loss, loss_summary

    def accuracy(self, x_test, y_test):
        """
        Returns accuracy of running model on x_test

        """
        val_accuracy = self.sess.run(
            (self.val_accuracy), feed_dict={self.x_test: x_test, self.y_test: y_test}
        )
        return val_accuracy

    def train_validate(self, x_train, y_train):
        """
        Returns accuracy of training set

        """
        train_accuracy, train_summary = self.sess.run(
            (self.train_accuracy, self.train_summary),
            feed_dict={self.x_train: x_train, self.y_train: y_train},
        )
        return train_accuracy, train_summary

    def validate(self, x_test, y_test, batch_size):
        """
        Returns accuracy of validation set

        """
        accuracy = 0.0
        for i in range(x_test.shape[0]):
            accuracy = accuracy + self.accuracy(
                x_test[i : (i + 1)], y_test[i : (i + 1)]
            )
        return accuracy / (x_test.shape[0])

    def predict(self, x):
        """
        Forward pass of the neural network. Predicts labels for images x.

        @params x: Numpy array of training images
        """
        prediction = self.sess.run(
            (tf.nn.softmax(self.val_pred)), feed_dict={self.x_test: x}
        )
        return prediction

    def visualize(self, x_test, y_test, val_output_dir):
        """
        Visualize images by outputting images in correctly or incorrectly classified folders

        @params val_output_dir: string of output directory for visualization images
        """
        out_dir = val_output_dir + "-" + str(self.sess.run(self.global_step))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        correct_dir = os.path.join(out_dir, "correct")
        if not os.path.exists(correct_dir):
            os.makedirs(correct_dir)

        miss_dir = os.path.join(out_dir, "miss")
        if not os.path.exists(miss_dir):
            os.makedirs(miss_dir)

        open(os.path.join(out_dir, "probabilities.txt"), "w")
        gts, probs = [], []
        for i in range(min(300, x_test.shape[0])):
            gt = np.argmax(y_test[i], 0)
            pred = np.argmax(self.predict(x_test[i : i + 1])[0], 0)

            if gt == pred:
                imsave(
                    os.path.join(correct_dir, "image_" + str(i) + ".jpg"),
                    x_test[i, :, :, 0],
                )
            else:
                imsave(
                    os.path.join(miss_dir, "image_" + str(i) + ".jpg"),
                    x_test[i, :, :, 0],
                )

            with open(os.path.join(out_dir, "probabilities.txt"), "a") as myfile:
                myfile.write("image_" + str(i))
                myfile.write("\t" + str(self.predict(x_test[i : i + 1])[0][gt]) + "\n")
                gts.append(gt)
                probs.append(self.predict(x_test[i : i + 1])[0][gt])

        from sklearn.metrics import roc_curve, auc

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fpr[0], tpr[0], _ = roc_curve(gts, probs)
        roc_auc[0] = auc(fpr[0], tpr[0])
        plt.plot(
            fpr[0],
            tpr[0],
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc[0],
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.savefig(os.path.join(out_dir, "roc_curve.png"))
        plt.close()
        with open(os.path.join(out_dir, "probabilities.txt"), "a") as myfile:
            myfile.write("ROC_Area")
            myfile.write("\t" + str(roc_auc[0]) + "\n")

    def network(self, input, keep_prob=0.5, reuse=None):
        with tf.variable_scope("network", reuse=reuse):
            pool_ = lambda x: nn.max_pool(x, 2, 2)
            max_out_ = lambda x: nn.max_out(x, 16)
            config = self.config

            conv_ = lambda x, output_depth, name, stride=1, padding="SAME", relu=True, filter_size=3: conv(
                x,
                filter_size,
                output_depth,
                stride,
                name=name,
                padding=padding,
                relu=relu,
            )
            fc_ = lambda x, features, name, relu=True: fc(x, features, name, relu=relu)

            VGG_MEAN = [config.mean, config.mean, config.mean]
            input = tf.concat(
                [input - VGG_MEAN[0], input - VGG_MEAN[1], input - VGG_MEAN[2]], axis=3
            )

            conv_1_1 = conv_(input, 64, "conv1_1")  # , trainable = False)
            conv_1_2 = conv_(conv_1_1, 64, "conv1_2")  # , trainable = False)

            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, 128, "conv2_1")  # , trainable = False)
            conv_2_2 = conv_(conv_2_1, 128, "conv2_2")  # , trainable = False)

            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, 256, "conv3_1")
            conv_3_2 = conv_(conv_3_1, 256, "conv3_2")
            conv_3_3 = conv_(conv_3_2, 256, "conv3_3")

            pool_3 = pool_(conv_3_3)

            conv_4_1 = conv_(pool_3, 512, "conv4_1")
            conv_4_2 = conv_(conv_4_1, 512, "conv4_2")
            conv_4_3 = conv_(conv_4_2, 512, "conv4_3")

            pool_4 = pool_(conv_4_3)

            conv_5_1 = conv_(pool_4, 512, "conv5_1")
            conv_5_2 = conv_(conv_5_1, 512, "conv5_2")
            conv_5_3 = conv_(conv_5_2, 512, "conv5_3")

            pool_5 = pool_(conv_5_3)
            flattened = tf.contrib.layers.flatten(
                pool_5
            )  # i.e. assume self.maxout=False

            fc_6 = nn.dropout(fc_(flattened, 4096, "fc6"), keep_prob)
            fc_7 = nn.dropout(fc_(fc_6, 4096, "fc7"), keep_prob)
            fc_8 = fc_(fc_7, config.label_dim, "fc8", relu=False)
            return fc_8

    def init_weights(self):
        pass
