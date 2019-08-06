
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random, math
import os, sys

from util import *
from scipy.misc import imread, imresize
from skimage.draw import circle
from skimage.filters import sobel
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure

def load_model(config, sess):
    return NN(config, sess)

def load_data(config, sess):
    data_dir = os.path.join(Data, config.data)
    sys.path.append(data_dir)
    from load_data import load_data as ld
    return ld(config, sess)

def train_print(i, j, loss, batch, batch_total, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.4} |".format(loss),
          "Data: {}/{} |".format(batch, batch_total), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")
    
def val_print(i, j, loss, acc, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.2} |".format(loss),
          "Acc: {} |".format(np.round(acc,3)), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")

def crop_data(img, segmentation, crop_max):
    '''
    Crops an image and its segmentation by some random integer amount between 0 and crop_max from each side of image
    Returns cropped image and segementation

    @params img: numpy array of an image
    @params segmentation: numpy array of a segmentation
    @params crop_max: integer of maximum amount cropped from each side of image
    '''
    ret_img = img.copy()
    ret_label = segmentation.copy()
    if crop_max:
        x_min = random.randint(0,crop_max)
        x_max = img.shape[0] - random.randint(0,crop_max)
        y_min = random.randint(0,crop_max)
        y_max = img.shape[1] - random.randint(0,crop_max)
        for i in range(img.shape[2]):
            crop = ret_img[:,:,i]
            crop = imresize(crop[x_min:x_max, y_min:y_max],(img.shape[0],img.shape[1]))
            ret_img[:,:,i] = crop
        for i in range(segmentation.shape[2]):
            crop = ret_label[:,:,i]
            crop = imresize(crop[x_min:x_max, y_min:y_max],(segmentation.shape[0],segmentation.shape[1]))
            ret_label[:,:,i] = crop
    return ret_img, ret_label

def blackout_data(img, segmentation, image_size, blackout_max):
    '''
    Black out a random circle in an image around the edge a segmentation

    @params img: numpy array of an image
    @params segmentation: numpy array of a segmentation
    @params blackout_max: integer of maximum circle radius of blackout
    '''
    img = img.copy()
    if blackout_max:
        segmentation = segmentation.copy()
        edges = gaussian_filter(sobel(segmentation.astype('uint8')),0.3)
        border = np.where(edges > 0)
        if len(border[0] != 0):
            point = random.randint(0,len(border[0])-1)
            rr, cc = circle(border[0][point], border[1][point],random.randint(1,blackout_max))
            img[np.clip(rr, 0, image_size-1), np.clip(cc, 0, image_size-1)] = random.randint(0,50)
    return img

def data_augmentation(x_train, y_train, config):
    '''
    Applies data augmentation to training images
    Returns augmented/altered training images
    
    @params x_train: numpy array of training images
    @params y_train: numpy array of segmentations
    @params config: dictionary of hyperparameters (includes data augmentation parameters)
    '''
    x_train_copy = x_train.copy()
    y_train_copy = y_train.copy()
    seg_num = y_train.shape[3]
    for i in range(x_train.shape[0]):
        x_train_copy[i] = blackout_data(x_train_copy[i], y_train_copy[i,:,:,random.randint(1,seg_num-1)], config.image_size, config.blackout_max)
        x_train_copy[i], y_train_copy[i] = crop_data(x_train_copy[i], y_train_copy[i], config.crop_max)
    return x_train_copy, y_train_copy

class NN(object):        
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.x_train = tf.placeholder(tf.float32, [None, config.image_size, config.image_size, config.feature_dim])
        self.y_train = tf.placeholder(tf.float32, [None, config.image_size,config.image_size, config.label_dim])
        self.x_test = tf.placeholder(tf.float32, [None, config.image_size, config.image_size, config.feature_dim])
        self.y_test = tf.placeholder(tf.float32, [None, config.image_size,config.image_size, config.label_dim])
        
        self.global_step = tf.Variable(0, trainable=False)
        self.output = self.network(self.x_train, config.mean, keep_prob=config.dropout)
        self.loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_train)) 
            + config.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]))
        self.opt = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss, global_step=self.global_step)
        
        self.pred = self.network(self.x_test, config.mean, keep_prob=1.0, reuse=True)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
  
    def train(self, x_train, y_train, x_test, y_test, saver, summary_writer, checkpoint_path, val_output_dir):
        '''
        Network training loop

        @params x_train: Numpy array of training images
        @params y_train: Numpy array of training labels
        @params x_test: Numpy array of validation images
        @params y_test: Numpy array of validation labels
        @params saver: Tensorflow class to save and restore models
        @params summary_writer: Tensorflow filewriter class to store loss and accuracies in tensorboard
        @params checkpoint_path: string of directory for storing models
        @params val_output_dir: string of directory for storing visualations on validation data
        '''
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

            for j in range(int(x_train.shape[0]/batch_size)):
                temp_indicies = indicies[j*batch_size:(j+1)*batch_size]
                x_train_temp, y_train_temp = data_augmentation(x_train[temp_indicies], y_train[temp_indicies], config)
                loss, loss_summary = self.fit_batch(x_train_temp, y_train_temp)
                
                if step % config.summary_interval == 0:
                    summary_writer.add_summary(loss_summary, step)
                if len(losses) == config.loss_smoothing:
                    losses.popleft()
                losses.append(loss)

                stop = timeit.default_timer()
                train_print(i, j, np.mean(losses), j*batch_size, x_train.shape[0], stop - start)
                step = step + 1

            stop = timeit.default_timer()
            acc = self.validate(x_test, y_test)
            summary = tf.Summary()
            for k in range(len(acc)):
                summary.value.add(tag="validation_acc_" + str(k), simple_value=acc[k])
            if summary_writer:    
                summary_writer.add_summary(summary, step)
            val_print(i, j, np.mean(losses), acc, stop - start)
            print()

            if (i+1) % config.epoch_save_interval == 0:
                saver.save(sess, checkpoint_path, global_step=step)
                self.visualize(x_test, y_test, val_output_dir)
        if (i+1) % config.epoch_save_interval != 0:
            saver.save(sess, checkpoint_path, global_step=step)
            self.visualize(x_test, y_test, val_output_dir)

        return True

    def validate(self, x_test, y_test):
        '''
        Returns accuracy of validation set

        '''
        scores = [0] * int(y_test.shape[3]-1)
        counts = [0] * int(y_test.shape[3]-1)
        for i in range(int(x_test.shape[0])):
            gt = np.argmax(y_test[i,:,:,:], 2)
            pred = np.argmax(self.predict(x_test[i:i+1])[0,:,:,:], 2)
            for j in range(int(y_test.shape[3]-1)):
                dice = iou(gt, pred, j+1)
                if not math.isnan(dice):
                    scores[j] = scores[j] + dice
                    counts[j] = counts[j] + 1
                
        return [score/counts[i] for i, score in enumerate(scores)]

    def fit_batch(self, x_train, y_train):
        '''
        Runs one step of gradient descent
        Returns loss value and loss_summary for displaying progress and tensorboard visualizations

        '''
        _, loss, loss_summary = self.sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    def predict(self, x):
        '''
        Forward pass of the neural network. Predicts labels for images x.

        @params x: Numpy array of training images
        '''
        prediction = self.sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    def visualize(self, x_test, y_test, val_output_dir):
        '''
        Visualize images, ground truth labels, and predicted labels as a matplotlib figure

        @params val_output_dir: string of output directory for visualization images
        '''
        out_dir = val_output_dir + '-' + str(self.sess.run(self.global_step))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for i in range(x_test.shape[0]):
            gt = np.argmax(y_test[i,:,:,:], 2)
            pred = np.argmax(self.predict(x_test[i:i+1])[0,:,:,:], 2)
            plt.figure(figsize = (10,4))
            plt.subplot(131)
            plt.imshow(x_test[i,:,:,0], cmap = 'gray'), plt.axis('off') # original image
            plt.subplot(132)
            plt.imshow(gt), plt.axis('off') # gt labels
            plt.subplot(133)
            plt.imshow(pred), plt.axis('off') # network prediction label
            out_filename = 'seg_' + str(i) + '.png'
            plt.savefig(os.path.join(out_dir,out_filename))
            plt.close()
            
    def network(self, input, mean, keep_prob=0.5, reuse=None):
        '''
        Neural network architecture
        Returns segmentation label predictions on input

        @params input: Numpy array of images
        @params mean: mean float value used to mean-subtract images
        @params keep_prob: dropout probability 
        @params reuse: Set to None for new session and True to use same variables in same session
        '''
        config = self.config
        with tf.variable_scope('network', reuse=reuse):
            input = input - mean
            pool_ = lambda x: max_pool(x, 2, 2)
            conv_ = lambda x, output_depth, name, stride=1, padding='SAME', relu=True, filter_size=3: conv(x, filter_size, output_depth, stride, name=name, padding=padding, relu=relu)
            deconv_ = lambda x, output_depth, name, filter_size=2, stride=2: deconv(x, filter_size, output_depth, stride, name=name)
            fc_ = lambda x, features, name, relu=True: fc(x, features, name, relu)
            
            conv_1_1 = conv_(input, 64, 'conv1_1')
            conv_1_2 = conv_(conv_1_1, 64, 'conv1_2')
            
            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, 128, 'conv2_1')
            conv_2_2 = conv_(conv_2_1, 128, 'conv2_2')
            
            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, 256, 'conv3_1')
            conv_3_2 = conv_(conv_3_1, 256, 'conv3_2')

            pool_3 = pool_(conv_3_2)

            conv_4_1 = conv_(pool_3, 512, 'conv4_1')
            conv_4_2 = conv_(conv_4_1, 512, 'conv4_2')

            pool_4 = pool_(conv_4_2)

            conv_5_1 = conv_(pool_4, 1024, 'conv5_1')
            conv_5_2 = conv_(conv_5_1, 1024, 'conv5_2')

            pool_5 = pool_(conv_5_2)

            conv_6_1 = tf.nn.dropout(conv_(pool_5, 2048, 'conv6_1'), keep_prob)
            conv_6_2 = tf.nn.dropout(conv_(conv_6_1, 2048, 'conv6_2'), keep_prob)
           
            up_7 = tf.concat([deconv_(conv_6_2, 1024, 'up7'), conv_5_2], 3)
            
            conv_7_1 = conv_(up_7, 1024, 'conv7_1')
            conv_7_2 = conv_(conv_7_1, 1024, 'conv7_2')
            
            up_8 = tf.concat([deconv_(conv_7_2, 512, 'up8'), conv_4_2], 3)
            
            conv_8_1 = conv_(up_8, 512, 'conv8_1')
            conv_8_2 = conv_(conv_8_1, 512, 'conv8_2')
            
            up_9 = tf.concat([deconv_(conv_8_2, 256, 'up9'), conv_3_2], 3)
            
            conv_9_1 = conv_(up_9, 256, 'conv9_1')
            conv_9_2 = conv_(conv_9_1, 256, 'conv9_2')

            up_10 = tf.concat([deconv_(conv_9_2, 128, 'up10'), conv_2_2], 3)
            
            conv_10_1 = conv_(up_10, 128, 'conv10_1')
            conv_10_2 = conv_(conv_10_1, 128, 'conv10_2')

            up_11 = tf.concat([deconv_(conv_10_2, 64, 'up11'), conv_1_2], 3)
            
            conv_11_1 = conv_(up_11, 64, 'conv11_1')
            conv_11_2 = conv_(conv_11_1, 64, 'conv11_2')
            
            conv_12 = conv_(conv_11_2, config.label_dim, 'conv12_2', filter_size=1, relu=False)
            return conv_12

    def init_weights(self):
        pass
