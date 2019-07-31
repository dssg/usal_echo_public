from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import timeit

from collections import deque
from sklearn.metrics import confusion_matrix
############################
# Neural Network Functions #
############################

# Convolution Layer
def conv(x, filter_size, num_filters, stride, weight_decay,  name, padding='SAME', groups=1, trainable=True):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable('W',
                                  shape=[filter_size, filter_size, input_channels // groups, num_filters],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=trainable,
                                  regularizer=regularizer,
                                  collections=['variables'])
        biases = tf.get_variable('b', shape=[num_filters], trainable=trainable, initializer=tf.zeros_initializer())

        if groups == 1:
            conv = convolve(x, weights)

        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, axis=3)
            weight_groups = tf.split(weights, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(output_groups, axis=3)

        return tf.nn.relu(conv + biases)

# Fully Connected Layer
def fc(x, num_out, weight_decay,  name, relu=True, trainable=True):
    num_in = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable('W',
                                  shape=[num_in, num_out], 
                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                  trainable=trainable, 
                                  regularizer=regularizer,
                                  collections=['variables'])
        biases = tf.get_variable('b', [num_out], initializer=tf.zeros_initializer(), trainable=trainable)
        x = tf.matmul(x, weights) + biases
        if relu:
            x = tf.nn.relu(x) 
    return x

# Local Response Normalization
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def max_pool(x, filter_size, stride, name=None, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


#################################
# Training/Validation Functions #
#################################


def validate(sess, model, x_test, y_test, batch_size):
    '''
    Calculates accuracy of validation set
    
    @params sess: Tensorflow Session
    @params model: Model defined from a neural network class
    @params x_test: Numpy array of validation images
    @params y_test: Numpy array of validation labels
    @params batch_size: Integer defining mini-batch size
    '''
    accuracy = 0.0
    for i in range(int(x_test.shape[0]/batch_size)):
        accuracy = accuracy + batch_size * model.validate(sess, x_test[i*batch_size:(i+1)*batch_size], 
                                                          y_test[i*batch_size:(i+1)*batch_size])
    # Tail Case
    if x_test.shape[0] % batch_size != 0:
        accuracy = accuracy + batch_size * model.validate(sess, x_test[(i+1)*batch_size:], 
                                                          y_test[(i+1)*batch_size:])
    return accuracy/(x_test.shape[0])

def validate_bagging(sess, model, x_test, y_test, batch_size, num_sets):
    '''
    Calculates accuracy of validation set by randomly sampling (with replacement)
    the validation set. Provides more accurate estimation of model accuracy.
    
    @params many same as validate()
    @params num_sets: Integer defining number of validation sets to test
    '''
    val_accs = []
    for i in range(num_sets):
        indicies = (np.random.sample((x_test.shape[0],))*x_test.shape[0]).astype(int)
        val_accs.append(validate(sess,model,x_test[indicies],y_test[indicies],batch_size))
    return np.mean(val_accs), np.std(val_accs)

def train_print(i, j, loss, train_acc, batch, batch_total, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.4} |".format(loss),
          "Training Acc: {:.4} |".format(train_acc), 
          "Data: {}/{} |".format(batch, batch_total), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")
    
def train(sess, model, x_train, y_train, x_test, y_test, epochs, batch_size, summary_writer = 0, train_validation = 5):
    '''
    Main function for training neural network model. 
    
    @params many identical to those in validate()
    @params summary_writer: Tf.summary.FileWriter used for Tensorboard variables
    @params batch_size: Integer defining mini-batch size
    @params train_validation: Integer defining how many train steps before running accuracy on training mini-batch
    '''
    losses = deque([])
    train_accs = deque([])
    step = 0
    for i in range(epochs):
        # Shuffle indicies
        indicies = range(x_train.shape[0])
        np.random.shuffle(indicies)
        # Start timer
        start = timeit.default_timer()

        for j in range(int(x_train.shape[0]/batch_size)):
            # Shuffle Data
            temp_indicies = indicies[j*batch_size:(j+1)*batch_size]
            loss, loss_summary = model.fit_batch(sess,x_train[temp_indicies], y_train[temp_indicies])
            if summary_writer:
                summary_writer.add_summary(loss_summary, step)
            if len(losses) == 50:
                losses.popleft()
            losses.append(loss)
            # How often to test accuracy on training batch
            if j%train_validation == 0:
                train_acc, train_summary = model.train_validate(sess,x_train[temp_indicies], y_train[temp_indicies])
                if summary_writer:
                    summary_writer.add_summary(train_summary, step)
                if len(train_accs) == 50:
                    train_accs.popleft()
                train_accs.append(train_acc)
            stop = timeit.default_timer()
            
            train_print(i, j, np.mean(losses), np.mean(train_accs), 
                        j*batch_size, x_train.shape[0], stop - start)
            step = step + 1

        # Tail case 
        if x_train.shape[0] % batch_size != 0:
            temp_indicies = indicies[(j+1)*batch_size:]
            loss, loss_summary = model.fit_batch(sess,x_train[temp_indicies], y_train[temp_indicies])
            if summary_writer:
                summary_writer.add_summary(loss_summary, step)
            if len(losses) == 50:
                losses.popleft()
            losses.append(loss)
            train_acc, train_summary = model.train_validate(sess,x_train[temp_indicies], y_train[temp_indicies])
            if summary_writer:
                summary_writer.add_summary(train_summary, step)
            if len(train_accs) == 50:
                train_accs.popleft()
            train_accs.append(train_acc)
            train_print(i, j, np.mean(losses), np.mean(train_accs), 
                        j*batch_size, x_train.shape[0], stop - start)
            step = step + 1
      
        # Accuracy on test set after every epoch
        if x_test is not None:
            val_acc = validate(sess,model,x_test,y_test,batch_size)
            summary = tf.Summary()
            summary.value.add(tag="validation_accuracy", simple_value=val_acc)
            if summary_writer:    
                summary_writer.add_summary(summary, step)
            stop = timeit.default_timer()
            print("Epoch {:1}|".format(i), 
                  "Iter {:1}|".format(j), 
                  "Loss: {:.4}|".format(np.mean(losses)),
                  "Training Acc: {:.4}|".format(np.mean(train_accs)),
                  "Val Acc: {:.4}|".format(val_acc), 
                  "Iter {}/{}|".format(x_train.shape[0],x_train.shape[0]),
                  "Time {:1.2}".format(stop-start), 
                  "   ", end="\r")
            print()
        


def prediction(sess, model, x_test, y_test, train_lst, val_lst, batch_size = 32):
    X_val = x_test
    Y_val = y_test
    preds = np.zeros((X_val.shape[0],))
    y_preds = np.zeros((X_val.shape[0],))
    for i in range(int(X_val.shape[0]/batch_size)):
        preds[batch_size*i:batch_size*(i+1)] = model.predict(sess, X_val[batch_size*i:batch_size*(i+1)])
        y_preds[batch_size*i:batch_size*(i+1)] = np.argmax(Y_val[batch_size*i:batch_size*(i+1)], axis = 1).astype('uint8')
    i = int(X_val.shape[0]/batch_size)
    preds[batch_size*i:] = model.predict(sess,X_val[batch_size*i:])
    y_preds[batch_size*i:] = np.argmax(Y_val[batch_size*i:], axis = 1).astype('uint8')
    return preds, y_preds

def plot_cm(preds, y_preds, class_num = 8):
    cm = confusion_matrix(y_preds,preds)
    norm_cm = cm.astype('float')/np.sum(cm,axis=1)
    plt.figure(figsize=(8,8))
    plt.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment = "center"
                 ,color = "white" if cm[i,j] > thresh else "black")
    plt.show()
    for i in range(class_num):
        print("Validation accuracy on label %d: " % (i), (norm_cm[i][i]))