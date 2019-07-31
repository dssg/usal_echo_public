import time
from optparse import OptionParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image
from scipy.misc import imresize
from skimage.color import rgb2gray, gray2rgb

from d00_utils.echocv_utils_v0 import *
from d00_utils.dcm_utils_v0 import create_imgdict_from_dicom


# # Hyperparams
parser=OptionParser()
parser.add_option("-d", "--dicomdir", dest="dicomdir", default = "dicomsample", help = "dicomdir")
parser.add_option("-g", "--gpu", dest="gpu", default = "0", help = "cuda device to use")
parser.add_option("-M", "--modeldir", default = "models", dest="modeldir")
params, args = parser.parse_args()
dicomdir = params.dicomdir
modeldir = params.modeldir

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu


class Unet(object):        
    def __init__(self, mean, weight_decay, learning_rate, label_dim, maxout = False):
        self.x_train = tf.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_train = tf.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.x_test = tf.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_test = tf.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        self.output = self.unet(self.x_train, mean)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.pred = self.unet(self.x_test, mean, keep_prob = 1.0, reuse = True)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
#         self.train_summary = tf.summary.scalar('training_accuracy', self.train_accuracy)
    
    # Gradient Descent on mini-batch
    
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary), 
                                         feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary

    
    def predict(self, sess, x):
        prediction = sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    
    def unet(self, input, mean, keep_prob = 0.5, reuse = None):
        width = 1
        weight_decay = 1e-12
        label_dim = self.label_dim
        
        with tf.variable_scope('vgg', reuse=reuse):
            input = input - mean
            pool_ = lambda x: max_pool(x, 2, 2)
            conv_ = lambda x, output_depth, name, padding = 'SAME', relu = True, filter_size = 3: conv(x, filter_size, output_depth, 1, weight_decay, name=name, padding=padding, relu=relu)
            deconv_ = lambda x, output_depth, name: deconv(x, 2, output_depth, 2, weight_decay, name=name)
            fc_ = lambda x, features, name, relu = True: fc(x, features, weight_decay, name, relu)
            
            conv_1_1 = conv_(input, int(64*width), 'conv1_1')
            conv_1_2 = conv_(conv_1_1, int(64*width), 'conv1_2')
            
            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, int(128*width), 'conv2_1')
            conv_2_2 = conv_(conv_2_1, int(128*width), 'conv2_2')
            
            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, int(256*width), 'conv3_1')
            conv_3_2 = conv_(conv_3_1, int(256*width), 'conv3_2')

            pool_3 = pool_(conv_3_2)

            conv_4_1 = conv_(pool_3, int(512*width), 'conv4_1')
            conv_4_2 = conv_(conv_4_1, int(512*width), 'conv4_2')

            pool_4 = pool_(conv_4_2)

            conv_5_1 = conv_(pool_4, int(1024*width), 'conv5_1')
            conv_5_2 = conv_(conv_5_1, int(1024*width), 'conv5_2')

            pool_5 = pool_(conv_5_2)

            conv_6_1 = tf.nn.dropout(conv_(pool_5, int(2048*width), 'conv6_1'), keep_prob)
            conv_6_2 = tf.nn.dropout(conv_(conv_6_1, int(2048*width), 'conv6_2'), keep_prob)
           
            up_7 = tf.concat([deconv_(conv_6_2, int(1024*width), 'up7'), conv_5_2], 3)
            
            conv_7_1 = conv_(up_7, int(1024*width), 'conv7_1')
            conv_7_2 = conv_(conv_7_1, int(1024*width), 'conv7_2')
            
            up_8 = tf.concat([deconv_(conv_7_2, int(512*width), 'up8'), conv_4_2], 3)
            
            conv_8_1 = conv_(up_8, int(512*width), 'conv8_1')
            conv_8_2 = conv_(conv_8_1, int(512*width), 'conv8_2')
            
            up_9 = tf.concat([deconv_(conv_8_2, int(256*width), 'up9'), conv_3_2], 3)
            
            conv_9_1 = conv_(up_9,int(256*width), 'conv9_1')
            conv_9_2 = conv_(conv_9_1, int(256*width), 'conv9_2')

            up_10 = tf.concat([deconv_(conv_9_2, int(128*width), 'up10'), conv_2_2], 3)
            
            conv_10_1 = conv_(up_10, int(128*width), 'conv10_1')
            conv_10_2 = conv_(conv_10_1, int(128*width), 'conv10_2')

            up_11 = tf.concat([deconv_(conv_10_2, int(64*width), 'up11'), conv_1_2], 3)
            
            conv_11_1 = conv_(up_11, int(64*width), 'conv11_1')
            conv_11_2 = conv_(conv_11_1, int(64*width), 'conv11_2')
            
            conv_12 = conv_(conv_11_2, label_dim, 'conv12_2', filter_size = 1, relu = False)
            
            return conv_12

        
def segmentChamber(videofile, dicomdir, view):
    """
    
    """
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
    sesses = []
    models = []
    global modeldir
    if view == "a4c":
        g_1 = tf.Graph()
        with g_1.as_default():
            label_dim = 6 #a4c
            sess1 = tf.Session()
            model1 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
            sess1.run(tf.local_variables_initializer())
            sess = sess1
            model = model1
        with g_1.as_default():
            saver = tf.train.Saver()
            saver.restore(sess1,os.path.join(modeldir,'a4c_45_20_all_model.ckpt-9000'))
    elif view == "a2c":
        g_2 = tf.Graph()
        with g_2.as_default():
            label_dim = 4 
            sess2 = tf.Session()
            model2 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
            sess2.run(tf.local_variables_initializer())
            sess = sess2
            model = model2
        with g_2.as_default():
            saver = tf.train.Saver()
            saver.restore(sess2,os.path.join(modeldir, 'a2c_45_20_all_model.ckpt-10600'))
    elif view == "a3c":
        g_3 = tf.Graph()
        with g_3.as_default():
            label_dim = 4 
            sess3 = tf.Session()
            model3 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
            sess3.run(tf.local_variables_initializer())
            sess = sess3
            model = model3
        with g_3.as_default():
            saver.restore(sess3,os.path.join(modeldir,'a3c_45_20_all_model.ckpt-10500'))
    elif view == "psax":
        g_4 = tf.Graph()
        with g_4.as_default():
            label_dim = 4 
            sess4 = tf.Session()
            model4 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
            sess4.run(tf.local_variables_initializer())
            sess = sess4
            model = model4
        with g_4.as_default():
            saver = tf.train.Saver()
            saver.restore(sess4,os.path.join(modeldir, 'psax_45_20_all_model.ckpt-9300'))
    elif view == "plax":
        g_5 = tf.Graph()
        with g_5.as_default():
            label_dim = 7 
            sess5 = tf.Session()
            model5 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
            sess5.run(tf.local_variables_initializer())
            sess = sess5
            model = model5
        with g_5.as_default():
            saver = tf.train.Saver()
            saver.restore(sess5,os.path.join(modeldir, 'plax_45_20_all_model.ckpt-9600'))
    outpath = "./segment/" + view + "/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    framedict = create_imgdict_from_dicom(dicomdir, videofile)
    images, orig_images = extract_images(framedict)
    if view == "a4c":
        a4c_lv_segs, a4c_la_segs, a4c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 4, 1)
        np.save(outpath + '/' + videofile + '_lv', np.array(a4c_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_la', np.array(a4c_la_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_lvo', np.array(a4c_lvo_segs).astype('uint8'))
    elif view == "a2c":
        a2c_lv_segs, a2c_la_segs, a2c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 3, 1)
        np.save(outpath + '/' + videofile + '_lv', np.array(a2c_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_la', np.array(a2c_la_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_lvo', np.array(a2c_lvo_segs).astype('uint8'))
    elif view == "psax":
        psax_lv_segs, psax_lvo_segs, psax_rv_segs, preds = extract_segs(images, orig_images, model, sess, 2, 1, 3)
        np.save(outpath + '/' + videofile + '_lv', np.array(psax_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_lvo', np.array(psax_lvo_segs).astype('uint8'))
    elif view == "a3c":
        a3c_lv_segs, a3c_la_segs, a3c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 3, 1)
        np.save(outpath + '/' + videofile + '_lvo', np.array(a3c_lvo_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_lv', np.array(a3c_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_la', np.array(a3c_la_segs).astype('uint8'))
    elif view == "plax":
        plax_lv_segs, plax_la_segs, plax_ao_segs, preds = extract_segs(images, orig_images, model, sess, 1, 5, 3)
        np.save(outpath + '/' + videofile + '_lv', np.array(plax_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_la', np.array(plax_la_segs).astype('uint8'))
    j = 0
    nrow = orig_images[0].shape[0]
    ncol = orig_images[0].shape[1]
    print(nrow, ncol)
    plt.figure(figsize = (5, 5))
    plt.axis('off')
    plt.imshow(imresize(preds, (nrow,ncol)))
    plt.savefig(outpath + '/' + videofile + '_' + str(j) + '_' + 'segmentation.png')
    plt.close() 
    plt.figure(figsize = (5, 5))
    plt.axis('off')
    plt.imshow(orig_images[0])
    plt.savefig(outpath + '/' + videofile + '_' + str(j) + '_' + 'originalimage.png')
    plt.close()   
    background = Image.open(outpath + '/' + videofile + '_' + str(j) + '_' + 'originalimage.png')
    overlay = Image.open(outpath + '/' + videofile + '_' + str(j) + '_' + 'segmentation.png')
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    outImage = Image.blend(background, overlay, 0.5)
    outImage.save(outpath + '/' + videofile + '_' + str(j) + '_' + 'overlay.png', "PNG")
    return 1


def segmentstudy(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax, dicomdir):
    for video in viewlist_a4c:
        segmentChamber(video, dicomdir, "a4c")
    for video in viewlist_a2c:
        segmentChamber(video, dicomdir, "a2c")
    for video in viewlist_psax:
        segmentChamber(video, dicomdir, "psax")
    for video in viewlist_plax:
        segmentChamber(video, dicomdir, "plax")
    return 1


def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output


def extract_images(framedict):
    images = []
    orig_images = []
    for key in list(framedict.keys()):
        image = np.zeros((384,384))
        image[:,:] = imresize(rgb2gray(framedict[key]), (384,384,1))
        images.append(image)
        orig_images.append(framedict[key])
    images = np.array(images).reshape((len(images), 384,384,1))
    return images, orig_images


def extract_segs(images, orig_images, model, sess, lv_label, la_label, lvo_label):
    segs = []
    preds = np.argmax(model.predict(sess, images[0:1])[0,:,:,:], 2)
    label_all = list(range(1, 8))
    label_good = [lv_label, la_label, lvo_label]
    for i in label_all:
        if not i in label_good:
            preds[preds == i] = 0
    for i in range(len(images)):
        seg = np.argmax(model.predict(sess, images[i:i+1])[0,:,:,:], 2)
        segs.append(seg)
    lv_segs = []
    lvo_segs = []
    la_segs = []
    for seg in segs:
        la_seg = create_seg(seg, la_label)
        lvo_seg = create_seg(seg, lvo_label)
        lv_seg = create_seg(seg, lv_label)
        lv_segs.append(lv_seg)
        lvo_segs.append(lvo_seg)
        la_segs.append(la_seg)
    return lv_segs, la_segs, lvo_segs, preds


def main():
    # To use dicomdir option set in global scope.
    global dicomdir
    # In case dicomdir is path with more than one part.
    dicomdir_basename = os.path.basename(dicomdir)
    viewfile = "probabilities/view_23_e5_class_11-Mar-2018_{}_probabilities.txt".format(dicomdir_basename)
    viewlist_a2c = []
    viewlist_a3c = []
    viewlist_a4c = []
    viewlist_plax = []
    viewlist_psax = []
    
    infile = open("viewclasses_view_23_e5_class_11-Mar-2018.txt")
    infile = infile.readlines()
    infile = [i.rstrip() for i in infile]

    viewdict = {}

    for i in range(len(infile)):
        viewdict[infile[i]] = i + 2
     
    probthresh = 0.5 #arbitrary choice of "probability" threshold for view classification

    infile = open(viewfile)
    infile = infile.readlines()
    infile = [i.rstrip() for i in infile]
    infile = [i.split('\t') for i in infile]

    start = time.time()
    for i in infile[1:]:
        dicomdir = i[0]
        filename = i[1]
        if eval(i[viewdict['psax_pap']]) > probthresh:
            viewlist_psax.append(filename)
        elif eval(i[viewdict['a4c']]) > probthresh:
            viewlist_a4c.append(filename)
        elif eval(i[viewdict['a2c']]) > probthresh:
            viewlist_a2c.append(filename)
        elif eval(i[viewdict['a3c']]) > probthresh:
            viewlist_a3c.append(filename)
        elif eval(i[viewdict['plax_plax']]) > probthresh:
            viewlist_plax.append(filename)
    print(viewlist_a2c, viewlist_a4c, viewlist_a3c, viewlist_psax, viewlist_plax)
    segmentstudy(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax, dicomdir)
    tempdir = os.path.join(dicomdir, "image")
    end = time.time()
    viewlist = viewlist_a2c + viewlist_a4c + viewlist_psax + viewlist_plax
    print("time:  " + str(end - start) + " seconds for " +  str(len(viewlist))  + " videos")
    #if os.path.exists(tempdir):
    #    shutil.rmtree(tempdir)

    
if __name__ == '__main__':
    main()

