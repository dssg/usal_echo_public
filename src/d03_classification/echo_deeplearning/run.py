#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import tensorflow as tf

import sys, os, time, json
from easydict import EasyDict as edict

from util import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

flags = tf.app.flags

# optional arguments
flags.DEFINE_boolean(
    "train", True, "True for training, False for testing phase. Default [True]"
)
flags.DEFINE_string("gpu", "0", "GPU number. Default [0]")
flags.DEFINE_string("val_split", "0", "Cross validation study split, Default: 0")
flags.DEFINE_boolean(
    "debug", False, "If true, train and validate for 1 iteration. Default [False]"
)
flags.DEFINE_boolean(
    "retrain",
    False,
    "If true, trains a new model and will override old models. Default [False]",
)
FLAGS = flags.FLAGS


def main(argv):
    # Must follow model argument format of "models/[model_name]/[model_experiment]"
    assert len(argv) == 2, "Only argument must be path to config directory"
    config_dir = os.path.abspath(argv[1])
    assert config_dir.startswith(Models), "Invalid config directory %s" % config_dir
    model_name, config_name = config_dir[len(Models) :].split("/")[:2]

    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    model_dir = os.path.join(Models, model_name)
    sys.path.append(model_dir)

    # load in model specific configurations
    config_dir = os.path.join(model_dir, config_name)
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, "r+") as f:
        config = edict(json.load(f))

    data_dir = os.path.join(Data, config.data)
    sys.path.append(data_dir)

    if FLAGS.debug:
        config.epochs = config.epoch_save_interval = 1

    train_dir = os.path.join(config_dir, "train", FLAGS.val_split)

    ckpt = tf.train.get_checkpoint_state(train_dir)

    if ckpt and not FLAGS.retrain:
        print("Latest checkpoint:", ckpt.model_checkpoint_path)
    elif not FLAGS.train:
        raise RuntimeError("Cannot find checkpoint to test from, exiting")

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # load model with the given configurations
    from load_model import load_model, load_data

    model = load_model(config, sess)
    x_train, x_test, y_train, y_test = load_data(config, FLAGS.val_split)

    print("DATA LOADED!")

    saver = tf.train.Saver()

    # initialize model
    if ckpt and not FLAGS.retrain:
        # print(ckpt.model_checkpoint_path)
        checkpoint_HARDCODE = train_dir + "/model.ckpt-0"
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        model.init_weights()

    val_output_dir = os.path.join(config_dir, "val", FLAGS.val_split, "output")

    if FLAGS.train:
        # train model
        summary_writer = tf.summary.FileWriter(train_dir)
        checkpoint_path = os.path.join(train_dir, "model.ckpt")

        success = model.train(
            x_train,
            y_train,
            x_test,
            y_test,
            saver,
            summary_writer,
            checkpoint_path,
            val_output_dir,
        )

        if not success:
            print("Exiting script")
            exit()
    else:
        print("Saving results")
        model.visualize(x_test, y_test, val_output_dir)


if __name__ == "__main__":
    tf.app.run()
