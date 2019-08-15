# coding: utf-8

import os
import datetime
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread

from d00_utils.db_utils import dbReadWriteClassification
from d03_classification import vgg
from d00_utils.log_utils import setup_logging

logger = setup_logging(__name__, __name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sys.path.append(os.getcwd())
f = open('/home/ubuntu/dvv/usal_echo/src/d03_classification/tf_vars.txt','r')
tf_vars = f.read()


view_classes = [
    "plax_far",
    "plax_plax",
    "plax_laz",
    "psax_az",
    "psax_mv",
    "psax_pap",
    "a2c_lvocc_s",
    "a2c_laocc",
    "a2c",
    "a3c_lvocc_s",
    "a3c_laocc",
    "a3c",
    "a4c_lvocc_s",
    "a4c_laocc",
    "a4c",
    "a5c",
    "other",
    "rvinf",
    "psax_avz",
    "suprasternal",
    "subcostal",
    "plax_lac",
    "psax_apex",
]

maps_dev = {
    "plax_far":"plax",
    "plax_plax":"plax",
    "plax_laz":"plax",
    "psax_az":"other",
    "psax_mv":"other",
    "psax_pap":"other",
    "a2c_lvocc_s":"a2c",
    "a2c_laocc":"a2c",
    "a2c":"a2c",
    "a3c_lvocc_s":"other",
    "a3c_laocc":"other",
    "a3c":"other",
    "a4c_lvocc_s":"a4c",
    "a4c_laocc":"a4c",
    "a4c":"a4c",
    "a5c":"other",
    "other":"other",
    "rvinf":"other",
    "psax_avz":"other",
    "suprasternal":"other",
    "subcostal":"other",
    "plax_lac":"plax",
    "psax_apex":"other",
}

maps_seg = {
    "plax_far":"other",
    "plax_plax":"plax",
    "plax_laz":"other",
    "psax_az":"other",
    "psax_mv":"other",
    "psax_pap":"other",
    "a2c_lvocc_s":"other",
    "a2c_laocc":"other",
    "a2c":"a2c",
    "a3c_lvocc_s":"other",
    "a3c_laocc":"other",
    "a3c":"other",
    "a4c_lvocc_s":"other",
    "a4c_laocc":"other",
    "a4c":"a4c",
    "a5c":"other",
    "other":"other",
    "rvinf":"other",
    "psax_avz":"other",
    "suprasternal":"other",
    "subcostal":"other",
    "plax_lac":"other",
    "psax_apex":"other",
}

def _classify(img_dir, feature_dim, label_dim, model_path):
    
    # Initialise tensorflow
    tf.reset_default_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model = vgg.Network(0.0, 0.0, feature_dim, label_dim, False)
    
    #saver = tf.train.Saver() # if running zhang's model
    
    #tf_vars = ['network/conv1_1/W:0', 'network/conv1_1/b:0']
    #tf_vars = ['conv1_1', 'conv1_2']
    #tf_vars = ['conv1_1/W:0', 'conv1_2/W:0']
    #tf_vars = tf.trainable_variables()
    #tf_vars = tf.global_variables()
    #note: tf.get_collection(tf.trainable_variables)) is an empty list
    #tf_vars = [v for v in tf.get_collection(tf.trainable_variables())]
    
    #saver = tf.train.Saver(tf_vars) # if running on retrained model
    saver = tf.train.Saver() # Thurs afternoon 2:41p

    #saver = tf.train.import_meta_graph(model_path + '.meta') # if new mod (?) 
    ckpt_files = model_path + '/model.ckpt-6460'
    saver.restore(sess, ckpt_files)#model_path)

    #sys.exit()

    # Classify views
    probabilities = {}
    for filename in os.listdir(img_dir):
        image = imread(os.path.join(img_dir, filename), flatten=True).astype("uint8")
        img_data = [image.reshape((224, 224, 1))]
        probabilities[filename] = np.around(
            model.probabilities(sess, img_data), decimals=3
        ).tolist()[0]

    return probabilities


def run_classify(img_dir, model_path, if_exists, feature_dim=1):
    """Writes classification probabilities of frames to database.
    :param img_dir: directory with jpg echo images for classification
    :param model_path: path to trained model for inferring probabilities
    :param if_exists (str): write action if table exists, must be 'replace' or 'append'
    :param feature_dim: default=1
    """
    label_dim = len(view_classes)
    model_name = os.path.basename(model_path)

    probabilities = _classify(img_dir, feature_dim, label_dim, model_path)

    df_columns = ['output_' + x for x in view_classes]
    df = (
        pd.DataFrame.from_dict(probabilities, columns=df_columns, orient='index')
        .rename_axis('file_name')
        .reset_index()
    ) 
    df['file_name'] = df['file_name'].apply(lambda x: x.split('.')[0])
    df['study_id'] = df['file_name'].apply(lambda x: x.split('_')[1])
    df['model_name'] = model_name
    df['date_run'] = datetime.datetime.now()
    df['img_dir'] = img_dir
    cols = ['study_id','file_name','model_name','date_run','img_dir'] + df_columns
    df = df[cols]

    io_classification = dbReadWriteClassification()
    io_classification.save_to_db(df, 'probabilities_frames', if_exists)

    logger.info(
        '{} prediction on frames with model {} (feature_dim={}).'.format(
            os.path.basename(img_dir), model_name, feature_dim
        )
    )


def agg_probabilities(if_exists):
    """Aggregates probabilities by study and instance.
    
    Fetches probabilities from classification.probabilities_frames and 
    calculates the mean. Returns dataframe with aggregated probabilities.
    :param if_exists (str): write action if table exists
    
    """
    io_classification = dbReadWriteClassification()

    probabilities_frames = io_classification.get_table('probabilities_frames')
    probabilities_frames['file_name'] = probabilities_frames['file_name'].apply(
        lambda x: x.rsplit('_', 1)[0]
    )

    mean_cols = ['output_' + x for x in view_classes]
    agg_cols = dict(zip(mean_cols, ['mean'] * len(mean_cols)))
    agg_cols['probabilities_frame_id'] = 'count' #TODO: debug error
    
    probabilities = (
        probabilities_frames.groupby(['study_id','file_name','model_name','date_run','img_dir'])
        .agg(agg_cols)
        .reset_index(drop=False)
    )
    probabilities.rename(columns={'probabilities_frame_id': 'frame_count'}, inplace=True)
    
    io_classification.save_to_db(probabilities, 'probabilities_instances', if_exists)
    logger.info('Aggregated probabilities saved to table classification.probabilities_instances')
    
    
def predict_views(if_exists):
    """Predicts and maps views based on maximum probability.
     
    :param if_exists (str): write action if table exists
    
    """    
    io_classification = dbReadWriteClassification()
    probabilities = io_classification.get_table('probabilities_instances')
                  
    #predictions = probabilities.drop(columns=['probabilities_instance_id', 'frame_count']
    #                                         ).set_index(['study_id','file_name','model_name','date_run','img_dir'])
    predictions = probabilities.drop(columns=['probabilities_instance_id']
                                              ).set_index(['study_id','file_name','model_name','date_run','img_dir'])
    
    predictions['view23_pred'] = predictions.idxmax(axis=1).apply(lambda x : x.split('_', 1)[1])
    predictions['view4_dev'] = predictions['view23_pred'].map(maps_dev)
    predictions['view4_seg'] = predictions['view23_pred'].map(maps_seg)

    df = predictions.loc[:,['view23_pred','view4_dev','view4_seg']]
    df.reset_index(inplace=True)     
    
    io_classification.save_to_db(df, 'predictions', if_exists)    
    logger.info('Predicted views saved to table classification.predictions')

