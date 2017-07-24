from __future__ import division
import os
import numpy as np
import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle, csv

from utils import *
from model import UNet3D, SurvivalVAE

flags = tf.app.flags
flags.DEFINE_integer("epoch", 4, "Epoch to train [4]")
flags.DEFINE_string("train_patch_dir", "patches", "Directory of the training data [patches]")
flags.DEFINE_bool("split_train", False, "Whether to split the train data into train and val [False]")
flags.DEFINE_string("train_data_dir", "../BraTS17TrainingData", "Directory of the train data [../BraTS17TrainingData]")
flags.DEFINE_string("deploy_data_dir", "../BraTS17ValidationData", "Directory of the test data [../BraTS17ValidationData]")
flags.DEFINE_string("deploy_output_dir", "output_validation", "Directory name of the output data [output]")
flags.DEFINE_string("train_csv", "../BraTS17TrainingData/survival_data.csv", "CSV path of the training data")
flags.DEFINE_string("deploy_csv", "../BraTS17ValidationData/survival_evaluation.csv", "CSV path of the validation data")
flags.DEFINE_integer("batch_size", 1, "Batch size [1]")
flags.DEFINE_integer("seg_features_root", 48, "Number of features in the first filter in the seg net [48]")
flags.DEFINE_integer("survival_features", 16, "Number of features in the survival net [16]")
flags.DEFINE_integer("conv_size", 3, "Convolution kernel size in encoding and decoding paths [3]")
flags.DEFINE_integer("layers", 3, "Encoding and deconding layers [3]")
flags.DEFINE_string("loss_type", "cross_entropy", "Loss type in the model [cross_entropy]")
flags.DEFINE_float("dropout", 0.5, "Drop out ratio [0.5]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save logs [logs]")
flags.DEFINE_boolean("train", False, "True for training, False for deploying [False]")
flags.DEFINE_boolean("run_seg", True, "True if run segmentation [True]")
flags.DEFINE_boolean("run_survival", False, "True if run survival prediction [True]")
FLAGS = flags.FLAGS

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    # Train
    all_train_paths = []
    for dirpath, dirnames, files in os.walk(FLAGS.train_data_dir):
        if os.path.basename(dirpath)[0:7] == 'Brats17':
            all_train_paths.append(dirpath)

    if FLAGS.split_train:
        if os.path.exists(os.path.join(FLAGS.train_patch_dir, 'files.log')):
            with open(os.path.join(FLAGS.train_patch_dir, 'files.log'), 'r') as f:
                training_paths, testing_paths = pickle.load(f)
        else:
            all_paths = [os.path.join(FLAGS.train_patch_dir, p) for p in sorted(os.listdir(FLAGS.train_data_dir))]
            np.random.shuffle(all_paths)
            n_training = int(len(all_paths) * 4 / 5)
            training_paths = all_paths[:n_training]
            testing_paths = all_paths[n_training:]
            # Save the training paths and testing paths
            with open(os.path.join(FLAGS.train_data_dir, 'files.log'), 'w') as f:
                pickle.dump([training_paths, testing_paths], f)

        training_ids = [os.path.basename(i) for i in training_paths]
        testing_ids = [os.path.basename(i) for i in testing_paths]

        training_survival_data = {}
        testing_survival_data = {}
        with open(FLAGS.train_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] in training_ids:
                    training_survival_data[row[0]] = (row[1], row[2])
                elif row[0] in testing_ids:
                    testing_survival_data[row[0]] = (row[1], row[2])

        training_survival_paths = [p for p in all_train_paths if os.path.basename(p) in training_survival_data.keys()]
        testing_survival_paths = [p for p in all_train_paths if os.path.basename(p) in testing_survival_data.keys()]
    else:
        training_paths = [os.path.join(FLAGS.train_patch_dir, name) for name in os.listdir(FLAGS.train_patch_dir)
                          if '.log' not in name]
        testing_paths = None

        training_ids = [os.path.basename(i) for i in training_paths]
        training_survival_paths = []
        testing_survival_paths = None
        training_survival_data = {}
        testing_survival_data = None

        with open(FLAGS.train_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] in training_ids:
                    training_survival_data[row[0]] = (row[1], row[2])
        training_survival_paths = [p for p in all_train_paths if os.path.basename(p) in training_survival_data.keys()]
        
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    
    # Segmentation net
    if FLAGS.run_seg:
        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as sess:
            unet = UNet3D(sess, checkpoint_dir=FLAGS.checkpoint_dir, log_dir=FLAGS.log_dir, training_paths=training_paths,
                          testing_paths=testing_paths, batch_size=FLAGS.batch_size, layers=FLAGS.layers,
                          features_root=FLAGS.seg_features_root, conv_size=FLAGS.conv_size,
                          dropout=FLAGS.dropout, loss_type=FLAGS.loss_type)

            if FLAGS.train:
                model_vars = tf.trainable_variables()
                slim.model_analyzer.analyze_vars(model_vars, print_info=True)

                train_config = {}
                train_config['epoch'] = FLAGS.epoch

                unet.train(train_config)
            else:
                # Deploy
                if not os.path.exists(FLAGS.deploy_output_dir):
                    os.makedirs(FLAGS.deploy_output_dir)
                unet.deploy(FLAGS.deploy_data_dir, FLAGS.deploy_output_dir)

        tf.reset_default_graph()
 
    # Survival net
    if FLAGS.run_survival:
        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as sess:
            survivalvae = SurvivalVAE(sess, checkpoint_dir=FLAGS.checkpoint_dir, log_dir=FLAGS.log_dir, 
                                      training_paths=training_survival_paths, testing_paths=testing_survival_paths,
                                      training_survival_data=training_survival_data,
                                      testing_survival_data=testing_survival_data)

            if FLAGS.train:
                model_vars = tf.trainable_variables()
                slim.model_analyzer.analyze_vars(model_vars, print_info=True)

                train_config = {}
                train_config['epoch'] = FLAGS.epoch * 100

                survivalvae.train(train_config)
            else:
                all_deploy_paths = []
                for dirpath, dirnames, files in os.walk(FLAGS.deploy_data_dir):
                    if os.path.basename(dirpath)[0:7] == 'Brats17':
                        all_deploy_paths.append(dirpath)
                deploy_survival_data = {}
                with open(FLAGS.deploy_csv, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if row[0] != 'Brats17ID':
                            deploy_survival_data[row[0]] = row[1]
                deploy_survival_paths = [p for p in all_deploy_paths if os.path.basename(p) in deploy_survival_data.keys()]
                survivalnet.deploy(FLAGS.deploy_survival_paths, FLAGS.deploy_survival_data)
        
if __name__ == '__main__':
    tf.app.run()