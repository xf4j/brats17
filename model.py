from __future__ import division
import tensorflow as tf
import os, re, time
import numpy as np
import pickle

from utils import *

def conv3d(input_, output_dim, f_size, is_training, scope='conv3d'):
    with tf.variable_scope(scope) as scope:
        # VGG network uses two 3*3 conv layers to effectively increase receptive field
        w1 = tf.get_variable('w1', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1 = tf.nn.conv3d(input_, w1, strides=[1, 1, 1, 1, 1], padding='SAME')
        b1 = tf.get_variable('b1', [output_dim], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.bias_add(conv1, b1)
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, scope='bn1', decay=0.9,
                                           zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r1 = tf.nn.relu(bn1)
        
        w2 = tf.get_variable('w2', [f_size, f_size, f_size, output_dim, output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2 = tf.nn.conv3d(r1, w2, strides=[1, 1, 1, 1, 1], padding='SAME')
        b2 = tf.get_variable('b2', [output_dim], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.bias_add(conv2, b2)
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope='bn2', decay=0.9,
                                           zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r2 = tf.nn.relu(bn2)
        return r2
    
def deconv3d(input_, output_shape, f_size, is_training, scope='deconv3d'):
    with tf.variable_scope(scope) as scope:
        output_dim = output_shape[-1]
        w = tf.get_variable('w', [f_size, f_size, f_size, output_dim, input_.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape, strides=[1, f_size, f_size, f_size, 1], padding='SAME')
        bn = tf.contrib.layers.batch_norm(deconv, is_training=is_training, scope='bn', decay=0.9,
                                          zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r = tf.nn.relu(bn)
        return r
    
def crop_and_concat(x1, x2):
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4)

def conv_relu(input_, output_dim, f_size, s_size, scope='conv_relu'):
    with tf.variable_scope(scope) as scope:
        w = tf.get_variable('w', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv3d(input_, w, strides=[1, s_size, s_size, s_size, 1], padding='VALID')
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        r = tf.nn.relu(conv)
        return r
    
class UNet3D(object):
    def __init__(self, sess, checkpoint_dir, log_dir, training_paths, testing_paths,
                 batch_size=1, layers=3, features_root=32, conv_size=3, dropout=0.5,
                 loss_type='cross_entropy', class_weights=None):
        self.sess = sess
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        self.testing_paths = testing_paths
        
        image, _ = read_patch(os.path.join(self.training_paths[0], '0'))
        
        self.nclass = 4
        self.batch_size = batch_size
        self.patch_size = image.shape[:-1]
        self.patch_stride = 4 # Used in deploy
        self.channel = image.shape[-1]
        self.layers = layers
        self.features_root = features_root
        self.conv_size = conv_size
        self.dropout = dropout
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.patches_per_image = len(os.listdir(self.training_paths[0]))
        
        self.build_model()
        
        self.saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection_ref('bn_collections'))
        
    def build_model(self):
        self.images = tf.placeholder(tf.float32, shape=[None, self.patch_size[0], self.patch_size[1], self.patch_size[2],
                                                        self.channel], name='images')
        self.labels = tf.placeholder(tf.float32, shape=[None, self.patch_size[0], self.patch_size[1], self.patch_size[2],
                                                        self.nclass], name='labels')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')
        
        conv_size = self.conv_size
        layers = self.layers

        deconv_size = 2
        pool_stride_size = 2
        pool_kernel_size = 3 # Use a larger kernel
        
        # Encoding path
        connection_outputs = []
        for layer in range(layers):
            features = 2**layer * self.features_root
            if layer == 0:
                prev = self.images
            else:
                prev = pool
                
            conv = conv3d(prev, features, conv_size, is_training=self.is_training, scope='encoding' + str(layer))
            connection_outputs.append(conv)
            pool = tf.nn.max_pool3d(conv, ksize=[1, pool_kernel_size, pool_kernel_size, pool_kernel_size, 1],
                                    strides=[1, pool_stride_size, pool_stride_size, pool_stride_size, 1],
                                    padding='SAME')
        
        bottom = conv3d(pool, 2**layers * self.features_root, conv_size, is_training=self.is_training, scope='bottom')
        bottom = tf.nn.dropout(bottom, self.keep_prob)
        
        # Decoding path
        for layer in range(layers):
            conterpart_layer = layers - 1 - layer
            features = 2**conterpart_layer * self.features_root
            if layer == 0:
                prev = bottom
            else:
                prev = conv_decoding
            
            shape = prev.get_shape().as_list()
            deconv_output_shape = [tf.shape(prev)[0], shape[1] * deconv_size, shape[2] * deconv_size,
                                   shape[3] * deconv_size, features]
            deconv = deconv3d(prev, deconv_output_shape, deconv_size, is_training=self.is_training,
                              scope='decoding' + str(conterpart_layer))
            cc = crop_and_concat(connection_outputs[conterpart_layer], deconv)
            conv_decoding = conv3d(cc, features, conv_size, is_training=self.is_training,
                                   scope='decoding' + str(conterpart_layer))
            
        with tf.variable_scope('logits') as scope:
            w = tf.get_variable('w', [1, 1, 1, conv_decoding.get_shape()[-1], self.nclass],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            logits = tf.nn.conv3d(conv_decoding, w, strides=[1, 1, 1, 1, 1], padding='SAME')
            b = tf.get_variable('b', [self.nclass], initializer=tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(logits, b)
        
        self.probs = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.probs, 4)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 4)), tf.float32))
                                  
        flat_logits = tf.reshape(logits, [-1, self.nclass])
        flat_labels = tf.reshape(self.labels, [-1, self.nclass])
        
        if self.class_weights is not None:
            class_weights = tf.constant(np.asarray(self.class_weights, dtype=np.float32))
            weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weights), axis=1)
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
            cross_entropy_loss = tf.reduce_mean(weighted_loss)
        else:
            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                        labels=flat_labels))
        eps = 1e-5
        dice_value = 0
        dice_loss = 0
        for i in range(1, self.nclass):
            slice_prob = tf.squeeze(tf.slice(self.probs, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1]), axis=4)
            slice_prediction = tf.cast(tf.equal(self.predictions, i), tf.float32)
            slice_label = tf.squeeze(tf.slice(self.labels, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1]), axis=4)
            intersection_prob = tf.reduce_sum(tf.multiply(slice_prob, slice_label), axis=[1, 2, 3])
            intersection_prediction = tf.reduce_sum(tf.multiply(slice_prediction, slice_label), axis=[1, 2, 3])
            union = eps + tf.reduce_sum(slice_prediction, axis=[1, 2, 3]) + tf.reduce_sum(slice_label, axis=[1, 2, 3])
            dice_loss += tf.reduce_mean(tf.div(intersection_prob, union))
            dice_value += tf.reduce_mean(tf.div(intersection_prediction, union))
        dice_value = dice_value * 2.0 / (self.nclass - 1)
        dice_loss = 1 - dice_loss * 2.0 / (self.nclass - 1)
        self.dice = dice_value
        
        if self.loss_type == 'cross_entropy':
            self.loss = cross_entropy_loss
        elif self.loss_type == 'dice':
            self.loss = cross_entropy_loss + dice_loss
        else:
            raise ValueError("Unknown cost function: " + self.loss_type)
        
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.dice_summary = tf.summary.scalar('dice', self.dice)
            
    def train(self, config):
        #optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        
        train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'train'), self.sess.graph)
        if self.testing_paths is not None:
            test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'test'))
            testing_orders = [(n, l) for n in range(len(self.testing_paths)) for l in range(self.patches_per_image)]
        
        merged = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.dice_summary])
                
        counter = 0
        training_orders = [(n, l) for n in range(len(self.training_paths)) for l in range(self.patches_per_image)]
        for epoch in range(config['epoch']):
            # Shuffle the orders
            epoch_training_orders = np.random.permutation(training_orders)
            
            # Go through all selected patches
            for f in range(len(epoch_training_orders) // self.batch_size):
                patches = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.channel),
                                   dtype=np.float32)
                labels = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.nclass),
                                  dtype=np.float32)
                for b in range(self.batch_size):
                    order = epoch_training_orders[f * self.batch_size + b]
                    patches[b], labels[b] = read_patch(os.path.join(self.training_paths[order[0]], str(order[1])))
                _, train_loss, summary = self.sess.run([optimizer, self.loss, merged],
                                                       feed_dict = { self.images: patches,
                                                                     self.labels: labels,
                                                                     self.is_training: True,
                                                                     self.keep_prob: self.dropout })
                train_writer.add_summary(summary, counter)
                counter += 1
                if np.mod(counter, 1000) == 0:
                    self.save(counter)
                    
                # Run test
                if self.testing_paths is not None and np.mod(counter, 100) == 0:
                    for b in range(self.batch_size):
                        order = testing_orders[np.random.choice(len(testing_orders))]
                        patches[b], labels[b] = read_patch(os.path.join(self.testing_paths[order[0]], str(order[1])))
                    test_loss, summary = self.sess.run([self.loss, merged],
                                                       feed_dict = { self.images: patches,
                                                                     self.labels: labels,
                                                                     self.is_training: True,
                                                                     self.keep_prob: 1 })
                    print(str(counter) + ":" + "train_loss: " + str(train_loss) + " test_loss: " + str(test_loss))
                    test_writer.add_summary(summary, counter)
                    
        # Save in the end
        self.save(counter)
       
    def deploy(self, input_path, output_path):
        # Step 1
        if not self.load()[0]:
            raise Exception("No model is found, please train first") 
        
        # Apply this to all subjects including the training cases
        # Read from files.log and pick the testing cases for analysis
        all_paths = []
        for dirpath, dirnames, files in os.walk(input_path):
            if os.path.basename(dirpath)[0:7] == 'Brats17':
                all_paths.append(dirpath)
                
        for path in all_paths:
            image = read_image(path, is_training=False)
            locations, padding = generate_test_locations(self.patch_size, self.patch_stride, image.shape[:-1])
            pad_image = np.pad(image, padding + ((0, 0),), 'constant')
            pad_result = np.zeros((pad_image.shape[:-1] + (self.nclass,)), dtype=np.float32)
            pad_add = np.zeros((pad_image.shape[:-1]), dtype=np.float32)
            for x in locations[0]:
                for y in locations[1]:
                    for z in locations[2]:
                        patch = pad_image[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                          int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                          int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2), :]
                        
                        patch = np.expand_dims(patch, axis=0)
                        
                        probs = self.sess.run(self.probs, feed_dict = { self.images: patch,
                                                                        self.is_training: True,
                                                                        self.keep_prob: 1 })
                        pad_result[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                   int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                   int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2), :] += probs[0]
                        pad_add[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2)] += 1
            pad_result = pad_result / np.tile(np.expand_dims(pad_add, axis=3), (1, 1, 1, pad_result.shape[-1]))
            result = pad_result[padding[0][0] : padding[0][0] + image.shape[0],
                                padding[1][0] : padding[1][0] + image.shape[1],
                                padding[2][0] : padding[2][0] + image.shape[2], :]
            print(path)
            np.save(os.path.join(output_path, os.path.basename(path) + '_probs'), result)
            
    def estimate_mean_std(self, training_orders):
        means = []
        stds = []
        # Strictly speaking, this is not the correct way to estimate std since the mean 
        # used in each image is not the global mean but the mean of the image, this would
        # cause an over-estimation of the std.
        # The correct way may need much more memory, and more importantly, it probably does not matter...
        for order in training_orders:
            patch, _ = read_patch(os.path.join(self.training_paths[order[0]], str(order[1])))
            means.append(np.mean(patch, axis=(0, 1, 2)))
            stds.append(np.std(patch, axis=(0, 1, 2)))
        return np.mean(np.asarray(means, dtype=np.float32), axis=0), np.mean(np.asarray(stds, dtype=np.float32), axis=0)
    
    @property
    def model_dir(self):
        return 'unet3d_layer{}_{}'.format(self.layers, self.loss_type)
    
    def save(self, step):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'unet3d'), global_step=step)
        
    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*\d)', ckpt_name)).group(0))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0

# This model is not working...        
class SurvivalNet(object):
    def __init__(self, sess, checkpoint_dir, log_dir, training_paths, testing_paths,
                 training_survival_data, testing_survival_data,
                 batch_size=1, features=16, dropout=0.5):
        self.sess = sess
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        self.testing_paths = testing_paths
        self.training_survival_data = training_survival_data
        self.testing_survival_data = testing_survival_data
        
        label = read_label(self.training_paths[0])
        
        self.label_size = label.shape[:-1]
        self.batch_size = batch_size
        self.channel = label.shape[-1]
        self.features = features
        self.dropout = dropout
        
        self.build_model()
        
        self.saver = tf.train.Saver()
    
    def build_model(self):
        self.labels = tf.placeholder(tf.float32, shape=[None, self.label_size[0], self.label_size[1], self.label_size[2],
                                                        self.channel], name='images')
        self.survival = tf.placeholder(tf.float32, shape=[None, 1], name='survival')
        self.age = tf.placeholder(tf.float32, shape=[None, 1], name='age')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')
        
        # variational autoencoders
        # 3 layers
        # 16*16*16->4*4*4->1*1*1
        f_sizes = [4, 4]
        s_sizes = [4, 4]
        for layer in range(2):
            if layer == 0:
                prev = self.labels
            else:
                prev = conv_relu_output
            conv_relu_output = conv_relu(prev, self.features, f_sizes[layer], s_sizes[layer], scope='layer' + str(layer))
            conv_relu_output = tf.nn.dropout(conv_relu_output, self.keep_prob)
        flat_image_input = tf.reshape(conv_relu_output, [-1, self.features])
        all_input = tf.concat([flat_image_input, self.age], 1)
        output = tf.layers.dense(all_input, 1)
        
        self.loss = tf.losses.mean_squared_error(self.survival, output)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
            
    def train(self, config):
        optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        
        train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'train'), self.sess.graph)
        if self.testing_paths is not None:
            test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'test'))
         
        counter = 0
        for epoch in range(config['epoch']):
            # Shuffle the orders
            training_paths = np.random.permutation(self.training_paths)
            
            # Go through all selected patches
            for f in range(len(training_paths) // self.batch_size):
                labels = np.empty((self.batch_size, self.label_size[0], self.label_size[1], self.label_size[2], self.channel),
                                  dtype=np.float32)
                survivals = np.empty((self.batch_size, 1), dtype=np.float32)
                ages = np.empty((self.batch_size, 1), dtype=np.float32)
                for b in range(self.batch_size):
                    labels[b] = read_label(training_paths[b])
                    survivals[b] = self.training_survival_data[os.path.basename(training_paths[b])][1]
                    ages[b] = self.training_survival_data[os.path.basename(training_paths[b])][0]
                _, train_loss, summary = self.sess.run([optimizer, self.loss, self.loss_summary],
                                                       feed_dict = { self.labels: labels,
                                                                     self.survival: survivals,
                                                                     self.age: ages,
                                                                     self.keep_prob: self.dropout })
                train_writer.add_summary(summary, counter)
                counter += 1
                if np.mod(counter, 1000) == 0:
                    self.save(counter)
                    
                # Run test
                if self.testing_paths is not None and np.mod(counter, 100) == 0:
                    testing_paths = np.random.permutation(self.testing_paths)
                    for b in range(self.batch_size):
                        labels[b] = read_label(testing_paths[b])
                        survivals[b] = self.testing_survival_data[os.path.basename(testing_paths[b])][1]
                        ages[b] = self.testing_survival_data[os.path.basename(testing_paths[b])][0]
                    test_loss, summary = self.sess.run([self.loss, self.loss_summary],
                                                       feed_dict = { self.labels: labels,
                                                                     self.survival: survivals,
                                                                     self.age: ages,
                                                                     self.keep_prob: 1 })
                    print(str(counter) + ":" + "train_loss: " + str(train_loss) + " test_loss: " + str(test_loss))
                    test_writer.add_summary(summary, counter)
                    
        # Save in the end
        self.save(counter)
       
    def deploy(self, input_path, output_path):
        # Step 1
        if not self.load()[0]:
            raise Exception("No model is found, please train first") 
        
        # Apply this to all subjects including the training cases
        # Read from files.log and pick the testing cases for analysis
        all_paths = []
        for dirpath, dirnames, files in os.walk(input_path):
            if os.path.basename(dirpath)[0:7] == 'Brats17':
                all_paths.append(dirpath)
                
        #mean, std = self.sess.run([self.mean, self.std])
        for path in all_paths:
            image = read_image(path, is_training=False)
            locations, padding = generate_test_locations(self.patch_size, self.patch_stride, image.shape[:-1])
            pad_image = np.pad(image, padding + ((0, 0),), 'constant')
            pad_result = np.zeros((pad_image.shape[:-1] + (self.nclass,)), dtype=np.float32)
            pad_add = np.zeros((pad_image.shape[:-1]), dtype=np.float32)
            for x in locations[0]:
                for y in locations[1]:
                    for z in locations[2]:
                        patch = pad_image[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                          int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                          int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2), :]
                        
                        patch = np.expand_dims(patch, axis=0)
                        #patch = (patch - mean) / std
                        
                        probs = self.sess.run(self.probs, feed_dict = { self.images: patch,
                                                                        self.is_training: True,
                                                                        self.keep_prob: 1 })
                        pad_result[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                   int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                   int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2), :] += probs[0]
                        pad_add[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2)] += 1
            pad_result = pad_result / np.tile(np.expand_dims(pad_add, axis=3), (1, 1, 1, pad_result.shape[-1]))
            result = pad_result[padding[0][0] : padding[0][0] + image.shape[0],
                                padding[1][0] : padding[1][0] + image.shape[1],
                                padding[2][0] : padding[2][0] + image.shape[2], :]
            print(path)
            np.save(os.path.join(output_path, os.path.basename(path) + '_probs'), result)
            
    def estimate_mean_std(self, training_orders):
        means = []
        stds = []
        # Strictly speaking, this is not the correct way to estimate std since the mean 
        # used in each image is not the global mean but the mean of the image, this would
        # cause an over-estimation of the std.
        # The correct way may need much more memory, and more importantly, it probably does not matter...
        for order in training_orders:
            patch, _ = read_patch(os.path.join(self.training_paths[order[0]], str(order[1])))
            means.append(np.mean(patch, axis=(0, 1, 2)))
            stds.append(np.std(patch, axis=(0, 1, 2)))
        return np.mean(np.asarray(means, dtype=np.float32), axis=0), np.mean(np.asarray(stds, dtype=np.float32), axis=0)
    
    @property
    def model_dir(self):
        return 'survival_feature{}'.format(self.features)
    
    def save(self, step):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'survivalnet'), global_step=step)
        
    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*\d)', ckpt_name)).group(0))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0
        
class SurvivalVAE(object):
    def __init__(self, sess, checkpoint_dir, log_dir, training_paths, testing_paths,
                 training_survival_data, testing_survival_data,
                 batch_size=6, n_hidden_1=500, n_hidden_2=500, n_z=20):
        self.sess = sess
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        self.testing_paths = testing_paths
        self.training_survival_data = training_survival_data
        self.testing_survival_data = testing_survival_data
        
        label = read_label(self.training_paths[0])
        
        self.label_size = label.shape[:-1]
        self.channel = label.shape[-1]
        self.n_input = label.size
        self.batch_size = batch_size
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_z = n_z
        
        self.build_model()
        
        self.saver = tf.train.Saver()
    
    def build_model(self):
        self.labels = tf.placeholder(tf.float32, shape=[None, self.label_size[0], self.label_size[1], self.label_size[2],
                                                        self.channel], name='images')
        self.survival = tf.placeholder(tf.float32, shape=[None, 1], name='survival')
        self.age = tf.placeholder(tf.float32, shape=[None, 1], name='age')
        
        # variational autoencoders
        self.input = tf.reshape(self.labels, [-1, self.n_input])
        with tf.variable_scope('enc') as scope:
            h1 = tf.get_variable('h1', [self.n_input, self.n_hidden_1],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable('b1', [self.n_hidden_1], initializer=tf.constant_initializer(0.0))
            layer_1 = tf.nn.relu(tf.add(tf.matmul(self.input, h1), b1))
            h2 = tf.get_variable('h2', [self.n_hidden_1, self.n_hidden_2],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable('b2', [self.n_hidden_2], initializer=tf.constant_initializer(0.0))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2))
            hmean = tf.get_variable('hmean', [self.n_hidden_2, self.n_z],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            bmean = tf.get_variable('bmean', [self.n_z], initializer=tf.constant_initializer(0.0))
            self.z_mean = tf.add(tf.matmul(layer_2, hmean), bmean)
            hsigma = tf.get_variable('hsigma', [self.n_hidden_2, self.n_z],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            bsigma = tf.get_variable('bsigma', [self.n_z], initializer=tf.constant_initializer(0.0))
            self.z_log_sigma_sq = tf.add(tf.matmul(layer_2, hsigma), bsigma)
        
        eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        
        with tf.variable_scope('dec') as scope:
            h2 = tf.get_variable('h2', [self.n_z, self.n_hidden_2],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable('b2', [self.n_hidden_2], initializer=tf.constant_initializer(0.0))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(self.z, h2), b2))
            h1 = tf.get_variable('h1', [self.n_hidden_2, self.n_hidden_1],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable('b1', [self.n_hidden_1], initializer=tf.constant_initializer(0.0))
            layer_1 = tf.nn.relu(tf.add(tf.matmul(layer_2, h1), b1))
            hmean = tf.get_variable('hmean', [self.n_hidden_1, self.n_input],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            bmean = tf.get_variable('bmean', [self.n_input], initializer=tf.constant_initializer(0.0))
            self.x_recon = tf.nn.sigmoid(tf.clip_by_value(tf.add(tf.matmul(layer_1, hmean), bmean), -30, 30))
            
        self.recon_loss = -tf.reduce_sum(self.input * tf.log(1e-10 + self.x_recon)
                                         + (1 - self.input) * tf.log(1e-10 + 1 - self.x_recon), 1)
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean)
                                                - tf.exp(self.z_log_sigma_sq), 1)
        self.loss = tf.reduce_sum(self.recon_loss + self.latent_loss)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
            
    def train(self, config):
        optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        
        train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'train'), self.sess.graph)
        if self.testing_paths is not None:
            test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'test'))
         
        counter = 0
        for epoch in range(config['epoch']):
            # Shuffle the orders
            training_paths = np.random.permutation(self.training_paths)
            
            # Go through all selected patches
            for f in range(len(training_paths) // self.batch_size):
                labels = np.empty((self.batch_size, self.label_size[0], self.label_size[1], self.label_size[2], self.channel),
                                  dtype=np.float32)
                survivals = np.empty((self.batch_size, 1), dtype=np.float32)
                ages = np.empty((self.batch_size, 1), dtype=np.float32)
                for b in range(self.batch_size):
                    labels[b] = read_label(training_paths[b])
                    survivals[b] = self.training_survival_data[os.path.basename(training_paths[b])][1]
                    ages[b] = self.training_survival_data[os.path.basename(training_paths[b])][0]
                _, train_loss, summary = self.sess.run([optimizer, self.loss, self.loss_summary],
                                                       feed_dict = { self.labels: labels,
                                                                     self.survival: survivals,
                                                                     self.age: ages })
                train_writer.add_summary(summary, counter)
                counter += 1
                if np.mod(counter, 1000) == 0:
                    self.save(counter)
                    
                # Run test
                if self.testing_paths is not None and np.mod(counter, 100) == 0:
                    testing_paths = np.random.permutation(self.testing_paths)
                    for b in range(self.batch_size):
                        labels[b] = read_label(testing_paths[b])
                        survivals[b] = self.testing_survival_data[os.path.basename(testing_paths[b])][1]
                        ages[b] = self.testing_survival_data[os.path.basename(testing_paths[b])][0]
                    test_loss, summary = self.sess.run([self.loss, self.loss_summary],
                                                       feed_dict = { self.labels: labels,
                                                                     self.survival: survivals,
                                                                     self.age: ages })
                    print(str(counter) + ":" + "train_loss: " + str(train_loss) + " test_loss: " + str(test_loss))
                    test_writer.add_summary(summary, counter)
                    
        # Save in the end
        self.save(counter)
       
    def deploy(self, input_path, output_path):
        print('deploy')
    
    @property
    def model_dir(self):
        return 'survival_vae'
    
    def save(self, step):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'survivalnet'), global_step=step)
        
    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*\d)', ckpt_name)).group(0))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0