import sys
import cv2
import os

import numpy as np
import tensorflow as tf

import Configuration as cfg

import AlexNet as an

import DataOperator as do

def print_batch_info(epoch_idx, batch_idx, loss_mean_value):
    print('Epoch : {0}, Batch : {1}, Loss Mean : {2}'.format(epoch_idx, batch_idx, loss_mean_value))

def print_epoch_info(epoch_idx, accuracy_mean_value):
    print('Epoch : {0}, Accuracy Mean : {1}'.format(epoch_idx, accuracy_mean_value))

def main():
    with tf.Session() as sess:
        alexnet_train_data, alexnet_train_mean = do.load_alexnet_train_data(sys.argv[1])
        alexnet_train_size = len(alexnet_train_data)

        alexnet_finetune_data = do.load_alexnet_finetune_data(sys.argv[2])
        alexnet_finetune_size = len(alexnet_finetune_data)

        image = tf.placeholder(tf.float32, [None, cfg.image_size_width, cfg.image_size_height, 3])
        label = tf.placeholder(tf.float32, [None, cfg.object_class_num])
        bbox = tf.placeholder(tf.float32, [None, 9])
        bbox_slice_idx = tf.placeholder(tf.int32, [None, 2])
        finetune_label = tf.placeholder(tf.float32, [None, cfg.object_class_num + 1])

        alexnet_model = an.AlexNet(None, alexnet_train_mean, True)
        with tf.name_scope('alexnet_content'):
            alexnet_model.build(image, label)
        with tf.name_scope('alexnet_finetune_content'):
            alexnet_model.build_finetune(bbox, bbox_slice_idx, finetune_label)

        writer = tf.summary.FileWriter('./log/', sess.graph)

        sess.run(tf.global_variables_initializer())

        print('Training AlexNet')
        for epoch_idx in range(cfg.training_max_epoch):
            for batch_idx in range(alexnet_train_size // cfg.batch_size):
                batch_image, batch_label = do.get_alexnet_train_batch_data(sess, alexnet_train_data, cfg.batch_size)
                feed_dict = {image:batch_image, label:batch_label}

                _, loss_mean_value = sess.run([alexnet_model.optimizer, alexnet_model.loss_mean], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_mean_value)

            batch_image, batch_label = do.get_alexnet_train_batch_data(sess, alexnet_train_data, cfg.batch_size)
            feed_dict = {image:batch_image, label:batch_label}

            accuracy_mean_value = sess.run(alexnet_model.accuracy_mean, feed_dict=feed_dict)
            print_epoch_info(epoch_idx, accuracy_mean_value)

        print('Finetuning AlexNet')
        for epoch_idx in range(cfg.finetuning_max_epoch):
            for batch_idx in range(alexnet_finetune_size // cfg.batch_size):
                batch_image, batch_bbox, batch_bbox_slice_idx, batch_label = do.get_alexnet_finetune_batch_data(sess, alexnet_finetune_data, cfg.batch_size)
                feed_dict = {image:batch_image, bbox:batch_bbox, bbox_slice_idx:batch_bbox_slice_idx, finetune_label:batch_label}

                _, loss_mean_value = sess.run([alexnet_model.finetune_optimizer, alexnet_model.finetune_loss_mean], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_mean_value)

            batch_image, batch_bbox, batch_bbox_slice_idx, batch_label = do.get_alexnet_finetune_batch_data(sess, alexnet_finetune_data, cfg.batch_size)
            feed_dict = {image:batch_image, bbox:batch_bbox, bbox_slice_idx:batch_bbox_slice_idx, finetune_label:batch_label}

            accuracy_mean_value = sess.run(alexnet_model.finetune_accuracy_mean, feed_dict=feed_dict)
            print_epoch_info(epoch_idx, accuracy_mean_value)

        do.save_model(sess, alexnet_model.var_dict, sys.argv[3])
        do.save_mean(alexnet_model.mean, sys.argv[4])

if __name__ == '__main__':
    main()
