import cv2
import random as rand

import numpy as np
import tensorflow as tf

import Configuration as cfg
import SelectiveSearch as ss

def load_model(model_path):
    model = np.load(model_path, encoding='latin1').item()
    return model

def save_model(sess, var_dict, model_path):
    data_dict = {}

    for (name, idx), var in list(var_dict.items()):
        var_out = sess.run(var)
        if name not in data_dict:
            data_dict[name] = {}
        data_dict[name][idx] = var_out

    np.save(model_path, data_dict)

def load_mean(mean_path):
    mean_file = open(mean_path, 'r')
    line = mean_file.readline()
    split_line = line.split(' ')
    mean = [float(split_line[0]), float(split_line[1]), float(split_line[2])]
    mean_file.close()
    return mean

def save_mean(mean, mean_path):
    mean_file = open(mean_path, 'w')
    mean_file.write('{0} {1} {2}'.format(mean[0], mean[1], mean[2]))
    mean_file.close()

def load_image(img_path):
    img = cv2.imread(img_path)
    height, width, channel = img.shape
    reshape_img = cv2.resize(img, dsize=(cfg.image_size_width, cfg.image_size_height), interpolation=cv2.INTER_CUBIC)
    np_img = np.asarray(reshape_img, dtype=float)
    expand_np_img = np.expand_dims(np_img, axis=0)
    return expand_np_img, width, height

def get_intersection_over_union(box1, box2):
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)

    area_intersection = inter_width * inter_height
    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union
    return iou

def load_alexnet_train_data(train_data_path):
    train_data = []
    train_mean = [0, 0, 0]

    train_file = open(train_data_path, 'r')
    all_line = train_file.readlines()
    for line in all_line:
        split_line = line.split(' ')
        train_data.append((split_line[0], int(split_line[1])))

        expand_np_img, _, _ = load_image(split_line[0])
        train_mean += np.mean(expand_np_img, axis=(0, 1, 2))
    train_file.close()

    train_mean /= len(train_data)

    return train_data, train_mean

def get_alexnet_train_batch_data(sess, train_data, batch_size):
    rand.shuffle(train_data)
    
    image = []
    label = []

    batch_data = train_data[:batch_size]
    for data in batch_data:
        expand_np_img, _, _ = load_image(data[0])
        image.append(expand_np_img)
        label.append(data[1])

    batch_image = np.concatenate(image)
    batch_label_op = tf.one_hot(label, on_value=1, off_value=0, depth=cfg.object_class_num)
    batch_label = sess.run(batch_label_op)

    return batch_image, batch_label
    
def load_alexnet_finetune_data(train_data_path):
    train_data = {}

    train_file = open(train_data_path, 'r')
    all_line = train_file.readlines()
    for line in all_line:
        split_line = line.split(' ')
        img = cv2.imread(split_line[0])
        ground_truth_label = int(split_line[1])
        ground_truth_bbox = (int(split_line[2]), int(split_line[3]), int(split_line[2]) + int(split_line[4]), int(split_line[3]) + int(split_line[5]))
        train_data[split_line[0]] = []

        proposal = ss.selective_search_image(cfg.sigma, cfg.k, cfg.min_size, cfg.smallest, cfg.largest, cfg.distortion, img)
        for region in proposal:
            label = ground_truth_label
            region_bbox = (region.rect.left, region.rect.top, region.rect.right, region.rect.bottom)
            iou = get_intersection_over_union(region_bbox, ground_truth_bbox)
            if iou < 0.5:
                label = cfg.object_class_num
            train_data[split_line[0]].append((label, ground_truth_bbox[0], ground_truth_bbox[1], ground_truth_bbox[2], ground_truth_bbox[3], region_bbox[0], region_bbox[1], region_bbox[2], region_bbox[3]))
    train_file.close()

    return train_data

def get_alexnet_finetune_batch_data(sess, train_data, batch_size):    
    image = []
    bbox = []
    bbox_slice_idx = []
    label = []

    train_data_key = list(train_data.keys())
    rand.shuffle(train_data_key)
    batch_data_key = train_data_key[:batch_size]

    for key_idx, data_key in enumerate(batch_data_key):
        expand_np_img, width, height = load_image(data_key)
        image.append(expand_np_img)

        train_data_value = train_data[data_key]
        rand.shuffle(train_data_value)
        batch_data_value = train_data_value[:cfg.region_per_batch]

        region_scale_width = cfg.image_size_width / width
        region_scale_height = cfg.image_size_height / height

        for value_idx, data_value in enumerate(batch_data_value):
            data_value[1] *= region_scale_width
            data_value[2] *= region_scale_height
            data_value[3] *= region_scale_width
            data_value[4] *= region_scale_height

            data_value[5] *= region_scale_width
            data_value[6] *= region_scale_height
            data_value[7] *= region_scale_width
            data_value[8] *= region_scale_height

            region_width = data_value[7] - data_value[5]
            region_hegith = data_value[8] - data_value[6]
            region_center_x = data_value[5] + region_width / 2
            region_center_y = data_value[6] + region_hegith / 2

            ground_truth_width = data_value[3] - data_value[1]
            ground_truth_height = data_value[4] - data_value[2]
            ground_truth_center_x = data_value[1] + ground_truth_width / 2
            ground_truth_center_y = data_value[2] + ground_truth_height / 2

            target_x = (ground_truth_center_x - region_center_x) / region_width
            target_y = (ground_truth_center_y - region_center_y) / region_hegith
            target_width = np.log(ground_truth_width / region_width)
            target_height = np.log(ground_truth_height / region_hegith)

            bbox.append((key_idx, data_value[6], data_value[5], data_value[8], data_value[7], target_x, target_y, target_width, target_height))

            bbox_slice_idx_label = data_value[0]
            if bbox_slice_idx_label == cfg.object_class_num:
                bbox_slice_idx_label = 0
            bbox_slice_idx.append(((key_idx * cfg.region_per_batch) + value_idx, bbox_slice_idx_label))

            label.append(data_value[0])

    batch_image = np.concatenate(image)
    batch_bbox_op = tf.convert_to_tensor(bbox, dtype=tf.float32)
    batch_bbox = sess.run(batch_bbox_op)
    batch_bbox_slice_idx_op = tf.convert_to_tensor(bbox_slice_idx, dtype=tf.float32)
    batch_bbox_slice_idx = sess.run(batch_bbox_slice_idx_op)
    batch_label_op = tf.one_hot(label, on_value=1, off_value=0, depth=cfg.object_class_num + 1)
    batch_label = sess.run(batch_label_op)

    return batch_image, batch_bbox, batch_bbox_slice_idx, batch_label
