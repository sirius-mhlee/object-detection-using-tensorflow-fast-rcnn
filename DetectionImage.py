import sys
import cv2
import os
import random as rand

import numpy as np
import tensorflow as tf

import Configuration as cfg

import AlexNet as an

import DataOperator as do
import SelectiveSearch as ss

def generate_image(label_file_path, img, nms_detect_list):
    label_file = open(label_file_path, 'r')
    synset = [line.strip() for line in label_file.readlines()]
    label_file.close()

    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(len(synset))]

    save_img = img.copy()
    height, width, channel = save_img.shape

    for detect in nms_detect_list:
        left = int(max(detect[2], 0))
        top = int(max(detect[3], 0))
        right = int(min(detect[4], width))
        bottom = int(min(detect[5], height))

        cv2.rectangle(save_img, (left, top), (right, bottom), color[detect[0]], 2)

        text_size, baseline = cv2.getTextSize(' ' + synset[detect[0]] + ' ', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(save_img, (left, top - text_size[1] - (baseline * 2)), (left + text_size[0], top), color[detect[0]], -1)
        cv2.putText(save_img, ' ' + synset[detect[0]] + ' ', (left, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return save_img

def main():
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, cfg.image_size_width, cfg.image_size_height, 3])

        model = do.load_model(sys.argv[1])
        mean = do.load_mean(sys.argv[2])
        alexnet_model = an.AlexNet(model, mean, False)
        with tf.name_scope('alexnet_content'):
            alexnet_model.build(image)
            alexnet_model.build_finetune()

        sess.run(tf.global_variables_initializer())

        detect_list = []
        img = cv2.imread(sys.argv[6])
        proposal = ss.selective_search_image(cfg.sigma, cfg.k, cfg.min_size, cfg.smallest, cfg.largest, cfg.distortion, img)
        for region in proposal:
            region_img = do.load_region_image(sys.argv[6], region.rect.left, region.rect.top, region.rect.right, region.rect.bottom)
            
            feed_dict = {image:region_img}
            region_prob, region_bbox = sess.run([alexnet_model.finetune_fc8, alexnet_model.finetune_bbox1], feed_dict=feed_dict)

            label = np.argmax(region_prob[0])
            if label != cfg.object_class_num:
                region_width = region.rect.right - region.rect.left
                region_hegith = region.rect.bottom - region.rect.top
                region_center_x = region.rect.left + region_width / 2
                region_center_y = region.rect.top + region_hegith / 2

                bbox_center_x = region_width * region_bbox[0][0] + region_center_x
                bbox_center_y = region_hegith * region_bbox[0][1] + region_center_y
                bbox_width = region_width * np.exp(region_bbox[0][2])
                bbox_height = region_hegith * np.exp(region_bbox[0][3])

                bbox_left = bbox_center_x - bbox_width / 2
                bbox_top = bbox_center_y - bbox_height / 2
                bbox_right = bbox_center_x + bbox_width / 2
                bbox_bottom = bbox_center_y + bbox_height / 2

                detect_list.append((label, region_prob[0][label], bbox_left, bbox_top, bbox_right, bbox_bottom))

        nms_detect_list = []
        for i in range(len(detect_list)):
            check_suppression = False
            for j in range(len(detect_list)):
                if i != j:
                    bbox1 = (detect_list[i][2], detect_list[i][3], detect_list[i][4], detect_list[i][5])
                    bbox2 = (detect_list[j][2], detect_list[j][3], detect_list[j][4], detect_list[j][5])
                    iou = do.get_intersection_over_union(bbox1, bbox2)
                    if iou > 0.3:
                        if detect_list[j][1] > detect_list[i][1]:
                            check_suppression = True
                            break;

            if not check_suppression:
                nms_detect_list.append(detect_list[i])

        save_img = generate_image(sys.argv[5], img, nms_detect_list)
        cv2.imwrite(sys.argv[7], save_img)

if __name__ == '__main__':
    main()
