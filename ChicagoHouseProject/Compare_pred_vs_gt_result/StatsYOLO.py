# Author: Tao Zang
# Date: Oct 20, 2020
# Contact: taozang2@illinois.edu

import json
import pandas as pd
import numpy as np
import os
import csv

def count(class_id, class_count: np.ndarray):
    # helper function to count the number of label 0 objects and label 1 objects shown in the image
    if class_id == 0:
        class_count[0] += 1
    if class_id == 1:
        class_count[1] += 1
    return class_count

def get_single_gt_class_count(labelfile):
    # function to count the number of label 0 objects and label 1 objects shown in a single image
    # labelname: the name of the text file storing the yolo ground truth label
    # output: 
    #      image_name: string
    #      gt_class_count: np.array([num_of_class_0, num_of_class_1])
    image_name = labelfile.split('.txt')[0] + '.jpg'
    gt = np.loadtxt('Test/' + labelfile)
    gt_class_count = np.zeros(2, dtype='int')
    if gt.size == 0: 
        print("Empty! %s contains no label."%labelname)
    if gt.ndim <= 1:
        class_id = gt[0]
        gt_class_count = count(class_id, gt_class_count)
    else:
        for n, bb in enumerate(gt):
            class_id = bb[0]
            gt_class_count = count(class_id, gt_class_count)
    return image_name, gt_class_count

def ground_truth_class_count():
    # function to count the number of label 0 objects and label 1 objects for entire dataset
    # output: 
    #      gt_list: the data contains class 0 and class 1 counts
    #      csv file: write using gt_list
    # reference: https://www.programiz.com/python-programming/writing-csv-files
    gt_list = [['imageName', 'gt_num_class_0', 'gt_num_class_1', 'gt_total_single_image']]
    total_num_gt_0 = 0
    total_num_gt_1 = 0
    for filename in os.listdir('Test/'):
        if filename.endswith('.txt'):
            image_name, gt_class_count = get_single_gt_class_count(filename)
            num_gt_0 = gt_class_count[0]
            total_num_gt_0 += num_gt_0
            num_gt_1 = gt_class_count[1]
            total_num_gt_1 += num_gt_1
            total_single_image = num_gt_0 + num_gt_1
            row = [image_name, num_gt_0, num_gt_1, total_single_image]
            gt_list.append(row)
    gt_list.append(['gt_total_dataset', total_num_gt_0, total_num_gt_1, total_num_gt_0 + total_num_gt_1])
    with open('gt_class_count.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(gt_list)
    print("The ground truth class count is stored as gt_class_count.csv.")
    return gt_list

def get_single_pred_class_count(curr_data, threshold):
    assert threshold <= 1
    pred_class_count = np.zeros(2, dtype='int')
    for n, box in enumerate(curr_data['objects']):
        if box['confidence'] >= threshold:
            class_id = box['class_id']
            pred_class_count = count(class_id, pred_class_count)
    return pred_class_count

def pred_class_count(jsonfile, threshold):
    pred_list = [['imageName', 'pred_num_class_0', 'pred_num_class_1', 'pred_total_single_img']]
    total_num_pred_0 = 0
    total_num_pred_1 = 0
    with open(jsonfile) as f:
        data = json.load(f)
    N = len(data)
    for i in range(N):
        image_name = data[i]['filename'].split('/')[2]
        pred_class_count = get_single_pred_class_count(data[i], threshold)
        num_pred_0 = pred_class_count[0]
        total_num_pred_0 += num_pred_0
        num_pred_1 = pred_class_count[1]
        total_num_pred_1 += num_pred_1
        row = [image_name, num_pred_0, num_pred_1, num_pred_0 + num_pred_1]
        pred_list.append(row)
    pred_list.append(['pred_total_dataset', total_num_pred_0, total_num_pred_1, total_num_pred_0 + total_num_pred_1])
    with open('pred_class_count.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(pred_list)
    print("The prediction class count is stored as pred_class_count.csv.")
    return pred_list