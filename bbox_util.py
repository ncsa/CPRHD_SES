import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import numpy as np
import matplotlib.pyplot as plt

def convertYOLO2Other(yolo_bb: np.ndarray):
    # This function converts yolo bounding box to coordinates accpeted by imgaug library.
    # yolo_bb: a numpy array generated from the result json
    H,W = 640, 640
    if yolo_bb.size == 0: return yolo_bb
    imgaug_bb = np.zeros(yolo_bb.shape)
    if yolo_bb.ndim <= 1:
        class_id = yolo_bb[0]
        center_x, center_y, w, h = yolo_bb[1] * W, yolo_bb[2] * H, yolo_bb[3] * W, yolo_bb[4] * H
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        conf = yolo_bb[5]
        imgaug_bb = np.array([class_id, x1, y1, x2, y2, conf])
        imgaug_bb = imgaug_bb.reshape(1,-1)
    else:
        for n, bb in enumerate(yolo_bb):
            class_id = bb[0]
            center_x, center_y, w, h = bb[1] * W, bb[2] * H, bb[3] * W, bb[4] * H
            x1 = center_x - w / 2
            y1 = center_y - h / 2
            x2 = center_x + w / 2
            y2 = center_y + h / 2
            conf = bb[5]
            imgaug_bb[n] = np.array([class_id, x1, y1, x2, y2, conf])
    return imgaug_bb

def read_ith_pred_result_yolo(data, i, threshold=0):
    # data: the json file
    # i: the index of the json file
    # threshold: confidence threshold
    assert threshold <= 1
    filename = data[i]['filename'][15:]
    img = imageio.imread('Test/' + filename)

    # data[i] is the i-th image
    num_bbox = len(data[i]['objects'])
    pred_bbox_single_img = np.zeros([num_bbox, 6])
    for n, box in enumerate(data[i]['objects']):
        if box['confidence'] >= threshold:
            tmp = np.zeros(6)
            tmp[0] = box['class_id']
            tmp[1] = box['relative_coordinates']['center_x']
            tmp[2] = box['relative_coordinates']['center_y']
            tmp[3] = box['relative_coordinates']['width']
            tmp[4] = box['relative_coordinates']['height']
            tmp[5] = box['confidence']
            pred_bbox_single_img[n, :] = tmp
    return filename, img, pred_bbox_single_img

def draw_ground_truth_label(labelname, img):
    yolo_bb = np.loadtxt('Test/' + labelname)
    H,W = 640, 640
    if yolo_bb.size == 0: return yolo_bb
    imgaug_bb = np.zeros(yolo_bb.shape)
    if yolo_bb.ndim <= 1:
        class_id = yolo_bb[0]
        center_x, center_y, w, h = yolo_bb[1] * W, yolo_bb[2] * H, yolo_bb[3] * W, yolo_bb[4] * H
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        imgaug_bb = np.array([class_id, x1, y1, x2, y2])
        imgaug_bb = imgaug_bb.reshape(1,-1)
    else:
        for n, bb in enumerate(yolo_bb):
            class_id = bb[0]
            center_x, center_y, w, h = bb[1] * W, bb[2] * H, bb[3] * W, bb[4] * H
            x1 = center_x - w / 2
            y1 = center_y - h / 2
            x2 = center_x + w / 2
            y2 = center_y + h / 2
            imgaug_bb[n] = np.array([class_id, x1, y1, x2, y2])
            
    gt = BoundingBoxesOnImage.from_xyxy_array(imgaug_bb[:, 1:], shape=img.shape)
    for n, bb in enumerate(gt):
        bb.label = imgaug_bb[n, 0].astype('int')
        
    return gt.draw_on_image(img, size=2, color=[0,0,255])

def analyze_pred_gt_image(data, i, threshold):
    filename, img, single_image_yolo_pred = read_ith_pred_result_yolo(data, i, threshold)
    single_image_imgaug_pred = convertYOLO2Other(single_image_yolo_pred)
    imgaug_bbox_single_img = BoundingBoxesOnImage.from_xyxy_array(single_image_imgaug_pred[:, 1:5], shape=img.shape)
    for n, bb in enumerate(imgaug_bbox_single_img):
        bb.label = single_image_imgaug_pred[n, 0].astype('int')

    print(filename)
    print(single_image_yolo_pred)
    print(imgaug_bbox_single_img)
    img_with_yolo = imgaug_bbox_single_img.draw_on_image(img, size=2)

    # draw the ground truth
    label_file = filename.split('.jpg')[0] + '.txt'
    img_with_gt = draw_ground_truth_label(label_file, img)
    
    ia.imshow(np.hstack([img_with_yolo, img_with_gt]))
    imageio.imwrite("sampleImages/pred_vs_gt_bbox_" + filename, np.hstack([img_with_yolo, img_with_gt]))
    
