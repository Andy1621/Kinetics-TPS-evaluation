# -*- coding:utf-8 -*-

from __future__ import division

import sys
import os
import os.path as osp
import json
import scipy
import scipy.optimize
import numpy as np

USE_BATCH_IOU = False

def bbox_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    # Correction: reject non-overlapping boxes
    if interW <=0 or interH <=0 :
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def batch_bbox_iou(gt_boxes, pred_boxes):
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return 0
    
    # 1. calculate the inters coordinate
    gt_boxes = gt_boxes[:, np.newaxis]
    pred_boxes = pred_boxes[np.newaxis, :]
    ixmin = np.maximum(pred_boxes[:, :, 0], gt_boxes[:, :, 0])
    ixmax = np.minimum(pred_boxes[:, :, 2], gt_boxes[:, :, 2])
    iymin = np.maximum(pred_boxes[:, :, 1], gt_boxes[:, :, 1])
    iymax = np.minimum(pred_boxes[:, :, 3], gt_boxes[:, :, 3])

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
    inters = iw * ih

    # 3.calculate the area of union
    uni = ((pred_boxes[:, :, 2] - pred_boxes[:, :, 0] + 1.) * (pred_boxes[:, :, 3] - pred_boxes[:, :, 1] + 1.) +
           (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1.) * (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1.) - inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
    iou = inters / uni
    
    return iou 

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    #import ipdb;ipdb.set_trace()
    if USE_BATCH_IOU:
        iou_matrix = batch_bbox_iou(bbox_gt, bbox_pred)
    else:
        iou_matrix = np.zeros((n_true, n_pred), dtype=np.float32)
        for i in range(n_true):
            for j in range(n_pred):
                iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label


def match_humans(gt_humans, pred_humans, IOU_THRESH=0.5):
    gt_boxes = []
    pred_boxes = []
    for i in range(len(gt_humans)):
        gt_box = gt_humans[i]['box']
        gt_boxes.append(gt_box)
    for i in range(len(pred_humans)):
        pred_box = pred_humans[i]['box']
        pred_boxes.append(pred_box)
    gt_boxes = np.float32(gt_boxes)
    pred_boxes = np.float32(pred_boxes)
    idx_gt, idx_pred, ious, valid = match_bboxes(gt_boxes, pred_boxes, IOU_THRESH)
    match_info = {}
    for i in range(len(idx_gt)):
        match_info[idx_gt[i]] = [idx_pred[i], ious[i], pred_humans[idx_pred[i]]]  # idx_pred, iou, pre_human_info
    #print(match_info)
    return match_info

def cal_pvacc(part_vid_acc, t):
    cor = 0
    total = len(part_vid_acc)
    for vid, pacc, vacc in part_vid_acc:
        if pacc >= t and vacc == 1:
            cor += 1
    return cor/total

def cal_area(part_vid_acc, steps=10000):
    area = 0
    for t in range(steps):
        if t == 0:
            continue
        t = t/steps
        area += 1.0 / steps * cal_pvacc(part_vid_acc, t)
    return area

def eval_human_TPS(gt_human, pred_human_info, human_iou, max_human_iou, max_part_iou):
    human_part_acc_sum = 0
    for part_name in gt_human['parts'].keys():
        part_correct = 0
        pred_part_num = 0

        gt_part = gt_human['parts'][part_name]
        gt_part_verb = gt_part['verb']

        if part_name in pred_human_info['parts'] and human_iou >= max_human_iou:
            pred_part_verbs = pred_human_info['parts'][part_name]['verb']
            pred_part_boxs = pred_human_info['parts'][part_name]['box']
            assert len(pred_part_verbs) <= 5, print('can only predict 5  proposals on each part!')
            pred_part_verbs = [pred_part_verbs]
            pred_part_boxs = [pred_part_boxs]

            gt_part_box = gt_part['box']
            pred_part_num += len(pred_part_verbs)
            #import ipdb;ipdb.set_trace()
            if USE_BATCH_IOU:
                part_ious = batch_bbox_iou(np.float32([gt_part_box]), np.float32(pred_part_boxs))[0]
            for j in range(len(pred_part_boxs)):
                pred_part_box = pred_part_boxs[j]
                pred_part_verb = pred_part_verbs[j]
                part_iou = part_ious[j] if USE_BATCH_IOU else bbox_iou(gt_part_box, pred_part_box)
                if part_iou >= max_part_iou and gt_part_verb == pred_part_verb:
                    part_correct = 1
                    break
        part_acc = 1.0 * part_correct / pred_part_num if pred_part_num > 0 else 0
        human_part_acc_sum += part_acc
    return human_part_acc_sum

def eval_video_TPS(vid, gt_part_result, pred_part_result, max_human_iou=0.5, max_part_iou=0.3):
    total_frames = len(gt_part_result[vid])
    #total_frames = len(pred_part_result[vid])
    video_part_acc_sum = 0
    
    for frame in gt_part_result[vid].keys():
        frame_part_counts = 0
        frame_part_acc_sum = 0

        gt_humans = gt_part_result[vid][frame]['humans']
        if len(gt_humans) == 0: # empty frame
            total_frames -= 1
            continue
        if frame not in pred_part_result[vid]:
            continue            

        pred_humans = pred_part_result[vid][frame]['humans']
        match_human_info = match_humans(gt_humans, pred_humans, max_human_iou)
        assert len(pred_humans) <= 10, print('can only predict 10 human proposals on each frame!')
        for i in range(len(gt_humans)): 
            gt_human = gt_humans[i]
            frame_part_counts += len(gt_human['parts'])
            human_part_acc_sum = 0
          #  import ipdb;ipdb.set_trace()
            if i in match_human_info:
                human_part_acc_sum = eval_human_TPS(gt_human, match_human_info[i][2], match_human_info[i][1],
                                                    max_human_iou, max_part_iou)
            frame_part_acc_sum += human_part_acc_sum
        if frame_part_counts == 0: # no parts on this frame, skip
            total_frames -= 1
        else:
            frame_part_acc_avg = 1.0 * frame_part_acc_sum / frame_part_counts
            video_part_acc_sum += frame_part_acc_avg

    if total_frames == 0: # no valid part on this video, part_acc set as 1
        video_part_acc_avg = 1.0
    else:
        video_part_acc_avg = 1.0 * video_part_acc_sum / total_frames

    return video_part_acc_avg


def cal_task_acc(gt_vid_result, gt_part_result, pred_vid_result, pred_part_result,
                 max_human_iou=0.5, max_part_iou=0.3):
    # List[video name, avg_part_acc, video_correct]
    part_vid_acc = []
    for vid in gt_vid_result:
        gt_cla = gt_vid_result[vid]
        pred_cla = pred_vid_result[vid]
        vid_correct = 1 if gt_cla == pred_cla else 0
        video_part_acc_avg = eval_video_TPS(vid, gt_part_result, pred_part_result, 
                             max_human_iou, max_part_iou)
        part_vid_acc.append([vid, video_part_acc_avg, vid_correct])
    return part_vid_acc

def eval(gt_dir = 'gt_dir', pred_dir = 'pred_dir'):
    gt_vid_result = json.load(open(osp.join(gt_dir, 'gt_vid_result.json')))
    gt_part_result = json.load(open(osp.join(gt_dir, 'gt_part_result.json')))
    pred_vid_result  = json.load(open(osp.join(pred_dir, 'pred_vid_result.json')))
    pred_part_result = json.load(open(osp.join(pred_dir, 'pred_part_result.json')))

    part_vid_acc = cal_task_acc(gt_vid_result, gt_part_result, pred_vid_result, pred_part_result, 
                    max_human_iou=0.5, max_part_iou=0.3)
    for hi in [0.001, 0.3, 0.5, 0.9, 0.99]:
        for pi in [0.001, 0.3, 0.5, 0.9, 0.99]:
            print('human iou: ', hi, "part iou: ", pi)
            part_vid_acc = cal_task_acc(gt_vid_result, gt_part_result, pred_vid_result, pred_part_result, 
                            max_human_iou=hi, max_part_iou=pi)
            for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                vacc = cal_pvacc(part_vid_acc, t)
                
                print(vacc)
            print("area: ", cal_area(part_vid_acc))


if __name__ == '__main__':
    eval()

