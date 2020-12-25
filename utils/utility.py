import pandas as pd
from skimage.draw import rectangle
import numpy as np
import torchvision
import torch

def csv_preprocess(csv):
    
    df = pd.read_csv(csv)
    label_list = list()
    
    for index, row in df.iterrows():
        
        bbox_list = list()
        bbox_list_gt = df.iloc[index]['labels'].split(" ")
        
        for j in range(0, len(bbox_list_gt), 5):
            
            uid, x, y, width, height = bbox_list_gt[j:j+5]
            bbox_list.append([uid, int(x) + int(width) // 2, int(y) + int(height) // 2, int(width), int(height)])    
        
        label_list.append([df.iloc[index]['image_id'], bbox_list])
        
    return label_list
        

def calc_iou(bbox_pred, _nms_index, val_list, dindex, imshape):
    mask_pred = np.zeros(imshape)
    mask_gt = np.zeros(imshape)
    
    for i in range(len(_nms_index)):
    
        top, left, bottom, right = bbox_pred[_nms_index[i]]
        
        top = int(top)
        left= int(left)
        bottom = int(bottom)
        right = int(right)
        
        start = (top, left)
        end = (bottom, right)
        
        rrf, ccf = rectangle(start, end=end, shape=mask_pred.shape)
        
        mask_pred[rrf, ccf] = 1
        
    for j in range(1, len(val_list[dindex][1])):
        x, y, width, height = val_list[dindex][1][j][1:5]
        
        
        top = y - height // 2
        left = x - width // 2
        bottom = y + height // 2
        right = x + width // 2
            
        start = (top, left)
        end = (bottom, right)
            
        rrf, ccf = rectangle(start, end=end, shape=mask_gt.shape)
            
        mask_gt[rrf, ccf] = 1
        
    intersection = np.multiply(mask_pred, mask_gt).sum()
    iou = intersection / (mask_pred.sum() + mask_gt.sum() - intersection)
    
    return iou

def _nms_eval_iou(gt, predict, img_width, img_height, output_size, nms_score, iou_threshold):
    
    
    mask_pred = get_pred_mask(predict, img_width, img_height, output_size, nms_score, iou_threshold)
    mask_gt = get_gt_mask(gt, img_width, img_height, output_size)
    
    intersection = np.multiply(mask_pred, mask_gt).sum()
    iou = intersection / (mask_pred.sum() + mask_gt.sum() - intersection)
        
    return iou

def get_pred_mask(predict, img_width, img_height, output_size, nms_score, iou_threshold):
    bbox = list()
    score_list = list()
    
    heatmap = predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]
    
    for i in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:

        row = i // output_size 
        col = i - row*output_size
        
        bias_x = offset_x[row, col] * (img_height / output_size)
        bias_y = offset_y[row, col] * (img_width / output_size)

        width = width_map[row, col] * output_size * (img_height / output_size)
        height = height_map[row, col] * output_size * (img_width / output_size)

        score_list.append(heatmap[row, col])

        row = row * (img_height / output_size) + bias_y
        col = col * (img_width / output_size) + bias_x

        top = row - width // 2
        left = col - height // 2
        bottom = row + width // 2
        right = col + height // 2

        start = (top, left)
        end = (bottom, right)

        bbox.append([top, left, bottom, right])
        
    if len(bbox) == 0:
        print('No object was found in the image')
        bbox.append([0, 0, 0, 0])
        score_list.append(0)
        
    _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)), iou_threshold=iou_threshold)
    
    mask_pred = np.zeros((img_height, img_width))
    for i in range(len(_nms_index)):
    
        top, left, bottom, right = bbox[_nms_index[i]]
        
        top = int(top)
        left= int(left)
        bottom = int(bottom)
        right = int(right)
        
        start = (top, left)
        end = (bottom, right)
        
        rrf, ccf = rectangle(start, end=end, shape=mask_pred.shape)
        
        mask_pred[rrf, ccf] = 1
    
    return mask_pred

def get_gt_mask(gt, img_width, img_height, output_size):
    bbox_gt = list()
    
    heatmap_gt = gt.data.cpu().numpy()[0, ..., 1]
    offset_y_gt = gt.data.cpu().numpy()[0, ..., 2]
    offset_x_gt = gt.data.cpu().numpy()[0, ..., 3]
    width_map_gt = gt.data.cpu().numpy()[0, ..., 4]
    height_map_gt = gt.data.cpu().numpy()[0, ..., 5]
    
    for j in np.where(heatmap_gt.reshape(-1, 1) == 1)[0]:

        row = j // output_size 
        col = j - row*output_size
        
        bias_x_gt = offset_x_gt[row, col] * (img_height / output_size)
        bias_y_gt = offset_y_gt[row, col] * (img_width / output_size)

        width = width_map_gt[row, col] * output_size * (img_height / output_size)
        height = height_map_gt[row, col] * output_size * (img_width / output_size)

        row = row * (img_height / output_size) + bias_y_gt
        col = col * (img_width / output_size) + bias_x_gt

        top = row - width // 2
        left = col - height // 2
        bottom = row + width // 2
        right = col + height // 2

        start = (top, left)
        end = (bottom, right)

        bbox_gt.append([top, left, bottom, right])
    
    mask_gt = np.zeros((img_height, img_width))
        
    for j in range(len(bbox_gt)):
        top, left, bottom, right = bbox_gt[j]
        
        top = int(top)
        left= int(left)
        bottom = int(bottom)
        right = int(right)
        
        start = (top, left)
        end = (bottom, right)
        
        rrf, ccf = rectangle(start, end=end, shape=mask_gt.shape)
        
        mask_gt[rrf, ccf] = 1
        
    return mask_gt
