import pandas as pd
import os
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import os
import pickle
import ast

df = pd.read_csv("output_file_final_5k.txt", header=None, names=['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max', 'in_or_out', 'meta', 'gp_bb_x1', 'gp_bb_y1', 'gp_bb_x2', 'gp_bb_y2', 'gaze_cls_id', 'bounding_boxes'], sep=',', skiprows=1)

def extract_bbox_cid(s):
    df['bounding_boxes'][0]
    
    # Convert the string to a list of tuples
    list_of_tuples = ast.literal_eval(s)
    
    # Extract bounding boxes and class ids
    bounding_boxes = [list(tup[:-1]) for tup in list_of_tuples]
    class_ids = [tup[-1] for tup in list_of_tuples]

    return bounding_boxes, class_ids


obj = []

for idx, r in df.iterrows():
    
    t_bbox, t_cid = extract_bbox_cid(r['bounding_boxes'])
    
    filename = r['image_path']
    width = 1920
    height = 1080
    l1 = [r['gp_bb_x1'], r['gp_bb_y1'], r['gp_bb_x2'], r['gp_bb_y2']]
    l2 = [ r['head_bbox_x_min'], r['head_bbox_y_min'], r['head_bbox_x_max'], r['head_bbox_y_max'] ]
    bbox_lst = [l1]
    for i in t_bbox:
        bbox_lst.append(i)
    bbox_lst.append(l2)
    
    labels = [r['gaze_cls_id']]
    for i in t_cid:
        labels.append(i)
    labels.append(81)
    
    
    ann = {'bboxes':np.array(bbox_lst), 'labels':np.array(labels), 'bboxes_ignore': None, 'labels_ignore': None, 'gt_bboxes_ignore': None, 'gt_labels_ignore': None}
    gaze_item = r['gaze_cls_id']
    gazeIdx = 0
    gaze_cx = r['gaze_x']
    gaze_cy = r['gaze_y']
    hx = r['eye_x']
    hy = r['eye_y']
    seg = np.array([0])
    occluded = False
    
    d = {'filename':filename, 'width':width, 'height':height, 'ann':ann, 'gaze_item':gaze_item, 'gazeIdx':gazeIdx, 'gaze_cx':gaze_cx, 'gaze_cy':gaze_cy, 'hx':hx, 'hy':hy, 'seg':seg, 'occluded':occluded}
    obj.append(d)
    

pickle_file_path = 'gazefollow_new_final.pickle'
# Save the list of dictionaries to the pickle file
with open(pickle_file_path, 'wb') as f:
    pickle.dump(obj, f)

