import pandas as pd
import os
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import ast

#Things that can lead to problem - filename in dict, height, width, use of -1 for none obj

labels_path = 'output_file_final_5k.txt'
column_names = ['image_path','id','body_bbox_x','body_bbox_y','body_bbox_width','body_bbox_height','eye_x','eye_y','gaze_x','gaze_y','head_bbox_x_min','head_bbox_y_min','head_bbox_x_max','head_bbox_y_max','in_or_out','meta','gp_bb_x1','gp_bb_y1','gp_bb_x2','gp_bb_y2','gaze_cls_id','bounding_box']

df_pred = pd.read_csv(
        labels_path,
        sep=",",
        names=column_names,
        usecols=column_names,
        index_col=False,
    )
    
labels_path_new = 'train_annotations_release_new1.txt'
column_names_new = ['image_path','id','body_bbox_x','body_bbox_y','body_bbox_width','body_bbox_height','eye_x','eye_y','gaze_x','gaze_y','head_bbox_x_min','head_bbox_y_min','head_bbox_x_max','head_bbox_y_max','in_or_out','dataset','dataset_img_path','gp_bb_x1','gp_bb_y1','gp_bb_x2','gp_bb_y2','gaze_cls_id']
df_true = pd.read_csv(
        labels_path_new,
        sep=",",
        names=column_names_new,
        usecols=column_names_new,
        index_col=False,
    )

# Define the path to your text file
file_path = 'yolo_class_mapping.txt'

# Read the content of the text file
with open(file_path, 'r') as file:
    file_content = file.read()

# Convert the file content (string) to a dictionary
yolo_dict = ast.literal_eval(file_content)

# Display the dictionary
# print(yolo_dict)
# 


file_path = 'coco_labels_paper.txt'

# Initialize an empty dictionary
coco_dict = {}

# Read the content of the text file line by line
with open(file_path, 'r') as file:
    for line_number, line in enumerate(file, start=1):
        # Strip the newline character and assign the line content to the dictionary
        coco_dict[line_number] = line.strip()

# Display the dictionary
print(coco_dict)

#preprocess new annotation file to remove array symbol []
# with open("train_annotations_release_new.txt", "r") as original_file:
#     # Open a new file for writing
#     with open("train_annotations_release_new1.txt", "w") as new_file:
#         # Iterate through each line in the original file
#         for line in original_file:
#             # Split the line by comma
#             parts = line.strip().split(',')
#             # Join the first 17 elements with comma
#             new_line = ','.join(parts[:17])
#             # Append the bounding box coordinates and class_id
#             new_line += ',' + parts[17].strip('[]"')+','+parts[18]+','+parts[19]+','+parts[20].strip('[]"')+','+parts[21]
#             # Write the new line to the new file
#             # print(new_line)
            
#             new_file.write(new_line + '\n')
          
#function to calculate iou
def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
    box1: list or numpy array, [x1, y1, x2, y2] representing the coordinates of the first bounding box
    box2: list or numpy array, [x1, y1, x2, y2] representing the coordinates of the second bounding box
    
    Returns:
    IoU: float, Intersection over Union between box1 and box2
    """
    box1 = convert_bbox_format(box1)
    
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    
    # Calculate areas of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def convert_bbox_format(bbox):
    """
    Convert bounding box from [x_min, y_min, width, height] format to [x_min, y_min, x_max, y_max] format.
    
    Args:
    bbox: list or numpy array, [x_min, y_min, width, height] representing the coordinates of the bounding box
    
    Returns:
    new_bbox: list, [x_min, y_min, x_max, y_max] representing the coordinates of the bounding box in the new format
    """
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    new_bbox = [x_min, y_min, x_max, y_max]
    return new_bbox

df_true.keys()
df_pred.iloc[0]

mismatch_lst = []
threshold = 0.85
tp = 0
fp = 0
cnt = 0
for idx, r in df_pred.iterrows():
    tval = df_true[df_true['image_path'] == r['image_path']].iloc[0]
    box1 = [int(tval['gp_bb_x1']), int(tval['gp_bb_y1']), int(tval['gp_bb_x2']), int(tval['gp_bb_y2'])]
    # print(tval['gp_bb_x1'])
    box2 = [r['gp_bb_x1'], r['gp_bb_y1'], r['gp_bb_x2'], r['gp_bb_y2']]
    # print(box1)
    # print(box2)
    # print()
    # print(tval['gaze_cls_id'], r['gaze_cls_id'])
    # print()
    match = False
    
    if r['gaze_cls_id'] != -1 and coco_dict[tval['gaze_cls_id']] == yolo_dict[r['gaze_cls_id']]:
        iou = calculate_iou(box1, box2)
        # print(iou)
        if iou > threshold:
            match = True
            
    if match == True:
        tp += 1
    else:
        mismatch_lst.append(tval['image_path'])
        fp += 1
        
    # if cnt == 100: 
    #     break
    # cnt += 1


precision = tp/(tp+fp)

print(precision)

recall = tp/len(df_true)
print(recall)

#########################################################



#########################################################

# i = 32
# img_path = 'Data/data_new/' + mismatch_lst[i] #train/00000014/00014240.jpg'
# img_path1 = mismatch_lst[i] #'train/00000014/00014240.jpg'
# print(img_path1)
# print(df_true[df_true['image_path'] == mismatch_lst[i]].iloc[0]['gaze_cls_id'])
# print(df_pred[df_pred['image_path'] == mismatch_lst[i]].iloc[0]['gaze_cls_id'])

# def plot_bbox_on_image(image_path, bbox_list):
#     """
#     Plot bounding boxes on an image.
    
#     Args:
#     image_path: str, path to the image file
#     bbox_list: list of bounding boxes in the format [x_min, y_min, x_max, y_max]
#     """
#     # Open the image
#     image = Image.open(image_path)
    
#     # Plot the image
#     plt.imshow(image)
    
#     # Get the current axis
#     ax = plt.gca()
    
#     # Add bounding boxes to the plot
#     for bbox in bbox_list:
#         x_min, y_min, x_max, y_max = bbox
#         width = x_max - x_min
#         height = y_max - y_min
#         rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
    
#     # Show the plot
#     plt.show()

# tval = df_true[df_true['image_path'] == img_path1].iloc[0] 
# box1 = [int(tval['gp_bb_x1']), int(tval['gp_bb_y1']), int(tval['gp_bb_x2']), int(tval['gp_bb_y2'])]
# box1 = convert_bbox_format(box1)
# r = df_pred[df_pred['image_path'] == img_path1].iloc[0] 
# box2 = [r['gp_bb_x1'], r['gp_bb_y1'], r['gp_bb_x2'], r['gp_bb_y2']]
# plot_bbox_on_image(img_path, [box1, box2])


