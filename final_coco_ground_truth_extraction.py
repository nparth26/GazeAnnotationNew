import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import ast

#importing ms coco ground truth
file_path = '/home/parth/Parth SBU/Sem2/CV/project/coco/annotations_trainval2014/annotations/instances_val2014.json'
with open(file_path, 'r') as file:
    data_val = json.load(file)
file_path = '/home/parth/Parth SBU/Sem2/CV/project/coco/annotations_trainval2014/annotations/instances_train2014.json'
with open(file_path, 'r') as file:
    data_train = json.load(file)
    
#function to get image id from image name
def get_img_id(img_coco, dataset):
  img_coco_id = 0
  if dataset == 'coco':
    for i in data_train['images']:
        if i['file_name'] == img_coco:
            img_coco_id = i['id']
  elif dataset == 'coco_val':
    for i in data_val['images']:
        if i['file_name'] == img_coco:
            img_coco_id = i['id']

  return img_coco_id

coco_val = COCO('coco/annotations_trainval2014/annotations/instances_val2014.json')
coco_train = COCO('coco/annotations_trainval2014/annotations/instances_train2014.json')

DATA_PATH = '/home/parth/Parth SBU/Sem2/CV/project/gazefollow_extended/'

labels_path = 'test_annotations_release.txt'

column_names = ['image_path','id','body_bbox_x','body_bbox_y','body_bbox_width','body_bbox_height','eye_x','eye_y','gaze_x','gaze_y','head_bbox_x_min','head_bbox_y_min','head_bbox_x_max','head_bbox_y_max','dataset','dataset_img_path']
column_names_new = ['image_path','id','body_bbox_x','body_bbox_y','body_bbox_width','body_bbox_height','eye_x','eye_y','gaze_x','gaze_y','head_bbox_x_min','head_bbox_y_min','head_bbox_x_max','head_bbox_y_max','dataset','dataset_img_path', 'gaze_obj_bbox', 'gaze_obj_category']
df = pd.read_csv(
        DATA_PATH+labels_path,
        sep=",",
        names=column_names,
        usecols=column_names,
        index_col=False,
    )

df_coco = df[(df['dataset'] == 'coco_val') | (df['dataset'] == 'coco')].reset_index(drop=True)
# df_coco = df_coco[df_coco['in_or_out'] == 1].reset_index(drop=True)


bbox_lst = []
cat_id_lst = []

no_bbox_lst = []
multi_bbox_lst = []
no_bbox = 0
multi_bbox = 0

df_final = pd.DataFrame(columns=column_names_new)

for idx, r in tqdm(df_coco.iterrows()):
  
  img_coco_id = get_img_id(r['dataset_img_path'], r['dataset'])
  temp_lst = []
  temp_cat_lst = []
  # print(img_coco_id)
  if r['dataset'] == 'coco':
    ann_ids = coco_train.getAnnIds(imgIds=img_coco_id)
    anns = coco_train.loadAnns(ann_ids)
  elif r['dataset'] == 'coco_val':
    ann_ids = coco_val.getAnnIds(imgIds=img_coco_id)
    anns = coco_val.loadAnns(ann_ids)

    
  # print(anns)
  #getting image shape
  img = plt.imread(DATA_PATH+r['image_path'])
  img_shape = img.shape
  # print(img_shape)
  #gaze point denormalize
  # px = round(r['gaze_x']*img_shape[1])
  # py = round(r['gaze_y']*img_shape[0])
  px = int(r['gaze_x']*img_shape[1])
  py = int(r['gaze_y']*img_shape[0])
  f = 0
  for ann in anns:
    if r['dataset'] == 'coco':
      mask = coco_train.annToMask(ann)
    elif r['dataset'] == 'coco_val':
      mask = coco_val.annToMask(ann)

    if mask[py, px] == 1:
      temp_lst.append(ann['bbox'])
      # bbox_lst.append(ann['bbox'])
      temp_cat_lst.append(ann['category_id'])
      # cat_id_lst.append(ann['category_id'])
      f += 1
  bbox_lst.append(temp_lst)
  cat_id_lst.append(temp_cat_lst)
  if f==1:
      new_row = []
      
      for i in r:
          new_row.append(i)
      new_row.append(temp_lst[0])
      new_row.append(temp_cat_lst[0])
      
      df_final.loc[len(df_final.index)] = new_row

  if f==0:
    no_bbox += 1
    no_bbox_lst.append(r['image_path'])
    
  if f > 1:
    multi_bbox += 1
    multi_bbox_lst.append(r['image_path'])

    
print('Total rows: ', len(df_coco))
print('Total rows: ', len(df_final))
print('Number of no bbox: ', no_bbox)
print('Number of multiple bbox: ', multi_bbox)

df_final.to_csv('test_annotations_release_new.txt', index=False)

#################################################################

img_coco_id = get_img_id('COCO_train2014_000000205086.jpg', 'coco')

data_train['annotations'][0]
cat = []
for i in data_train['annotations']:
    if i['image_id'] == img_coco_id:
        cat.append(i['category_id'])



# Define the path to your text file
file_path = 'yolo_class_mapping.txt'

# Read the content of the text file
with open(file_path, 'r') as file:
    file_content = file.read()

# Convert the file content (string) to a dictionary
data_dict = ast.literal_eval(file_content)

# Display the dictionary
print(data_dict)



file_path = 'coco_labels_paper.txt'

# Initialize an empty dictionary
line_dict = {}

# Read the content of the text file line by line
with open(file_path, 'r') as file:
    for line_number, line in enumerate(file, start=1):
        # Strip the newline character and assign the line content to the dictionary
        line_dict[line_number] = line.strip()

# Display the dictionary
print(line_dict)
