import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import os

DATA_PATH = '/home/parth/Parth SBU/Sem2/CV/project/gazefollow_extended/'

labels_path = 'train_annotations_release_new.txt'

# column_names = ['image_path','id','body_bbox_x','body_bbox_y','body_bbox_width','body_bbox_height','eye_x','eye_y','gaze_x','gaze_y','head_bbox_x_min','head_bbox_y_min','head_bbox_x_max','head_bbox_y_max','in_or_out','dataset','dataset_img_path']
column_names_new = ['image_path','id','body_bbox_x','body_bbox_y','body_bbox_width','body_bbox_height','eye_x','eye_y','gaze_x','gaze_y','head_bbox_x_min','head_bbox_y_min','head_bbox_x_max','head_bbox_y_max','dataset','dataset_img_path', 'gaze_obj_bbox', 'gaze_obj_category']

df = pd.read_csv(
        labels_path,
        sep=",",
        names=column_names_new,
        usecols=column_names_new,
        index_col=False,
    )

s = df.loc[0, 'image_path'].split('/')
s = s[0]+'/'+s[1]
s
source_file = 'gazefollow_extended/'+df.loc[0, 'image_path']
destination_directory = 'gazefollow_extended_reduced/'+s
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)
shutil.copy(source_file, destination_directory)

for idx, r in df.iterrows():
    img_path = r['image_path']
    s = r['image_path'].split('/')
    s = s[0]+'/'+s[1]
    source_file = 'gazefollow_extended/'+img_path
    destination_directory = 'gazefollow_extended_reduced/'+s
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    shutil.copy(source_file, destination_directory)
    
#################################################################

# import pandas as pd
# import os
# import numpy as np
# from pycocotools.coco import COCO
# from tqdm import tqdm
# import shutil
# import os
# import pickle

# obj = pd.read_pickle('test_gazefollow_new_final.pickle')

# cnt = 0
# for i in obj:
    
#     img_path = i['filename']
    
#     source_file = 'gazefollow_extended/'+img_path
    
#     dest_dir_path = 'gazefollow_extended_reduced/test_data/data_proc/images1'
#     new_file_name = str(cnt)+'.jpg'
#     dest_file_path = os.path.join(dest_dir_path, new_file_name)
    
#     shutil.copy(source_file, dest_file_path)
    
#     obj1 = i.copy()
#     obj1['filename'] = new_file_name
#     pickle_file_path = 'gazefollow_extended_reduced/test_data/data_proc/data1/'+str(cnt)+'.pickle'
#     # Save the list of dictionaries to the pickle file
#     with open(pickle_file_path, 'wb') as f:
#         pickle.dump(obj1, f) 
#     cnt += 1
    
# file_path = 'gazefollow_extended_reduced/test_data/data_proc/test.txt'
# # Open the file in write mode
# with open(file_path, 'w') as file:
#     # Write integers from 0 to 998, each on a new line
#     for i in range(999):
#         file.write(f"{i}\n")   

# obj = pd.read_pickle('gatector/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction/test_data/data_proc/data1/0.pickle')

# src_file_path = 'gazefollow_extended_reduced/train/00000005/00005001.jpg'
# dest_dir_path = 'gazefollow_extended_reduced/test_data/data_proc/images1'
# new_file_name = '0.jpg'
# dest_file_path = os.path.join(dest_dir_path, new_file_name)
    
# # Copy the file and rename it in the destination directory
# shutil.copy(src_file_path, dest_file_path)


    
    
    
    