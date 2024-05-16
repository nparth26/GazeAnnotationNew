from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import pandas as pd
import os
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from PIL import Image
from io import BytesIO

HOME = os.getcwd()
checkpoint_path = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)

predictor = SamPredictor(sam)


from tqdm import tqdm

def ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:  #Check if the tensor has only one element
            return x.item()  #Return as a Python scalar
        else:
            return x.cpu().numpy()  #Convert the entire tensor to a numpy array
    return x

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

train_file_path = './train_annotations_release_new.txt'
test_file_path = './test_annotations_release.txt'
# Define the column names
column_names_train = [
      "image_path", "id", "body_bbox_x", "body_bbox_y", "body_bbox_width", "body_bbox_height",
      "eye_x", "eye_y", "gaze_x", "gaze_y", "head_bbox_x_min", "head_bbox_y_min",
      "head_bbox_x_max", "head_bbox_y_max", "in_or_out", "meta","ground_truth","category_ground_truth"
]

new_col_names = ["image_path", "id", "body_bbox_x", "body_bbox_y", "body_bbox_width", "body_bbox_height",
      "eye_x", "eye_y", "gaze_x", "gaze_y", "head_bbox_x_min", "head_bbox_y_min",
      "head_bbox_x_max", "head_bbox_y_max", "in_or_out", "meta","gp_bb_x1","gp_bb_y1","gp_bb_x2","gp_bb_y2","gaze_cls_id","bounding_boxes"]

df_new = pd.DataFrame(columns=new_col_names)
itr = 0
#Reading the text file into a DataFrame
df = pd.read_csv(train_file_path, header=None, names=column_names_train,sep=',',index_col=False)

for index, row in tqdm(df.iterrows(),total=df.shape[0]):
  """if itr < 5001:
    continue"""
  """if index <= 5001:
    continue"""
  
  image_path = "./"+row["image_path"]
  gaze_label = np.array([1])
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  shape = image.shape
  gaze_point = np.array([[row["gaze_x"]*shape[1],row["gaze_y"]*shape[0]]])
  
  predictor.set_image(image)
  masks, scores, logits = predictor.predict(
  point_coords=gaze_point,
  point_labels=gaze_label,
  multimask_output=True,
  )

  mask1, mask2, mask3 = masks[0], masks[1], masks[2]

  masked_image_1 = image * mask1[:, :, None]  #Expand mask dimensions to match image
  masked_image_2 = image * mask2[:, :, None]  #Expand mask dimensions to match image
  masked_image_3 = image * mask3[:, :, None]  #Expand mask dimensions to match image
  masked_imgs = [masked_image_1, masked_image_2, masked_image_3]
  # Displaying the masked image
  """for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    #show_points(gaze_point, gaze_label, plt.gca())
    #show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  """

  model = YOLO('yolov8n.pt')  #Ensuring the model path is correct
  id = 0
  max_conf_score = -1.0
  max_bb_x1, max_bb_y1, max_bb_x2, max_bb_y2 = -1.0,-1.0,-1.0,-1.0
  max_cls_id = -1
  for masked_image in masked_imgs:
    id += 1  

    #Converting the masked image back to a PIL Image for processing
    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))
    results = model(masked_image_pil)

    # Assuming results[0] contains the necessary details
    boxes = results[0].boxes.xyxy.cpu().numpy()  #boxes are in the format [xmin, ymin, xmax, ymax]
    classes = results[0].boxes.cls  #class ids
    names = results[0].names  #class names from the model
    confidences = results[0].boxes.conf  #confidence scores

    #Plotting setup
    #fig, ax = plt.subplots(1, figsize=(12, 8))
    #ax.imshow(image)

    #Iterating through the results
    for box, cls, conf in zip(boxes, classes, confidences):
          x1, y1, x2, y2 = box
          name = names[int(cls)]  #Fetching the name using class ID
          color = 'red'
          #rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
          #ax.add_patch(rect)
          #plt.text(x1, y1, f'{name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor=color, alpha=0.5))
          if conf > max_conf_score:   #for fetching the bouding box with max score
            max_conf_score = conf
            max_cls_id = int(cls)
            max_bb_x1, max_bb_y1, max_bb_x2, max_bb_y2 = x1, y1, x2, y2
    # Plotting setup
  """fig, ax = plt.subplots(1, figsize=(12, 8))
  ax.imshow(image)"""

  #name = names[int(max_cls_id)]  # Fetching the name using class ID
  color = 'red'
  """rect = patches.Rectangle((max_bb_x1, max_bb_y1), max_bb_x2-max_bb_x1, max_bb_y2-max_bb_y1, linewidth=2, edgecolor=color, facecolor='none')
  ax.add_patch(rect)
  plt.text(x1, y1, f'{name} {max_conf_score:.2f}', color='white', fontsize=12, bbox=dict(facecolor=color, alpha=0.5))
          
"""

  #plt.axis('off')
  #plt.show()

  itr += 1
  #Detect all objs
  bounding_boxes = []

  bb_results = model(image)

  #Assuming results[0] contains the required details
  bb_boxes = bb_results[0].boxes.xyxy.cpu().numpy()  # boxes are in the format [xmin, ymin, xmax, ymax]
  bb_classes = bb_results[0].boxes.cls  # class ids
  bb_names = bb_results[0].names  # class names from the model
  bb_confidences = bb_results[0].boxes.conf  # confidence scores

  for box, cls, conf in zip(bb_boxes, bb_classes, bb_confidences):
          x1, y1, x2, y2 = box
          name = names[int(cls)]  #Fetching the name using class ID
          bb_tuple = (x1,y1,x2,y2,int(cls))
          bounding_boxes.append(bb_tuple)
          #rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
          #ax.add_patch(rect)
          #plt.text(x1, y1, f'{name} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor=color, alpha=0.5))
  df_new.loc[index-1] = [row["image_path"],row["id"],ensure_numpy(row["body_bbox_x"]),ensure_numpy(row["body_bbox_y"]),ensure_numpy(row["body_bbox_width"]),ensure_numpy(row["body_bbox_height"]),
      ensure_numpy(row["eye_x"]),ensure_numpy(row["eye_y"]),ensure_numpy(row["gaze_x"]),ensure_numpy(row["gaze_y"]),ensure_numpy(row["head_bbox_x_min"]),ensure_numpy(row["head_bbox_y_min"]),
      ensure_numpy(row["head_bbox_x_max"]),ensure_numpy(row["head_bbox_y_max"]),ensure_numpy(row["in_or_out"]),ensure_numpy(row["meta"]),ensure_numpy(max_bb_x1), ensure_numpy(max_bb_y1), ensure_numpy(max_bb_x2), ensure_numpy(max_bb_y2),ensure_numpy(max_cls_id),ensure_numpy(bounding_boxes)]
df_new.to_csv('/home/projuser/kaustubh_parth_gaze/gaze_detection/gazefollow_extended_reduced/output_file_final_5kn.txt', sep=',',index=False)
#print(df_new.head())
