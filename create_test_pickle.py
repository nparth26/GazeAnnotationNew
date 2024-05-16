import pandas as pd
import os
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import os
import pickle

obj = pd.read_pickle('gazefollow_new_final.pickle')

obj_train = []
for i in range(4000):
    obj_train.append(obj[i])
    
   
pickle_file_path = 'train_gazefollow_new_final.pickle'
# Save the list of dictionaries to the pickle file
with open(pickle_file_path, 'wb') as f:
    pickle.dump(obj_train, f)
    
obj_test = []
for i in range(4000,len(obj)):
    obj_test.append(obj[i])
    
pickle_file_path = 'test_gazefollow_new_final.pickle'
# Save the list of dictionaries to the pickle file
with open(pickle_file_path, 'wb') as f:
    pickle.dump(obj_test, f)  
