# GazeFollow_Annotate
A repository to create object level annotations for GazeFollow Dataset

1. extract_annotations contains the code for extracting the gazefollow annotations using SAM model.

It requires follwing packages to be installed in the virtual env. Please use the following commands to download the packages:

!pip install 'git+https://github.com/facebookresearch/segment-anything.git'
!pip install -q roboflow supervision
!wget -q 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

(for downloading weights for SAM Model)
!mkdir -p {HOME}/weights
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights 


!pip install ultralytics
import ultralytics
ultralytics.checks()

2. To extract the ground truth from ms coco dataset run 'final_coco_ground_truth_extraction.py' file

Change the path as per local system in the following variable for 'final_coco_ground_truth_extraction.py' file - file_path, DATA_PATH, coco_val, coco_train

Current state of the file 'final_coco_ground_truth_extraction.py' is set to extract ground truth for test subset of Gazefollow. To extract the ground truth for train subset change the labels_path to 'train_annotations_release.txt' and use the column_names and column_names_new in which the field 'in_out' is present (Comment the current column_names and column_names_new initialization and uncomment the below second initialization). For train annotations change the output file command as per from 'df_final.to_csv('test_annotations_release_new.txt', index=False)' to 'df_final.to_csv('train_annotations_release_new.txt', index=False)'. coco annotation link - http://images.cocodataset.org/annotations/annotations_trainval2014.zip

3. Run the file 'sam_det_evaluation.py' to get the precision of the newly annotated data against the ms coco grount truth

Threshold can be changed in the 'threshold' variable

4. Run 'extract_reduced_img.py' to copy the subset of the Gazefollow_extended dataset to another folder. Change the DATA_PATH. Gazefollow_extended link - https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?e=1&dl=0

5. Run 'convert_to_goo_format_new.py" to create pickle file for gatector model.

6. Run 'create_test_pickle.py' to split the 'gazefollow_new_final.pickle' into train and test

7. Clone the git repo - https://github.com/CodeMonsterPHD/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction.git

8. Use the ./gatector/main.py in place of the main.py of the above repo

9. Place the ./gatector/voc_classes1.txt in the ./data/anchors

10. Run the following command for training

python main.py --train_mode 0 --train_dir './data/gazefollow_extended_reduced/' --train_annotation './data/gazefollow_extended_reduced/train_gazefollow_new_final.pickle' --test_dir './data/gazefollow_extended_reduced/' --test_annotation './data/gazefollow_extended_reduced/test_gazefollow_new_final.pickle'

12. Place the gazefollow_extended_reduce in the ./data

13. Place the train_gazefollow_new_final.pickle and test_gazefollow_new_final.pickle in ./data/gazefollow_new_final

14. 
