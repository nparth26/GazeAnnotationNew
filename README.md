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

2. 