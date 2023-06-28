'''
- Create dataset for yolo training, 
  get the surface of defect component from it's 3D point clouds,
  and use findContour to label the defect area in yolo's format:
      ([x, y, w, h], [label])

- Includes a brightness_factor to control the brightness of image 
  for data augmentation

- Save the results in data_path and label_path
'''

import argparse
from roi import *

parser = argparse.ArgumentParser()
parser.add_argument('--component_path', default="./yolo/component_extraction/", type=str)
parser.add_argument('--ply_path', default="./yolo/RGBCLOUD/20221207/", type=str)
parser.add_argument('--data_path', default="./yolo/dataset/data/", type=str)
parser.add_argument('--label_path', default="./yolo/dataset/label/", type=str)
parser.add_argument("--extract_component", help="get the component from point cloud", action="store_true")
args = parser.parse_args()

component_path = args.component_path
ply_path = args.ply_path
data_path = args.data_path
label_path = args.label_path

# Create directories
mkdir(component_path)
mkdir(data_path)
mkdir(label_path)

# Extracts component froms 3D point clouds and store in args.component_path
if args.extract_component:
    getComponent(ply_path, component_path)

brightness_factor = [0.5, 0.6, 0.8, 0.9]
findContour(component_path, data_path, label_path, brightness_factor, visualize=True)