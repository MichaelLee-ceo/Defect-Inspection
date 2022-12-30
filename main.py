from roi import *

component_path = "./yolo/component_extraction/"
data_path = "./yolo/dataset/data/"
label_path = "./yolo/dataset/label/"

mkdir(component_path)
mkdir(data_path)
mkdir(label_path)

# ply_path = './yolo/RGBCLOUD/20221207/'
# getComponent(ply_path, component_path)

findContour(component_path, data_path, label_path)