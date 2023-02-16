from roi import *

component_path = "./new/component/"           # "./yolo/component_extraction/"
data_path = "./new/visualize/dataset/data/"      # "./yolo/visualize/dataset/data/"
label_path = "./new/visualize/dataset/label/"    # "./yolo/visualize/dataset/label/"

mkdir(component_path)
mkdir(data_path)
mkdir(label_path)

ply_path = "./new/after/cloud/"                 # './yolo/RGBCLOUD/20221207/'
tmp_path = "./new/before/cloud"
getComponent(ply_path, component_path, tmp_path)

# findContour(component_path, data_path, label_path)