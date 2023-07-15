### Run the command for finding and labeling the carbon area
```
python3 main.py
```

| Parameter | Description |
| --- | --- |
| --component_path | Folder to save 2D images extracted from 3D point clouds |
| --ply_path | Folder of 3D point clouds |
| --data_path | Folder to save images for training |
| --label_path | Folder to save labels for training |
| --extract_component | Specify if needed to extract component from 3D point clouds |


### Run the command for detection
```
python3 darknet_images.py --config_file ../src/cfg_defect/yolov4.cfg --data_file ../src/cfg_defect/obj.data --weights ../src/cfg_defect/weights/yolov4_last.weights --input ../../calibration/carbon/data/component.png --dont_show
```