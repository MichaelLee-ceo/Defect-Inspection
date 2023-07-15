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
parser.add_argument("--calibration_path", default="./calibration/carbon/", type=str)
parser.add_argument("--component_path", default="./calibration/carbon/component/", type=str)
parser.add_argument('--data_path', default="./calibration/carbon/data/", type=str)
parser.add_argument("--extract_component", help="get the component from point cloud", action="store_true")
args = parser.parse_args()

calibration_path = args.calibration_path
component_path = args.component_path
data_path = args.data_path

# Create directories
mkdir(component_path)
mkdir(data_path)

# Extracts component froms 3D point clouds and store in args.component_path
if args.extract_component:
    getComponent(calibration_path, component_path)

# Get the corner point of component for coordinate transformation
img_files = getFiles(component_path)
for idx, img_path in enumerate(img_files):
    img = cv2.imread(img_path)
    img = cv2.rotate(img, cv2.ROTATE_180)
    height, width, channel = img.shape

    y = int(height / 6)
    x = int(width / 4)
    crop_img = img[y:y * 5, x:x * 3]

    # w, h = 512, 512
    # center = crop_img.shape
    # x = center[1]/2 - 512/2
    # y = center[0]/2 - 512/2
    # crop_img = crop_img[int(y):int(y+h), int(x):int(x+w)]

    crop_img = cv2.resize(crop_img, (512, 512))

    # convert img to grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = 255-gray

    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1)
    thresh = 255-thresh

    # apply morphology
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # separate horizontal and vertical lines to filter out spots outside the rectangle
    kernel = np.ones((7,3), np.uint8)
    vert = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3,7), np.uint8)
    horiz = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # combine
    rect = cv2.add(horiz,vert)

    # thin
    kernel = np.ones((3,3), np.uint8)
    rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)

    # get largest contour
    contours = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        area_thresh = 0
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c

    # get rotated rectangle from contour
    rot_rect = cv2.minAreaRect(big_contour)
    bbox = cv2.boxPoints(rot_rect)
    bbox = np.int0(bbox)
    print(bbox)

    for i in range(len(bbox)):
        for j in range(2):
            if bbox[i][j] < 256:
                bbox[i][j] += 10
            else:
                bbox[i][j] -= 10

    print("New bbox:", bbox)        

    # draw rotated rectangle on copy of img
    rot_bbox = crop_img.copy()
    # cv2.drawContours(rot_bbox,[bbox],0,(0,0,255),2)
    
    for box in bbox:
        cv2.circle(rot_bbox, box, 2, (0, 0, 255), 2)
        cv2.putText(rot_bbox, "[{:.2f}, {:.2f}]".format(box[0], box[1]),
                    (box[0] - 75, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)

    # write img with red rotated bounding box to disk
    # cv2.imwrite("rectangle_thresh.png", thresh)
    # cv2.imwrite("rectangle_outline.png", rect)
    cv2.imwrite(data_path + "component.png", rot_bbox)

    # display it
    # cv2.imshow("BBOX", rot_bbox)
    # cv2.waitKey(0)