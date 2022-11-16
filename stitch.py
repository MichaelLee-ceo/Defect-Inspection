import os
import cv2


mypath = './images/2022-09-19/2DR'
mypath = os.path.join(os.getcwd(), mypath)

files = [os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
print('Find', len(files), 'files')

images = []

for file in files:
    img = cv2.imread(file)
    img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)), interpolation=cv2.INTER_AREA)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    images.append(img)

stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)  # 我的是OpenCV4

(status, pano) = stitcher.stitch(images)

stitched = cv2.rotate(pano, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite('tmp_y.png', stitched)