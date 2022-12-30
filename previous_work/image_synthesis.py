import cv2
import os
import numpy as np
from PIL import Image, ImageChops


mypath = './images/2022-09-19/2D'
mypath = os.path.join(os.getcwd(), mypath)

files = [os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
print('Find', len(files), 'files')

images = []

for file in files:
    img = cv2.imread(file)
    images.append(img)
    # print(file, 'Width:', width, ',Height:', height)

img1, img2, img3 = images[0], images[1], images[2]

# cv2.circle(img1, (220, 600), radius=10, color=(0, 0, 255), thickness=-1)
cv2.circle(img1, (350, 1020), radius=10, color=(0, 0, 255), thickness=-1)
#
# cv2.circle(img2, (135, 600), radius=10, color=(0, 255, 0), thickness=-1)
cv2.circle(img2, (350, 650), radius=10, color=(0, 255, 0), thickness=-1)

cv2.circle(img3, (350, 280), radius=10, color=(255, 0, 255), thickness=-1)

# delta_x1, delta_y1 = -1, 1
# t1 = np.float32([
#     [1, 0, delta_x1],
#     [0, 1, delta_y1],
#     [0, 0, 1],
# ])
# rows, cols, dim = img2.shape
# img2 = cv2.warpPerspective(img2, t1, (cols, rows))


'''
# === find contour ===
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_DEFAULT)

# 輪廓化(edge cascade) 小技巧，可以先將圖片模糊化，再進行輪廓化，可以抓到比較少雜訊。
# canny = cv2.Canny(blur, 127, 255)

ret, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img1, contours, -1, (255, 0, 255), 7)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 0:
        cv2.drawContours(img1, cnt, -1, (255, 0, 255), 7)
cv2.imshow('Image', img1)
cv2.waitKey(0)
'''

'''
prev_img = images[0]
start = 150
bias = 70
for i in range(1, len(images)):
    prev_img = prev_img[:, :start + bias*(i-1) -20, :]
    current_img = images[i][:, bias-10:, :]
    prev_img = np.concatenate((prev_img, current_img), axis=1)
'''

prev_img = images[0]
start = 1020
bias = 375
for i in range(1, len(images)):
    prev_img = prev_img[:start + bias*(i-1)-15, :, :]
    current_img = images[i][start - bias-15:, :, :]
    prev_img = np.concatenate((prev_img, current_img), axis=0)



# finding difference
# diff = ImageChops.difference(Image.fromarray(img1), Image.fromarray(img2))
# diff.show()
# img1 = img1[:1020, :, :]
# img2 = img2[1020-370:, :, :]
# img3 = img3[:, 60:, :]

# prev_img = np.concatenate((img1, img2), axis=0)
# prev_img = np.concatenate((prev_img, img3), axis=0)
#
# prev_img = np.concatenate((prev_img[:, :200+50, :], img3), axis=1)
cv2.imwrite('tmp_y.png', prev_img)
print('After concatenated:', prev_img.shape)
