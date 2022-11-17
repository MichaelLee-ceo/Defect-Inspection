import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans

Color_Lower = (65, 65, 65)
Color_Upper = (200, 200, 200)

def findContour(img_path, result_path):
    img = cv2.imread(img_path)
    invert = cv2.bitwise_not(img)
    gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(gray, (11, 11), 0)
    # dilate = cv2.dilate(frame, None, iterations=2)
    # erode = cv2.erode(dilate, None, iterations=2)
    ret, threshed_img = cv2.threshold(frame, 200, 65, cv2.THRESH_BINARY)

    # cv2.imshow('Image', threshed_img)
    # cv2.waitKey(0)

    (contours, _) = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2, lineType=8, shift=0)

    cv2.imwrite(result_path + img_path.split('/')[-1] , img)
    print("[INFO] save result to:", result_path + img_path.split('/')[-1])
    # cv2.imshow("Frame", img)
    # cv2.waitKey(0)



filtered_path = "./20221024/component_extraction/"
if not os.path.exists(filtered_path):
    os.makedirs(filtered_path)
    print("[+] Creating component dir", filtered_path)

detection_path = "./20221024/defect_detection/"
if not os.path.exists(detection_path):
    os.makedirs(detection_path)
    print("[+] Creating defection dir", detection_path)


# ply_path = './20221024/20221024RGB/'
# ply_path = os.path.join(os.getcwd(), ply_path)
#
# ply_files = [os.path.join(ply_path,f) for f in os.listdir(ply_path) if os.path.isfile(os.path.join(ply_path, f))]
# print('Find', len(ply_files), 'files')
#
# kmeans = KMeans(n_clusters=2)
#
# for i in range(0, len(ply_files)):
#     simple_pcd = o3d.io.read_point_cloud(ply_files[i])                      # 讀點雲 .ply 檔案
#     simple_pcd = simple_pcd.voxel_down_sample(voxel_size=0.0018)
#     # aabb = simple_pcd.get_axis_aligned_bounding_box()
#     # aabb.color = (1, 0, 0)
#
#     xyz_load = np.asarray(simple_pcd.points)
#     color_load = np.asarray(simple_pcd.colors)
#
#     x, y, z = xyz_load[:, 0], xyz_load[:, 1], xyz_load[:, 2]
#     print('3D Point Cloud shape:', xyz_load.shape)
#
#     result = kmeans.fit(z.reshape(-1, 1))
#     center = result.cluster_centers_
#     predict_label = result.labels_
#     # print(predict_label.shape)
#
#     ''' 透過 cluster 的中心點，把加工物件與桌面分離 '''
#     if center[0] < center[1]:
#         object = np.where(predict_label == 0)
#         background = np.where(predict_label == 1)
#     else:
#         object = np.where(predict_label == 1)
#         background = np.where(predict_label == 0)
#
#     filtered_component = np.take(xyz_load, object, axis=0).squeeze(0)
#     print('Filtered Point Cloud: ', filtered_component.shape)
#
#
#     ''' color thresholding '''
#     # for i in range(color_load.shape[0]):
#     #     for j in range(3):
#     #         if color_load[i][j] > 0.2:
#     #             color_load[i][j] = 0.95
#
#     filtered_color = np.take(color_load, object, axis=0).squeeze(0)
#     # filtered_color *= 255
#     print('Filter Color: ', filtered_color.shape)
#
#     ''' 用取出來的加工物件跟對應的顏色，建立新的 3D Point Cloud '''
#     points = o3d.geometry.PointCloud()
#     points.points = o3d.utility.Vector3dVector(filtered_component)
#     points.colors = o3d.utility.Vector3dVector(filtered_color)
#     # o3d.visualization.draw_geometries([points])
#
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(points)
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(filtered_path + ply_files[i].split('/')[-1].replace('ply', 'png'))
#     vis.destroy_window()


    # gray = cv2.cvtColor(color_load, cv2.COLOR_BGR2GRAY)  # convert roi into gray
    # print(gray.shape, color_load.shape)
    # ret, threshed_img = cv2.threshold(gray, 50, 150, cv2.THRESH_BINARY)
    # cv2.imwrite('./20221020/result/' + 'result_' + str(i) + '.png', images[i])
    # plt.close()
    # print('Write ' + str(i) + ' file')

img_path = os.path.join(os.getcwd(), filtered_path)
img_files = [os.path.join(img_path, f) for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
print('Find', len(img_files), 'files')

for file in img_files:
    findContour(file, detection_path)