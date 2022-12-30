import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print("[+] Creating dir", dirpath)


def getFiles(filepath):
    file_path = os.path.join(os.getcwd(), filepath)
    files = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    print('Find', len(files), 'files in', filepath)

    return files


def saveImage_txt(image, image_path, content, content_path):
    cv2.imwrite(image_path, image)
    with open(content_path, "w") as label_file:
        label_file.write(content)

    print("[INFO] save result to:", image_path, content_path)


def findContour(image_path, data_path, label_path):
    img_files = getFiles(image_path)
    count = 0
    for idx, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        height, width, channel = img.shape

        y = int(height / 6)
        x = int(width / 4)
        crop_img = img[y:y * 5, x:x * 3]
        crop_img = cv2.resize(crop_img, (512, 512))

        invert = cv2.bitwise_not(crop_img)
        gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(gray, (9, 9), 0)
        # dilate = cv2.dilate(frame, None, iterations=2)
        # erode = cv2.erode(dilate, None, iterations=2)
        ret, threshed_img = cv2.threshold(frame, 170, 70, cv2.THRESH_BINARY)
        # cv2.imshow("Frame", threshed_img)
        # cv2.waitKey(0)

        (contours, _) = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(crop_img, contours, -1, (0, 255, 0), 2)
        # cv2.imshow("Frame", crop_img)
        # cv2.waitKey(0)

        ''' YOLO format: <object-class> <x_center> <y_center> <width> <height> '''
        content = ""
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2, lineType=8, shift=0)

            x_center, y_center = x + w/2, y + h/2
            content += ("0" + " " + str(x_center/512) + " " + str(y_center/512) + " " + str(w/512) + " " + str(h/512) + "\n")

        save_path_img = data_path + str(count) + ".png"
        save_path_txt = label_path + str(count) + ".txt"
        saveImage_txt(crop_img, save_path_img, content, save_path_txt)

        # gammas = [0.3, 0.5, 0.7]
        gammas = []
        for gamma in gammas:
            count += 1
            save_path_img = data_path + str(count) + ".png"
            save_path_txt = label_path + str(count) + ".txt"
            aug_img = add_light(crop_img, gamma)
            saveImage_txt(aug_img, save_path_img, content, save_path_txt)

        count += 1


def add_light(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_image = cv2.LUT(image, table)
    return new_image


def getComponent(ply_path, component_path):
    ply_files = getFiles(ply_path)
    kmeans = KMeans(n_clusters=3)
    for i in range(0, len(ply_files)):
        simple_pcd = o3d.io.read_point_cloud(ply_files[i])                      # 讀點雲 .ply 檔案
        simple_pcd = simple_pcd.voxel_down_sample(voxel_size=0.002)
        # aabb = simple_pcd.get_axis_aligned_bounding_box()
        # aabb.color = (1, 0, 0)

        xyz_load = np.asarray(simple_pcd.points)
        color_load = np.asarray(simple_pcd.colors)

        x, y, z = xyz_load[:, 0], xyz_load[:, 1], xyz_load[:, 2]
        print('3D Point Cloud shape:', xyz_load.shape)

        result = kmeans.fit(z.reshape(-1, 1))
        center = result.cluster_centers_
        predict_label = result.labels_

        ''' 透過 cluster 的中心點，把加工物件與桌面分離 '''
        center_min = np.argmin(center)
        object = np.where(predict_label == center_min)

        filtered_component = np.take(xyz_load, object, axis=0).squeeze(0)
        print('Filtered Point Cloud: ', filtered_component.shape)

        ''' color thresholding '''
        # for i in range(color_load.shape[0]):
        #     for j in range(3):
        #         if color_load[i][j] > 0.2:
        #             color_load[i][j] = 0.95

        filtered_color = np.take(color_load, object, axis=0).squeeze(0)
        # filtered_color *= 255
        print('Filter Color: ', filtered_color.shape)

        ''' 用取出來的加工物件跟對應的顏色，建立新的 3D Point Cloud '''
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(filtered_component)
        points.colors = o3d.utility.Vector3dVector(filtered_color)
        # o3d.visualization.draw_geometries([points])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(points)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(component_path + ply_files[i].split('/')[-1].replace('ply', 'png'))
        vis.destroy_window()