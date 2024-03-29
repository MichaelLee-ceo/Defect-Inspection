import os
import cv2
import copy
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.7,
                                      front=[0, 0, 1],
                                      lookat=[0.037595129930056065, -0.017835838099320725, 0.89047966202100115],
                                      up=[0, 1, 0]
                                      )

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
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


def findContour(image_path, data_path, label_path, gammas, visualize=False):
    img_files = getFiles(image_path)
    count = 0
    for idx, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        height, width, channel = img.shape

        y = int(height / 6)
        x = int(width / 4)
        crop_img = img[y:y * 5, x:x * 3]
        crop_img = cv2.resize(crop_img, (512, 512))

        # contrast_img = crop_img * float(2.2)
        # contrast_img[contrast_img > 255] = 255
        # contrast_img = np.round(contrast_img)
        # contrast_img = contrast_img.astype(np.uint8)

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(gray)

        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # dilate = cv2.dilate(frame, None, iterations=2)
        # erode = cv2.erode(dilate, None, iterations=2)
        ret, threshed_img = cv2.threshold(invert, 155, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Frame", np.concatenate((gray, threshed_img), axis=1))
        # cv2.waitKey(0)

        (contours, _) = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(crop_img, contours, -1, (0, 255, 0), 2)
        # cv2.imshow("Frame", crop_img)
        # cv2.waitKey(0)

        ''' YOLO format: <object-class> <x_center> <y_center> <width> <height> '''
        content = ""
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if visualize:
                cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2, lineType=8, shift=0)

            x_center, y_center = x + w/2, y + h/2
            content += ("0" + " " + str(x_center/512) + " " + str(y_center/512) + " " + str(w/512) + " " + str(h/512) + "\n")

        save_path_img = data_path + str(count) + ".png"
        save_path_txt = label_path + str(count) + ".txt"
        saveImage_txt(crop_img, save_path_img, content, save_path_txt)

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
    content = ""
    for i in range(0, len(ply_files)):
        print(ply_files[i])
        simple_pcd = o3d.io.read_point_cloud(ply_files[i])                      # 讀點雲 .ply 檔案
        simple_pcd = simple_pcd.voxel_down_sample(voxel_size=0.002)
        # simple_pcd = simple_pcd.uniform_down_sample(every_k_points=10)
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

        # cl, ind = points.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.01)
        # inlier_cloud = cl.select_by_index(ind)
        # display_inlier_outlier(points, ind)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(points)

        ctr = vis.get_view_control()
        ctr.set_lookat([0.037595129930056065, -0.017835838099320725, 0.89047966202100115])
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])

        # content += str(ctr.convert_to_pinhole_camera_parameters().extrinsic) + "\n\n"
        # content += str(ctr.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix) + "\n\n"

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(component_path + ply_files[i].split('/')[-1].replace('ply', 'png'))
        vis.destroy_window()

        del ctr
        del vis

        # intrinsic_matrix = ctr.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix
        # extrinsic_matrix = ctr.convert_to_pinhole_camera_parameters().extrinsic
        # print("Intrinsic Matrix:", intrinsic_matrix.shape)
        # print("Extrinsic Matrix:", extrinsic_matrix.shape)

        # transformed = copy.deepcopy(points).transform(extrinsic_matrix)
        # tmp = np.transpose(np.asarray(transformed.points))
        #
        # trans_xyz = np.transpose(np.dot(intrinsic_matrix, tmp))
        # print(trans_xyz.shape)

        # xyz = np.asarray(points.points)
        # x1, y1, z1 = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # x2, y2, z2 = trans_xyz[:, 0], trans_xyz[:, 1], trans_xyz[:, 2]

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.view_init(azim=0, elev=90)
        # ax.scatter(x1, y1, z1, c='deepskyblue', label='Original 3D')
        # ax.scatter(x2, y2, z2, c='orange', label='Transformed')
        # ax.legend()
        # plt.show()

    # with open('extrinsic.txt', "w") as label_file:
    #     label_file.write(content)
