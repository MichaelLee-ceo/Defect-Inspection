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
            cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2, lineType=8, shift=0)

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


def getComponent(before_path, after_path):
    before_plys = getFiles(before_path)
    after_plys = getFiles(after_path)

    for i in range(0, len(before_plys)):
        print("Reading:", before_plys[i])
        before_pcd = o3d.io.read_point_cloud(before_plys[i])  # 讀點雲 .ply 檔案
        before_pcd = before_pcd.voxel_down_sample(voxel_size=0.002)

        after_pcd = o3d.io.read_point_cloud(after_plys[i])  # 讀點雲 .ply 檔案
        after_pcd = after_pcd.voxel_down_sample(voxel_size=0.002)

        before_load, before_color = np.asarray(before_pcd.points), np.asarray(before_pcd.colors)
        after_load, after_color = np.asarray(after_pcd.points), np.asarray(after_pcd.colors)

        # x, y, z = before_load[:, 0], before_load[:, 1], before_load[:, 2]
        # print('3D Point Cloud shape:', before_load.shape)

        # assign color to point cloud
        before_color = np.ones(before_load.shape) * 0.5
        after_color = np.ones(after_load.shape) * 0.1
        before_pcd.colors = o3d.utility.Vector3dVector(before_color)
        after_pcd.colors = o3d.utility.Vector3dVector(after_color)
        o3d.visualization.draw_geometries([before_pcd, after_pcd])


        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False)
        # vis.add_geometry(points)
        #
        # ctr = vis.get_view_control()
        # ctr.set_lookat([0.037595129930056065, -0.017835838099320725, 0.89047966202100115])
        # ctr.set_front([0, 0, -1])
        # ctr.set_up([0, -1, 0])
        #
        # # content += str(ctr.convert_to_pinhole_camera_parameters().extrinsic) + "\n\n"
        # # content += str(ctr.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix) + "\n\n"
        #
        # vis.poll_events()
        # vis.update_renderer()
        # vis.capture_screen_image(component_path + ply_files[i].split('/')[-1].replace('ply', 'png'))
        # vis.destroy_window()
        #
        # del ctr
        # del vis

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