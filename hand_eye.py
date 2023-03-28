import open3d as o3d
import numpy as np
import copy
from roi import *

def get_object_from_pc(ply_file):
    source_pcd = o3d.io.read_point_cloud(ply_file)  # 讀點雲 .ply 檔案
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.002)

    source_np = np.asarray(source_pcd.points)
    source_color = np.ones(source_np.shape) * 0.8
    source_pcd.colors = o3d.utility.Vector3dVector(source_color)

    # get the bounding box of the object in the point cloud
    bbox = o3d.geometry.AxisAlignedBoundingBox([-0.3, -0.15, 0.8], [0.3, 0, 1])
    # bbox.color = (1, 0, 0)

    # 選 bounding box 裡面的點雲
    pcd_crop = source_pcd.crop(bbox)
    pcd_crop_np = np.asarray(pcd_crop.points)
    pcd_crop.colors = o3d.utility.Vector3dVector(np.ones(pcd_crop_np.shape) * 0.1)

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([mesh, source_pcd, pcd_crop])

    # 對選出來的點雲取平均 (找中心點)
    centroid_np = np.mean(pcd_crop_np, axis=0)
    print('Object center:', centroid_np)

    return pcd_crop_np, centroid_np


ply_file = './test/point_cloud_00000.ply'
crop_pcd, centroid_pcd = get_object_from_pc(ply_file)

# target_pcd = o3d.io.read_point_cloud(after_plys[i])  # 讀點雲 .ply 檔案
# target_pcd = target_pcd.voxel_down_sample(voxel_size=0.002)

# target_np = np.asarray(target_pcd.points)
# target_color = np.ones(target_np.shape) * 0.1       # 比較黑的
# target_pcd.colors = o3d.utility.Vector3dVector(target_color)

# calculate the centroids of the two point clouds
# centroid_src = np.mean(source_np, axis=0)
# centroid_tgt = np.mean(target_np, axis=0)

# shift the point clouds to their respective centroids
# source = source_np - centroid_src
# target = target_np - centroid_tgt

# calculate the cross-covariance matrix
# C = np.dot(source.T, target)
# perform SVD on the cross-covariance matrix
# U, S, Vt = np.linalg.svd(C)
# calculate the optimal rotation matrix
# R = np.dot(Vt.T, U.T)
# calculate the optimal translation matrix
# t = centroid_tgt.T - np.dot(R, centroid_src.T)

# create the transformation matrix
# T = np.identity(4)
# T[0:3, 0:3] = R
# T[0:3, 3] = t

# print the transformation matrix
# print("Transformation Matrix:\n", T)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=0, elev=90)
ax.scatter(crop_pcd[:, 0], crop_pcd[:, 1], crop_pcd[:, 2], c="gray", label="pcd_crop")
ax.scatter(centroid_pcd[0], centroid_pcd[1], centroid_pcd[2], c="orange", s=50, label="center")
ax.legend()
plt.show()

# source_trans = copy.deepcopy(source_pcd).transform(T)
# source_trans.colors = o3d.utility.Vector3dVector(np.ones(source_np.shape) * 0.5)
# o3d.visualization.draw_geometries([mesh, source_trans, target_pcd])

# trans_np = np.asarray(source_trans.points)
# print("Transformation deviation:", np.sum(trans_np - target_np))
