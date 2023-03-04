import open3d as o3d
import numpy as np
import copy
from roi import *

# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

# T = np.eye(4)
# T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
# T[:, 3] = [0, 1, 0, 1]
# matrix = mesh.get_rotation_matrix_from_xyz((1, 0, 0))
# print(matrix)

# mesh_t = copy.deepcopy(mesh).transform(T)
# o3d.visualization.draw_geometries([mesh, mesh_t])

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(mesh)
# ctr = vis.get_view_control()
# print(ctr.convert_to_pinhole_camera_parameters().extrinsic)
# intrinsic_matrix, extrinsic_matrix = ctr.convert_to_pinhole_camera_parameters()
# print(intrinsic_matrix)
# print(extrinsic_matrix)


before_path = "./new/before/cloud"
after_path = "./new/after/cloud/"

before_plys = getFiles(before_path)
after_plys = getFiles(after_path)

for i in range(0, len(before_plys)):
    print("Reading:", before_plys[i])
    source_pcd = o3d.io.read_point_cloud(before_plys[i])  # 讀點雲 .ply 檔案
    # source_pcd = source_pcd.voxel_down_sample(voxel_size=0.002)

    target_pcd = o3d.io.read_point_cloud(after_plys[i])  # 讀點雲 .ply 檔案
    # target_pcd = target_pcd.voxel_down_sample(voxel_size=0.002)

    source_np = np.asarray(source_pcd.points)
    target_np = np.asarray(target_pcd.points)

    # assign color to point clouds
    source_color = np.ones(source_np.shape) * 0.5
    target_color = np.ones(target_np.shape) * 0.1       # 比較黑的
    source_pcd.colors = o3d.utility.Vector3dVector(source_color)
    target_pcd.colors = o3d.utility.Vector3dVector(target_color)

    # calculate the centroids of the two point clouds
    centroid_src = np.mean(source_np, axis=0)
    centroid_tgt = np.mean(target_np, axis=0)

    # shift the point clouds to their respective centroids
    source = source_np - centroid_src
    target = target_np - centroid_tgt

    # calculate the cross-covariance matrix
    C = np.dot(source.T, target)
    # perform SVD on the cross-covariance matrix
    U, S, Vt = np.linalg.svd(C)
    # calculate the optimal rotation matrix
    R = np.dot(Vt.T, U.T)
    # calculate the optimal translation matrix
    t = centroid_tgt.T - np.dot(R, centroid_src.T)

    # create the transformation matrix
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    # print the transformation matrix
    print("Transformation Matrix:\n", T)
 
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.view_init(azim=0, elev=90)
    # # ax.scatter(centroid_src[0], centroid_src[1], centroid_src[2], c='deepskyblue', label='src_center')
    # # ax.scatter(centroid_tgt[0], centroid_tgt[1], centroid_tgt[2], c='blue', label='tgt_center')
    # ax.scatter(trans[:, 0], trans[:, 1], trans[:, 2], c="bisque", label="src_pcd")
    # ax.scatter(target[:, 0], target[:, 1], target[:, 2], c="orange", label="tgt_pcd")
    # ax.legend()
    # plt.show()

    source_trans = copy.deepcopy(source_pcd).transform(T)
    # o3d.visualization.draw_geometries([source_trans, target_pcd])

    trans_np = np.asarray(source_trans.points)
    print("Transformation deviation:", np.sum(trans_np - target_np))
