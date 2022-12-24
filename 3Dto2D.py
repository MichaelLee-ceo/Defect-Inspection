import open3d as o3d

pcd_full = o3d.io.read_point_cloud("newply0.ply")

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd_full)
vis.update_geometry(pcd_full)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("1.jpg")
vis.destroy_window()