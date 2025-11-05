import open3d as o3d

pcd = o3d.io.read_point_cloud("result/arch/dense_model/dense_geometric_loose.ply")
print(pcd)
print("num points:", len(pcd.points))
o3d.visualization.draw_geometries([pcd])


