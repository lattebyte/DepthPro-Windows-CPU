# LICENSE_2 applies to this file
# Author JZ from LatteByte.ai 2024

import depth_pro
import torch
import os
import numpy as np
import cv2  # OpenCV library
from PIL import Image  # Import PIL's Image module
import open3d as o3d  # Open3D for 3D visualization
import matplotlib.pyplot as plt

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image using OpenCV
current_dir = os.path.dirname(__file__)
#relative_path = 'data/example.jpg'
relative_path = 'data/road_1.png'
image_path = os.path.join(current_dir, relative_path)

# Load image with OpenCV and convert from BGR to RGB format
original_rgb = cv2.imread(image_path)
if original_rgb is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)

# Preprocess the image for the model
image = Image.fromarray(original_rgb)  # Convert to PIL image
image = transform(image)

# Ensure the image has a batch dimension if needed by the model
if image.dim() == 3:
    image = image.unsqueeze(0)  # Add batch dimension if necessary

# Run inference to get depth prediction
with torch.no_grad():  # Disable gradient computation for inference
    prediction = model.infer(image)

# Extract depth map and convert to numpy
depth_np = prediction["depth"].squeeze().cpu().numpy()

# Downsample for performance and to reduce visual clutter
downsample_factor = 1
depth_np = depth_np[::downsample_factor, ::downsample_factor]
original_rgb = original_rgb[::downsample_factor, ::downsample_factor]

# Scale depth values by a factor of 10 for enhanced 3D effect
#depth_scale = 2000  # for nearby object, e.g. indoor
depth_scale = 5  # for image included sky

z = depth_np * depth_scale  # Scale depth values

# Generate x, y coordinates and flatten arrays for point cloud
h, w = depth_np.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

# Use original RGB colors
colors = original_rgb.reshape(-1, 3) / 255.0  # Normalize colors to [0, 1]

# Create Open3D point cloud
points = np.vstack((x_flat, y_flat, z_flat)).T  # Stack x, y, and z into (N, 3) array
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)  # Use original RGB colors

# Visualize the point cloud with custom camera view
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point cloud to the visualizer
vis.add_geometry(point_cloud)

# Set up a custom camera view for enhanced 3D effect
view_control = vis.get_view_control()
view_control.set_front([0.5, -0.5, -1])  # Set a custom front direction
view_control.set_lookat([w / 2, h / 2, np.max(z_flat) / 2])  # Center on the point cloud
view_control.set_up([0, -1, 0])  # Adjust up direction
view_control.set_zoom(0.5)  # Adjust zoom level for perspective

# Run the visualizer
vis.run()
vis.destroy_window()
