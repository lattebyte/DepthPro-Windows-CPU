# LICENSE_2 applies to this file
# Author JZ from LatteByte.ai 2024

from PIL import Image
import depth_pro
import matplotlib.pyplot as plt
import torch
import os

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()


# Load and preprocess an image.
current_dir = os.path.dirname(__file__)
relative_path = 'data/example.jpg'
image_path=os.path.join(current_dir, relative_path)
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Ensure the image has a batch dimension if needed by the model
if image.dim() == 3:
    image = image.unsqueeze(0)  # Add batch dimension if necessary

# Run inference.
with torch.no_grad():  # Disable gradient computation for inference
    prediction = model.infer(image, f_px=f_px)

# Extract depth map (in meters).
depth = prediction["depth"]

# Convert depth tensor to numpy for visualization
depth_np = depth.squeeze().cpu().numpy()

# Display depth map using matplotlib
plt.imshow(depth_np, cmap='plasma')  # Use a colormap like 'plasma' for visualization
plt.colorbar(label="Depth (meters)")  # Optional colorbar for the depth scale
plt.title("Inferred Depth Map")
plt.show()

