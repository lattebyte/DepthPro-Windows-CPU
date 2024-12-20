![](data/inferred_1.png)
![](data/inferred_0.png)

## YouTube Demo

https://youtu.be/3Vm2yJsiLZY
https://youtu.be/ZKJg3GJ4SSY

## Getting Started

We recommend setting up a virtual environment. Using e.g. Anaconda run as administration,

https://www.anaconda.com/download/success  
for MacOS, choose Apple sillicon for M1, M2  
otherwise usually Intel Chip option

the `depth_pro` package can be installed via:

```bash
cd /target/folder
conda create -n depth-pro-win-cpu -y python=3.9
conda activate depth-pro-win-cpu
git clone https://github.com/lattebyte/DepthPro-Windows-CPU.git
cd DepthPro-Windows-CPU
pip install -e .
pip install opencv-python
```

To download pretrained checkpoints follow the steps below:

```bash
# create directory
mkdir checkpoints
```

Then, manually download model by entering the following into web browser:
`https://huggingface.co/lattebyte-ai/Depth_Est/blob/main/depth_pro.pt`

Once completed, move `depth_pro.pt` into `checkpoints`

### Running from python

LICENSE_2 applies to these files
Note that file path for MacOS is WIP, it might be different to Windows OS
For 2D depth mapping:

```bash
python cpu_img_inference_2D_color_map.py
```

For 3D point cloud:

```bash
python cpu_img_inference_3D_point_cloud.py
```

## Original References

### Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

This software project accompanies the research paper:
**[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**,
_Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun_.

We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image.

The model in this repository is a reference implementation, which has been re-trained. Its performance is close to the model reported in the paper but does not match it exactly.
![](data/depth-pro-teaser.jpg)

### Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

This software project accompanies the research paper:
**[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**,
_Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun_.

We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image.

The model in this repository is a reference implementation, which has been re-trained. Its performance is close to the model reported in the paper but does not match it exactly.

### Citation

If you find our work useful, please cite the following paper:

```bibtex
@article{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun}
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  journal    = {arXiv},
  year       = {2024},
  url        = {https://arxiv.org/abs/2410.02073},
}
```

### License

This sample code is released under the [LICENSE](LICENSE) terms.

The model weights are released under the [LICENSE](LICENSE) terms.

### Acknowledgements

Our codebase is built using multiple opensource contributions, please see [Acknowledgements](ACKNOWLEDGEMENTS.md) for more details.

Please check the paper for a complete list of references and datasets used in this work.
