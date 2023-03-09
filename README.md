# HITrack

**HITrack** or **Human Inertial Tracking** is a pipeline consisting of 3 human recognition state-of-the-art neural networks 
([yolov7](https://github.com/WongKinYiu/yolov7), 
[VitPose](https://github.com/ViTAE-Transformer/ViTPose) and 
[MHFormer](https://github.com/Vegetebird/MHFormer)) 
linked together by specially designed **Inertial Tracking** to produce a 3D scene on a monocular image.

https://user-images.githubusercontent.com/46619252/223071638-9cb6990e-6ba1-42b2-81f5-1d6ed856a4b3.mp4
## Quick start

```
pip install HITrack
```

```python
from HITrack import HITrack
hit = HITrack('videos/dance.mp4')

# 2D keypoints + tracking
hit.compute_2d(yolo='yolov7x', vitpose='b')

# merging recovered tracks and broken tracks manually
hit.recover_2d({2:4, 3:5})

# 2D to 3D by tracking
hit.compute_3d()

# 3D to scene
hit.compute_scene()

# visualising any of these steps
hit.visualize('3D_scene', compress=True, original_sound=True)
```

## License
This project is licensed under the terms of the MIT license.