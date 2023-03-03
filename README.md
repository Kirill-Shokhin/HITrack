# HITrack

**Human Inertial Tracking** is a pipeline consisting of 3 human recognition state-of-the-art neural networks 
([yolov7](https://github.com/WongKinYiu/yolov7), 
[VitPose](https://github.com/ViTAE-Transformer/ViTPose) and 
[MHFormer](https://github.com/Vegetebird/MHFormer)) 
linked together by specially designed **Inertial Tracking** to produce a 3D scene on a monocular image.

https://user-images.githubusercontent.com/46619252/222317959-678a6505-c4ac-498f-9fba-54cdce71df96.mp4

## Quick start

```python
from HITrack import HITrack
hit = HITrack('videos/dance.mp4')

# 2D keypoints + tracking
hit.compute_2d()

# merging recovered tracks and broken tracks manually
hit.recover_2d({1:6, 2:5, 3:4})

# 2D to 3D by tracking
hit.compute_3d()

# 3D to scene
hit.compute_scene()

# visualising any of these steps
hit.visualize('3D_scene', compress=True, original_sound=True)
```

## License
This project is licensed under the terms of the MIT license.