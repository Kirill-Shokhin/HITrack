# HITrack

This repository is a pipeline of three state-of-the-art human recognition models on a monocular image: [yolov7](https://github.com/WongKinYiu/yolov7), 
[VitPose](https://github.com/ViTAE-Transformer/ViTPose) and [MHFormer](https://github.com/Vegetebird/MHFormer) - to create bounding boxes, 2D poses and 
3D poses respectively. By combining them, a 3D scene with people can be produced. 
But the transition from 2D to 3D requires stable tracking, so an Human Inertial Tracking system was developed, after which the repository is named.

https://user-images.githubusercontent.com/46619252/222317959-678a6505-c4ac-498f-9fba-54cdce71df96.mp4
