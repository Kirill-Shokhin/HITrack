from .Networks import YOLOv7, VITPOSE, MHFORMER
from .KeypointTracker import KeypointTracker
from .constants import WEIGHTS_COCO
from .utils import OpenVideo, VideoDataKeypoints, recover_kps, merge_track, poses2scene
from .HumanVisualizer import Visualizer
import numpy as np
import os


class HITrack:
    def __init__(self, video_path, wait_recovery=20):
        self.wait_recovery = wait_recovery

        self.video_path = video_path
        self.npz_path = ''.join(video_path.split('.')[:-1])+'.npz'
        self.data = VideoDataKeypoints(self.npz_path)

        if os.path.exists(self.npz_path):
            self.data.load()
            print(f'Loaded from file {self.npz_path}')

    def __call__(self, merge_dict=None):
        if self.data.keypoints_2d is None:
            self.compute_2d()
        self.recover_2d(merge_dict)
        self.compute_3d()
        self.compute_scene()
        self.visualize('3D_scene')

    def compute_2d(self, save=True):
        self.ht = HumanTrack(self.video_path, wait_recovery=self.wait_recovery)
        keypoints_2d = self.ht.compute()
        self.data.update(keypoints_2d, save=save)

    def recover_2d(self, merge_dict=None, save=True):
        keypoints_2d = self.data.keypoints_2d
        if merge_dict is not None:
            keypoints_2d = merge_track(keypoints_2d, merge_dict)

        keypoints_2d = recover_kps(keypoints_2d, self.wait_recovery)
        self.data.update(keypoints_2d, save=save)

    def compute_3d(self, save=True):
        lifter = MHFORMER('351')
        keypoints_3d = lifter(self.data.keypoints_2d)
        self.data.update(keypoints_3d=keypoints_3d, save=save)

    def compute_scene(self, save=True):
        scene = poses2scene(self.data)
        self.data.update(scene=scene, save=save)

    def visualize(self, how='scene_3D', id_=None, skeleton_format="H36", end=None):
        # scene_matplotlib(self.data.scene, video_path='videos/dance2.VitPose_b.tracking.mp4')
        vis = Visualizer(self.data, video_path=self.video_path, skeleton_format=skeleton_format)
        vis(how, id_, end)


class HumanTrack:
    def __init__(self, video_path, yolo_model='yolov7', vitpose_model='b', wait_recovery=15):

        self.cap = OpenVideo(video_path)
        self.video_path = video_path

        self.wait_recovery = wait_recovery
        self.keypoint_weights = WEIGHTS_COCO

        self.pose = VITPOSE(vitpose_model)
        self.det = YOLOv7(yolo_model)
        self.keypoints = []

    def compute(self):
        for i in range(self.cap.length):
            rgb = self.cap.read()
            boxes = self.det(rgb)
            skeletons = self.pose(rgb, boxes)

            if i == 0:
                self.t = KeypointTracker(skeletons, wait_recovery=self.wait_recovery,
                                         keypoint_weights=self.keypoint_weights)
            else:
                self.t.step(skeletons)

            skeletons, ids = self.t.current()
            self.keypoints.append([i, skeletons, ids])

        keypoints_2d = np.zeros((self.t.cur_max_id + 1, self.cap.length, len(self.keypoint_weights), 2))
        for i, skeletons, ids in self.keypoints:
            keypoints_2d[ids, i] = skeletons[:, :, :2]

        return keypoints_2d