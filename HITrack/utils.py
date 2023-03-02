from tqdm.notebook import tqdm
from itertools import groupby
import numpy as np
import requests
import warnings
import torch
import cv2
import gc
import os
warnings.filterwarnings("ignore")


def download_models(cid, path='', title=None):
    if len(os.path.basename(path).split('.')) == 2:
        path = os.path.dirname(path)

    url = f'https://drive.google.com/uc?export=download&confirm=no_antivirus&id={cid}'
    with requests.get(url, stream=True) as response:
        total_size = int(response.headers["Content-Length"])
        filename = response.headers['Content-Disposition'].split('"')[1]
        pbar = tqdm(response.iter_content(1024), title,
                    total=total_size, unit="B", unit_scale=True, unit_divisor=1024)
        with open(os.path.join(path, filename), "wb") as f:
            for data in pbar.iterable:
                f.write(data)
                pbar.update(len(data))
    pbar.close()


def clear():
    torch.cuda.empty_cache()
    gc.collect()


def id_existence(keypoints):
    return ~(keypoints == 0).all(2).all(2)


def recover_kps(keypoints, max_=10):
    keypoints = keypoints.copy()
    exist = ~(keypoints == 0).all(2).all(2)
    for id_ in range(len(keypoints)):
        rep = [[k, len(list(g))] for k, g in groupby(exist[id_])]
        starts = np.array([0] + [x[1] for x in rep]).cumsum()

        for i, (exs, val) in enumerate(rep[1:-1], start=1):
            if not exs and val < max_:
                left = keypoints[id_][starts[i] - 1]
                right = keypoints[id_][starts[i + 1]]

                for k in range(val):
                    alpha = (k + 1) / (val + 1)
                    keypoints[id_][starts[i] + k] = left * (1 - alpha) + right * alpha

                print(f'id:{id_}, frames recovered: {starts[i]}-{starts[i + 1]}')
    return keypoints


def merge_track(keypoints, merge_dict: dict):
    exist = ~(keypoints == 0).all(2).all(2)
    keypoints = keypoints.copy()
    for old, new in merge_dict.items():
        keypoints[old][exist[new]] = keypoints[new][exist[new]]
    keypoints = np.delete(keypoints, list(merge_dict.values()), axis=0)
    return keypoints


def world2cam(pose, pose3d, cameraMatrix, distCoeffs, rvec=None, tvec=None):
    use = False if rvec is None else True
    rvec, tvec = cv2.solvePnP(pose3d, pose, cameraMatrix, distCoeffs, rvec, tvec, use)[1:]
    R = cv2.Rodrigues(rvec)[0]
    x, y, z = R @ pose3d.T + tvec
    x, y, z = x, z, -y
    return np.array((x, y, z)).T, rvec, tvec


def CreateCamera(img_size, focus=1):
    fx = fy = focus * np.hypot(*img_size)
    cx, cy = img_size[0] / 2, img_size[1] / 2

    distCoeffs = np.zeros(4, np.float32)
    cameraMatrix = np.float32([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])
    return cameraMatrix, distCoeffs


def poses2scene(data, img_size=(1920, 1080)):
    exist = data.ids != -1
    scene = np.zeros_like(data.keypoints_3d)
    cameraMatrix, distCoeffs = CreateCamera(img_size)
    
    lis = []
    rvec, tvec = None, None
    for pose, pose3d in zip(coco2h36(data.keypoints_2d)[exist], data.keypoints_3d[exist]):
        sc, rvec, tvec = world2cam(pose, pose3d, cameraMatrix, distCoeffs, rvec, tvec) 
        lis.append(sc)
        
    scene[exist] = lis

#     scene[exist] = [world2cam(pose, pose3d, cameraMatrix, distCoeffs) for pose, pose3d
#                                         in zip(coco2h36(data.keypoints_2d)[exist], data.keypoints_3d[exist])]
    return scene


class VideoDataKeypoints:

    def __init__(self, path='test.npz', keypoints_2d=None, keypoints_3d=None, scene=None):
        self.keypoints_2d, self.keypoints_3d, self.scene = self.check([keypoints_2d, keypoints_3d, scene])
        self.shape, self.ids = self.create_ids()
        self.path = path

    def save(self):
        kps2d, kps3d, scne = [arr[self.ids != -1] if arr is not None else None
                              for arr in [self.keypoints_2d, self.keypoints_3d, self.scene]]

        np.savez_compressed(self.path, keypoints_2d=kps2d, keypoints_3d=kps3d, scene=scne, ids=self.ids,
                            shape=self.shape)

    def load(self):
        with np.load(self.path, allow_pickle=True) as npz:
            self.ids, self.shape = npz['ids'], npz['shape']

            arrays = []
            for name, dims in zip(npz.files[:3], [2, 3, 3]):
                array = None
                if npz[name].ndim:
                    array = np.zeros((*self.shape, dims))
                    array[self.ids != -1] = npz[name]
                arrays.append(array)

        self.keypoints_2d, self.keypoints_3d, self.scene = arrays

    def update(self, keypoints_2d=None, keypoints_3d=None, scene=None, save=False):
        arrays = self.check([keypoints_2d, keypoints_3d, scene])
        if keypoints_2d is not None:
            self.keypoints_2d, self.keypoints_3d, self.scene = arrays
        else:
            if keypoints_3d is not None:
                self.keypoints_3d = arrays[1]
            if scene is not None:
                self.scene = arrays[2]

        self.shape, self.ids = self.create_ids()
        if save:
            self.save()

    def create_ids(self):
        for arr in [self.keypoints_2d, self.keypoints_3d, self.scene]:
            if arr is not None:
                n_ids, n_frames, _, _ = arr.shape
                ids = np.mgrid[:n_ids, :n_frames][0]
                ids[~id_existence(arr)] = -1
                return arr.shape[:3], ids
        return None, None

    @staticmethod
    def check(arrays):
        shapes = [arr.shape for arr in arrays if arr is not None]
        if len(shapes) > 0:
            assert all([len(shape) == 4 for shape in shapes]), 'All arrays must be 4-dimensional'
            assert np.all(np.diff(np.array(shapes)[:, :3], axis=0) == 0), 'Shapes of arrays must be the same'
            assert np.all(np.diff([id_existence(arr) for arr in arrays if arr is not None],
                                  axis=0) == 0), 'Arrays do not match by the number of ids on some frames'

            arrays = [np.float64(arr) if arr is not None else None for arr in arrays]
        return arrays


class OpenVideo:
    def __init__(self, video_path, end=None, pbar=True):
        self.cap = cv2.VideoCapture(video_path)
        self.shape = int(self.cap.get(3)), int(self.cap.get(4))
        self.fps = self.cap.get(5)
        self.length = end if end else int(self.cap.get(7))
        self.pbar = pbar
        self.frame = 0
        
        if pbar:
            self.tqdm = tqdm(total=self.length)

    def read(self):
        ret, bgr = self.cap.read()
        self.frame += 1
        if self.pbar:
            self.tqdm.update(1)
            
        if self.frame == self.length:
            self.cap.release()
            if self.pbar:
                self.tqdm.close()

        if ret:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        else:
            self.cap.release()
            if self.pbar:
                self.tqdm.close()


def coco2h36(keypoints):
    # shape (ids, frames, 17, 2)
    h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
    coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    spple_keypoints = [10, 8, 0, 7]

    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float64)
    htps_keypoints = np.zeros((*keypoints.shape[:2], 4, 2), dtype=np.float64)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[:, :, 0, 0] = np.mean(keypoints[:, :, 1:5, 0], axis=2)
    htps_keypoints[:, :, 0, 1] = np.sum(keypoints[:, :, 1:3, 1], axis=2) - keypoints[:, :, 0, 1]
    htps_keypoints[:, :, 1, :] = np.mean(keypoints[:, :, 5:7, :], axis=2)
    htps_keypoints[:, :, 1, :] += (keypoints[:, :, 0, :] - htps_keypoints[:, :, 1, :]) / 3

    htps_keypoints[:, :, 2, :] = np.mean(keypoints[:, :, 11:13, :], axis=2)
    htps_keypoints[:, :, 3, :] = np.mean(keypoints[:, :, [5, 6, 11, 12], :], axis=2)

    keypoints_h36m[:, :, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, :, h36m_coco_order, :] = keypoints[:, :, coco_order, :]

    keypoints_h36m[:, :, 9, :] -= (keypoints_h36m[:, :, 9, :] - np.mean(keypoints[:, :, 5:7, :], axis=2)) / 4
    keypoints_h36m[:, :, 7, 0] += 0.3 * (keypoints_h36m[:, :, 7, 0] - np.mean(keypoints_h36m[:, :, [0, 8], 0], axis=2))
    keypoints_h36m[:, :, 8, 1] -= (np.mean(keypoints[:, :, 1:3, 1], axis=2) - keypoints[:, :, 0, 1]) * 2 / 3

    return keypoints_h36m
