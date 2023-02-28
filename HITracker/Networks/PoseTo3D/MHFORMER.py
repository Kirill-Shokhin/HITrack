from .MHFormer.model.mhformer import Model
from .MHFormer.common.camera import camera_to_world
from HITracker.utils import coco2h36, clear

import glob
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch
import numpy as np
import os


class MHFORMER:
    def __init__(self, model_name='351', batch_size='auto'):
        self.model_name = model_name
        if self.model_name not in ['27', '81', '243', '351']:
            raise AssertionError('Only 27, 81, 243, 351 models exist')

        self.layers, self.channel, self.d_hid, self.frames = 3, 512, 1024, int(self.model_name)
        self.n_joints, self.out_joints = 17, 17
        self.pad = self.frames // 2
        self.model_dir = os.path.join(os.path.dirname(__file__),
                                      f'MHFormer/checkpoint/pretrained/{self.model_name}')
        self.batch_size = self.get_batch_size() if batch_size == 'auto' else int(batch_size)
        self.model = self.load_model()

    def __call__(self, keypoints_2d):

        keypoints_2d = coco2h36(keypoints_2d)

        seq_of_seq, len_seq = self.create_input(keypoints_2d)
        loader = DataLoader(seq_of_seq, batch_size=self.batch_size, shuffle=False)

        poses_3D = []
        for batch in tqdm(loader):
            batch_flip = self.flip_skeleton(batch.clone())
            batch = torch.concatenate((batch[None], batch_flip[None])).cuda()

            post_out = self.estimation(batch)
            poses_3D.append(post_out)

        poses_3D = np.concatenate(poses_3D)
        poses_3D = [poses_3D[len_seq[i]:len_seq[i + 1]] for i in range(len(len_seq) - 1)]
        clear()

        keypoints_3d = np.zeros((*keypoints_2d.shape[:-1], 3))
        for i in range(len(keypoints_2d)):
            keypoints_3d[i, (keypoints_2d[i] != 0).all(1).all(1)] = poses_3D[i]

        keypoints_3d *= [[[[-1, -1, 1]]]]
        return keypoints_3d

    @torch.no_grad()
    def estimation(self, batch):
        output_3D_non_flip = self.model(batch[0])[:, self.pad]
        output_3D_flip = self.model(batch[1])[:, self.pad]
        output_3D_flip = self.flip_skeleton(output_3D_flip)
        output_3D = (output_3D_non_flip + output_3D_flip) / 2
        output_3D[:, 0] = 0
        post_out = self.rotate_out(output_3D.cpu().detach().numpy())
        return post_out

    def create_input(self, keypoints):
        seq_of_seq = []
        len_seq = [0]
        for sequence in keypoints:
            sequence = sequence[(sequence != 0).all(1).all(1)]
            sequences = np.array(
                [self.normalize(self.make_sequence(sequence, i)) for i in range(len(sequence))])
            seq_of_seq.append(sequences.astype(np.float32))
            len_seq.append(len(sequences))

        seq_of_seq = np.concatenate(seq_of_seq)
        len_seq = np.array(len_seq).cumsum()
        return seq_of_seq, len_seq

    def make_sequence(self, keypoints, i):
        start = max(0, i - self.pad)
        end = min(i + self.pad, len(keypoints) - 1)
        sequence = keypoints[start:end + 1]

        left_pad, right_pad = 0, 0
        if sequence.shape[0] != self.frames:
            if i < self.pad:
                left_pad = self.pad - i
            if i > len(keypoints) - self.pad - 1:
                right_pad = i + self.pad - (len(keypoints) - 1)

            sequence = np.pad(sequence, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        return sequence

    @staticmethod
    def normalize(sequence):
        transp = sequence.transpose(1, 0, 2)
        shifted = transp - sequence.mean(axis=1)
        input_2D = shifted / (sequence.max(axis=1) - sequence.min(axis=1)).mean() / 2
        return input_2D.transpose(1, 0, 2)

    @staticmethod
    def flip_skeleton(sequence):
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        if len(sequence.shape) == 4:
            sequence[:, :, :, 0] *= -1
            sequence[:, :, joints_left + joints_right] = sequence[:, :, joints_right + joints_left]
        elif len(sequence.shape) == 3:
            sequence[:, :, 0] *= -1
            sequence[:, joints_left + joints_right] = sequence[:, joints_right + joints_left]
        else:
            raise AssertionError('Len of shape must be equal 3 or 4')
        return sequence

    @staticmethod
    def get_batch_size():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        return int(total // 30)

    @staticmethod
    def rotate_out(post_out):
        rot = np.float32([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088])
        return camera_to_world(post_out, R=rot, t=0)

    def load_model(self):
        model = Model(self).cuda()
        model_dict = model.state_dict()
        model_path = sorted(glob.glob(os.path.join(self.model_dir, '*.pth')))[0]
        pre_dict = torch.load(model_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model.load_state_dict(model_dict)
        model.eval()
        return model
