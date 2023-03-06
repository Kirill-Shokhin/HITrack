from ..constants import H36_PERSON_SKELETON, COLORS
from . import HumanVisualizer
from ..utils import coco2h36, OpenVideo
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import plotly.express as px
import io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.switch_backend('agg')


class Visualizer:
    def __init__(self, data, video_path=None, skeleton_format='COCO'):
        self.data = data
        self.video_path = video_path
        self.size = (1080, 1080)
        self.fps = 30
        self.pairs = np.array(H36_PERSON_SKELETON) - 1
        self.skeleton_format = skeleton_format

    def __call__(self, how='3D_scene', id_=None, end=None, compress=False, original_sound=False):
        self.video_out = f'{self.video_path}.{how}.mp4'
        self.len_video = self.data.shape[1] if end is None else int(end)

        if how == '3D_scene':
            flat = self.data.scene[self.data.ids != -1].reshape(-1, 3)
            range_axes = np.concatenate((flat.min(0)[None], flat.max(0)[None])).T
            scale_axes = np.diff(range_axes)[:, 0]

        elif how == '3D_single':
            range_axes = [[-0.7, 0.7] for _ in range(3)]
            scale_axes = [1, 1, 1]
            if id_ is None:
                raise AssertionError('Specify id_')

        else:
            assert how in ['2D', '3D_single', '3D_scene']

        if self.video_path:
            cap = OpenVideo(self.video_path, self.len_video, False)
            self.fps = cap.fps
            w, h = cap.shape
            self.size = (w, h) if how == '2D' else (w + h, h)

            vis_2d = HumanVisualizer(skeleton_format=self.skeleton_format)
            keypoints_2d = self.data.keypoints_2d
            if self.skeleton_format == 'H36':
                keypoints_2d = coco2h36(keypoints_2d)
        
        fig = plt.figure(figsize=(10, 10), dpi=self.size[1] / 10)
        result = cv2.VideoWriter(self.video_out, cv2.VideoWriter_fourcc(*'MP4V'), self.fps, self.size)
        for i in tqdm(range(self.len_video)):

            if how == '3D_scene':
                ids = self.data.ids[:, i]
                poses = self.data.scene[:, i]

            elif how == '3D_single':
                ids = [id_]
                poses = self.data.keypoints_3d[ids, i]

            if how != '2D':
                image_3d = self.matplotlib_3D(fig, poses, ids, range_axes, scale_axes)

            if self.video_path:
                rgb = cap.read()
                rgb = vis_2d(rgb, keypoints_2d[:, i], self.data.ids[:, i])
                if how != '2D':
                    image = np.concatenate((rgb, image_3d), axis=1)
                else:
                    image = rgb
            else:
                image = image_3d

            result.write(image[:, :, ::-1])
        result.release()

        if compress or original_sound:
            print('ffmpeg starting...')
            self.ffmpeg(compress, original_sound)
            print('Done')

    def matplotlib_3D(self, fig, poses, ids, range_axes, scale_axes):
        ax = Axes3D(fig, elev=15, azim=80, auto_add_to_figure=False)
        fig.add_axes(ax)

        for id_, pose in zip(ids, poses):
            if id_ != -1:
                for pair in pose[self.pairs]:
                    x, y, z = pair.T
                    ax.plot(x, y, z, lw=2, color=np.array(COLORS[id_]) / 255)

        ax.set_xlim3d(range_axes[0][::-1])
        ax.set_ylim3d(range_axes[1][::-1])
        ax.set_zlim3d(range_axes[2])
        ax.set_box_aspect(scale_axes)

        white = (1.0, 1.0, 1.0, 0.0)
        ax.xaxis.set_pane_color(white)
        ax.yaxis.set_pane_color(white)
        ax.zaxis.set_pane_color(white)

        ax.tick_params('x', labelbottom=False)
        ax.tick_params('y', labelleft=False)
        ax.tick_params('z', labelleft=False)

        image = save_ax_nosave()
        fig.clf()

        return image

    def ffmpeg(self, compress=True, original_sound=True, replace=True):
        # ffmpeg required
        import subprocess, os
        if original_sound:
            subprocess.call(f'ffmpeg -i {self.video_path} -q:a 0 -map a temp.mp3', shell=True)

            cmd = f"ffmpeg -i {self.video_out} -i temp.mp3 -c copy -map 0:v:0 -map 1:a:0 temp.mp4"
            subprocess.call(cmd, shell=True)

            os.remove('temp.mp3')
            os.remove(self.video_out)
            os.rename('temp.mp4', self.video_out)

        if compress:
            subprocess.call('ffmpeg -i {} -acodec libmp3lame -ab 192 {} -y'.format(self.video_out,
                                                    self.video_out + '.mp4'), stderr=subprocess.DEVNULL, shell=True)
            if replace:
                os.remove(self.video_out)
                os.rename(self.video_out + '.mp4', self.video_out)


def save_ax_nosave():
    buff = io.BytesIO()
    plt.savefig(buff, format="png", bbox_inches='tight', pad_inches=0)
    buff.seek(0)
    im = plt.imread(buff)
    return np.uint8(im[:, :, :3] * 255)


def scene_plotly(scene, thickness=4, show_axes=False):
    assert len(scene.shape) == 4
    exist = ~(scene == 0).all(2).all(2)
    n_ids, n_frames = exist.shape
    pairs = np.array(H36_PERSON_SKELETON) - 1
    Nones = np.full((n_ids, n_frames, len(pairs), 1, 3), None)
    px_poses = np.concatenate((scene[:, :, pairs], Nones), 3)
    px_poses = px_poses[exist].reshape(-1, 3)

    ids_frames = np.tile(np.mgrid[0:n_ids, 0:n_frames][:, exist].T, len(pairs) * 3).reshape(-1, 2)
    df = pd.DataFrame(np.concatenate((px_poses, ids_frames), 1),
                      columns=['x', 'y', 'z', 'id_', 'frame']).apply(pd.to_numeric)

    fig = px.line_3d(df, x='x', y='y', z='z', color='id_', animation_frame='frame',
                     color_discrete_sequence=rgb2hex(COLORS))

    min_max = np.concatenate((df.min()[:3].values[None], df.max()[:3].values[None])).T
    diff = np.diff(min_max)[:, 0]
    scale = diff.mean()
    title = None if show_axes else ''

    scene_aspectratio = dict(x=diff[0], y=diff[1], z=diff[2])
    scene = dict(xaxis=dict(nticks=5, range=min_max[0], showticklabels=show_axes, title=title),
                 yaxis=dict(nticks=5, range=min_max[1], showticklabels=show_axes, title=title),
                 zaxis=dict(nticks=5, range=min_max[2], showticklabels=show_axes, title=title))

    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=-0.5 * scale, y=-1.7 * scale, z=0.6 * scale))

    fig.update_layout(showlegend=False,
                      width=900, height=500, margin=dict(t=10, r=0, l=0, b=0), scene=scene,
                      scene_camera=camera, scene_aspectmode='manual', scene_aspectratio=scene_aspectratio,
                      )

    fig.update_traces(line_width=thickness)
    return fig


def rgb2hex(colors):
    return ["".join([hex(rgb)[2:] for rgb in c]) for c in colors]
