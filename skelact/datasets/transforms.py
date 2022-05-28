# Copyright (c) Hikvision Research Institute. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

from mmaction.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PoseRandomCrop:
    """Sample a clip randomly with a random clip length ranging from min_ratio
    to max_ratio from the video.

    Required keys are "total_frames", added or modified key is "frame_inds".

    Args:
        min_ratio (float): Minimal sampling ratio.
        max_ratio (float): Maximal sampling ratio.
        min_len (int): Minimal length of each sampled output clip.
    """

    def __init__(self, min_ratio=0.5, max_ratio=1.0, min_len=64):

        assert 0 < min_ratio <= 1
        assert 0 < max_ratio <= 1
        assert min_ratio <= max_ratio

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_len = min_len

    def __call__(self, results):
        num_frames = results['total_frames']

        min_frames = int(num_frames * self.min_ratio)
        max_frames = int(num_frames * self.max_ratio)
        clip_len = np.random.randint(min_frames, max_frames + 1)
        clip_len = min(max(clip_len, self.min_len), num_frames)

        start = np.random.randint(0, num_frames - clip_len + 1)
        inds = np.arange(start, start + clip_len)

        results['frame_inds'] = inds.astype(np.int)
        return results


@PIPELINES.register_module()
class PoseCenterCrop:
    """Sample a clip from the video.

    Required keys are "total_frames", added or modified key is "frame_inds".

    Args:
        clip_ratio (float): Sampling ratio.
    """

    def __init__(self, clip_ratio=0.9):

        assert 0 < clip_ratio <= 1
        self.clip_ratio = clip_ratio

    def __call__(self, results):
        num_frames = results['total_frames']

        clip_len = int(num_frames * self.clip_ratio)
        start = (num_frames - clip_len) // 2
        inds = np.arange(start, start + clip_len)

        results['frame_inds'] = inds.astype(np.int)
        return results


@PIPELINES.register_module()
class PoseResize:
    """Resize the input video to the given clip length.

    Required keys are "keypoint", added or modified keys
    are "keypoint" and "frame_inds".

    Args:
        clip_len (int): clip length.
    """

    def __init__(self, clip_len=32):

        self.clip_len = clip_len

    def __call__(self, results):
        frame_inds = results['frame_inds']
        keypoint = results['keypoint'][:, frame_inds]

        m, t, v, c = keypoint.shape
        keypoint = keypoint.transpose((0, 3, 1, 2))  # M T V C -> M C T V
        keypoint = F.interpolate(
            torch.from_numpy(keypoint),
            size=(self.clip_len, v),
            mode='bilinear',
            align_corners=False)
        keypoint = keypoint.permute((0, 2, 3, 1)).numpy()
        results['keypoint'] = keypoint

        inds = np.arange(self.clip_len)
        results['frame_inds'] = inds.astype(np.int)

        return results


@PIPELINES.register_module()
class PoseRandomRotate:
    """Random rotate the input skeleton sequence.

    Required key is "keypoint", modified key is "keypoint".

    Args:
        rand_rotate (float): strength of rotation.
    """

    def __init__(self, rand_rotate=0.1):
        self.theta = rand_rotate * np.pi

    def __call__(self, results):
        keypoint = results['keypoint']

        keypoint = keypoint.transpose((3, 1, 2, 0))  # M T V C -> C T V M
        keypoint = self.random_rotate(keypoint)
        keypoint = keypoint.transpose((3, 1, 2, 0))  # C T V M -> M T V C
        results['keypoint'] = keypoint

        return results

    def random_rotate(self, keypoint):
        theta = np.random.uniform(-self.theta, self.theta, 3)
        cos = np.cos(theta)
        sin = np.sin(theta)

        # rotate by Z
        rot_z = np.eye(3)
        cos_z, sin_z = cos[0], sin[0]
        rot_z[0, 0] = cos_z
        rot_z[1, 0] = sin_z
        rot_z[0, 1] = -sin_z
        rot_z[1, 1] = cos_z

        # rotate by Y
        rot_y = np.eye(3)
        cos_y, sin_y = cos[1], sin[1]
        rot_y[0, 0] = cos_y
        rot_y[0, 2] = sin_y
        rot_y[2, 0] = -sin_y
        rot_y[2, 2] = cos_y

        # rotate by X
        rot_x = np.eye(3)
        cos_x, sin_x = cos[2], sin[2]
        rot_x[1, 1] = cos_x
        rot_x[2, 1] = sin_x
        rot_x[1, 2] = -sin_x
        rot_x[2, 2] = cos_x
        rot = np.matmul(np.matmul(rot_z, rot_y), rot_x)

        c, t, v, m = keypoint.shape
        keypoint = np.matmul(rot, keypoint.reshape(c, -1))
        keypoint = keypoint.reshape(c, t, v, m).astype(np.float32)
        return keypoint
