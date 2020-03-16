#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from tqdm import tqdm

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


CORNER_QUALITY = 0.0001

kek = False

def filter_too_close(img_shape, new_points, good_points, minDistance):
    mask = np.ones(img_shape, dtype=np.uint8)
    good_points_int = good_points.astype(np.int32)
    good_points_int = np.maximum(good_points_int, 0)
    #print(good_points_int[:, :, 0].max(), good_points_int[:, :, 1].max())
    good_points_int[:, :, 0] = np.minimum(good_points_int[:, :, 0], img_shape[1] - 1)
    good_points_int[:, :, 1] = np.minimum(good_points_int[:, :, 1], img_shape[0] - 1)

    mask.T[good_points_int[:, :, 0], good_points_int[:, :, 1]] = 0

    kernel = np.ones((minDistance, minDistance), dtype=np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)


    cv2.imwrite('kek.png', mask * 255)

    #print(mask.sum(), img_shape[0] * img_shape[1])

    new_points_int = new_points.astype(np.int32)
    new_points_int = np.maximum(new_points_int, 0)
    new_points_int[:, :, 0] = np.minimum(new_points_int[:, :, 0], img_shape[1] - 1)
    new_points_int[:, :, 1] = np.minimum(new_points_int[:, :, 1], img_shape[0] - 1)


    to_take = mask.T[new_points_int[:, :, 0], new_points_int[:, :, 1]] == 1
    #print(to_take.shape, new_points[to_take].shape)
    return new_points[to_take]


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    image_0_gray = np.array(image_0 * 255, dtype=np.uint8)
    h, w = image_0_gray.shape
    print(h, w)
    blockSize = 7
    #maxCorners = int(20 * h * w / (np.pi * (blockSize / 2) ** 2))
    maxCorners = 30000
    minDistance = 7
    points = cv2.goodFeaturesToTrack(
        image_0,
        mask=None,
        maxCorners=maxCorners,
        qualityLevel=CORNER_QUALITY,
        minDistance=minDistance,
        blockSize=blockSize,
        k=0.04,
        gradientSize=3,
        useHarrisDetector=False
    )
    ids = np.arange(len(points))
    cur_id = len(points)
    corners = FrameCorners(ids, points, 7 * np.ones(len(points)))
    builder.set_corners_at_frame(0, corners)

    lk_params = dict(winSize=(30, 30),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    for frame, image_1 in tqdm(enumerate(frame_sequence[1:], 1)):
        image_1_gray = np.array(image_1 * 255, dtype=np.uint8)

        # calculate optical flow
        points1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=image_0_gray, nextImg=image_1_gray, prevPts=points, nextPts=None, **lk_params)
        st = st.reshape(-1)
        err = err.reshape(-1)

        to_take = np.logical_and(st == 1, err < 2.5)

        assert len(points1) == len(points)
        good_points = points1[to_take]
        good_ids = ids[to_take]

        new_points = cv2.goodFeaturesToTrack(
            image_1,
            mask=None,
            maxCorners=maxCorners,
            qualityLevel=CORNER_QUALITY,
            minDistance=minDistance,
            blockSize=blockSize,
            k=0.04,
            gradientSize=3,
            useHarrisDetector=False
        )
        #new_points = np.array([point for point in new_points if np.linalg.norm(good_points - point, axis=-1).min() > minDistance], dtype=np.float32)
        new_points = filter_too_close(image_1_gray.shape, new_points, good_points, minDistance)
        new_ids = np.arange(cur_id, cur_id + len(new_points))
        cur_id += len(new_points)

        # Now update the previous frame and previous points
        points = np.concatenate([good_points.reshape(-1, 1, 2), new_points.reshape(-1, 1, 2)])[:maxCorners]
        ids = np.concatenate([good_ids, new_ids])[:maxCorners]
        corners = FrameCorners(ids, points, 7 * np.ones(len(points)))
        builder.set_corners_at_frame(frame, corners)

        image_0_gray = image_1_gray


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
