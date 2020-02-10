#! /usr/bin/env /usr/local/bin/python3

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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    image_0_gray = np.array(image_0 * 255, dtype=np.uint8)
    points = cv2.goodFeaturesToTrack(image_0_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    corners = FrameCorners(np.array(list(range(len(points)))), points, 7 * np.ones(len(points)))
    builder.set_corners_at_frame(0, corners)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.001))

    print(len(frame_sequence))
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1_gray = np.array(image_1 * 255, dtype=np.uint8)


        # calculate optical flow
        points1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=image_0_gray, nextImg=image_1_gray, prevPts=points, nextPts=None, **lk_params)
        good_new = points1[st == 1]

        # Now update the previous frame and previous points
        image_0_gray = image_1_gray
        points = good_new.reshape(-1, 1, 2)
        corners = FrameCorners(np.array(list(range(len(points)))), points, 7 * np.ones(len(points)))
        builder.set_corners_at_frame(frame, corners)


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
    kek = builder.build_corner_storage()
    print(kek.__getitem__(0).points)
    return kek


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
