#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    Correspondences,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    rodrigues_and_translation_to_view_mat3x4,
    triangulate_correspondences,
    TriangulationParameters)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    print(intrinsic_mat)


    id0, mat0 = known_view_1
    id1, mat1 = known_view_2
    mat0 = pose_to_view_mat3x4(mat0)
    mat1 = pose_to_view_mat3x4(mat1)

    #n_iter = 5
    #for iter in range(n_iter):
    kmat0 = intrinsic_mat @ mat0
    kmat1 = intrinsic_mat @ mat1

    points0, ids0 = corner_storage[id0].points, corner_storage[id0].ids
    points1, ids1 = corner_storage[id1].points, corner_storage[id1].ids

    j = 0

    common0 = []
    common1 = []
    common_ids = []
    for i, cur0 in enumerate(ids0):
        while j < len(ids1) and ids1[j] < cur0:
            j += 1
        if j == len(ids1):
            break
        if ids1[j] == cur0:
            common0.append(points0[i])
            common1.append(points1[j])
            common_ids.append(cur0)
    common0 = np.array(common0)
    common1 = np.array(common1)
    common_ids = np.array(common_ids)

    points_3d = []
    for point0, point1 in zip(common0, common1):
        A = np.vstack([
            point0.reshape(-1, 1) @ kmat0[2, :].reshape(1, -1) - kmat0[:2, :],
            point1.reshape(-1, 1) @ kmat1[2, :].reshape(1, -1) - kmat1[:2, :]
        ])
        b = -A[:, 3]
        A = A[:, :3]
        x = A.T @ (A @ A.T)**(-1) @ b
        points_3d.append(x)
    points_3d = np.array(points_3d)

    a1, a2, b = triangulate_correspondences(
        Correspondences(common_ids, common0, common1), mat0, mat1, intrinsic_mat,
        TriangulationParameters(
            max_reprojection_error=2000,
            min_triangulation_angle_deg=0.0001,
            min_depth=3
        )
    )
    print(a1, a2, b)


    view_mats = [mat0, mat1]

    min_id = -1
    min_error = -1

    #points0, ids0 = points1, ids1
    for n_frame, corners in enumerate(corner_storage[2:], 2):
        points1, ids1 = corners.points, corners.ids

        common_obj = []
        common_img = []
        common_all_ids = []
        j = 0
        for i, cur0 in enumerate(common_ids):
            while j < len(ids1) and ids1[j] < cur0:
                j += 1
            if j == len(ids1):
                break
            if ids1[j] == cur0:
                common_obj.append(points_3d[i])
                common_img.append(points1[j])
                common_all_ids.append(cur0)

        common_obj = np.array(common_obj)
        common_img = np.array(common_img)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(common_obj, common_img, intrinsic_mat, None, flags=cv2.SOLVEPNP_EPNP)
        mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        view_mats.append(mat)

        projections = np.array([intrinsic_mat @ mat @ np.hstack([x, 1]) for x in common_obj])
        zs = projections[:, 2]
        projections = projections[:, :2]
        projections[:, 0] /= zs
        projections[:, 1] /= zs

        reprojection_error = np.mean(np.linalg.norm(common_img - projections, axis=-1))

        if min_id == -1 or reprojection_error < min_error:
            min_error = reprojection_error
            min_id = n_frame

        print(f"Processing frame #{n_frame}. Number of inliers: {0 if inliers is None else len(inliers)}."
              f"Reprojection error: {reprojection_error}. Tracking points: {len(common_img)}")

        id0 = id1
        mat0 = mat1
        id1 = min_id
        mat1 = view_mats[min_id]

    point_cloud_builder = PointCloudBuilder(np.array(common_ids), np.array(points_3d))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
