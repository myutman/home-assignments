#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    rodrigues_and_translation_to_view_mat3x4,
    triangulate_correspondences,
    TriangulationParameters, build_correspondences)
from _corners import FrameCorners
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose

RANSAC_REPROJECTION_ERROR=10
MIN_TRISNGULATION_ANGLE=1.5
INITIAL_TRIANGULATION_PARAMETERS = TriangulationParameters(
    max_reprojection_error=15,
    min_triangulation_angle_deg=1,
    min_depth=1e-5
)
NEW_TRIANGULATION_PARAMETERS = {
    1: TriangulationParameters(
        max_reprojection_error=5,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=1e-5
    ),
    2: TriangulationParameters(
        max_reprojection_error=5,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=1e-5
    ),
    4: TriangulationParameters(
        max_reprojection_error=5,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=1e-5
    ),
    8: TriangulationParameters(
        max_reprojection_error=5,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=1e-5
    ),
    16: TriangulationParameters(
        max_reprojection_error=5,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=1e-5
    ),
    32: TriangulationParameters(
        max_reprojection_error=5,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=1e-5
    )
}

def calc_starting_points(
    intrinsic_mat: np.ndarray,
    corner_storage: CornerStorage,
    known_view_1: Tuple[int, Pose],
    known_view_2: Tuple[int, Pose]
):
    id1, pose1 = known_view_1
    id2, pose2 = known_view_2

    corners1 = corner_storage[id1]
    corners2 = corner_storage[id2]

    mat1 = pose_to_view_mat3x4(pose1)
    mat2 = pose_to_view_mat3x4(pose2)

    correspondences = build_correspondences(corners1, corners2)
    points_3d, points_ids, _ = triangulate_correspondences(
        correspondences,
        mat1,
        mat2,
        intrinsic_mat,
        INITIAL_TRIANGULATION_PARAMETERS
    )
    return  points_3d, points_ids

def get_common_points(
    corners: FrameCorners,
    common_ids: List[int],
    points_3d: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
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
    return common_obj, common_img


def build_view_mat(
    common_obj: np.ndarray,
    common_img: np.ndarray,
    intrinsic_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        common_obj,
        common_img,
        intrinsic_mat,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=RANSAC_REPROJECTION_ERROR,
        iterationsCount=1000
    )
    mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    projections = np.array([intrinsic_mat @ mat @ np.hstack([x, 1]) for x in common_obj])
    zs = projections[:, 2]
    projections = projections[:, :2]
    projections[:, 0] /= zs
    projections[:, 1] /= zs

    reprojection_error = np.linalg.norm(common_img - projections, axis=-1).mean()
    inliers = np.array(inliers).reshape(-1)

    return mat, inliers, reprojection_error

def merge_sets(points_3d, points_ids, new_points_3d, new_points_ids):
    res_points_3d = []
    res_points_ids = []

    j = 0
    for point, id in zip(new_points_3d, new_points_ids):
        while j < len(points_ids) and points_ids[j] < id:
            res_points_3d.append(points_3d[j])
            res_points_ids.append(points_ids[j])
            j += 1
        res_points_3d.append(point)
        res_points_ids.append(id)
        if j < len(points_ids) and points_ids[j] == id:
            j += 1
    while j < len(points_ids):
        res_points_3d.append(points_3d[j])
        res_points_ids.append(points_ids[j])
        j += 1

    return res_points_3d, res_points_ids

def calc_new_points(
    cur_id: int,
    intrinsic_mat: np.ndarray,
    corner_storage: CornerStorage,
    view_mats: List[np.ndarray],
    points_3d: List[np.ndarray],
    points_ids: List[int]
) -> Tuple[List[np.ndarray], List[int]]:
    for d in [-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32]:
        i = cur_id + d
        if i < 0 or i >= len(corner_storage) or view_mats[i][0, 0] is None:
            continue
        mat1 = view_mats[i]
        mat2 = view_mats[cur_id]

        corners1 = corner_storage[i]
        corners2 = corner_storage[cur_id]

        correspondences = build_correspondences(corners1, corners2)
        new_points_3d, new_points_ids, _ = triangulate_correspondences(
            correspondences,
            mat1,
            mat2,
            intrinsic_mat,
            NEW_TRIANGULATION_PARAMETERS[abs(d)]
        )

        points_3d, points_ids = merge_sets(
            points_3d,
            points_ids,
            new_points_3d,
            new_points_ids
        )

    return points_3d, points_ids


def track_and_calc_colors(
    camera_parameters: CameraParameters,
    corner_storage: CornerStorage,
    frame_sequence_path: str,
    known_view_1: Optional[Tuple[int, Pose]] = None,
    known_view_2: Optional[Tuple[int, Pose]] = None
) -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    print(intrinsic_mat)

    points_3d, points_ids = calc_starting_points(intrinsic_mat, corner_storage, known_view_1, known_view_2)

    #initial_points_3d, initial_points_ids = points_3d.copy(), points_ids.copy()

    n_points = corner_storage.max_corner_id() + 1
    res_points_3d = np.full((n_points, 3), None)
    #res_points_3d[points_ids] = points_3d

    view_mats = [np.full((3, 4), None) for _ in range(len(corner_storage))]
    view_mats[known_view_1[0]], view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(known_view_2[1])

    points_3d, points_ids = list(points_3d), list(points_ids)

    id1 = known_view_1[0]
    for n_frame, corners in enumerate(corner_storage[id1 + 1:], id1 + 1):
        common_obj, common_img = get_common_points(corners, points_ids, points_3d)

        mat, inliers, reprojection_error = build_view_mat(common_obj, common_img, intrinsic_mat)
        view_mats[n_frame] = mat

        points_3d = list(np.array(points_3d)[inliers])
        points_ids = list(np.array(points_ids)[inliers])
        #res_points_3d[points_ids] = points_3d

        points_3d, points_ids = calc_new_points(
            n_frame,
            intrinsic_mat,
            corner_storage,
            view_mats,
            points_3d,
            points_ids
        )

        print(f"Processing frame #{n_frame}. Number of inliers: {0 if inliers is None else len(inliers)}. "
              f"Reprojection error: {reprojection_error}. Tracking points: {len(common_img)}")

    for n_frame, corners in list(enumerate(corner_storage))[::-1]:
        common_obj, common_img = get_common_points(corners, points_ids, points_3d)

        mat, inliers, reprojection_error = build_view_mat(common_obj, common_img, intrinsic_mat)
        view_mats[n_frame] = mat

        points_3d = list(np.array(points_3d)[inliers])
        points_ids = list(np.array(points_ids)[inliers])
        #res_points_3d[points_ids] = points_3d

        points_3d, points_ids = calc_new_points(
            n_frame,
            intrinsic_mat,
            corner_storage,
            view_mats,
            points_3d,
            points_ids
        )

        print(f"Processing frame #{n_frame}. Number of inliers: {0 if inliers is None else len(inliers)}. "
              f"Reprojection error: {reprojection_error}. Tracking points: {len(common_img)}")

    # Approximating
    n_iter = 3
    for iter in range(n_iter):
        for n_frame, corners in enumerate(corner_storage):
            common_obj, common_img = get_common_points(corners, points_ids, points_3d)

            mat, inliers, reprojection_error = build_view_mat(common_obj, common_img, intrinsic_mat)
            view_mats[n_frame] = mat

            points_3d = list(np.array(points_3d)[inliers])
            points_ids = list(np.array(points_ids)[inliers])

            if iter == n_iter - 1:
                res_points_3d[points_ids] = points_3d

            points_3d, points_ids = calc_new_points(
                n_frame,
                intrinsic_mat,
                corner_storage,
                view_mats,
                points_3d,
                points_ids
            )

            print(f"Processing frame #{n_frame}. Number of inliers: {0 if inliers is None else len(inliers)}. "
                  f"Reprojection error: {reprojection_error}. Tracking points: {len(common_img)}")

        for n_frame, corners in list(enumerate(corner_storage))[::-1]:
            common_obj, common_img = get_common_points(corners, points_ids, points_3d)

            mat, inliers, reprojection_error = build_view_mat(common_obj, common_img, intrinsic_mat)
            view_mats[n_frame] = mat

            points_3d = list(np.array(points_3d)[inliers])
            points_ids = list(np.array(points_ids)[inliers])
            if iter == n_iter - 1:
                res_points_3d[points_ids] = points_3d

            points_3d, points_ids = calc_new_points(
                n_frame,
                intrinsic_mat,
                corner_storage,
                view_mats,
                points_3d,
                points_ids
            )

            print(f"Processing frame #{n_frame}. Number of inliers: {0 if inliers is None else len(inliers)}. "
                  f"Reprojection error: {reprojection_error}. Tracking points: {len(common_img)}")


    res_points_ids = np.array([i for i, x in enumerate(res_points_3d) if x[0] is not None])
    res_points_3d = np.array(res_points_3d[res_points_ids], dtype=float)

    point_cloud_builder = PointCloudBuilder(ids=res_points_ids, points=res_points_3d)

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
