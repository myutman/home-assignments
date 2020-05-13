#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp
from tqdm import tqdm

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
    TriangulationParameters,
    build_correspondences,
    _remove_correspondences_with_ids,
    compute_reprojection_errors, eye3x4)
from _corners import FrameCorners
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose

RANSAC_REPROJECTION_ERROR=10
TRIANGULATION_REPROJECTION_ERROR=10
MIN_TRISNGULATION_ANGLE=1.5
INITIAL_TRIANGULATION_PARAMETERS = TriangulationParameters(
    max_reprojection_error=TRIANGULATION_REPROJECTION_ERROR,
    min_triangulation_angle_deg=1,
    min_depth=0
)
NEW_TRIANGULATION_PARAMETERS = {
    1: TriangulationParameters(
        max_reprojection_error=TRIANGULATION_REPROJECTION_ERROR,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=0
    ),
    2: TriangulationParameters(
        max_reprojection_error=TRIANGULATION_REPROJECTION_ERROR,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=0
    ),
    4: TriangulationParameters(
        max_reprojection_error=5,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=0
    ),
    8: TriangulationParameters(
        max_reprojection_error=TRIANGULATION_REPROJECTION_ERROR,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=0
    ),
    16: TriangulationParameters(
        max_reprojection_error=TRIANGULATION_REPROJECTION_ERROR,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=0
    ),
    32: TriangulationParameters(
        max_reprojection_error=TRIANGULATION_REPROJECTION_ERROR,
        min_triangulation_angle_deg=MIN_TRISNGULATION_ANGLE,
        min_depth=0
    )
}

def calc_starting_points(
    intrinsic_mat: np.ndarray,
    corner_storage: CornerStorage,
    known_view_1: Tuple[int, Pose],
    known_view_2: Tuple[int, Pose]
) -> Tuple[np.ndarray, np.ndarray]:
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
    return points_3d, points_ids

def get_common_points(
    corners: FrameCorners,
    common_ids: np.ndarray,
    points_3d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points1, ids1 = corners.points, corners.ids

    common_all_ids, (ids_obj, ids_img) = snp.intersect(common_ids.reshape(-1), np.array(ids1).reshape(-1), indices=True)

    #common_obj = []
    #common_img = []
    #common_all_ids = []
    #j = 0
    #for i, cur0 in enumerate(common_ids):
    #    while j < len(ids1) and ids1[j] < cur0:
    #        j += 1
    #    if j == len(ids1):
    #        break
    #    if ids1[j] == cur0:
    #        common_obj.append(points_3d[i])
    #        common_img.append(points1[j])
    #        common_all_ids.append(cur0)

    common_obj = points_3d[ids_obj]
    common_img = points1[ids_img]
    return common_obj, common_img, common_all_ids


def build_view_mat(
    common_obj: np.ndarray,
    common_img: np.ndarray,
    common_ids: np.ndarray,
    intrinsic_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        common_obj,
        common_img,
        intrinsic_mat,
        None,
        #flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=RANSAC_REPROJECTION_ERROR,
        iterationsCount=100
    )
    mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    inliers = np.array(inliers).reshape(-1)
    inlier_points = common_obj[inliers]
    inlier_img = common_img[inliers]
    inlier_ids = common_ids[inliers]
    reprojection_errors = compute_reprojection_errors(inlier_points, inlier_img, intrinsic_mat @ mat)
    #to_take = reprojection_errors < RANSAC_REPROJECTION_ERROR
    #reprojection_error = compute_reprojection_errors(inlier_points, inlier_img, intrinsic_mat @ mat).mean()

    #reprojection_error = np.linalg.norm(common_img - projections, axis=-1).mean()

    #return mat, inlier_points[to_take], inlier_ids[to_take], reprojection_errors[to_take].mean()
    return mat, inlier_points, inlier_ids, reprojection_errors.mean()

def merge_sets(
    points_3d: np.ndarray,
    points_ids: np.ndarray,
    new_points_3d: np.ndarray,
    new_points_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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

    return np.array(res_points_3d), np.array(res_points_ids)

def calc_new_points(
    cur_id: int,
    intrinsic_mat: np.ndarray,
    corner_storage: CornerStorage,
    view_mats: List[np.ndarray],
    points_3d: np.ndarray,
    points_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    for d in [-32, 32, -16, 16, -8, 8, -4, 4, -2, 2 -1, 1]:
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


def pose_by_frames(
    frame1: FrameCorners,
    frame2: FrameCorners,
    intrinsic_mat: np.ndarray
) -> Tuple[Pose, int]:
    correspondences = build_correspondences(frame1, frame2)
    mat, mask = cv2.findEssentialMat(
        correspondences.points_1,
        correspondences.points_2,
        intrinsic_mat,
        cv2.RANSAC,
        0.99,
        1
    )

    if mat is None or mat.shape != (3, 3):
        return None, 0

    if mask is not None:
        correspondences = _remove_correspondences_with_ids(correspondences, np.argwhere(mask.flatten() == 0))

    R1, R2, t = cv2.decomposeEssentialMat(mat)
    max_pose = None
    max_npoints = 0
    for mat in [R1.T, R2.T]:
        for vec in [t, -t]:
            pose = Pose(mat, mat @ vec)
            points, _, _ = triangulate_correspondences(
                correspondences,
                eye3x4(),
                pose_to_view_mat3x4(pose),
                intrinsic_mat,
                INITIAL_TRIANGULATION_PARAMETERS
            )
            if len(points) > max_npoints:
                max_pose = pose
                max_npoints = len(points)
    return max_pose, max_npoints


#--camera-poses ../../dataset/soda_free_motion/ground_truth.yml

def initial_camera_views(
    corner_storage: CornerStorage,
    intrinsic_mat: np.ndarray
) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    max_pose = None
    max_npoints = 0
    maxj = -1
    maxi = -1

    enum_corner_storage = list(enumerate(corner_storage))
    for j, base_frame in tqdm(enum_corner_storage[::32]):
        base_frame = corner_storage[0]
        for i, frame in enum_corner_storage[j+1:j+32]:
            pose, npoints = pose_by_frames(base_frame, frame, intrinsic_mat)
            print(i, j, npoints)
            if npoints > max_npoints:
                max_npoints = npoints
                maxi = i
                maxj = j
                max_pose = pose

    print(max_npoints)
    return (maxj, view_mat3x4_to_pose(eye3x4())), (maxi, max_pose)


def track_and_calc_colors(
    camera_parameters: CameraParameters,
    corner_storage: CornerStorage,
    frame_sequence_path: str,
    known_view_1: Optional[Tuple[int, Pose]] = None,
    known_view_2: Optional[Tuple[int, Pose]] = None
) -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = initial_camera_views(corner_storage, intrinsic_mat)
    print(known_view_1[1])
    print(known_view_2[1])

    print(known_view_1[0], known_view_2[0])


    print(intrinsic_mat)

    points_3d, points_ids = calc_starting_points(intrinsic_mat, corner_storage, known_view_1, known_view_2)

    #initial_points_3d, initial_points_ids = points_3d.copy(), points_ids.copy()

    n_points = corner_storage.max_corner_id() + 1
    res_points_3d = np.full((n_points, 3), None)
    #res_points_3d[points_ids] = points_3d

    view_mats = [np.full((3, 4), None) for _ in range(len(corner_storage))]
    view_mats[known_view_1[0]], view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(known_view_2[1])

    print(f'len(corner_storage):{len(corner_storage)}')

    id1 = known_view_1[0]
    for n_frame, corners in enumerate(corner_storage[id1 + 1:], id1 + 1):
        common_obj, common_img, common_ids = get_common_points(corners, points_ids, points_3d)

        mat, points_3d, points_ids, reprojection_error = build_view_mat(common_obj, common_img, common_ids, intrinsic_mat)
        points_ids = np.array(points_ids)
        view_mats[n_frame] = mat

        is_null = np.array([(x[0] is None) for x in res_points_3d[points_ids]])
        #res_points_3d[points_ids[is_null]] = points_3d[is_null]

        n_inliers = len(points_3d)

        points_3d, points_ids = calc_new_points(
            n_frame,
            intrinsic_mat,
            corner_storage,
            view_mats,
            points_3d,
            points_ids
        )

        print(f"Processing frame #{n_frame}. Number of inliers: {n_inliers}. "
              f"Reprojection error: {reprojection_error}. Tracking points: {len(common_img)}")

    for n_frame, corners in list(enumerate(corner_storage))[::-1]:
        common_obj, common_img, common_ids = get_common_points(corners, points_ids, points_3d)

        mat, points_3d, points_ids, reprojection_error = build_view_mat(common_obj, common_img, common_ids,
                                                                        intrinsic_mat)
        view_mats[n_frame] = mat

        is_null = np.array([(x[0] is None) for x in res_points_3d[points_ids]])
        #res_points_3d[points_ids[is_null]] = points_3d[is_null]

        n_inliers = len(points_3d)

        points_3d, points_ids = calc_new_points(
            n_frame,
            intrinsic_mat,
            corner_storage,
            view_mats,
            points_3d,
            points_ids
        )

        print(f"Processing frame #{n_frame}. Number of inliers: {n_inliers}. "
              f"Reprojection error: {reprojection_error}. Tracking points: {len(common_img)}")

    # Approximating

    n_iter = 1
    for iter in range(n_iter):
        for n_frame, corners in enumerate(corner_storage):
            common_obj, common_img, common_ids = get_common_points(corners, points_ids, points_3d)

            mat, points_3d, points_ids, reprojection_error = build_view_mat(common_obj, common_img, common_ids,
                                                                            intrinsic_mat)

            view_mats[n_frame] = mat
            if iter == n_iter - 1:
                res_points_3d[points_ids] = points_3d

            n_inliers = len(points_3d)

            points_3d, points_ids = calc_new_points(
                n_frame,
                intrinsic_mat,
                corner_storage,
                view_mats,
                points_3d,
                points_ids
            )

            print(f"Processing frame #{n_frame}. Number of inliers: {n_inliers}. "
                  f"Reprojection error: {reprojection_error}. Tracking points: {len(common_img)}")

        for n_frame, corners in list(enumerate(corner_storage))[::-1]:
            common_obj, common_img, common_ids = get_common_points(corners, points_ids, points_3d)

            mat, points_3d, points_ids, reprojection_error = build_view_mat(common_obj, common_img, common_ids,
                                                                            intrinsic_mat)

            view_mats[n_frame] = mat
            if iter == n_iter - 1:
                res_points_3d[points_ids] = points_3d

            n_inliers = len(points_3d)

            points_3d, points_ids = calc_new_points(
                n_frame,
                intrinsic_mat,
                corner_storage,
                view_mats,
                points_3d,
                points_ids
            )

            print(f"Processing frame #{n_frame}. Number of inliers: {n_inliers}. "
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
