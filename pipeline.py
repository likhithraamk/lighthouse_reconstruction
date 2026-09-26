import os
import re
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# LIGHTHOUSE / SPACE 3D RECONSTRUCTION
# ROBUST INCREMENTAL STRUCTURE FROM MOTION
# + POINT CLOUD
# + MESH
# ============================================================


# ============================================================
# PROJECT SETTINGS
# ============================================================

IMAGE_DIR = "images"

OUTPUT_ROOT = "outputs"

OUT_POINT_CLOUD = os.path.join(
    OUTPUT_ROOT,
    "point_cloud"
)

OUT_MESH = os.path.join(
    OUTPUT_ROOT,
    "mesh"
)

OUT_SCREENSHOTS = os.path.join(
    OUTPUT_ROOT,
    "screenshots"
)

OUT_DEPTH = os.path.join(
    OUTPUT_ROOT,
    "depth_maps"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

# Use all available images.
MAX_IMAGES = 200

# Resize for feature processing.
# 1024 is a good balance between quality and RAM.
RESIZE_WIDTH = 1024


# ============================================================
# CAMERA SETTINGS
# ============================================================

# Original image width supplied for this dataset.
ORIGINAL_WIDTH = 7152

# Estimated / supplied focal length.
ORIGINAL_FOCAL = 9299.0


# ============================================================
# SIFT SETTINGS
# ============================================================

SIFT_FEATURES = 8000

SIFT_CONTRAST = 0.012

SIFT_EDGE = 10

SIFT_SIGMA = 1.6


# ============================================================
# MATCHING
# ============================================================

RATIO_TEST = 0.78

MIN_INITIAL_MATCHES = 50

MIN_PNP_MATCHES = 20

MIN_PNP_INLIERS = 15


# ============================================================
# GEOMETRY
# ============================================================

ESSENTIAL_THRESHOLD = 1.5

PNP_REPROJECTION_ERROR = 3.0

PNP_CONFIDENCE = 0.999

PNP_ITERATIONS = 500

MAX_REPROJECTION_ERROR = 3.0

MIN_TRIANGULATION_ANGLE = 1.0


# ============================================================
# POINT CLOUD FILTERING
# ============================================================

MAX_POINT_DISTANCE_FACTOR = 15.0

STATISTICAL_NEIGHBORS = 30

STATISTICAL_STD = 1.5

VOXEL_FACTOR = 0.001


# ============================================================
# MESH SETTINGS
# ============================================================

POISSON_DEPTH = 9

POISSON_DENSITY_QUANTILE = 0.03

MESH_SAMPLE_POINTS = 200000


# ============================================================
# VISUALIZATION
# ============================================================

MAX_DISPLAY_POINTS = 60000


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUT_POINT_CLOUD,
    exist_ok=True
)

os.makedirs(
    OUT_MESH,
    exist_ok=True
)

os.makedirs(
    OUT_SCREENSHOTS,
    exist_ok=True
)

os.makedirs(
    OUT_DEPTH,
    exist_ok=True
)


# ============================================================
# NATURAL SORT
# ============================================================

def natural_key(text):

    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(
            r"(\d+)",
            text
        )
    ]


# ============================================================
# FIND UNIQUE IMAGES
# ============================================================

def find_images():

    if not os.path.isdir(IMAGE_DIR):

        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    images = []

    for filename in os.listdir(IMAGE_DIR):

        path = os.path.join(
            IMAGE_DIR,
            filename
        )

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in (
            ".jpg",
            ".jpeg"
        ):

            images.append(path)

    images.sort(
        key=lambda x: natural_key(
            os.path.basename(x)
        )
    )

    return images


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        return None

    h, w = image.shape[:2]

    if w > RESIZE_WIDTH:

        scale = (
            RESIZE_WIDTH /
            float(w)
        )

        new_w = RESIZE_WIDTH

        new_h = int(
            h * scale
        )

        image = cv2.resize(
            image,
            (
                new_w,
                new_h
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


# ============================================================
# CAMERA MATRIX
# ============================================================

def create_camera_matrix(image):

    h, w = image.shape[:2]

    focal = (
        ORIGINAL_FOCAL *
        w /
        float(ORIGINAL_WIDTH)
    )

    cx = w / 2.0
    cy = h / 2.0

    K = np.array(
        [
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ],
        dtype=np.float64
    )

    return K


# ============================================================
# SIFT
# ============================================================

def create_sift():

    return cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=SIFT_CONTRAST,
        edgeThreshold=SIFT_EDGE,
        sigma=SIFT_SIGMA
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    image,
    sift
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            None
        )
    )

    return (
        keypoints,
        descriptors
    )


# ============================================================
# FEATURE MATCHING
# ============================================================

def match_features(
    desc1,
    desc2
):

    if desc1 is None:
        return []

    if desc2 is None:
        return []

    if len(desc1) < 2:
        return []

    if len(desc2) < 2:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    raw = matcher.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good = []

    for pair in raw:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < (
            RATIO_TEST * n.distance
        ):

            good.append(m)

    good.sort(
        key=lambda x: x.distance
    )

    return good


# ============================================================
# PROJECTION MATRIX
# ============================================================

def projection_matrix(
    K,
    R,
    t
):

    return K @ np.hstack(
        (
            R,
            t.reshape(3, 1)
        )
    )


# ============================================================
# REPROJECTION
# ============================================================

def reprojection_error(
    points3d,
    points2d,
    K,
    R,
    t
):

    if len(points3d) == 0:
        return np.array([])

    rvec, _ = cv2.Rodrigues(R)

    projected, _ = cv2.projectPoints(
        points3d.astype(np.float64),
        rvec,
        t.astype(np.float64),
        K,
        None
    )

    projected = projected.reshape(
        -1,
        2
    )

    return np.linalg.norm(
        projected - points2d,
        axis=1
    )


# ============================================================
# TRIANGULATION
# ============================================================

def triangulate_points(
    pts1,
    pts2,
    K,
    R1,
    t1,
    R2,
    t2
):

    if len(pts1) < 8:

        return (
            np.empty((0, 3)),
            np.zeros(
                len(pts1),
                dtype=bool
            )
        )

    P1 = projection_matrix(
        K,
        R1,
        t1
    )

    P2 = projection_matrix(
        K,
        R2,
        t2
    )

    points4d = cv2.triangulatePoints(
        P1,
        P2,
        pts1.T.astype(np.float64),
        pts2.T.astype(np.float64)
    )

    w = points4d[3]

    valid_w = (
        np.abs(w) > 1e-10
    )

    output_mask = np.zeros(
        len(pts1),
        dtype=bool
    )

    if not np.any(valid_w):

        return (
            np.empty((0, 3)),
            output_mask
        )

    valid_indices = np.where(
        valid_w
    )[0]

    points4d = points4d[
        :,
        valid_w
    ]

    points3d = (
        points4d[:3] /
        points4d[3]
    ).T

    finite = np.all(
        np.isfinite(points3d),
        axis=1
    )

    points3d = points3d[
        finite
    ]

    valid_indices = valid_indices[
        finite
    ]

    if len(points3d) == 0:

        return (
            np.empty((0, 3)),
            output_mask
        )

    # --------------------------------------------------------
    # POSITIVE DEPTH
    # --------------------------------------------------------

    cam1 = (
        R1 @ points3d.T +
        t1.reshape(3, 1)
    ).T

    cam2 = (
        R2 @ points3d.T +
        t2.reshape(3, 1)
    ).T

    positive = (
        cam1[:, 2] > 0
    ) & (
        cam2[:, 2] > 0
    )

    points3d = points3d[
        positive
    ]

    valid_indices = valid_indices[
        positive
    ]

    if len(points3d) == 0:

        return (
            np.empty((0, 3)),
            output_mask
        )

    # --------------------------------------------------------
    # REPROJECTION ERROR
    # --------------------------------------------------------

    error1 = reprojection_error(
        points3d,
        pts1[valid_indices],
        K,
        R1,
        t1
    )

    error2 = reprojection_error(
        points3d,
        pts2[valid_indices],
        K,
        R2,
        t2
    )

    good = (
        error1 <
        MAX_REPROJECTION_ERROR
    ) & (
        error2 <
        MAX_REPROJECTION_ERROR
    )

    points3d = points3d[
        good
    ]

    valid_indices = valid_indices[
        good
    ]

    output_mask[
        valid_indices
    ] = True

    return (
        points3d,
        output_mask
    )


# ============================================================
# IMAGE COLORS
# ============================================================

def sample_colors(
    image,
    points
):

    h, w = image.shape[:2]

    colors = []

    for p in points:

        x = int(
            round(p[0])
        )

        y = int(
            round(p[1])
        )

        if x < 0:
            x = 0

        if y < 0:
            y = 0

        if x >= w:
            x = w - 1

        if y >= h:
            y = h - 1

        b, g, r = image[y, x]

        colors.append(
            [
                r / 255.0,
                g / 255.0,
                b / 255.0
            ]
        )

    return np.asarray(
        colors,
        dtype=np.float64
    )


# ============================================================
# INITIAL PAIR
# ============================================================

def find_initial_pair(
    images,
    K,
    sift
):

    print()
    print(
        "=" * 70
    )

    print(
        "SEARCHING FOR GOOD INITIAL PAIR"
    )

    print(
        "=" * 70
    )

    # Try several gaps.
    candidate_pairs = []

    maximum = min(
        len(images) - 1,
        50
    )

    for i in range(maximum):

        for gap in (
            1,
            2,
            3,
            4,
            5
        ):

            j = i + gap

            if j < len(images):

                candidate_pairs.append(
                    (i, j)
                )

    best = None

    for i, j in candidate_pairs:

        img1 = load_image(
            images[i]
        )

        img2 = load_image(
            images[j]
        )

        if img1 is None or img2 is None:
            continue

        kp1, des1 = extract_features(
            img1,
            sift
        )

        kp2, des2 = extract_features(
            img2,
            sift
        )

        if des1 is None or des2 is None:
            continue

        matches = match_features(
            des1,
            des2
        )

        if len(matches) < MIN_INITIAL_MATCHES:
            continue

        pts1 = np.float32(
            [
                kp1[m.queryIdx].pt
                for m in matches
            ]
        )

        pts2 = np.float32(
            [
                kp2[m.trainIdx].pt
                for m in matches
            ]
        )

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ESSENTIAL_THRESHOLD
        )

        if E is None:
            continue

        if E.shape[0] > 3:

            E = E[:3, :]

        try:

            _, R, t, pose_mask = (
                cv2.recoverPose(
                    E,
                    pts1,
                    pts2,
                    K
                )
            )

        except Exception:

            continue

        if pose_mask is None:
            continue

        pose_mask = (
            pose_mask.ravel() > 0
        )

        pose_count = int(
            np.sum(pose_mask)
        )

        if pose_count < 35:
            continue

        R1 = np.eye(3)

        t1 = np.zeros(3)

        R2 = R

        t2 = t.reshape(3)

        points3d, tri_mask = (
            triangulate_points(
                pts1,
                pts2,
                K,
                R1,
                t1,
                R2,
                t2
            )
        )

        count = len(points3d)

        print(
            f"Pair {i:03d}->{j:03d} "
            f"matches={len(matches):4d} "
            f"pose={pose_count:4d} "
            f"3D={count:4d}"
        )

        if count < 50:
            continue

        score = count

        if best is None or score > best["score"]:

            best = {
                "i": i,
                "j": j,
                "image1": img1,
                "image2": img2,
                "kp1": kp1,
                "kp2": kp2,
                "des1": des1,
                "des2": des2,
                "matches": matches,
                "tri_mask": tri_mask,
                "points3d": points3d,
                "R1": R1,
                "t1": t1,
                "R2": R2,
                "t2": t2,
                "score": score
            }

    return best


# ============================================================
# PNP FRAME
# ============================================================

def estimate_camera_pose(
    previous_keypoints,
    previous_descriptors,
    previous_point_ids,
    current_keypoints,
    current_descriptors,
    points3d,
    K
):

    matches = match_features(
        previous_descriptors,
        current_descriptors
    )

    if len(matches) < MIN_PNP_MATCHES:
        return None

    object_points = []

    image_points = []

    match_list = []

    used_points = set()

    for m in matches:

        old_feature = m.queryIdx

        if old_feature >= len(
            previous_point_ids
        ):
            continue

        point_id = previous_point_ids[
            old_feature
        ]

        if point_id < 0:
            continue

        if point_id >= len(points3d):
            continue

        if point_id in used_points:
            continue

        used_points.add(
            point_id
        )

        object_points.append(
            points3d[point_id]
        )

        image_points.append(
            current_keypoints[
                m.trainIdx
            ].pt
        )

        match_list.append(m)

    if len(object_points) < MIN_PNP_MATCHES:
        return None

    object_points = np.asarray(
        object_points,
        dtype=np.float64
    )

    image_points = np.asarray(
        image_points,
        dtype=np.float64
    )

    success, rvec, tvec, inliers = (
        cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            None,
            iterationsCount=PNP_ITERATIONS,
            reprojectionError=PNP_REPROJECTION_ERROR,
            confidence=PNP_CONFIDENCE,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    )

    if not success:
        return None

    if inliers is None:
        return None

    inliers = inliers.ravel()

    if len(inliers) < MIN_PNP_INLIERS:
        return None

    # Refine pose.
    try:

        rvec, tvec = cv2.solvePnP(
            object_points[inliers],
            image_points[inliers],
            K,
            None,
            rvec,
            tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    except Exception:
        pass

    try:

        if hasattr(
            cv2,
            "solvePnPRefineLM"
        ):

            rvec, tvec = (
                cv2.solvePnPRefineLM(
                    object_points[inliers],
                    image_points[inliers],
                    K,
                    None,
                    rvec,
                    tvec
                )
            )

    except Exception:
        pass

    R, _ = cv2.Rodrigues(
        rvec
    )

    t = tvec.reshape(3)

    # Verify reprojection.
    errors = reprojection_error(
        object_points[inliers],
        image_points[inliers],
        K,
        R,
        t
    )

    good_inliers = (
        errors <
        MAX_REPROJECTION_ERROR
    )

    if np.sum(good_inliers) < MIN_PNP_INLIERS:
        return None

    # Current feature -> global point.
    current_point_ids = np.full(
        len(current_keypoints),
        -1,
        dtype=np.int64
    )

    for local_index in inliers:

        if not good_inliers[
            list(inliers).index(
                local_index
            )
        ]:
            continue

        m = match_list[
            local_index
        ]

        point_id = previous_point_ids[
            m.queryIdx
        ]

        current_point_ids[
            m.trainIdx
        ] = point_id

    return {
        "R": R,
        "t": t,
        "matches": matches,
        "match_list": match_list,
        "point_ids": current_point_ids,
        "inliers": int(
            np.sum(good_inliers)
        )
    }


# ============================================================
# TRIANGULATE NEW TRACKS
# ============================================================

def create_new_points(
    previous_keypoints,
    current_keypoints,
    matches,
    K,
    R_prev,
    t_prev,
    R_current,
    t_current,
    current_point_ids,
    image,
    points3d,
    colors
):

    if len(matches) < 10:
        return 0

    candidate = []

    for m in matches:

        if m.trainIdx >= len(
            current_point_ids
        ):
            continue

        # Don't recreate an existing point.
        if current_point_ids[
            m.trainIdx
        ] >= 0:
            continue

        candidate.append(m)

    if len(candidate) < 10:
        return 0

    pts1 = np.float32(
        [
            previous_keypoints[
                m.queryIdx
            ].pt
            for m in candidate
        ]
    )

    pts2 = np.float32(
        [
            current_keypoints[
                m.trainIdx
            ].pt
            for m in candidate
        ]
    )

    new_points, valid_mask = (
        triangulate_points(
            pts1,
            pts2,
            K,
            R_prev,
            t_prev,
            R_current,
            t_current
        )
    )

    if len(new_points) == 0:
        return 0

    valid_indices = np.where(
        valid_mask
    )[0]

    valid_2d = pts2[
        valid_indices
    ]

    new_colors = sample_colors(
        image,
        valid_2d
    )

    added = 0

    for i, source_index in enumerate(
        valid_indices
    ):

        m = candidate[
            source_index
        ]

        feature = m.trainIdx

        if current_point_ids[
            feature
        ] >= 0:
            continue

        point_id = len(
            points3d
        )

        points3d.append(
            new_points[i]
        )

        colors.append(
            new_colors[i]
        )

        current_point_ids[
            feature
        ] = point_id

        added += 1

    return added


# ============================================================
# REMOVE GROSS GLOBAL OUTLIERS
# ============================================================

def remove_global_outliers(
    points,
    colors
):

    if len(points) < 100:
        return points, colors

    points = np.asarray(
        points,
        dtype=np.float64
    )

    colors = np.asarray(
        colors,
        dtype=np.float64
    )

    finite = np.all(
        np.isfinite(points),
        axis=1
    )

    points = points[
        finite
    ]

    colors = colors[
        finite
    ]

    # --------------------------------------------------------
    # Robust median distance.
    # --------------------------------------------------------

    center = np.median(
        points,
        axis=0
    )

    distances = np.linalg.norm(
        points - center,
        axis=1
    )

    median_distance = np.median(
        distances
    )

    mad = np.median(
        np.abs(
            distances -
            median_distance
        )
    )

    if mad < 1e-9:

        return points, colors

    threshold = (
        median_distance +
        12.0 * mad
    )

    keep = (
        distances <
        threshold
    )

    points = points[
        keep
    ]

    colors = colors[
        keep
    ]

    return points, colors


# ============================================================
# OPEN3D CLEANING
# ============================================================

def clean_point_cloud(
    points,
    colors
):

    print()
    print(
        "=" * 70
    )

    print(
        "CLEANING GLOBAL POINT CLOUD"
    )

    print(
        "=" * 70
    )

    points, colors = (
        remove_global_outliers(
            points,
            colors
        )
    )

    print(
        f"After global filtering: "
        f"{len(points)}"
    )

    if len(points) < 50:

        return points, colors

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    cloud.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    # --------------------------------------------------------
    # Statistical filtering
    # --------------------------------------------------------

    neighbors = min(
        STATISTICAL_NEIGHBORS,
        len(points) - 1
    )

    if neighbors >= 10:

        cloud, _ = (
            cloud.remove_statistical_outlier(
                nb_neighbors=neighbors,
                std_ratio=STATISTICAL_STD
            )
        )

    print(
        f"After statistical filtering: "
        f"{len(cloud.points)}"
    )

    # --------------------------------------------------------
    # Radius filtering
    # --------------------------------------------------------

    if len(cloud.points) > 100:

        distances = np.asarray(
            cloud.compute_nearest_neighbor_distance()
        )

        if len(distances) > 20:

            median_nn = np.median(
                distances
            )

            radius = (
                median_nn * 4.0
            )

            if radius > 0:

                cloud, _ = (
                    cloud.remove_radius_outlier(
                        nb_points=4,
                        radius=radius
                    )
                )

    print(
        f"After radius filtering: "
        f"{len(cloud.points)}"
    )

    # --------------------------------------------------------
    # Voxel downsample
    # --------------------------------------------------------

    if len(cloud.points) > 100:

        bbox = (
            cloud.get_axis_aligned_bounding_box()
        )

        extent = np.asarray(
            bbox.get_extent()
        )

        diagonal = np.linalg.norm(
            extent
        )

        if diagonal > 0:

            voxel = (
                diagonal *
                VOXEL_FACTOR
            )

            cloud = (
                cloud.voxel_down_sample(
                    voxel
                )
            )

    print(
        f"Final clean points: "
        f"{len(cloud.points)}"
    )

    return (
        np.asarray(
            cloud.points
        ),
        np.asarray(
            cloud.colors
        )
    )


# ============================================================
# SAVE POINT CLOUD
# ============================================================

def save_point_cloud(
    points,
    colors
):

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    cloud.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    path = os.path.join(
        OUT_POINT_CLOUD,
        "lighthouse_clean.ply"
    )

    o3d.io.write_point_cloud(
        path,
        cloud
    )

    print()
    print(
        "[✓] CLEAN POINT CLOUD:"
    )

    print(path)

    return cloud


# ============================================================
# CREATE MESH
# ============================================================

def create_mesh(
    cloud
):

    print()
    print(
        "=" * 70
    )

    print(
        "CREATING 3D MESH"
    )

    print(
        "=" * 70
    )

    if len(cloud.points) < 100:

        print(
            "[!] Not enough points for mesh."
        )

        return None

    # --------------------------------------------------------
    # Estimate normals
    # --------------------------------------------------------

    print(
        "Estimating normals..."
    )

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=30
        )
    )

    cloud.orient_normals_consistent_tangent_plane(
        20
    )

    # --------------------------------------------------------
    # Poisson reconstruction
    # --------------------------------------------------------

    print(
        "Running Poisson reconstruction..."
    )

    try:

        mesh, densities = (
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud,
                depth=POISSON_DEPTH,
                width=0,
                scale=1.1,
                linear_fit=True
            )
        )

    except Exception as e:

        print(
            "Poisson reconstruction failed:"
        )

        print(e)

        return None

    densities = np.asarray(
        densities
    )

    if len(densities) > 0:

        threshold = np.quantile(
            densities,
            POISSON_DENSITY_QUANTILE
        )

        keep = (
            densities >
            threshold
        )

        mesh.remove_vertices_by_mask(
            ~keep
        )

    # --------------------------------------------------------
    # Crop mesh to point cloud bounds
    # --------------------------------------------------------

    bbox = (
        cloud.get_axis_aligned_bounding_box()
    )

    mesh = mesh.crop(
        bbox
    )

    # --------------------------------------------------------
    # Remove tiny / isolated geometry
    # --------------------------------------------------------

    try:

        triangle_clusters, cluster_n_triangles, _ = (
            mesh.cluster_connected_triangles()
        )

        triangle_clusters = np.asarray(
            triangle_clusters
        )

        cluster_n_triangles = np.asarray(
            cluster_n_triangles
        )

        if len(cluster_n_triangles) > 0:

            largest_cluster = np.argmax(
                cluster_n_triangles
            )

            remove_mask = (
                triangle_clusters !=
                largest_cluster
            )

            mesh.remove_triangles_by_mask(
                remove_mask
            )

            mesh.remove_unreferenced_vertices()

    except Exception:
        pass

    # --------------------------------------------------------
    # Save mesh
    # --------------------------------------------------------

    mesh_path = os.path.join(
        OUT_MESH,
        "lighthouse_mesh.ply"
    )

    o3d.io.write_triangle_mesh(
        mesh_path,
        mesh
    )

    print()
    print(
        "[✓] 3D MESH:"
    )

    print(mesh_path)

    print(
        f"Vertices: "
        f"{len(mesh.vertices)}"
    )

    print(
        f"Triangles: "
        f"{len(mesh.triangles)}"
    )

    return mesh


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    points,
    colors
):

    if len(points) == 0:
        return

    if len(points) > MAX_DISPLAY_POINTS:

        rng = np.random.default_rng(
            42
        )

        ids = rng.choice(
            len(points),
            MAX_DISPLAY_POINTS,
            replace=False
        )

        points = points[
            ids
        ]

        colors = colors[
            ids
        ]

    # Robust display bounds.
    low = np.percentile(
        points,
        2,
        axis=0
    )

    high = np.percentile(
        points,
        98,
        axis=0
    )

    center = (
        low + high
    ) / 2.0

    radius = (
        np.max(
            high - low
        ) / 2.0
    )

    if radius <= 0:
        radius = 1.0

    fig = plt.figure(
        figsize=(16, 12)
    )

    views = [
        ("Front", 20, -70),
        ("Left", 20, 20),
        ("Top", 75, -70),
        ("Back", 20, 110)
    ]

    for i, (
        name,
        elevation,
        azimuth
    ) in enumerate(views):

        ax = fig.add_subplot(
            2,
            2,
            i + 1,
            projection="3d"
        )

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=0.8,
            alpha=0.75
        )

        ax.set_title(
            name
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.view_init(
            elev=elevation,
            azim=azimuth
        )

        ax.set_xlim(
            center[0] - radius,
            center[0] + radius
        )

        ax.set_ylim(
            center[1] - radius,
            center[1] + radius
        )

        ax.set_zlim(
            center[2] - radius,
            center[2] + radius
        )

    fig.suptitle(
        "Lighthouse Complete 3D Reconstruction",
        fontsize=18
    )

    plt.tight_layout()

    path = os.path.join(
        OUT_SCREENSHOTS,
        "complete_3d_reconstruction.png"
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "[✓] Screenshot saved:"
    )

    print(path)


# ============================================================
# SAVE CAMERA POSES
# ============================================================

def save_camera_poses(
    poses
):

    path = os.path.join(
        OUT_POINT_CLOUD,
        "camera_poses.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for index in sorted(
            poses.keys()
        ):

            R = poses[index]["R"]

            t = poses[index]["t"]

            f.write(
                f"\nIMAGE {index}\n"
            )

            f.write(
                "R:\n"
            )

            for row in R:

                f.write(
                    " ".join(
                        f"{x:.8f}"
                        for x in row
                    )
                )

                f.write("\n")

            f.write(
                "t: "
                +
                " ".join(
                    f"{x:.8f}"
                    for x in t
                )
                +
                "\n"
            )

    print(
        "[✓] Camera poses saved:"
    )

    print(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "LIGHTHOUSE COMPLETE 3D RECONSTRUCTION"
    )
    print(
        "ROBUST INCREMENTAL SfM + MESH"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    images = find_images()

    print()
    print(
        f"[✓] Found {len(images)} unique images"
    )

    if len(images) < 2:

        print(
            "ERROR: Need at least 2 images."
        )

        return

    # --------------------------------------------------------
    # SELECT IMAGES
    # --------------------------------------------------------

    if len(images) > MAX_IMAGES:

        indices = np.linspace(
            0,
            len(images) - 1,
            MAX_IMAGES,
            dtype=int
        )

        images = [
            images[i]
            for i in indices
        ]

    print(
        f"[✓] Using {len(images)} images"
    )

    # --------------------------------------------------------
    # LOAD FIRST IMAGE
    # --------------------------------------------------------

    first = load_image(
        images[0]
    )

    if first is None:

        print(
            "ERROR: Cannot read first image."
        )

        return

    # --------------------------------------------------------
    # CAMERA MATRIX
    # --------------------------------------------------------

    K = create_camera_matrix(
        first
    )

    print()
    print(
        "Camera matrix:"
    )

    print(K)

    # --------------------------------------------------------
    # SIFT
    # --------------------------------------------------------

    print()
    print(
        "Initializing SIFT..."
    )

    sift = create_sift()

    # --------------------------------------------------------
    # INITIAL PAIR
    # --------------------------------------------------------

    initial = find_initial_pair(
        images,
        K,
        sift
    )

    if initial is None:

        print()
        print(
            "❌ Could not find a suitable initial pair."
        )

        return

    print()
    print(
        "=" * 70
    )

    print(
        "INITIALIZATION SUCCESSFUL"
    )

    print(
        "=" * 70
    )

    print(
        f"Initial pair: "
        f"{initial['i']} -> {initial['j']}"
    )

    print(
        f"Initial points: "
        f"{len(initial['points3d'])}"
    )

    # --------------------------------------------------------
    # GLOBAL POINTS
    # --------------------------------------------------------

    points3d = [
        x.copy()
        for x in initial["points3d"]
    ]

    # --------------------------------------------------------
    # INITIAL COLORS
    # --------------------------------------------------------

    valid_indices = np.where(
        initial["tri_mask"]
    )[0]

    valid_matches = [
        initial["matches"][i]
        for i in valid_indices
    ]

    color_2d = np.float32(
        [
            initial["kp2"][
                m.trainIdx
            ].pt
            for m in valid_matches
        ]
    )

    colors = list(
        sample_colors(
            initial["image2"],
            color_2d
        )
    )

    # Ensure equal size.
    count = min(
        len(points3d),
        len(colors)
    )

    points3d = points3d[
        :count
    ]

    colors = colors[
        :count
    ]

    # --------------------------------------------------------
    # FEATURE -> POINT ID
    # --------------------------------------------------------

    point_ids = np.full(
        len(initial["kp2"]),
        -1,
        dtype=np.int64
    )

    for point_id, m in enumerate(
        valid_matches[:count]
    ):

        point_ids[
            m.trainIdx
        ] = point_id

    # --------------------------------------------------------
    # CAMERA POSES
    # --------------------------------------------------------

    poses = {}

    poses[
        initial["i"]
    ] = {
        "R": initial["R1"],
        "t": initial["t1"]
    }

    poses[
        initial["j"]
    ] = {
        "R": initial["R2"],
        "t": initial["t2"]
    }

    # --------------------------------------------------------
    # CURRENT FRAME
    # --------------------------------------------------------

    previous_index = initial["j"]

    previous_image = initial["image2"]

    previous_keypoints = initial["kp2"]

    previous_descriptors = initial["des2"]

    successful = 2

    skipped = 0

    # --------------------------------------------------------
    # PROCESS REMAINING IMAGES
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "BUILDING GLOBAL 3D SPACE"
    )

    print(
        "=" * 70
    )

    for current_index in tqdm(
        range(
            previous_index + 1,
            len(images)
        ),
        desc="3D reconstruction"
    ):

        current_image = load_image(
            images[current_index]
        )

        if current_image is None:

            skipped += 1

            continue

        current_keypoints, current_descriptors = (
            extract_features(
                current_image,
                sift
            )
        )

        if current_descriptors is None:

            skipped += 1

            continue

        # ----------------------------------------------------
        # ESTIMATE CAMERA
        # ----------------------------------------------------

        result = estimate_camera_pose(
            previous_keypoints,
            previous_descriptors,
            point_ids,
            current_keypoints,
            current_descriptors,
            points3d,
            K
        )

        if result is None:

            skipped += 1

            continue

        R_current = result["R"]

        t_current = result["t"]

        current_point_ids = result[
            "point_ids"
        ]

        # ----------------------------------------------------
        # NEW TRIANGULATION
        # ----------------------------------------------------

        # Use feature matches between previous
        # and current frames.
        all_matches = result[
            "matches"
        ]

        new_added = create_new_points(
            previous_keypoints,
            current_keypoints,
            all_matches,
            K,
            poses[previous_index]["R"],
            poses[previous_index]["t"],
            R_current,
            t_current,
            current_point_ids,
            current_image,
            points3d,
            colors
        )

        # ----------------------------------------------------
        # SAVE CAMERA
        # ----------------------------------------------------

        poses[
            current_index
        ] = {
            "R": R_current,
            "t": t_current
        }

        # ----------------------------------------------------
        # PRINT STATUS
        # ----------------------------------------------------

        print(
            f"\nFrame {current_index:03d} | "
            f"PnP inliers={result['inliers']} | "
            f"new points={new_added} | "
            f"total points={len(points3d)}"
        )

        # ----------------------------------------------------
        # NEXT FRAME
        # ----------------------------------------------------

        previous_index = current_index

        previous_image = current_image

        previous_keypoints = current_keypoints

        previous_descriptors = current_descriptors

        point_ids = current_point_ids

        successful += 1

    # ========================================================
    # FINISHED SfM
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "GLOBAL SfM FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        f"Images:          {len(images)}"
    )

    print(
        f"Successful:      {successful}"
    )

    print(
        f"Skipped:         {skipped}"
    )

    print(
        f"Raw 3D points:   {len(points3d)}"
    )

    if len(points3d) < 100:

        print()
        print(
            "❌ Not enough reliable 3D points."
        )

        return

    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

    points_np = np.asarray(
        points3d,
        dtype=np.float64
    )

    colors_np = np.asarray(
        colors,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    points_np, colors_np = (
        clean_point_cloud(
            points_np,
            colors_np
        )
    )

    if len(points_np) < 100:

        print(
            "❌ Too few points after filtering."
        )

        return

    # ========================================================
    # SAVE POINT CLOUD
    # ========================================================

    cloud = save_point_cloud(
        points_np,
        colors_np
    )

    # ========================================================
    # SAVE CAMERA POSES
    # ========================================================

    save_camera_poses(
        poses
    )

    # ========================================================
    # CREATE MESH
    # ========================================================

    mesh = create_mesh(
        cloud
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    create_visualization(
        points_np,
        colors_np
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)

    print(
        "✅ COMPLETE 3D RECONSTRUCTION FINISHED"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "POINT CLOUD:"
    )

    print(
        os.path.join(
            OUT_POINT_CLOUD,
            "lighthouse_clean.ply"
        )
    )

    print()
    print(
        "MESH:"
    )

    print(
        os.path.join(
            OUT_MESH,
            "lighthouse_mesh.ply"
        )
    )

    print()
    print(
        "CAMERA POSES:"
    )

    print(
        os.path.join(
            OUT_POINT_CLOUD,
            "camera_poses.txt"
        )
    )

    print()
    print(
        "3D PREVIEW:"
    )

    print(
        os.path.join(
            OUT_SCREENSHOTS,
            "complete_3d_reconstruction.png"
        )
    )

    print()
    print(
        f"FINAL POINTS: {len(points_np)}"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()