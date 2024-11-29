import cv2
import cv2.aruco as aruco
import numpy as np
from os import listdir
from coordinates import referenceMapping
from math import acos, pi


def load_image(path_to_file: str) -> cv2.typing.MatLike:
    return cv2.imread(path_to_file, cv2.IMREAD_COLOR)


def combine_scenery_with_image(scenery: cv2.typing.MatLike, image: cv2.typing.MatLike):
    assert scenery.shape == image.shape
    gray_poster = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_inverse = cv2.threshold(gray_poster, 1, 255, cv2.THRESH_BINARY_INV)
    scenery = cv2.bitwise_and(scenery, scenery, mask=mask_inverse)
    return cv2.addWeighted(scenery, 1, image, 1, 0)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray):
    return acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / pi




debug_mode = False
eval_lines = True
input_files = listdir("images/")
poster = load_image("loadbearingposter.png")
aruco_diameter = 64
poster_height, poster_width, _ = poster.shape

for file in input_files:

    img = load_image(f"images/{file}")
    print("reading file " + file)

    dictionary = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

    marker_ids = np.zeros(0)
    marker_corners = np.zeros(0)
    rejected_candidates = np.zeros(0)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    aruco_detections = detector.detectMarkers(img, marker_corners, marker_ids, rejected_candidates)
    if aruco_detections[1] is None:
        print(f"No ArUco detected in file: {file}")
        continue

    height, width, _ = img.shape
    detected_corners = aruco_detections[0][0][0]

    actual_corners = np.float32([
        [poster_height / 2 - aruco_diameter / 2, poster_width / 2 - aruco_diameter / 2],
        [poster_height / 2 + aruco_diameter / 2, poster_width / 2 - aruco_diameter / 2],
        [poster_height / 2 + aruco_diameter / 2, poster_width / 2 + aruco_diameter / 2],
        [poster_height / 2 - aruco_diameter / 2, poster_width / 2 + aruco_diameter/2]
    ])

    transformation_matrix = cv2.getPerspectiveTransform(actual_corners, detected_corners)
    transformation_matrix_inv = cv2.getPerspectiveTransform(detected_corners, actual_corners)
    identity = np.matmul(transformation_matrix, transformation_matrix_inv)

    warped_poster = cv2.warpPerspective(poster, transformation_matrix, (width, height))

    if debug_mode:
        combined = cv2.addWeighted(img, 0.5, warped_poster, 0.5, 0)
    else:
        combined = combine_scenery_with_image(img, warped_poster)

    for corner in detected_corners:
        combined[int(corner[1])][int(corner[0])] = [0, 0, 255]

    coordinate_element = referenceMapping.get(file, None)

    if coordinate_element is not None and eval_lines:
        combined = cv2.line(combined, coordinate_element.h1, coordinate_element.h2, (0, 255, 0), 8)
        combined = cv2.line(combined, coordinate_element.v1, coordinate_element.v2, (128, 0, 128), 8)

        retransformed_h1 = np.matmul(transformation_matrix_inv, np.append(np.array(coordinate_element.h1), 1))
        retransformed_h2 = np.matmul(transformation_matrix_inv, np.append(np.array(coordinate_element.h2), 1))
        horizontal_eval_vector = retransformed_h2 - retransformed_h1

        retransformed_v1 = np.matmul(transformation_matrix_inv, np.append(np.array(coordinate_element.v1), 1))
        retransformed_v2 = np.matmul(transformation_matrix_inv, np.append(np.array(coordinate_element.v2), 1))
        vertical_eval_vector = retransformed_v2 - retransformed_v1

        horizontal_vector = np.array((1, 0, 0))
        vertical_vector = np.array((0, 1, 0))

        horizontal_angle = angle_between_vectors(horizontal_eval_vector, horizontal_vector)
        vertical_angle = angle_between_vectors(vertical_eval_vector, vertical_vector)
        print(f'$\\alpha$={round(horizontal_angle, 2)}°, $\\beta$={round(vertical_angle, 2)}°')

    cv2.imwrite(f'output/combined_{file}', cv2.resize(combined, (0, 0), fx=0.25, fy=0.25))

