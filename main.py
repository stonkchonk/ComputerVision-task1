import cv2
import cv2.aruco as aruco
import numpy as np
from os import listdir


def load_image(path_to_file: str) -> cv2.typing.MatLike:
    return cv2.imread(path_to_file, cv2.IMREAD_COLOR)


def add_two_images(first: cv2.typing.MatLike, second: cv2.typing.MatLike):
    assert first.shape == second.shape
    original_shape = first.shape
    first_flat = first.flat
    second_flat = second.flat
    for idx, val in enumerate(second_flat):
        if val > 0:
            first_flat[idx] = val
    return np.reshape(first_flat, original_shape)


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

    M = cv2.getPerspectiveTransform(actual_corners, detected_corners)
    warped = cv2.warpPerspective(poster, M, (width, height))

    combined = cv2.addWeighted(img, 1, warped, 1, 0) #add_two_images(img, warped)

    for corner in detected_corners:
        combined[int(corner[1])][int(corner[0])] = [0, 0, 255]

    cv2.imwrite(f'output/combined_{file}', combined)
