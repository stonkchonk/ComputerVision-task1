import cv2
import cv2.aruco as aruco
import numpy as np

input_files = [
    '20221115_113319.jpg',
    '20221115_113328.jpg',
    '20221115_113340.jpg',
    '20221115_113346.jpg',
    '20221115_113356.jpg',
    '20221115_113401.jpg',
    '20221115_113412.jpg',
    '20221115_113424.jpg',
    '20221115_113437.jpg',
    '20221115_113440.jpg',
    '20221115_113653.jpg',
]

for file in input_files:

    img = cv2.imread(f"images/{file}", cv2.IMREAD_COLOR)  # images/20221115_113319.jpg
    print("reading file " + file + "\n")

    dictionary = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

    marker_ids = np.zeros(0)
    marker_corners = np.zeros(0)
    rejected_candidates = np.zeros(0)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    try:
        aruco_detections = detector.detectMarkers(img, marker_corners, marker_ids, rejected_candidates)
    except:
        print(f"No ArUco detected in file: {file}")
        continue

    print(aruco_detections[0][0][0])
    print("---")
    print(aruco_detections[1])
    print("---")

    height, width, _ = img.shape
    print(height, width)
    detected_corners = aruco_detections[0][0][0]
    actual_corners = np.float32([[224, 224], [288, 224], [288, 288], [224, 288]])

    transformable_image = cv2.imread("loadbearingposter.png", cv2.IMREAD_COLOR)
    M = cv2.getPerspectiveTransform(actual_corners, detected_corners)
    print("M:", M)
    warped = cv2.warpPerspective(transformable_image, M, (width, height))

    combined = cv2.addWeighted(img, 1, warped, 1, 0)
    cv2.imwrite(f'output/combined_{file}', combined)



