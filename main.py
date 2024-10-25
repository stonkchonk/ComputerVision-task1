import cv2
import cv2.aruco as aruco
import numpy as np

img = cv2.imread("images/20221115_113328.jpg", 0) #images/20221115_113319.jpg
print(img.ndim)

aruco_marker_image = np.zeros((48, 48), dtype=np.uint8)
dictionary = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
#for i in range(0, 1):
#    cv2.aruco.generateImageMarker(dictionary, i, 48, aruco_marker_image, 1)
#    cv2.imwrite(f'aruco{i}.png', aruco_marker_image)
#    print(i)
marker_ids = np.zeros(0)
marker_corners = np.zeros(0)
rejected_candidates = np.zeros(0)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
res = detector.detectMarkers(img, marker_corners, marker_ids, rejected_candidates)

print(res[0])
print("---")
print(res[1])
print("---")
#print(res[2])
print("ids", marker_ids)
print("corners", marker_corners)
print("rejected", rejected_candidates)

out_img = img.copy()

cv2.aruco.drawDetectedMarkers(out_img, marker_corners, marker_ids)

#cv2.namedWindow("Resize", cv2.WINDOW_NORMAL)
# Using resizeWindow()
#cv2.resizeWindow("Resize", 1000, 1000)
#cv2.imshow("Resize", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
