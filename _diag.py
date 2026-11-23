import cv2, numpy as np, zxingcpp

img = cv2.imread('data/test/image4.png', cv2.IMREAD_GRAYSCALE)
crop = img[476:600, 310:420]

th = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY_INV, 31, 15)
# 找 L 型实心边框: 用 Hough 线
edges = cv2.Canny(crop, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=5)
print('lines:', 0 if lines is None else len(lines))
angles = []
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = np.degrees(np.arctan2(y2-y1, x2-x1))
        angles.append(ang)
        print('  line (%d,%d)-(%d,%d) ang=%.1f len=%.1f' % (x1,y1,x2,y2, ang, np.hypot(x2-x1,y2-y1)))
if angles:
    print('median angle:', np.median(angles))