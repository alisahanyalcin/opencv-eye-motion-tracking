import cv2

video = "Video.mp4"

cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    roi = frame[100: 10000, 200: 10000]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.circle(roi, (x + int(w / 2), y + int(h / 2)), int((x + y + w + h) / 14), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("Gray Roi", gray_roi)
    cv2.imshow("Roi", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
