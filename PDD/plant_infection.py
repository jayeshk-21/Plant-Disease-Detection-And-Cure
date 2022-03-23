import cv2
import numpy as np
import argparse


def infec_region(img_path):
    img1 = cv2.imread(img_path)
    # '.\\0d2e2971-f1c9-4278-b35c-91dd8a22a64d___RS_Early.B_7581.JPG')
    img = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    original = img.copy()
    neworiginal = img.copy()

    blur1 = cv2.GaussianBlur(img, (3, 3), 1)

    newimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)

    blur = cv2.GaussianBlur(img, (11, 11), 1)

    kernel = np.ones((5, 5), np.uint8)
    canny = cv2.Canny(blur, 200, 290)
    res = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('Canny', res)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 25, 25])
    upper = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in contours:
        cnt = cv2.contourArea(i)
        #M = cv2.momens(i)
        #cx = int(M['m10']/M['m00'])
        if cnt > 1000:
            cv2.drawContours(img, [i], 0, (0, 0, 255), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    Tarea = cv2.contourArea(cnt)
    # cv2.imshow('img', img)

    height, width, _ = canny.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    frame = canny.copy()

    for contour, hier in zip(contours, hierarchy):

        (x, y, w, h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w >= 0 and h >= 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi = img[y:y+h, x:x+w]
            originalroi = original[y:y+h, x:x+w]
    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    img = roi

    imghls = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    imghls[np.where((imghls == [30, 200, 2]).all(axis=2))] = [0, 200, 0]

    huehls = imghls[:, :, 0]

    huehls[np.where(huehls == [0])] = [35]

    # Thresholding on hue image
    ret, thresh = cv2.threshold(huehls, 28, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('thresh', thresh)

    mask = cv2.bitwise_and(originalroi, originalroi, mask=thresh)

    contours, heirarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    Infarea = 0

    for x in range(len(contours)):

        cv2.drawContours(originalroi, contours[x], -1, (0, 0, 255), 2)

        # cv2.imshow('Contour masked', originalroi)

        # Calculating area of infected region
        Infarea += cv2.contourArea(contours[x])

    # if Infarea > Tarea:
        #Tarea = img.shape[0]*img.shape[1]

    print('______________________\n| Total area: ' +
          str(Tarea) + ' |\n|____________________|')

    # Finding the percentage of infection in the leaf
    print('\n__________________________\n| Infected area: ' +
          str(Infarea) + ' |\n|________________________|')

    try:
        per = 100 * Infarea/Tarea

    except ZeroDivisionError:
        per = 0

    print('\n_________________________________________________\n| Percentage of infected region: ' +
          str(per) + ' |\n|_______________________________________________|')

    # cv2.imshow('orig', original)

    return per


# per = infec_region(
#     '.\\0d2e2971-f1c9-4278-b35c-91dd8a22a64d___RS_Early.B_7581.JPG')
# print(per)
