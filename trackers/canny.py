def bounding_boxes(frame):
    edged = cv2.Canny(frame, 1, 250)
    # cv2.imshow('new', edged)
    # cv2.waitKey(0)
    # find contours in the edge map
    img, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    objects = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # ensure that the approximated contour is "roughly" rectangular
        if len(approx) < 4:
            continue
        # compute the bounding box of the approximated contour and
        # use the bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        # aspectRatio = w / float(h)

        # compute the solidity of the original contour
        area = cv2.contourArea(c)
        hullArea = max(0.1, cv2.contourArea(cv2.convexHull(c)))
        solidity = area / float(hullArea)

        # compute whether or not the width and height, solidity, and
        # aspect ratio of the contour falls within appropriate bounds
        keepDims = w > 25 and h > 25
        keepSolidity = solidity > 0.3
        # keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

        # ensure that the contour passes all our tests
        if keepDims and keepSolidity:
            # draw an outline around the target and update the status
            # text
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
            # compute the center of the contour region and draw the
            # crosshairs
            M = cv2.moments(approx)
            (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
            (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
            cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
            cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 1)
            objects.append(Trackable((x, y, w, h), np.array([cX, cY])))
    # cv2.imshow('Image', frame)
    # cv2.waitKey(0)
    return objects
