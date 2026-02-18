# loading dependencies

import cv2
import numpy as np
import matplotlib.pyplot as plt

# function to crop bboxes

def get_crop_coords(img):
    img_display = img.copy()

    drawing = False
    ix, iy = -1, -1
    mx, my = -1, -1
    rect_out = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img_display, mx, my, rect_out

        mx, my = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_display = img.copy()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0,255,0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            img_display = img.copy()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0,255,0), 2)
            rect_out = (ix, iy, x, y)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        disp = img_display.copy()

        if mx >= 0 and my >= 0:
            cv2.circle(disp, (mx, my), 3, (0,255,0), -1)
        
        cv2.putText(
        disp,
        "SELECT YOUR MAMMAL !!! - PRESS ESC AFTER CHOOSING BOX",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2,
        cv2.LINE_AA
        )

        cv2.imshow("Image", disp)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

    return rect_out

def get_mask(img, bbox):
    x1, y1, x2, y2 = bbox

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Invalid bounding box or zero crop area.")

    mask = np.zeros(crop.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, crop.shape[1] - 2, crop.shape[0] - 2)

    cv2.grabCut(
        crop,
        mask,
        rect,
        bgModel,
        fgModel,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    result_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    return result_mask

def apply_mask_transparent(img, crop_mask, bbox):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]

    alpha = crop_mask.copy()
    alpha[alpha > 0] = 255

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    return rgba