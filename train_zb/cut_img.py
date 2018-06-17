# -*- coding: utf-8 -*-
# @Author: Yujie Pan
# @Date:   2018-06-05 19:39:35
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-17 21:35:24

import cv2
import numpy as np
from mmcq import MMCQ

# 可调参数：threshold， line 146 的function里面的大量数字


def cut_img(img, width=25, height=25, xstep=25, ystep=25, threshold=0.22, write_mode=False, output_folder=None):
    """
    Inputs
    ------
    width, height:      Size of the output.
    xstep, ystep:       Step length when looking for desired mini-images. 
                        'x' means moving rightwards, and 'y' means downwards. 
    threshold = 0.5:    If roi_area / total_area > threshold, then this cut is qualified. 
    write_mode:         If true, the result images will write into output_folder.
    output_folder:      If writing the results on disk is required. 

    Outputs
    -------
    cut_pics:           List of result mini-images.
    cut_pos:            List of cut positions (x,y).
    chosen_demo:        A picture showing what positions are accepted in results.
    """

    if img is None:
        return None, None, None
    chosen_demo = img.copy()

    mask = roi_mask(img)

    cnt = 0
    cut_pics = []
    cut_pos = []
    for y in range(0, img.shape[0] - height + 1, ystep):
        for x in range(0, img.shape[1] - width + 1, xstep):
            cut_mask = mask[y:y + height, x:x + width]
            if not is_qualified(cut_mask, threshold):
                continue
            try:
                cut_rect = img[y:y + height, x:x + width]
                if write_mode:
                    success = cv2.imwrite(os.path.join(
                        output_folder, str(cnt) + ".jpg"), cut_rect)
                    if not success:
                        continue
                # if cnt>1000 : break
                cut_pics.append(cut_rect)
                cut_pos.append((x, y))
                cv2.rectangle(chosen_demo, (x + 1, y + 1),
                              (x + width - 1, y + height - 1), (0, 255, 0), 1)
                cnt += 1
            except Exception as e:
                print("ERROR: at (x=%d, y=%d\n\t)".format(x, y))
                raise e
    return cut_pics, cut_pos, chosen_demo

def roi_mask(bgr):
    bgcolor_bgr = get_background(bgr)  # background color
    pmcolor_bgrs = get_primary_color(cv2.resize(
        bgr, (150, 150)), num=2, exclude_color=bgcolor_bgr)  # primary color
    mask = np.zeros(bgr.shape[:2]).astype(np.uint8)
    # print(mask.shape)
    for c in pmcolor_bgrs: 
        mask = cv2.bitwise_or(get_similar_color_mask(bgr, c), mask)
    return mask


def get_background(bgr):
    """
    Select 8 small sub-images (2 at each corner) of 50 * 1 size 
    to determine the background color (in BGR).
    """
    h, w = bgr.shape[0], bgr.shape[1]
    sel = np.zeros([0, 50, 3], dtype=np.uint8)
    for y in [1, h - 2]:
        for x in [1, w - 51]:
            sel = np.append(sel, bgr[y:y + 1, x:x + 50], axis=0)
    for y in [51, h - 1]:
        for x in [1, w - 2]:
            sel_line = cv2.transpose(
                bgr[y - 50:y, x:x + 1], cv2.ROTATE_90_CLOCKWISE)
            sel = np.append(sel, sel_line, axis=0)
    return get_primary_color(sel).astype(np.uint8)


def get_primary_color(bgr, exclude_color=None, num=1, bias=3000):
    img = bgr.copy()
    if exclude_color is not None:
        mask = cv2.inRange(img, exclude_color, np.array([255, 255, 255], np.uint8))
        mask = np.bitwise_not(mask)
        img = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imwrite("PPPPP.jpg", img)
    
    mmcq = MMCQ(img, 4)
    theme = mmcq.quantize()

    if exclude_color is None or num == 1:
        return np.array(theme[0]).astype(np.uint8)  # b-g-r
    show = np.zeros([200, 0, 3], np.uint8)
    for i in range(4):
        out = np.zeros([200, 200, 3]) + theme[i]
        show = np.append(show, out, axis=1)
    # cv2.imshow(str(bgr), show.astype(np.uint8));cv2.waitKey(0)
#    if np.sum(np.square(theme[0] - exclude_color)) > bias:
#       return np.array(theme[0]).astype(np.uint8)  # b-g-r

    res = []
    for i in theme[:-1]:
        if np.sum(i) < 30: continue
        res.append(np.array(i).astype(np.uint8))
    return res


def get_similar_color_mask(bgr, pc_bgr):
    # print(pc_bgr)
    pc_hsv = cv2.cvtColor(np.array([[pc_bgr]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # print(pc_hsv)
    hue = pc_hsv[0]
    if hue < 20:
        low1 = np.array([0, 13, 13], np.uint8)
        high1 = np.array([hue + 10, 255, 255], np.uint8)
        low2 = np.array([180 - (20 - hue), 13, 13], np.uint8)
        high2 = np.array([180, 255, 255], np.uint8)
        return cv2.inRange(hsv, low1, high1) + cv2.inRange(hsv, low2, high2)
    if hue > 160:
        low1 = np.array([hue - 20, 13, 13], np.uint8)
        high1 = np.array([180, 255, 255], np.uint8)
        low2 = np.array([0, 13, 13], np.uint8)
        high2 = np.array([20 - (180 - hue), 255, 255], np.uint8)
        return cv2.inRange(hsv, low1, high1) + cv2.inRange(hsv, low2, high2)
    low = np.array([hue - 20, 13, 13], np.uint8)
    high = np.array([hue + 20, 255, 255], np.uint8)
    return cv2.inRange(hsv, low, high)


def is_qualified(mask, threshold):
    return np.sum(mask) / 255 >= threshold * mask.shape[0] * mask.shape[1]


def show_color(bgr):
    out = np.zeros([200, 200, 3]) + bgr
    cv2.imshow(str(bgr), out.astype(np.uint8))
    cv2.waitKey(0)


if __name__ == '__main__':
    pic = ["20703_A_9_4.jpg", "23427_A_9_4.jpg", "24277_A_7_4.jpg", "27588_A_7_4.jpg", "30973_A_8_4.jpg", "7289_A_8_4.jpg", "76189_A_7_4.jpg", "33125_A_7_4.jpg", "37065_A_9_4.jpg", "71080_A_7_4.jpg", "120267_A_7_4.jpg"]

    for i in pic:
        pics, positions, chosen_demo = cut_img(i)
        cv2.imwrite(i + ".res.jpg", chosen_demo)
        print(i, "OK", flush=True)
