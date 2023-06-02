"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math,time
import pyclipper
from shapely.geometry import *
from shapely.ops import nearest_points
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
""" auxilary functions """
# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel, iterations=1)
        #kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        #segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel1, iterations=1)


        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

def getDetBoxes_new(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    #ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, text_score_comb = cv2.threshold(linkmap, link_threshold, 1, 0)

    #text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(linkmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        #segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel, iterations=1)
        #kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        #segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel1, iterations=1)


        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)



        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 30 or h < 30:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_new(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys


def getCharBoxes(scores,text_threshold,low_text ,fast):
    #image = np.clip(image, 0, 255)
    #image = np.array(image,np.uint8)
    region_scores = np.uint8(np.clip(scores, 0, 1) * 255)
    #bgr_region_scores = cv2.resize(region_scores, (image.shape[1], image.shape[0]))
    bgr_region_scores = cv2.cvtColor(region_scores, cv2.COLOR_GRAY2BGR)
    #start =time.time()
    pursedo_bboxes = watershed2(bgr_region_scores, text_threshold, low_text, fast,False)
    #print ('time3:',time.time()-start)
    _tmp = []
    for i in range(pursedo_bboxes.shape[0]):
        if np.mean(pursedo_bboxes[i].ravel()) > 2:
            _tmp.append(pursedo_bboxes[i])
        else:
            print("filter bboxes", pursedo_bboxes[i])
    pursedo_bboxes = np.array(_tmp, np.float32)
    if pursedo_bboxes.shape[0] > 1:
        index = np.argsort(pursedo_bboxes[:, 0, 0])
        pursedo_bboxes = pursedo_bboxes[index]
    bboxes = pursedo_bboxes
    #bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
    #bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)
    return bboxes

def getDetCharBoxes_core(textmap, fast,text_threshold=0.5, low_text=0.4):
    # prepare data
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8),
                                                                        connectivity=4)

    det = []
    mapper = []
    #print ('nlabels',nLabels)
    #segmaps = []
    for k in range(1, nLabels):

        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        #start = time.time()
        if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        # segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]

        if fast:
            sx, ex, sy, ey = x, x + w + 1 , y , y + h + 1

            # boundary check
            if sx < 0: sx = 0
            if sy < 0: sy = 0
            if ex >= img_w: ex = img_w
            if ey >= img_h: ey = img_h
            box = np.array([[sx, sy], [ex, sy], [ex, ey], [sx, ey]], dtype=np.float32)




        else:
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)

            #print ('type', type(w),type(niter))
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            if sx < 0: sx = 0
            if sy < 0: sy = 0
            if ex >= img_w: ex = img_w
            if ey >= img_h: ey = img_h

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)
        #print ('time8:', time.time() -start )
        #print ('time3:',time.time()-start)

    return det, labels, mapper


def watershed2(image, text_threshold, low_text,fast,viz=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    boxes, _, _ = getDetCharBoxes_core(gray,fast,text_threshold, low_text)

    return np.array(boxes)

def getMeshBoxes(meshmap, threshold):
    # prepare data
    t0 = time.time()
    meshmap = meshmap.copy()
    img_h, img_w = meshmap.shape

    """ labeling method """
    #meshmap = (np.clip(meshmap, 0, 1) * 255).astype(np.uint8)
    ret, text_score_comb = cv2.threshold(meshmap, threshold, 1, 1)
    text_score_comb = (np.clip(text_score_comb, 0, 1) * 255)
    #cv2.imwrite('temp.jpg',text_score_comb)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    print ('nLabels of boxes',nLabels)
    #print ('labels',labels)

    det = []
    polys = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        #if np.max(meshmap[labels==k]) > threshold: continue

        # make segmentation map
        segmap = np.zeros(meshmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        #segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]

        #niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        niter = 2
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        #print ('zb1',time.time()-t0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))

        #print ('zb2',time.time()-t0)
        #erode& dilate
        ##todo
        #segmap[sy:ey, sx:ex] = cv2.morphologyEx(segmap[sy:ey, sx:ex], cv2.MORPH_OPEN, kernel, iterations=2)
#        segmap[sy:ey, sx:ex] = cv2.erode(segmap[sy:ey, sx:ex], kernel, iterations=2)
#        _nLabels, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(segmap, connectivity=4)
#        if _nLabels > 2:
#            continue

        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        #print ('zb3',time.time()-t0)

        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)

        # make box

        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)


        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        if size/(w*h) < 0.65:#not valid cell
            #print ('sb2')
            continue

        '''
        # convexity defects detect
        _,contours,_ = cv2.findContours(segmap,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        extreme_defects = []
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                #print ('convexity',start,end,far)
                far_depth = min(distance_btw2points(start,far),distance_btw2points(end,far))
                startend_depth = distance_btw2points(start,end)
                print (far_depth,startend_depth)
                if startend_depth/far_depth<0.3:
                    print (d,w,h)
                    print (box)
                    extreme_defects.append([s,e,f])

        if extreme_defects !=[]:
            continue
        '''
        # align diamond-shape
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        #t0 = time.time()
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)
        det.append(box)
        #print ('zb4',time.time()-t0)


        hull = cv2.convexHull(np_contours)

        polys.append(hull)


        #print ('zb4',time.time()-t0)
        #polys.append(box)

    return det, polys

def getMeshBoxes_new(meshmap, threshold):
    # prepare data
    det = []
    polys = []
    t0 = time.time()
    meshmap = meshmap.copy()
    img_h, img_w = meshmap.shape

    ret, text_score_comb = cv2.threshold(meshmap, threshold, 1, 1)
    text_score_comb = (np.clip(text_score_comb, 0, 1) * 255)
    try:
        _,contours, hierarchy = cv2.findContours(text_score_comb.astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, hierarchy = cv2.findContours(text_score_comb.astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num_contours = len(contours)

    ##remove invalid hierarchy level
    first_level, last_level, middle_level, standalone_level = [],[],[],[]
    for i in range(num_contours):

        [Next, Previous, Child, Parent] = hierarchy[0][i]
        if Parent == -1 and Child == -1:
            standalone_level.append(i)
        elif Parent == -1:
            first_level.append(i)
        elif Child == -1:
            last_level.append(i)
        else:
            middle_level.append(i)

    first_len, last_len, middle_len, standalone_len = len(first_level),len(last_level),len(middle_level), len(standalone_level)

    print ('hierarchy',first_len, last_len, middle_len, standalone_len)

    max_len = max(first_len, last_len, middle_len)

    if first_len == max_len:
        max_level = first_level
    elif middle_len == max_len:
        max_level = middle_level
    elif last_len == max_len:
        max_level = last_level

    max_level.extend(standalone_level)

    print ('nLabels of boxes',num_contours)
    for i, contour in enumerate(contours):

        if i not in max_level :
            continue

        size = cv2.contourArea(contour)

        if size < 10:
            continue

        rectangle = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rectangle)
        #print (box)

        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        if size/(w*h) < 0.65:#not valid cell
            #print ('sb2')
            continue

        # align diamond-shape
#        box_ratio = max(w, h) / (min(w, h) + 1e-5)
#        if abs(1 - box_ratio) <= 0.1:
#            l, r = min(np_contours[:,0]), max(np_contours[:,0])
#            t, b = min(np_contours[:,1]), max(np_contours[:,1])
#            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        #t0 = time.time()
        #box = unclip(box)[0]
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)


        #print (box)
        det.append(box)
        #print ('zb4',time.time()-t0)


        hull = cv2.convexHull(contour)
        hull = unclip(hull).reshape(-1, 1, 2)


        polys.append(hull)

    return det, polys


def unclip(box):
    box = box.reshape(-1,2)
    #print (box)
    #poly = Polygon(box)
    distance = 0.5#poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded



def getMeshLines(meshmap, threshold, up_threshold=None):
    # prepare data
    meshmap = meshmap.copy()
    img_h, img_w = meshmap.shape

    """ labeling method """
    #meshmap = (np.clip(meshmap, 0, 1) * 255).astype(np.uint8)
    ret, text_score_comb = cv2.threshold(meshmap, threshold, 1, 0)
    text_score_comb = (np.clip(text_score_comb, 0, 1) * 255)
    #cv2.imwrite('temp.jpg',text_score_comb)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    print ('nLabels of lines',nLabels)
    #print ('labels',labels)

    #compute adaptive up threshold
    if up_threshold is None:
        up_threshold = np.mean(meshmap[labels>0])
    #if up_threshold > 0.6:
    #    up_threshold = 0.7
    print ('up threshold', up_threshold)


    det = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 100:
            continue

        # thresholding

        if np.max(meshmap[labels==k]) < up_threshold: continue

        # make segmentation map
        segmap = np.zeros(meshmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        #segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]

        #niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        niter = 2
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        #erode& dilate
        ##todo
        #segmap[sy:ey, sx:ex] = cv2.morphologyEx(segmap[sy:ey, sx:ex], cv2.MORPH_OPEN, kernel, iterations=2)
#        segmap[sy:ey, sx:ex] = cv2.erode(segmap[sy:ey, sx:ex], kernel, iterations=2)
#        _nLabels, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(segmap, connectivity=4)
#        if _nLabels > 2:
#            continue

        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel, iterations=1)
        try:
            _,contours,_ = cv2.findContours(segmap,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        except:
            contours,_ = cv2.findContours(segmap,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnt = np.array(contours[0]).reshape(-1,2)

        det.append(cnt)


        #debug
#        img = np.zeros([img_h,img_w,3],np.uint8)
#        import time
#        date_str = str(time.time())
#        img = cv2.polylines(img, [cnt], True, (0,0,255), 2)
#        cv2.imwrite(date_str + '.jpg',img)
    return det, up_threshold




def line_rmse(s1, s2):
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(s1.reshape(-1,1), s2.reshape(-1,1))
    # Predict
    predicted = regression_model.predict(s1.reshape(-1,1))
    # model evaluation
    rmse = mean_squared_error(s2.reshape(-1,1), predicted)
    #rmse = rmse/(ys.reshape(-1,1)**2- y_predicted**2).max()**2

    #print (rmse)
    return rmse

def lineFit(xs,ys):
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(xs.reshape(-1,1), ys.reshape(-1,1))
    return regression_model

def curve_line_rmse(s1,s2):
    ##polynomial curve fitting

    #speed up
    s1,s2 = sparse(s1),sparse(s2)

    #print (s1.shape)
    #print ('zb0:', time.time()-t0)

    reg = np.polyfit(s1, s2,2)
    #print ('zb2:', time.time()-t0)
    predicted = np.polyval(reg, s1)
    #print ('zb3:', time.time()-t0)
    rmse = mean_squared_error(s2, predicted)
    #print ('zb4:', time.time()-t0)
    #print (rmse)
    return reg,rmse

def if_point_fit_line(line,pt,type_):
    if type_=='h':
        reg, mse1 = curve_line_rmse(line[:,0], line[:,1])
        predicted = np.polyval(reg, pt.x)
        #print (predicted)
        mse2 = mean_squared_error([pt.y], [predicted])
    elif type_=='v':
        reg, mse1 = curve_line_rmse(line[:,1], line[:,0])
        predicted = np.polyval(reg, pt.y)
        mse2 = mean_squared_error([pt.x], [predicted])
    if 1:#(mse2 - mse1)/mse1 < 0.25:
        return True
    else:
        return False

def find_end_point(line, x0, type_):
    if type_ == 'h':
        for (x,y) in list(line.exterior.coords):
            if x == x0:
                return (x,y)
    if type_ == 'v':
        for (y,x) in list(line.exterior.coords):
            if x == x0:
                return (y,x)
    assert 1==0
    return None

def get_extend_line(line,pt,type_):
    #short line do not use curve fit, use linear fit
    length = Polygon(line).length
    if length > 500:
        deg = 2
    else:
        deg = 1

    if type_ == 'left':
        reg = np.polyfit(line[:,0], line[:,1],deg)
        X = np.arange(0,pt.x)
        Y = np.polyval(reg, X)
        extend_line = np.stack([X,Y],axis=1)
    elif type_ == 'right':
        reg = np.polyfit(line[:,0], line[:,1],deg)
        X = np.arange(pt.x,5000)
        Y = np.polyval(reg, X)
        extend_line = np.stack([X,Y],axis=1)
    elif type_ == 'up':
        reg = np.polyfit(line[:,1], line[:,0],deg)
        X = np.arange(0,pt.y)
        Y = np.polyval(reg, X)
        extend_line = np.stack([Y,X],axis=1)
    elif type_ == 'down':
        reg = np.polyfit(line[:,1], line[:,0],deg)
        X = np.arange(pt.y,5000)
        Y = np.polyval(reg, X)
        extend_line = np.stack([Y,X],axis=1)
    return extend_line



def find_line_cross_pt(line_e,line_):
    try:
        intersect = Polygon(line_e).buffer(1).intersection(line_)
        #if len(intersect.geoms)>1:
        try:
            intersect = list(intersect)[0]
        except:
            ...

        cross_pts = list(intersect.exterior.coords)
        if len(cross_pts) > 0:
            return cross_pts[0]
        else:
            return None
    except:
        return None


def if_connect_lines(line1, line2, v_lines, type_):
    ##check weather connecting the hlines with the same id
    line1 = Polygon(line1)
    line2 = Polygon(line2)
    #length1 = line1.length
    #length2 = line2.length
    (minx1, miny1, maxx1, maxy1) = line1.bounds
    (minx2, miny2, maxx2, maxy2) = line2.bounds
    if type_ == 'h':
        min1, max1, min2, max2 = minx1, maxx1, minx2, maxx2
    else:
        min1, max1, min2, max2 = miny1, maxy1, miny2, maxy2

    if min1 > max2:
        #line2 --- line1
        (x1,y1) = find_end_point(line1,min1,type_)
        (x2,y2) = find_end_point(line2,max2,type_)
        pt1 = Point(x1,y1)
        pt2 = Point(x2,y2)
    elif max1 < min2:
        #line1 --- line2
        (x1,y1) = find_end_point(line1,max1,type_)
        (x2,y2) = find_end_point(line2,min2,type_)
        pt1 = Point(x1,y1)
        pt2 = Point(x2,y2)
    else:
        return False, None,None,None,None


    flag1, flag2 = False, False
    for v_line in v_lines:
        v_line = Polygon(v_line)
        dist1 = pt1.distance(v_line)
        dist2 = pt2.distance(v_line)
        if dist1 > 20:
            flag1 = True
        if dist2 > 20:
            flag2 = True
    if flag1 and flag2:
        return True, int(x1),int(y1),int(x2),int(y2)

    return False, None,None,None,None

def if_connect_lines_new(line1, line2, v_lines, type_):
    ##check weather connecting the hlines with the same id
    #speed up
    t0 =time.time()
    line1, line2 = sparse(line1),sparse(line2)

    line1 = Polygon(line1)
    line2 = Polygon(line2)


    (minx1, miny1, maxx1, maxy1) = line1.bounds
    (minx2, miny2, maxx2, maxy2) = line2.bounds

    #print ('sb1:', time.time()-t0)
    if type_ == 'h':
        min1, max1, min2, max2 = minx1, maxx1, minx2, maxx2
    else:
        min1, max1, min2, max2 = miny1, maxy1, miny2, maxy2

    if max1 > min2:
        return False, None,None,None,None


    [pt1,pt2] = nearest_points(line1, line2)
    #print ('sb1:', time.time()-t0)
    #print (pt1,pt2)
    #flag1, flag2 = True, True
    flag = True
    for v_line in v_lines:

        v_line = Polygon(sparse(v_line))
        t1 = time.time()

        dist1 = pt1.distance(v_line)
        dist2 = pt2.distance(v_line)
        #print ('sbbb', time.time()-t1)
        #print (dist1, dist2)
        if dist1 < 10 or dist2 < 10:
            flag = False
            break
        #if dist2 < 10:
        #    flag2 = False


    #if flag1 and flag2:
    if flag:
        return True, int(pt1.x),int(pt1.y),int(pt2.x),int(pt2.y)

    return False, None,None,None,None

def if_close_lines_new(line, pt1, v_lines, type_):

    lefts,rights,ups,downs = [],[],[],[]

    for v_line0 in v_lines:

        v_line = Polygon(v_line0)

        dist = pt1.distance(v_line)

        if dist > 0:
            t0 =time.time()
            #speed up
            v_line1 = sparse(v_line0)
            v_line1 = Polygon(v_line1)
            [pt1_,pt2] = nearest_points(pt1, v_line1)
            assert pt1 == pt1_
            if pt1.x > pt2.x:
                lefts.append([pt1,pt2,dist,v_line])
            if pt1.x < pt2.x:
                rights.append([pt1,pt2,dist,v_line])
            if pt1.y > pt2.y:
                ups.append([pt1,pt2,dist,v_line])
            if pt1.y < pt2.y:
                downs.append([pt1,pt2,dist,v_line])
            #print ('sb1:', time.time()-t0)
        else:
            return 0, None,None,None,None # close line


    #print ('s'*100)
    #print (lefts)
    #print ('sf'*100)
    #print (rights)

    if type_ == 'left':
        sort_ids = sorted(range(len(lefts)), key=lambda k: lefts[k][2])
        lefts = [lefts[i] for i in sort_ids]
        for [pt1,pt2,dist,line_] in lefts:
            #if dist == 0:
            #    break
            line_e = get_extend_line(line,pt1,type_)
            cross_pt = find_line_cross_pt(line_e,line_)
            if cross_pt is not None:
                 return 1, int(pt1.x),int(pt1.y),int(cross_pt[0]),int(cross_pt[1])
            #if if_point_fit_line(line,pt2,'h'):
            #    return True, int(pt1.x),int(pt1.y),int(pt2.x),int(pt2.y)

    elif type_ == 'right':
        sort_ids = sorted(range(len(rights)), key=lambda k: rights[k][2])
        rights = [rights[i] for i in sort_ids]
        #print (rights)
        for [pt1,pt2,dist,line_] in rights:
            #if dist == 0:
            #    break
            line_e = get_extend_line(line,pt1,type_)
            cross_pt = find_line_cross_pt(line_e,line_)
            if cross_pt is not None:
                 return 1, int(pt1.x),int(pt1.y),int(cross_pt[0]),int(cross_pt[1])
            #if if_point_fit_line(line,pt2,'h'):
            #    return True, int(pt1.x),int(pt1.y),int(pt2.x),int(pt2.y)

    elif type_ == 'up':
        sort_ids = sorted(range(len(ups)), key=lambda k:ups[k][2])
        ups = [ ups[i] for i in sort_ids]
        #print (ups)
        for [pt1,pt2,dist,line_] in ups:
            #if dist == 0:
            #    break
            line_e = get_extend_line(line,pt1,type_)
            cross_pt = find_line_cross_pt(line_e,line_)
            if cross_pt is not None:
                 return 1, int(pt1.x),int(pt1.y),int(cross_pt[0]),int(cross_pt[1])
            #if if_point_fit_line(line,pt2,'v'):
            #    return True, int(pt1.x),int(pt1.y),int(pt2.x),int(pt2.y)

    elif type_ == 'down':
        sort_ids = sorted(range(len(downs )), key=lambda k:downs[k][2])
        downs = [downs [i] for i in sort_ids]
        for [pt1,pt2,dist,line_] in downs :
            #if dist == 0:
            #    break
            line_e = get_extend_line(line,pt1,type_)
            cross_pt = find_line_cross_pt(line_e,line_)
            if cross_pt is not None:
                 return 1, int(pt1.x),int(pt1.y),int(cross_pt[0]),int(cross_pt[1])
            #if if_point_fit_line(line,pt2,'v'):
            #    return True, int(pt1.x),int(pt1.y),int(pt2.x),int(pt2.y)

    return -1, None,None,None,None # unclose line


def lines_sort_merge(h_lines, v_lines):
    ##1.lines sort

    h_lines_midy = []
    for h_line in h_lines:
        #print (h_line)
        (minx, miny, maxx, maxy) = Polygon(h_line).bounds
        h_lines_midy.append((miny+maxy)/2)
    #h_lines_midy, h_lines = zip(*sorted(zip(h_lines_midy, h_lines)))
    sort_ids = sorted(range(len(h_lines_midy)), key=lambda k: h_lines_midy[k])
    h_lines = [h_lines[i] for i in sort_ids]

    v_lines_midx = []
    for v_line in v_lines:
        (minx, miny, maxx, maxy) = Polygon(v_line).bounds
        #print (minx, miny, maxx, maxy)
        v_lines_midx.append((minx+maxx)/2)
    #print (len(v_lines))
    sort_ids = sorted(range(len(v_lines_midx)), key=lambda k: v_lines_midx[k])
    v_lines = [v_lines[i] for i in sort_ids]
    #v_lines_midx, v_lines = zip(*sorted(zip(v_lines_midx, v_lines)))
    h_lines_id = [ i for i in range(len(h_lines))]
    v_lines_id = [ i for i in range(len(v_lines))]

    ##2.lines merge
    #debug
#    img = np.zeros([1000,1000,3],np.uint8)
#    for i in range(len(v_lines)):
#        v_line1 = v_lines[i]
#        regression_model1 = lineFit(v_line1[:,1].reshape(-1,1), v_line1[:,0].reshape(-1,1))
#        x_predicted1 = regression_model1.predict(v_line1[:,1].reshape(-1,1))
#
#        points_list = np.hstack((x_predicted1,v_line1[:,1].reshape(-1,1))).tolist()
#        print (len(points_list))
#
#        if i %2 == 0:
#            color = (0,0,255)
#        else:
#            color = (0,255,0)
#        for point in points_list:
#            point = tuple(int(x) for x in point)
#            cv2.circle(img, point, 1, color , 4)
#    cv2.imwrite('temp.jpg',img)

    #
    nexts = 5 ##

    for i in range(len(h_lines)-1):
        for j in range(i+1, min(len(h_lines),i+nexts)): #speed up for next x lines
            t0 =time.time()
            h_line1 = np.vstack((h_lines[i],h_lines[i]))
            h_line2 = np.vstack((h_lines[i],h_lines[j]))
            #print ('SSb0:', time.time()-t0)
            _,mse1 = curve_line_rmse(h_line1[:,0], h_line1[:,1])
            _,mse2 = curve_line_rmse(h_line2[:,0], h_line2[:,1])
            #print ('SSb1:', time.time()-t0)
            if (mse2 - mse1)/mse1 < 0.25:
                h_lines_id[j] = h_lines_id[i]



    for i in range(len(v_lines)-1):
        for j in range(i+1, min(len(v_lines),i+nexts)):
            v_line1 = np.vstack((v_lines[i],v_lines[i]))
            v_line2 = np.vstack((v_lines[i],v_lines[j]))
            _,mse1 = curve_line_rmse(v_line1[:,1],v_line1[:,0])
            _,mse2 = curve_line_rmse(v_line2[:,1],v_line2[:,0])
            if (mse2 - mse1)/mse1 < 0.25:
                v_lines_id[j] = v_lines_id[i]



    print (h_lines_id)
    print (v_lines_id)
    return h_lines, v_lines, h_lines_id, v_lines_id

def lines_connect_close(h_lines, v_lines, h_lines_id, v_lines_id, score_h, score_v):
    t0 =time.time()
    ##lines connect
    connect_h_lines = []
    for i in range(len(h_lines)):
        this_lines = []
        for j,h_line_id in enumerate(h_lines_id):
            if h_line_id == i:
                this_lines.append(h_lines[j])
        if len(this_lines)>1:
            this_lines_midx = []

            for this_line in this_lines:
                (minx, miny, maxx, maxy) = Polygon(this_line).bounds
                this_lines_midx.append((minx+maxx)/2)

            sort_ids = sorted(range(len(this_lines_midx)), key=lambda k: this_lines_midx[k])
            #print (sort_ids)
            this_lines = [this_lines[i] for i in sort_ids]
            #print (this_lines)
            for j in range(len(this_lines)-1):

                flag, x1,y1,x2,y2 = if_connect_lines_new(this_lines[j], this_lines[j+1], v_lines, 'h')

                if flag:
                    print ('h connect')
                    cv2.line(score_h, (x1,y1),(x2,y2),1,3)
                    connect_h_lines.append([this_lines[j],this_lines[j+1]])

    print ('sb1:', time.time()-t0)
    connect_v_lines = []
    for i in range(len(v_lines)-1):
        this_lines = []
        for j,v_line_id in enumerate(v_lines_id):
            if v_line_id == i:
                this_lines.append(v_lines[j])
        if len(this_lines)>1:
            this_lines_midy = []
            for this_line in this_lines:
                (minx, miny, maxx, maxy) = Polygon(this_line).bounds
                this_lines_midy.append((miny+maxy)/2)
            sort_ids = sorted(range(len(this_lines_midy)), key=lambda k: this_lines_midy[k])
            this_lines = [this_lines[i] for i in sort_ids]
            for j in range(len(this_lines)-1):
                t1= time.time()
                flag, x1,y1,x2,y2 = if_connect_lines_new(this_lines[j], this_lines[j+1], h_lines,'v')
                #print ('zb',time.time()-t1)
                if flag:
                    print ('v connect')
                    cv2.line(score_v, (x1,y1),(x2,y2),1,3)
                    connect_v_lines.append([this_lines[j],this_lines[j+1]])

    print ('sb2:', time.time()-t0)
    ##lines close
    #both side of the lines should be close
    #connected side of the lines do not need be close

    unclose_left_pts = []
    unclose_right_pts = []
    unclose_up_pts = []
    unclose_down_pts = []

    for h_line in h_lines:

        left_flag, right_flag = True, True
        for [line1, line2] in connect_h_lines:
            #print (h_line)
            #print (line1)
            if h_line.shape == line1.shape and (h_line==line1).all():
                right_flag = False

            if h_line.shape == line2.shape and (h_line==line2).all():
                left_flag = False
        #print (left_flag, right_flag)

        if left_flag:

            (minx1, miny1, maxx1, maxy1) = Polygon(h_line).bounds

            (x1,y1) = find_end_point(Polygon(h_line),minx1,'h')
            #
            pt = Point(x1,y1)
            #
            flag, x1,y1,x2,y2 = if_close_lines_new(h_line,pt, v_lines, 'left')
            #
            if flag == 1 :
                print ('left')
                cv2.line(score_h, (x1,y1),(x2,y2),1,3)
            elif flag == -1:
                unclose_left_pts.append(pt)


        if right_flag:
            (minx1, miny1, maxx1, maxy1) = Polygon(h_line).bounds
            #print (minx1, miny1, maxx1, maxy1)

            (x1,y1) = find_end_point(Polygon(h_line),maxx1,'h')

            pt = Point(x1,y1)
            #print (pt)
            #t1 = time.time()
            v_lines_ = v_lines.copy()
            v_lines_.reverse()
            flag, x1,y1,x2,y2 = if_close_lines_new(h_line,pt, v_lines_, 'right')
            #print ('zb', time.time()-t1)
            if flag == 1:
                print ('right')
                cv2.line(score_h, (x1,y1),(x2,y2),1,3)
            elif flag == -1:
                unclose_right_pts.append(pt)

        #print ('zb3:', time.time()-t0)

    print ('sb3-1:', time.time()-t0)
    for v_line in v_lines:
        up_flag, down_flag = True, True
        for [line1, line2] in connect_v_lines:
            if v_line.shape == line1.shape and (v_line==line1).all():
                down_flag = False
            if v_line.shape == line2.shape  and (v_line==line2).all():
                up_flag = False

        if up_flag:
            (minx1, miny1, maxx1, maxy1) = Polygon(v_line).bounds
            (x1,y1) = find_end_point(Polygon(v_line),miny1,'v')
            pt = Point(x1,y1)
            #t1 = time.time()
            flag, x1,y1,x2,y2 = if_close_lines_new(v_line,pt, h_lines, 'up')
            #print ('sbs', time.time()-t1)
            if flag == 1:
                print ('up')
                cv2.line(score_v, (x1,y1),(x2,y2),1,3)
            elif flag == -1:
                unclose_up_pts.append(pt)

        if down_flag:
            (minx1, miny1, maxx1, maxy1) = Polygon(v_line).bounds
            (x1,y1) = find_end_point(Polygon(v_line),maxy1,'v')
            pt = Point(x1,y1)
            #t1 = time.time()
            h_lines_ = h_lines.copy()
            h_lines_.reverse()
            flag, x1,y1,x2,y2 = if_close_lines_new(v_line,pt, h_lines_, 'down')
            #print ('sbss', time.time()-t1)
            if flag == 1:
                print ('down')
                cv2.line(score_v, (x1,y1),(x2,y2),1,3)
            elif flag == -1:
                unclose_down_pts.append(pt)

        #print ('zb4:', time.time()-t0)

    print ('sb3:', time.time()-t0)

    ##border line add
    # compose all one-side end points of unclosed lines
    # LineString's length should be similar to distance between two end points
    if  if_linestring_valid(unclose_left_pts, h_lines):
        unclose_left_pts = np.asarray(LineString(unclose_left_pts)).astype(np.int32)
        print ('border')
        cv2.polylines(score_v, [unclose_left_pts],0,1,3)

    if  if_linestring_valid(unclose_right_pts, h_lines):
        unclose_right_pts = np.asarray(LineString(unclose_right_pts)).astype(np.int32)
        print ('border')
        cv2.polylines(score_v, [unclose_right_pts],0,1,3)

    if if_linestring_valid(unclose_up_pts, v_lines):
        unclose_up_pts = np.asarray(LineString(unclose_up_pts)).astype(np.int32)
        print ('border')
        cv2.polylines(score_h, [unclose_up_pts],0,1,3)

    if if_linestring_valid(unclose_down_pts, v_lines):
        unclose_down_pts = np.asarray(LineString(unclose_down_pts)).astype(np.int32)
        print ('border')
        cv2.polylines(score_h, [unclose_down_pts],0,1,3)



    return  score_h, score_v

def if_linestring_valid(pts, lines):
    if len(pts)<=1 or (len(pts)<=3 and len(pts)/len(lines)<0.2):
        return False

    #print (list(LineString(pts).coords))
    length = LineString(pts).length
    distance = pts[0].distance(pts[-1])
    print (length, distance)
    if abs(length-distance)/length < 0.02:
        return True
    else:
        return False


def predict_cells_struct_on_mesh(boxes,h_lines,v_lines,h_lines_id, v_lines_id):

    ##connect boxes and lines
    cells_info = []
    max_row_id = 0
    max_col_id = 0

    #boxes_ = [tuple(box) for box in boxes]
    #boxes_ = MultiPolygon(boxes_)

    h_lines_p = []
    v_lines_p = []
    t0 = time.time()

    for h_line in h_lines:
        h_lines_p.append(Polygon(h_line))

    for v_line in v_lines:
        v_lines_p.append(Polygon(v_line))

    #print ('sdb', time.time() -t0)


    for box in boxes:
        #print (box)
        box = box.reshape(-1,2)

        #print ('box',box)

        box = Polygon(box)

        #print (list(box.interiors))

        #box_bound = box.bounds

        #find the nearest lines
        cell_info = {}

        this_row_ids =[]
        for i,h_line in enumerate(h_lines_p):
            #h_line = Polygon(h_line)

            #h_line_bound = h_line.bounds

            #if if_overlap(box_bound,h_line_bound):
                #print ('snn')
            if box.intersects(h_line):
                this_row_ids.append(h_lines_id[i])
            #else:
                #print ('conti')

        if this_row_ids == []:
            cell_info['row_id_start'] = -1
            cell_info['row_id_end'] = -1
        else:
            cell_info['row_id_start'] = min(this_row_ids)
            cell_info['row_id_end'] = max(this_row_ids)
        max_row_id = max(max_row_id,cell_info['row_id_end'])

        this_col_ids =[]
        for i,v_line in enumerate(v_lines_p):
            #v_line = Polygon(v_line)

            #v_line_bound = v_line.bounds
            #if if_overlap(box_bound,v_line_bound):
            if box.intersects(v_line):
                this_col_ids.append(v_lines_id[i])
        #print (this_col_ids)
        if this_col_ids == []:
            cell_info['col_id_start'] = -1
            cell_info['col_id_end'] = -1
        else:
            cell_info['col_id_start'] = min(this_col_ids)
            cell_info['col_id_end'] = max(this_col_ids)
        max_col_id = max(max_col_id,cell_info['col_id_end'])

        cells_info.append(cell_info)



    ## squeeze empty rowcols
    squeeze_row_ids = list(range(max_row_id+1))
    squeeze_col_ids = list(range(max_col_id+1))
    for cell_info in cells_info:
        row_id_start = cell_info['row_id_start']
        row_id_end = cell_info['row_id_end']
        col_id_start = cell_info['col_id_start']
        col_id_end = cell_info['col_id_end']
        if row_id_start in squeeze_row_ids:
            squeeze_row_ids.remove(row_id_start)
        if row_id_end in squeeze_row_ids:
            squeeze_row_ids.remove(row_id_end)
        if col_id_start in squeeze_col_ids:
            squeeze_col_ids.remove(col_id_start)
        if col_id_end in squeeze_col_ids:
            squeeze_col_ids.remove(col_id_end)

    squeeze_row_ids.sort(reverse=True)
    squeeze_col_ids.sort(reverse=True)

    for squeeze_row_id in squeeze_row_ids:
        for i,cell_info in enumerate(cells_info):
            if cell_info['row_id_start'] > squeeze_row_id:
                cells_info[i]['row_id_start'] -= 1
            if cell_info['row_id_end'] > squeeze_row_id:
                cells_info[i]['row_id_end'] -= 1
    for squeeze_col_id in squeeze_col_ids:
        for i,cell_info in enumerate(cells_info):
            if cell_info['col_id_start'] > squeeze_col_id:
                cells_info[i]['col_id_start'] -= 1
            if cell_info['col_id_end'] > squeeze_col_id:
                cells_info[i]['col_id_end'] -= 1

    #remove “-1” involved cell
    #for cell_info in cells_info:
    #    if -1 in [cell_info['row_id_start'], cell_info['row_id_end'],cell_info['col_id_start'],cell_info['col_id_end']]:
    #        cells_info.remove(cell_info)


    return cells_info

def distance_btw2points(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2 )

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2.0):
    if len(polys) > 0:
        #polys = np.array(polys)
        #print (polys)
        #print (polys[0])

        for k in range(len(polys)):
            if polys[k] is not None:

                polys[k] = polys[k].astype(np.float64)
                #print (polys[k])
                #polys[k].dtype = np.float64
                #print (polys[k])
                #print ('-----')
                #print (ratio_w * ratio_net,ratio_h * ratio_net)

                polys[k] *= (ratio_w * ratio_net,ratio_h * ratio_net)


    return polys


def sparse(x):
    if x.shape[0] > 100:
        x = x[np.arange(0,x.shape[0],10)]
    return x

def if_overlap(bound1,bound2):
    (x1,y1,x2,y2) = bound1
    (X1,Y1,X2,Y2) = bound2
    left_line = max(x1,X1)
    right_line = min(x2,X2)
    top_line = max(y1,Y1)
    bottom_line = min(y2,Y2)
    if left_line > right_line or top_line > bottom_line:
        return False
    else:
        return True
