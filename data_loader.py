


###for icdar2015####



import torch
import torch.utils.data as data
import scipy.io as scio
from gaussian import GaussianTransformer
#from watershed import watershed
#import re
#import itertools
from file_utils import *
#from mep import mep
import random
from PIL import Image
import cv2
import torchvision.transforms as transforms
import craft_utils
import Polygon as plg
import time,copy
from math import *
import numpy as np
import glob



def sp_noise(image,prob=0.5):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.01):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

def src2dst(x,y,M):
    [M11,M12,M13] = M[0]
    [M21,M22,M23] = M[1]
    [M31,M32,M33] = M[2]
    x,y = (M11*x+M12*y+M13)/(M31*x+M32*y+M33), (M21*x+M22*y+M23)/(M31*x+M32*y+M33)
    return int(x),int(y)

def rad(x):
    return x*np.pi/180

def ComputeDestWH(w,h,anglex,angley,anglez,fov=60):
    #anglex= 10
    #angley = 10
    #anglez = 10
    #fov = 42
    #the dist between lens and image. Computing Z value to make sure show the whole image.
    #fov/2 is the half view scape
    z=np.sqrt(w**2 + h**2)/2/np.tan(rad(fov/2))
    #print(anglex,angley,anglez,fov,z)
    #homogeneous transform matrix
    rx = np.array([[1,                  0,                          0,                          0],
                  [0,                  np.cos(rad(anglex)),        -np.sin(rad(anglex)),      0],
                  [0,                -np.sin(rad(anglex)),        np.cos(rad(anglex)),        0,],
                  [0,                  0,                          0,                          1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0,                        np.sin(rad(angley)),      0],
                  [0,                  1,                        0,                          0],
                  [-np.sin(rad(angley)),0,                        np.cos(rad(angley)),        0,],
                  [0,                  0,                        0,                          1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)),      0,                          0],
                  [-np.sin(rad(anglez)), np.cos(rad(anglez)),      0,                          0],
                  [0,                  0,                          1,                          0],
                  [0,                  0,                          0,                          1]], np.float32)

    r = rx.dot(ry).dot(rz)

    #generate four point pairs
    pcenter = np.array([w/2, h/2, 0, 0], np.float32)

    p1 = np.array([0,0,  0,0], np.float32) - pcenter
    p2 = np.array([w,0,  0,0], np.float32) - pcenter
    p3 = np.array([w,h,  0,0], np.float32) - pcenter
    p4 = np.array([0,h,  0,0], np.float32) - pcenter


    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]
    #print(list_dst)

    org = np.array([[0,0],
                    [w,0],
                    [w,h],
                    [0,h]], np.float32)

    dst = np.zeros((4,2), np.float32)

    #project to film plane
    minx = None     #attension: minx=-1 some times in python2.7, np's near zero equal -1.000e+00
    maxx = None
    miny = None
    maxy = None
    for i in range(4):
        dst[i,0] = list_dst[i][0]*z/(z-list_dst[i][2]) + pcenter[0]
        if minx is None or dst[i,0]<minx:
            minx = copy.deepcopy(dst[i,0])
        if maxx is None or dst[i,0]>maxx:
            maxx = copy.deepcopy(dst[i,0])
        dst[i,1] = list_dst[i][1]*z/(z-list_dst[i][2]) + pcenter[1]
        if miny is None or dst[i,1]<miny:
            miny = copy.deepcopy(dst[i,1])
        if maxy is None or dst[i,1]>maxy:
            maxy = copy.deepcopy(dst[i,1])
    new_w = maxx - minx
    new_h = maxy - miny
    #print('minx,maxx,miny,maxy',minx,maxx,miny,maxy)
    #print('pcenter[0],pcenter[1]',pcenter[0],pcenter[1])
    #print(org,dst,new_w,new_h,anglez)
    polypoint_list = []
    #polypoint_list.append([dst[0,0]-minx,dst[0,1]-miny])
    #polypoint_list.append([dst[1,0]-minx,dst[1,1]-miny])
    #polypoint_list.append([dst[3,0]-minx,dst[3,1]-miny])
    #polypoint_list.append([dst[2,0]-minx,dst[2,1]-miny])
    #----
    polypoint_list.append([dst[0,0],dst[0,1]])
    polypoint_list.append([dst[1,0],dst[1,1]])
    polypoint_list.append([dst[2,0],dst[2,1]])
    polypoint_list.append([dst[3,0],dst[3,1]])

    return polypoint_list, minx,maxx,miny,maxy


def ratio_area(h, w, box):
    area = h * w
    ratio = 0
    for i in range(len(box)):
        poly = plg.Polygon(box[i])
        box_area = poly.area()
        tem = box_area / area
        if tem > ratio:
            ratio = tem
    return ratio, area

def rescale_img(img, box, h, w):
    image = np.zeros((768,768,3),dtype = np.uint8)
    length = max(h, w)
    scale = 768 / length           ###768 is the train image size
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    image[:img.shape[0], :img.shape[1]] = img
    box *= scale
    return image

def random_scale(img, bboxes, min_size, words= None, min_sep_width = 8):
    h, w = img.shape[0:2]
    # ratio, _ = ratio_area(h, w, bboxes)
    # if ratio > 0.5:
    #     image = rescale_img(img.copy(), bboxes, h, w)
    #     return image
    scale = 1.0
    if max(h, w) * scale  >1280:###1280
        scale = 1280.0 / max(h, w)

    random_scale = np.array([0.5, 0.75, 1.0, 1.5])  ###[0.5, 1.0, 1.5, 2.0]
    #random_scale = np.array([1.0,1.5,2.0]) #for table mesh
    scale1 = np.random.choice(random_scale)

    if min(h, w) * scale * scale1 <= min_size:
        scale = (min_size+10) * 1.0 / min(h, w)
    else:
        scale = scale * scale1

    #text_h(10) * scale > 8 -> scale > 0.8,
    #insure min text_h surpass 8
    if min_sep_width *scale <8:

        scale = 8/min_sep_width
    #print ('scale',scale)
    bboxes *= scale
    if words is not None:
        words *= scale
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    return img

def random_scale_by_32(img, bboxes, min_size, words, min_sep_width = 8):
    h, w = img.shape[0:2]
    # ratio, _ = ratio_area(h, w, bboxes)
    # if ratio > 0.5:
    #     image = rescale_img(img.copy(), bboxes, h, w)
    #     return image
    scale = 1.0
    #if max(h, w) * scale  >1280:###1280
    #    scale = 1280.0 / max(h, w)

    random_scale = np.array([0.5, 0.75, 1.0, 1.5])  ###[0.5, 1.0, 1.5, 2.0]
    #random_scale = np.array([1.0,1.5,2.0]) #for table mesh
    scale1 = np.random.choice(random_scale)

    if min(h, w) * scale * scale1 <= min_size:
        scale = (min_size+10) * 1.0 / min(h, w)
    else:
        scale = scale * scale1

    #text_h(10) * scale > 8 -> scale > 0.8,
    #insure min text_h surpass 8
    if min_sep_width *scale <4:
        scale = 4/min_sep_width
    elif min_sep_width *scale > 16:
        scale = 16/min_sep_width
    #print ('scale',scale)

    """resize image to be divisible by 32
    """
    h_,w_ = h*scale, w*scale

    resize_h = int(h_) if h_ % 32 == 0 else int(h_ / 32) * 32
    resize_w = int(w_) if w_ % 32 == 0 else int(w_ / 32) * 32

    ratio_h = resize_h / h
    ratio_w = resize_w / w

    #print (ratio_h, ratio_w)
    img = cv2.resize(img,(resize_w, resize_h))
    #img = cv2.resize(img, dsize=None, fx=ratio_w, fy=ratio_h)

    bboxes[:,:,0] *= ratio_w
    bboxes[:,:,1] *= ratio_h

    if len(words)>0:
        words[:,:,0]*=ratio_w
        words[:,:,1]*=ratio_h

    return img

def padding_image(image,imgsize,conf=False):
    length = max(image.shape[0:2])

    if len(image.shape) == 3:
        img = np.zeros((imgsize, imgsize, len(image.shape)), dtype = np.uint8)
    elif conf:
        img = np.zeros((imgsize, imgsize), dtype = np.float32)
    else:
        img = np.zeros((imgsize, imgsize), dtype = np.uint8)
    scale = imgsize / length
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale,interpolation=cv2.INTER_NEAREST)
    if len(image.shape) == 3:
        img[:image.shape[0], :image.shape[1], :] = image
    else:
        img[:image.shape[0], :image.shape[1]] = image
    return img

def padding_image_(image,imgsize,conf=False):
    length = max(image.shape[0:2])

    if len(image.shape) == 3:
        img = np.zeros((imgsize, imgsize, len(image.shape)), dtype = np.uint8)
    elif conf:
        img = np.zeros((imgsize, imgsize), dtype = np.float32)
    else:
        img = np.zeros((imgsize, imgsize), dtype = np.uint8)
    scale = imgsize / length
    #image = cv2.resize(image, dsize=None, fx=scale, fy=scale,interpolation=cv2.INTER_NEAREST)
    if len(image.shape) == 3:
        img[:image.shape[0], :image.shape[1], :] = image
    else:
        img[:image.shape[0], :image.shape[1]] = image
    return img

def random_crop(imgs, img_size, character_bboxes):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w == tw and h == th:
        return imgs

    word_bboxes = []
    if len(character_bboxes) > 0:
        for bboxes in character_bboxes:
            word_bboxes.append(
                [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    #else:
    #    word_bboxes = Word_bboxes
    word_bboxes = np.array(word_bboxes, np.int32)

    #### IC15 for 0.6, MLT for 0.35 #####
    if random.random() > 0.6 and len(word_bboxes) > 0:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]
        left = max(sample_bboxes[1, 0] - img_size[0], 0)
        top = max(sample_bboxes[1, 1] - img_size[0], 0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    else:
        ### train for IC15 dataset####
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)

        #### train for MLT dataset ###
        i, j = 0, 0
        crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            if idx == len(imgs)-1:
                imgs[idx] = padding_image(imgs[idx], tw, True)
            else:
                imgs[idx] = padding_image(imgs[idx], tw)

    return imgs

def random_warp(imgs):
    offset4p = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if random.random() < 0.3:
        if random.random() > 0.5:
            anglex= random.sample(list(range(-30,-15))+list(range(15,30)),1)[0]
            angley= random.sample(list(range(-3,-0))+list(range(0,3)),1)[0]
        else:
            anglex= random.sample(list(range(-3,-0))+list(range(0,3)),1)[0]
            angley= random.sample(list(range(-30,-15))+list(range(15,30)),1)[0]
        anglez= random.sample(list(range(-15,-6))+list(range(6,15)),1)[0]
        for i in range(len(imgs)):
            img = imgs[i]
            h, w = img.shape[:2]
            polypoint_list,minx,maxx,miny,maxy = ComputeDestWH(w,h,anglex,angley,anglez)
            new_w = maxx - minx
            new_h = maxy - miny
            x = y = 0
            org = np.array([[x,y], [x+w,y], [x+w,y+h],[x,y+h]], np.float32)
            dst = np.zeros((4,2), np.float32)
            dst[0,0] = polypoint_list[0][0]+x
            dst[0,1] = polypoint_list[0][1]+y
            dst[1,0] = polypoint_list[1][0]+x
            dst[1,1] = polypoint_list[1][1]+y
            dst[2,0] = polypoint_list[2][0]+x
            dst[2,1] = polypoint_list[2][1]+y
            dst[3,0] = polypoint_list[3][0]+x
            dst[3,1] = polypoint_list[3][1]+y

            warpR = cv2.getPerspectiveTransform(org, dst)
            imgs[i] = cv2.warpPerspective(img, warpR, (h,w))

            if i ==0 :
                invwarpR = cv2.getPerspectiveTransform(dst, org)
                x1,y1 = src2dst(0,0,invwarpR)
                x2,y2 = src2dst(w-1,0,invwarpR)
                x3,y3 = src2dst(w-1,h-1,invwarpR)
                x4,y4 = src2dst(0,h-1,invwarpR)
                x1 = (x1-0)/w
                y1 = (y1-0)/h
                x2 = (x2-w)/w
                y2 = (y2-0)/h
                x3 = (x3-w)/w
                y3 = (y3-h)/h
                x4 = (x4-0)/w
                y4 = (y4-h)/h
                offset4p = [x1,y1,x2,y2,x3,y3,x4,y4]
    # remove black border by padding or warp
    h, w = imgs[0].shape[:2]
    bg_imgslist = glob.glob('{0}/*.jpg'.format('./data/background'))
    bigbg_imgslist = glob.glob('{0}/*.jpg'.format('./data/bigBG'))
    bg_imgfile = random.sample(bigbg_imgslist+bg_imgslist,1)[0]
    bg_img = cv2.imread(bg_imgfile)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = cv2.resize(bg_img,(w,h))
    confidence_mask = cv2.resize(imgs[-1],(w,h))
    _,bi_mask = cv2.threshold(confidence_mask,0,1,cv2.THRESH_BINARY_INV)
    bi_mask = bi_mask.astype(np.uint8)
    #print (bi_mask.shape)
    try:
        cnt,_ = cv2.findContours(bi_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    except:
        _,cnt,_ = cv2.findContours(bi_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((w,h),dtype=np.uint8)
    cv2.fillPoly(mask, cnt, 255)
#    mask = np.zeros(np.shape(bg_img),dtype=np.uint8)
#    mask = cv2.warpPerspective(mask, warpR, (h,w),borderValue = 255)
#    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    bg_img_canvas = cv2.add(bg_img,np.zeros(np.shape(bg_img),dtype=np.uint8),mask=mask)
    imgs[0] = cv2.add(imgs[0],bg_img_canvas)


    #debug
#    org = np.array([[0,0], [w-1,0], [w-1,h-1],[0,h-1]], np.float32)
#    [m1,n1,m2,n2,m3,n3,m4,n4] =offset4p
#    X1 = int(m1*w)
#    Y1 = int(n1*h)
#    X2 = int(m2*w)+w
#    Y2 = int(n2*h)
#    X3 = int(m3*w)+w
#    Y3 = int(n3*h)+h
#    X4 = int(m4*w)
#    Y4 = int(n4*h)+h
#    dst = np.array([[X1,Y1], [X2,Y2],[X3,Y3],[X4,Y4]], np.float32)
#    warpR = cv2.getPerspectiveTransform(org, dst)
#    imgs[0] = cv2.warpPerspective(imgs[0], warpR, (h,w))

    #print (offset4p)

    return imgs, offset4p

def random_noise(imgs):
    if random.random() < 0.2:
        img = imgs[0]
        var = random.random() * 0.05
        img = gasuss_noise(img,var = var)
        imgs[0] = img


    return imgs

def randombox_crop(imgs, img_size, word_bboxes):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w == tw and h == th:
        return imgs

    word_bboxes = np.array(word_bboxes, np.int32)

    #### 1 for table completeness #####
    #### 0 for table resolution #####


    if h > th and w > tw:#random.random() > 0.6 and len(word_bboxes) > 0:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]
        left = max(sample_bboxes[1, 0] - img_size[0], 0)
        top = max(sample_bboxes[1, 1] - img_size[0], 0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    elif h > th:
        i = random.randint(0, h - th)
        j = 0
        crop_h, crop_w = th, w
    elif w > tw:
        i = 0
        j = random.randint(0, w - tw)
        crop_h, crop_w = h, tw
    else:
        i,j = 0,0
        crop_h, crop_w = h, w

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw
        #print (i,crop_h,j,crop_w)
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w >= tw or crop_h >= th:
            imgs[idx] = padding_image(imgs[idx], tw)
        elif crop_w < tw and crop_h < th:
            imgs[idx] = padding_image_(imgs[idx], tw)

    return imgs


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    if random.random() < 0.4:
        max_angle = 5
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = imgs[i]
            h, w = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = img_rotation
    return imgs

def random_uncolor(imgs):
    if random.random() <= 0.3:

        img = imgs[0]
        #print (img)
        h, w = img.shape[:2]
        if len(img.shape)>2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = img
        uncolor_img = np.zeros(gray_img.shape, np.uint8)
        for i in range(w):
            for j in range(h):
                uncolor_img[i,j]=255-gray_img[i,j]
        if len(img.shape)>2:
            imgs[0] = cv2.cvtColor(uncolor_img, cv2.COLOR_GRAY2RGB)
        else:
            imgs[0] = uncolor_img

    return imgs


def check_overlap(r1,r2):
    [[x1,y1],_,[x2,y2],_]=r1
    [[X1,Y1],_,[X2,Y2],_]=r2
    x1=int(x1)
    y1=int(y1)
    width1=int(x2-x1)
    height1=int(y2-y1)

    X1=int(X1)
    Y1=int(Y1)
    width2=int(X2-X1)
    height2=int(Y2-Y1)

    endx = max(x1+width1,X1+width2)
    startx = min(x1,X1)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,Y1+height2)
    starty = min(y1,Y1)
    height = height1+height2-(endy-starty)

    if width>0.3*min(width1,width2) and height>0.4*(min(height1,height2)):
        return True
    else:
        return False

def box_size(r1):
    [[x1,y1],_,[x2,y2],_]=r1
    width=int(x2-x1)
    height=int(y2-y1)
    return width*height



class craft_base_dataset(data.Dataset):
    def __init__(self, target_size=768, viz=False, debug=False):
        self.target_size = target_size
        self.viz = viz
        self.debug = debug
        self.gaussianTransformer = GaussianTransformer(imgSize=1024, region_threshold=0.35, affinity_threshold=0.15)

    def load_image_gt_and_confidencemask(self, index):
        '''
        根据索引值返回图像、字符框、文字行内容、confidence mask
        :param index:
        :return:
        '''
        return None, None, None, None, None

    def crop_image_by_bbox(self, image, box):
        w = (int)(np.linalg.norm(box[0] - box[1]))
        h = (int)(np.linalg.norm(box[0] - box[3]))
        width = w
        height = h
        if h > w * 1.5:
        #if 0:
            width = h
            height = w
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
        else:
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

        warped = cv2.warpPerspective(image, M, (width, height))
        return warped, M

    def get_confidence(self, real_len, pursedo_len):
        #print (real_len, pursedo_len)
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

    # def inference_pursedo_bboxes(self, net, image, word_bbox, word, viz=False, conf_threshold = 0.5):

    #     word_image, MM = self.crop_image_by_bbox(image, word_bbox)

    #     real_word_without_space = word.replace('\s', '')
    #     real_char_nums = len(real_word_without_space)
    #     input = word_image.copy()
    #     scale = 64.0 / input.shape[0]
    #     input = cv2.resize(input, None, fx=scale, fy=scale)

    #     #cv2.imwrite('./temp/crop_' + str(random.random()) + '.jpg', input)

    #     img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
    #                                                                variance=(0.229, 0.224, 0.225)))
    #     img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
    #     img_torch = img_torch.type(torch.FloatTensor).cuda()
    #     scores, _ = net(img_torch)
    #     region_scores = scores[0, :, :, 0].cpu().data.numpy()
    #     region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
    #     bgr_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
    #     bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2BGR)
    #     #print ('input',input)
    #     pursedo_bboxes = watershed(input, bgr_region_scores, False)

    #     _tmp = []
    #     for i in range(pursedo_bboxes.shape[0]):
    #         if np.mean(pursedo_bboxes[i].ravel()) > 2:
    #             _tmp.append(pursedo_bboxes[i])
    #         else:
    #             print("filter bboxes", pursedo_bboxes[i])
    #     pursedo_bboxes = np.array(_tmp, np.float32)
    #     if pursedo_bboxes.shape[0] > 1:
    #         index = np.argsort(pursedo_bboxes[:, 0, 0])
    #         pursedo_bboxes = pursedo_bboxes[index]

    #     confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))

    #     bboxes = []

    #     ### for box prediction
    #     if confidence  <= conf_threshold:
    #         width = input.shape[1]
    #         height = input.shape[0]

    #         width_per_char = width / len(word)
    #         for i, char in enumerate(word):
    #             if char == ' ':
    #                 continue
    #             left = i * width_per_char
    #             right = (i + 1) * width_per_char
    #             bbox = np.array([[left, 0], [right, 0], [right, height],
    #                              [left, height]])
    #             bboxes.append(bbox)

    #         bboxes = np.array(bboxes, np.float32)
    #         confidence = 0.5

    #     else:
    #         bboxes = pursedo_bboxes
    #     if False:
    #         _tmp_bboxes = np.int32(bboxes.copy())
    #         _tmp_bboxes[:, :, 0] = np.clip(_tmp_bboxes[:, :, 0], 0, input.shape[1])
    #         _tmp_bboxes[:, :, 1] = np.clip(_tmp_bboxes[:, :, 1], 0, input.shape[0])
    #         for bbox in _tmp_bboxes:
    #             cv2.polylines(np.uint8(input), [np.reshape(bbox, (-1, 1, 2))], True, (255, 0, 0))
    #         region_scores_color = cv2.applyColorMap(np.uint8(region_scores), cv2.COLORMAP_JET)
    #         region_scores_color = cv2.resize(region_scores_color, (input.shape[1], input.shape[0]))
    #         target = self.gaussianTransformer.generate_region(region_scores_color.shape, [_tmp_bboxes])
    #         target_color = cv2.applyColorMap(target, cv2.COLORMAP_JET)
    #         viz_image = np.hstack([input[:, :, ::-1], region_scores_color, target_color])
    #         cv2.imshow("crop_image", viz_image)
    #         cv2.waitKey()
    #     bboxes /= scale

    #     for j in range(len(bboxes)):
    #         ones = np.ones((4, 1))
    #         tmp = np.concatenate([bboxes[j], ones], axis=-1)
    #         I = np.matrix(MM).I
    #         ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
    #         bboxes[j] = ori[:, :2]

    #     bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
    #     bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)

    #     return bboxes, region_scores, confidence

#    def inference_pursedo_bboxes_detectwarp(self, net, image, word_bbox, word, viz=False, conf_threshold = 0.5):
#
#        word_image, MM = self.crop_image_by_bbox(image, word_bbox)
#
#        real_word_without_space = word.replace('\s', '')
#        real_char_nums = len(real_word_without_space)
#        input = word_image.copy()
#        scale = 64.0 / input.shape[0]
#        input = cv2.resize(input, None, fx=scale, fy=scale)
#
#        #cv2.imwrite('./temp/crop_' + str(random.random()) + '.jpg', input)
#
#        img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
#                                                                   variance=(0.229, 0.224, 0.225)))
#        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
#        img_torch = img_torch.type(torch.FloatTensor).cuda()
#        scores, _, _ = net(img_torch)
#        region_scores = scores[0, :, :, 0].cpu().data.numpy()
#        region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
#        bgr_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
#        bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2BGR)
#        #print ('input',input)
#        pursedo_bboxes = watershed(input, bgr_region_scores, False)
#
#        _tmp = []
#        for i in range(pursedo_bboxes.shape[0]):
#            if np.mean(pursedo_bboxes[i].ravel()) > 2:
#                _tmp.append(pursedo_bboxes[i])
#            else:
#                print("filter bboxes", pursedo_bboxes[i])
#        pursedo_bboxes = np.array(_tmp, np.float32)
#        if pursedo_bboxes.shape[0] > 1:
#            index = np.argsort(pursedo_bboxes[:, 0, 0])
#            pursedo_bboxes = pursedo_bboxes[index]
#
#        confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))
#
#        bboxes = []
#
#        ### for box prediction
#        if confidence  <= conf_threshold:
#            width = input.shape[1]
#            height = input.shape[0]
#
#            width_per_char = width / len(word)
#            for i, char in enumerate(word):
#                if char == ' ':
#                    continue
#                left = i * width_per_char
#                right = (i + 1) * width_per_char
#                bbox = np.array([[left, 0], [right, 0], [right, height],
#                                 [left, height]])
#                bboxes.append(bbox)
#
#            bboxes = np.array(bboxes, np.float32)
#            confidence = 0.5
#
#        else:
#            bboxes = pursedo_bboxes
#        if False:
#            _tmp_bboxes = np.int32(bboxes.copy())
#            _tmp_bboxes[:, :, 0] = np.clip(_tmp_bboxes[:, :, 0], 0, input.shape[1])
#            _tmp_bboxes[:, :, 1] = np.clip(_tmp_bboxes[:, :, 1], 0, input.shape[0])
#            for bbox in _tmp_bboxes:
#                cv2.polylines(np.uint8(input), [np.reshape(bbox, (-1, 1, 2))], True, (255, 0, 0))
#            region_scores_color = cv2.applyColorMap(np.uint8(region_scores), cv2.COLORMAP_JET)
#            region_scores_color = cv2.resize(region_scores_color, (input.shape[1], input.shape[0]))
#            target = self.gaussianTransformer.generate_region(region_scores_color.shape, [_tmp_bboxes])
#            target_color = cv2.applyColorMap(target, cv2.COLORMAP_JET)
#            viz_image = np.hstack([input[:, :, ::-1], region_scores_color, target_color])
#            cv2.imshow("crop_image", viz_image)
#            cv2.waitKey()
#        bboxes /= scale
#
#        for j in range(len(bboxes)):
#            ones = np.ones((4, 1))
#            tmp = np.concatenate([bboxes[j], ones], axis=-1)
#            I = np.matrix(MM).I
#            ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
#            bboxes[j] = ori[:, :2]
#
#        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
#        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)
#
#        return bboxes, region_scores, confidence

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def get_imagename(self, index):
        return None

    def saveInput(self, imagename, image, region_scores, affinity_scores, confidence_mask):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes, polys = craft_utils.getDetBoxes(region_scores / 255, affinity_scores / 255, 0.7, 0.4, 0.4, False)
        boxes = np.array(boxes, np.int32) * 2
        if len(boxes) > 0:
            np.clip(boxes[:, :, 0], 0, image.shape[1])
            np.clip(boxes[:, :, 1], 0, image.shape[0])
            for box in boxes:
                cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 0, 255))
        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        gt_scores = np.hstack([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color])
        confidence_mask_gray = np.hstack([np.zeros_like(confidence_mask_gray), confidence_mask_gray])
        output = np.concatenate([gt_scores, confidence_mask_gray],
                                axis=0)
        #print (image.shape, output.shape)
        output = np.hstack([image, output])
        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + './output'), "%s_input.jpg" % imagename)
        #print(outpath)
        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath,output )

    def saveImage(self, imagename, image, bboxes, affinity_bboxes, region_scores, affinity_scores, confidence_mask):
        output_image = np.uint8(image.copy())
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        #print (bboxes.shape)
        if len(bboxes) > 0:
            affinity_bboxes = np.int32(affinity_bboxes)
            for i in range(affinity_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
            for i in range(len(bboxes)):
                _bboxes = np.int32(bboxes[i])
                cv2.polylines(output_image, [np.reshape(_bboxes, (-1, 1, 2))], True, (0, 0, 255))
#                for j in range(_bboxes.shape[0]):
#                    #print (_bboxes[j])
#                    cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        heat_map = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=1)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        merge_map = target_gaussian_heatmap_color + target_gaussian_affinity_heatmap_color + image
        output = np.concatenate([output_image, heat_map, merge_map], axis=1)

        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + './output'), imagename)
        #print (imagename)

        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)


    def pull_mesh(self, index):
        # if self.get_imagename(index) == 'img_59.jpg':
        #     pass
        # else:
        #     return [], [], [], [], np.array([0])
        image, word_bboxes, directions, character_bboxes,  confidence_mask, confidences = self.load_image_gt_and_confidencemask(index)
        #print ('image',image.shape)
        #print ('conf',len(confidences))
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()
        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_bboxes = []
        #region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
        affinity_scores1, affinity_scores2, affinity_bboxes = self.gaussianTransformer.generate_mesh(region_scores.shape,
                                                                                          word_bboxes,directions)
        if self.viz:
            self.saveImage(self.get_imagename(index), image.copy(), character_bboxes, affinity_bboxes, affinity_scores1,
                           affinity_scores2,
                           confidence_mask)
        random_transforms = [image, region_scores, affinity_scores1,affinity_scores2, confidence_mask]
        random_transforms = randombox_crop(random_transforms, (self.target_size, self.target_size), word_bboxes)
        #random_transforms = random_horizontal_flip(random_transforms)
        #random_transforms = random_rotate(random_transforms)

        cvimage, region_scores, affinity_scores1,affinity_scores2 , confidence_mask = random_transforms

        h,w,_ = cvimage.shape

#        region_scores = cv2.resize(region_scores,(w//2,h//2))
#        affinity_scores1 = cv2.resize(affinity_scores1,(w//2,h//2),interpolation=cv2.INTER_NEAREST)
#        affinity_scores2 = cv2.resize(affinity_scores2,(w//2,h//2),interpolation=cv2.INTER_NEAREST)
#        confidence_mask = cv2.resize(confidence_mask,(w//2,ceil(h/2)))
        region_scores = self.resizeGt(region_scores)
        affinity_scores1 = self.resizeGt(affinity_scores1)
        affinity_scores2 = self.resizeGt(affinity_scores2)
        confidence_mask = self.resizeGt(confidence_mask)

        #print (self.get_imagename(index))
        if self.viz:
            self.saveInput(self.get_imagename(index), cvimage, affinity_scores1, affinity_scores2, confidence_mask)
        image = Image.fromarray(cvimage)
        image = image.convert('RGB')
        image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores / 255).float()
        affinity_scores1_torch = torch.from_numpy(affinity_scores1 / 255).float()
        affinity_scores2_torch = torch.from_numpy(affinity_scores2 / 255).float()
        confidence_mask_torch = torch.from_numpy(confidence_mask).float()
        return image, affinity_scores1_torch, affinity_scores2_torch, confidence_mask_torch, confidences



class TableMesh(craft_base_dataset):
    def __init__(self, receipt_folder, target_size=768, viz=False, debug=False):
        super(TableMesh, self).__init__(target_size, viz, debug)
        #self.net = net
        #self.net.eval()
        self.img_folder = os.path.join(receipt_folder, 'train_images')
        self.gt_folder = os.path.join(receipt_folder, 'train_gts')
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for i,imagename in enumerate(imagenames):
            if '_unline_' in imagename:#ignore unline tables
                continue
            self.images_path.append(imagename)
            #if i > 5000:
            #    break

    def __getitem__(self, index):
        #return self.pull_item(index)
        return self.pull_mesh(index)
        #return self.pull_cnt(index)

    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]

    def load_image_gt_and_confidencemask(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框，
        '''
        imagename = self.images_path[index]
        gt_path = os.path.join(self.gt_folder, "%s.txt" % os.path.splitext(imagename)[0])
        word_bboxes, words, directions = self.load_gt(gt_path)
        word_bboxes = np.float32(word_bboxes)
        words = np.float32(words)
        image_path = os.path.join(self.img_folder, imagename)
        #print (image_path)
        image = cv2.imread(image_path)
        #print ('sb')
        #image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),-1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #print ('image0',image.shape)

        image = random_scale_by_32(image, word_bboxes, self.target_size,words)
        #print ('image0',image.shape)
        confidence_mask = np.zeros((image.shape[0], image.shape[1]), np.float32)
        character_bboxes = []

        confidences = []
        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)):
                confidence = 0.5
                confidences.append(confidence)
                #character_bboxes.append(word)
            #horizontal first , vertical then, to avoid cross point adding
            if words!=[]:
                word_distances = []
                word_center_ys = []
                for i in range(len(words)):
                    word = np.array(words[i])
                    wxmin = word[:, 0].min()
                    wxmax = word[:, 0].max()
                    wymin = word[:, 1].min()
                    wymax = word[:, 1].max()
                    w_x = wxmax - wxmin + 1
                    w_y = wymax - wymin + 1
                    word_distances.append(min(w_x,w_y))
                    word_center_ys.append((wymin+wymax)/2)
                min_distance = np.mean(word_distances)

                hori_ids = []
                vert_ids = []
                neighbor_words = []
                for i in range(len(word_bboxes)):
                    xmin = word_bboxes[i][:, 0].min()
                    xmax = word_bboxes[i][:, 0].max()
                    ymin = word_bboxes[i][:, 1].min()
                    ymax = word_bboxes[i][:, 1].max()
                    width = xmax - xmin + 1
                    height = ymax - ymin + 1
                    if width > height:
                        hori_ids.append(i)
                        #find neighbor_words
                        this_neighbor_words = []
                        y_center = (ymin+ymax)/2
                        for j,word_center_y in enumerate(word_center_ys):
                            if abs(word_center_y - y_center) < 2*min_distance:
                                this_neighbor_words.append(words[j])
                        #print (len(this_neighbor_words))

                        if len(this_neighbor_words)==0:
                            this_neighbor_words = words

                        neighbor_words.append(this_neighbor_words)


                    else:
                        vert_ids.append(i)

                for i,j in enumerate(hori_ids):
                    confidence_mask = self.gaussianTransformer.add_mesh_confidence(confidence_mask,np.int32(word_bboxes[j]),neighbor_words[i],min_distance)
                for i in vert_ids:
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0.5))


        #for i in range(image.shape[0]):
        #    for j in range(image.shape[1]):
        #        if confidence_mask[i,j] == 0:
        #            confidence_mask[i,j] = 1


        return image, word_bboxes, directions, character_bboxes, confidence_mask, confidences


    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        directions = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(len(ori_box)-1)]
            box = np.array(box, np.int32).reshape(-1, 2)
            if ori_box[-1] != 'text' and ori_box[-1] != 'cell':
            #if ori_box[-1] =='|||':#hori only or vertical only
                bboxes.append(box)
                if ori_box[-1] == '---':
                    directions.append('hori')
                elif ori_box[-1] == '|||':
                    directions.append('vertical')
                else:
                    rect = cv2.minAreaRect(box)
                    rect_w = rect[1][0]
                    rect_h = rect[1][1]
                    rect_angle = rect[2]
                    #print (rect_angle)
                    if (rect_w < rect_h and rect_angle > -45) or (rect_w > rect_h and rect_angle < -45):
                        directions.append('vertical')
                    else:
                        directions.append('hori')

            elif ori_box[-1] == 'text':
                words.append(box)
        #print (len(bboxes),len(directions))
        return bboxes, words, directions









if __name__ == '__main__':
    # synthtextloader = Synth80k('/home/jiachx/publicdatasets/SynthText/SynthText', target_size=768, viz=True, debug=True)
    # train_loader = torch.utils.data.DataLoader(
    #     synthtextloader,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)
    # train_batch = iter(train_loader)
    # image_origin, target_gaussian_heatmap, target_gaussian_affinity_heatmap, mask = next(train_batch)
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    from craft import CRAFT
    #from dewarp import CRAFT_DEWARP
    from torchutil import copyStateDict

    net = CRAFT(freeze=True)
#    #net.load_state_dict(
#    #    copyStateDict(torch.load('./weights/CRAFT_clr_40.pth')))
#    net.load_state_dict(
#        copyStateDict(torch.load('./pretrain/91.pth')),strict=False)
#
#
#    net = net.cuda()
#    net = torch.nn.DataParallel(net)
#    net.eval()
    #dataloader = ICDAR2015(net, './data/IC15', target_size=768, viz=True)
    #dataloader = ReceiptBox(net, './data/all_aug', target_size=768,viz=True)
    #dataloader = Receipt(net, './data/shangchao', target_size=768,viz=True)
    #dataloader = Receipt(net, './data/all_aug', target_size=1024,viz=True)
    #dataloader = Pdf('./data/pdf1', target_size=2240,viz=True)
    #dataloader = TableMesh(net, './data/xhqd_tablecell', target_size=768,viz=True)
    #dataloader = TableMesh(net, './data/ICDAR2013_table', target_size=512,viz=True)
    #dataloader = TableSplerge('./data/PubTabNet', target_size=768,viz=True)
    #dataloader = TableSplerge('./data/TableSplerge_line', target_size=768,viz=True, line_flag=True, char_line=False)
    #dataloader = TableSplerge('./data/WTW', target_size=768,viz=True, line_flag=True, char_line=False)
    dataloader = TableMesh('./data/TableSplerge_line', target_size=768,viz=True)
    #dataloader = ReceiptWarp(net, './data/KJDZ0003_warp', target_size=1024,viz=True)
    #dataloader = PdfDetectWarp(net, './data/pdf1', target_size=1024,viz=True)
    #dataloader = TableMesh(net, './data/receipts_cnt', target_size=768,viz=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    total = 0
    total_sum = 0
    MS= [[] for i in range(8)]
    for index, (opimage,_,_,_,_) in enumerate(train_loader):
        pass
#        total += 1
#        # confidence_mean = confidences_mean.mean()
#        # total_sum += confidence_mean
#        # print(index, confidence_mean)
#        offset4p = offset4p.tolist()[0]
#        print (offset4p)
#        for i in range(8):
#            MS[i].append(offset4p[i])
#    #print("mean=", total_sum / total)
#    for i in range(8):
#        hist, bin_edges = np.histogram(MS[i])
#        print (hist)
#        print (bin_edges)


