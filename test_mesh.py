# -*- coding: utf-8 -*-
#import sys
import os
import time
import argparse
#import onnx
#import onnxruntime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#from PIL import Image

import cv2
#from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
#import json
#import zipfile
#from craft import CRAFT
#from refinenet import RefineNet
from dtgs import DTGS_1
#import random,copy

from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/ppm1_line_2022070412.pth', type=str, help='pretrained model')
#parser.add_argument('--craft_model', default='weights/CRAFT_2021042928.pth', type=str, help='pretrained refiner model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.8, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.6, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=3000, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
#parser.add_argument('--test_folder', default='data/WTW/test_images', type=str, help='folder path to input images')
parser.add_argument('--test_folder', default='data/temp', type=str, help='folder path to input images')
#parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
#parser.add_argument('--refiner_model', default='weights/refine_box_2021050127.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net0, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 2048, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()


    # forward pass
    with torch.no_grad():
        y, feature = net0(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    with torch.no_grad():
        y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    #print (boxes)
    #print ('-'*100)
    #print (polys)
    #print ('*'*100)

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    Polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(Polys)):
        if Polys[k] is None: Polys[k] = boxes[k]

    #compute avg text height
    text_hs = []
    for r in boxes:
        [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] = r
        if y3 == y2:
            continue
        text_hs.append(y3-y2)

    if len(text_hs)>0:
        avg_text_h = np.mean(text_hs)
    else:
        avg_text_h = 25


    # resize image according to text_h
    mag_ratio = 30 / avg_text_h
    #mag_ratio = max(0.8, 30 / avg_text_h)
    print ('mag_ratio',mag_ratio)


    #x.to('cpu')

    #-------------------------------------------------------------

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if 1:#cuda:
        x = x.cuda()

    # forward pass
    # onnx forward pass
    #ort_session = onnxruntime.InferenceSession("tablemesh2.onnx")
    #onnx_runtime_input = x.detach().numpy()
    #ort_inputs = {ort_session.get_inputs()[0].name: onnx_runtime_input}

    t0 = time.time()
    with torch.no_grad():
        y, feature = net(x)


    #y1, feature = ort_session.run(None, ort_inputs)

    print ('mesh net time:', time.time()-t0)

    t0 = time.time()
    # make score and link map

    #score_h = y1[0,:,:,0].copy()
    #score_v = y1[0,:,:,1].copy()


    score_h = y[0,:,:,0].cpu().data.numpy()
    score_v = y[0,:,:,1].cpu().data.numpy()
    #print (score_h.shape)


    #score_link += score_text
    # Post-processing

    h_lines, up_h_threshold = craft_utils.getMeshLines(score_h, 0.3)
    v_lines, up_v_threshold = craft_utils.getMeshLines(score_v, 0.3)
    print ('mesh  postprocess0 time:', time.time()-t0)

    #score_h0, score_v0 = copy.copy(score_h), copy.copy(score_v)

    h_lines, v_lines, h_lines_id, v_lines_id = craft_utils.lines_sort_merge(h_lines, v_lines)
    print ('mesh  postprocess1 time:', time.time()-t0)

    score_h, score_v = craft_utils.lines_connect_close(h_lines, v_lines, h_lines_id, v_lines_id, score_h, score_v)
    print ('mesh  postprocess2 time:', time.time()-t0)

    ## re get hv_lines based on new score_h, score_v
    h_lines,_ = craft_utils.getMeshLines(score_h, 0.3, up_h_threshold)
    v_lines,_ = craft_utils.getMeshLines(score_v, 0.3, up_v_threshold)
    print ('mesh  postprocess3 time:', time.time()-t0)
    #score_h0, score_v0 = copy.copy(score_h), copy.copy(score_v)

    h_lines, v_lines, h_lines_id, v_lines_id = craft_utils.lines_sort_merge(h_lines, v_lines)

    print ('mesh  postprocess4 time:', time.time()-t0)



    boxes,polys = craft_utils.getMeshBoxes_new(score_h+score_v, 0.3)
    #boxes,polys = craft_utils.getMeshBoxes(score_h+score_v, 0.3)
    print ('mesh  postprocess5 time:', time.time()-t0)

    #print (h_lines)
    #print ('='*100)
    #h_lines = craft_utils.adjustResultCoordinates(h_lines, ratio_w, ratio_h)
    #v_lines = craft_utils.adjustResultCoordinates(v_lines, ratio_w, ratio_h)
    #polys = craft_utils.adjustResultCoordinates(polys, target_ratio, target_ratio, ratio_net = 0.5)
    print (len(h_lines),len(v_lines))
    cells_info = None
    cells_info = craft_utils.predict_cells_struct_on_mesh(polys,h_lines,v_lines,h_lines_id, v_lines_id)

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    print ('mesh  postprocess6 time:', time.time()-t0)


    output = imgproc.cvt2HeatmapImg(score_h+score_v)
    render_img = output


    #if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return render_img,  cells_info, boxes,polys, Polys



if __name__ == '__main__':

    # """
    # text detection phase
    # """
    # # load net
    # net0 = CRAFT()     # initialize

    # print('Loading weights from checkpoint (' + args.craft_model + ')')
    # if args.cuda:
    #     net0.load_state_dict(copyStateDict(torch.load(args.craft_model)))
    # else:
    #     net0.load_state_dict(copyStateDict(torch.load(args.craft_model, map_location='cpu')))

    # if args.cuda:
    #     net0 = net0.cuda()
    #     net0 = torch.nn.DataParallel(net0)
    #     cudnn.benchmark = False

    # net0.eval()

    # # LinkRefiner
    # refine_net = RefineNet()
    # print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    # if args.cuda:
    #     refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
    #     refine_net = refine_net.cuda()
    #     refine_net = torch.nn.DataParallel(refine_net)
    # else:
    #     refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    # refine_net.eval()
    # args.poly = True

    """
    table mesh phase
    """
    # load net
    net = DTGS_1(direction=0)     # initialize

    print('Loading table mesh weights from checkpoint (' + args.trained_model + ')')
    if 1:#args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if 1:#args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    t = time.time()
    #args.poly = False
    # load data
    for k, image_path in enumerate(image_list):
        #if random.random() > 0.1:
        #    continue
        #if 'O1CN01LNvVLP1MxxSPAggUc_!!6000000001502-0-lxb_0' not in image_path:
        #    continue
        #if '21a3eb706add0474b42f8403d454ad71_0' not in image_path:
        #    continue
        #if 'mit_google_image_search-10918758-c50123706ab410e65cb2bcc25b226e23bb5e86d1_0' not in image_path:
        #    continue
        #if 'customs-declaration-06742_0' not in image_path:
        #    continue
        #if 'caiwu_79' not in image_path:
        #    continue


        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path))
        image = imgproc.loadImage(image_path)

        render_img, cells_info,  boxes, cells, Polys = test_net(net0, net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        #filename = ''.join(list(filter(str.isalnum, filename)))
        mask_file = result_folder + filename + '_mask.jpg'
        cv2.imwrite(mask_file, render_img)
        print ('Polys',len(Polys))
        if len(Polys)==0:
            continue



        ## cell struct(deprecated)
        #from predict_cells_struct import p_main
        #main = p_main()
        #cells_info = main.predict(boxes)

        ##vis
        # output cell_info.txt
        # format: row_idx, row_idx_end, col_idx, col_idx_end, poly coords
        #result_txt = open(result_folder + filename +'_cell.txt','w')

        for poly in Polys:
            poly = np.int32(poly)
            poly.reshape((-1,1,2))
            cv2.polylines(image, [poly], True, (0, 255, 0),2)

        for cell_info,box,poly in zip(cells_info,boxes, cells):
            poly = np.int32(poly)
            poly.reshape((-1,1,2))
            #print (box)


            #print (cell_info)
            try:
                row_idx = cell_info['row_id_start']
                row_idx_end = cell_info['row_id_end']
                col_idx = cell_info['col_id_start']
                col_idx_end = cell_info['col_id_end']
                if -1 in [row_idx,row_idx_end,col_idx,col_idx_end]:
                    continue
                rowcol_id = ','.join([str(row_idx),str(row_idx_end),str(col_idx),str(col_idx_end)])
                [x1,y1] = box[0]
                cv2.putText(image, rowcol_id , (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                #poly = [x1,y1,x2,y2,x3,y3,x4,y4]
                #poly = [str(int(x)) for x in poly]
                #result_txt.write(rowcol_id + ',' + ','.join(poly) + '\n')
            except:
                pass
            cv2.polylines(image, [poly], True, (255, 0, 0),2)


        mask_file = result_folder + filename + '_box.jpg'
        cv2.imwrite(mask_file, image)
        if k >=200:
            break


    print("elapsed time : {}s".format(time.time() - t))
