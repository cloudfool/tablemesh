from math import exp
import numpy as np
import cv2,copy
import os
import imgproc
from shapely.geometry import Polygon
import pyclipper

def Distance( xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(
        xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(
        xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(
        point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
        (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 *
                     square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(
        square_distance_1, square_distance_2))[cosin < 0]
    # self.extend_line(point_1, point_2, result)
    return result


class GaussianTransformer(object):

    def __init__(self, imgSize=512, region_threshold=0.4,
                 affinity_threshold=0.2):
        distanceRatio = 3.34
        #distanceRatio = 3
        #print (distanceRatio)
        scaledGaussian = lambda x: exp(-(1 / 2) * (x ** 2))
        self.region_threshold = region_threshold
        self.imgSize = imgSize
        self.standardGaussianHeat = self._gen_gaussian_heatmap(imgSize, distanceRatio)
        self.boxGaussianHeat = self._gen_box_gaussian_heatmap(imgSize, distanceRatio)



        _, binary = cv2.threshold(self.standardGaussianHeat, region_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        # print("regionbox", self.regionbox)
        _, binary = cv2.threshold(self.standardGaussianHeat, affinity_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.affinitybox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        # print("affinitybox", self.affinitybox)
        _, binary = cv2.threshold(self.boxGaussianHeat, affinity_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.boxbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        self.oribox = np.array([[0, 0, 1], [imgSize - 1, 0, 1], [imgSize - 1, imgSize - 1, 1], [0, imgSize - 1, 1]],
                               dtype=np.int32)

    def _gen_gaussian_heatmap(self, imgSize, distanceRatio):
        scaledGaussian = lambda x: exp(-(1 / 2) * (x ** 2))
        heat = np.zeros((imgSize, imgSize), np.uint8)
        for i in range(imgSize):
            for j in range(imgSize):
                distanceFromCenter = np.linalg.norm(np.array([i - imgSize / 2, j - imgSize / 2]))
                distanceFromCenter = distanceRatio * distanceFromCenter / (imgSize / 2)
                scaledGaussianProb = scaledGaussian(distanceFromCenter)
                heat[i, j] = np.clip(scaledGaussianProb * 255, 0, 255)
        return heat

    def _gen_box_gaussian_heatmap(self, imgSize, distanceRatio):
        scaledGaussian = lambda x:1
        heat = np.zeros((imgSize, imgSize), np.uint8)
        for i in range(imgSize):
            for j in range(imgSize):
                distanceFromCenter = np.linalg.norm(np.array([i - imgSize / 2, j - imgSize / 2]))
                distanceFromCenter = distanceRatio * distanceFromCenter / (imgSize / 2)
                #distanceFromY = min(imgSize-j,j)/imgSize
                #distanceFromX = min(imgSize-i,i)/imgSize
                #print ('X',distanceFromX)
                scaledGaussianProb = scaledGaussian(distanceFromCenter)
                heat[i, j] = np.clip(scaledGaussianProb * 255, 0, 255)
        return heat



    def _test(self):
        sigma = 10
        spread = 3
        extent = int(spread * sigma)
        center = spread * sigma / 2
        gaussian_heatmap = np.zeros([extent, extent], dtype=np.float32)

        for i_ in range(extent):
            for j_ in range(extent):
                gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))

        gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        threshhold_guassian = cv2.applyColorMap(gaussian_heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'test_guassian.jpg'), threshhold_guassian)

    def add_region_character(self, image, target_bbox, regionbox=None):

        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True

        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        oribox = np.array(
            [[[0, 0], [self.imgSize - 1, 0], [self.imgSize - 1, self.imgSize - 1], [0, self.imgSize - 1]]],
            dtype=np.float32)
        test1 = cv2.perspectiveTransform(np.array([regionbox], np.float32), M)[0]
        real_target_box = cv2.perspectiveTransform(oribox, M)[0]
        # print("test\ntarget_bbox", target_bbox, "\ntest1", test1, "\nreal_target_box", real_target_box)
        real_target_box = np.int32(real_target_box)
        real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
        real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

        # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
        # warped = np.array(warped, np.uint8)
        # image = np.where(warped > image, warped, image)
        if np.any(target_bbox[0] < real_target_box[0]) or (
                target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
                target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
                target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):
            # if False:
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
            warped = np.array(warped, np.uint8)
            image = np.where(warped > image, warped, image)
            # _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            # warped = np.array(warped, np.uint8)
            #
            # # if affi:
            # # print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # # cv2.imshow("1123", warped)
            # # cv2.waitKey()
            # image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
            #                                        image[ymin:ymax, xmin:xmax])
        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = target_bbox.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))

            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            warped = np.array(warped, np.uint8)
            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return image
            # if affi:
            # print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # cv2.imshow("1123", warped)
            # cv2.waitKey()
            image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
                                                   image[ymin:ymax, xmin:xmax])
        return image
    '''
    def add_box_region_character(self, image, target_bbox, regionbox=None):

        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True

        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        oribox = np.array(
            [[[0, 0], [self.imgSize - 1, 0], [self.imgSize - 1, self.imgSize - 1], [0, self.imgSize - 1]]],
            dtype=np.float32)
        test1 = cv2.perspectiveTransform(np.array([regionbox], np.float32), M)[0]
        real_target_box = cv2.perspectiveTransform(oribox, M)[0]
        # print("test\ntarget_bbox", target_bbox, "\ntest1", test1, "\nreal_target_box", real_target_box)
        real_target_box = np.int32(real_target_box)
        real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
        real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

        # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
        # warped = np.array(warped, np.uint8)
        # image = np.where(warped > image, warped, image)
        if np.any(target_bbox[0] < real_target_box[0]) or (
                target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
                target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
                target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):
            # if False:
            if affi:
                warped = cv2.warpPerspective(self.boxGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
            else:

                warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
            warped = np.array(warped, np.uint8)
            image = np.where(warped > image, warped, image)
            # _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            # warped = np.array(warped, np.uint8)
            #
            # # if affi:
            # # print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # # cv2.imshow("1123", warped)
            # # cv2.waitKey()
            # image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
            #                                        image[ymin:ymax, xmin:xmax])
        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = target_bbox.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            if affi:
                warped = cv2.warpPerspective(self.boxGaussianHeat.copy(), M, (width, height))
            else:
                warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            warped = np.array(warped, np.uint8)
            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return image
            # if affi:
            # print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # cv2.imshow("1123", warped)
            # cv2.waitKey()
            image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
                                                   image[ymin:ymax, xmin:xmax])
        return image
    '''



    def add_affinity_character(self, image, target_bbox):
        return self.add_region_character(image, target_bbox, self.affinitybox)


    def add_affinity(self, image, bbox_1, bbox_2):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

        affinity = np.array([tl, tr, br, bl])

        #print (tl,tr,br,bl)
        #ratio_v = (bl[1] - tl[1]) / (br[1] - tr[1])
        #ratio_h = (tr - tl) / (br - bl)
        #if ratio_v >= 4 or ratio_v <= 0.25:
        #    return image, None



        return self.add_affinity_character(image, affinity.copy()), np.expand_dims(affinity, axis=0)

    def add_box(self, canvas, polygon):
        shrink_ratio = 0.4

        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        #print (polygon)

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(0)[0])
        #padded_polygon = np.array(padding.Execute(distance)[0])
        #cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = Distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
             distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

        return canvas

    def add_mesh(self, canvas, polygon,direction):
        shrink_ratio = 0.4
        gaussian = True
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        #print (polygon)

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)

        padded_polygon = np.array(padding.Execute(0)[0])
        #padded_polygon = np.array(padding.Execute(distance)[0])
        #cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        #print (xmin,xmax,ymin,ymax)

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            ##check direction of polygon edge
            #hori polygon consider hori direction edge; vertical polygon consider vertical direction edge
            x1,y1 = polygon[i][0],polygon[i][1]
            x2,y2 = polygon[j][0],polygon[j][1]
            if gaussian:
                if (direction == 'hori' and x1 == x2) or (direction == 'vertical' and y1 == y2):
                    distance_map[i] = np.clip(1, 0, 1)
                    continue
                absolute_distance = Distance(xs, ys, polygon[i], polygon[j])
                if direction == 'hori':
                    distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
                elif direction == 'vertical':
                    distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
            else:
                distance_map[i] = np.clip(1, 0, 1)

        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 2)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 2)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 2)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 2)
        #print (xmin_valid,xmax_valid,ymin_valid,ymax_valid)

        this_canvas = np.zeros(np.shape(canvas),dtype=np.float32)
        this_canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            this_canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

        mask = np.zeros(np.shape(canvas),dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(padded_polygon)], 255)
        this_canvas = cv2.add(this_canvas,np.zeros(np.shape(canvas),dtype=np.float32),mask = mask)

        canvas = cv2.add(canvas,this_canvas)
        return canvas

    def add_mesh_confidence(self, canvas,polygon,words,min_distance):

        shrink_ratio = 0.4

        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        #print (polygon)
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(0)[0])
        #padded_polygon = np.array(padding.Execute(distance)[0])
        #cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (len(words)*4, height, width), dtype=np.float32)
        #word_distances = []

        for i in range(len(words)):
            word = np.array(words[i])
            word[:, 0] = word[:, 0] - xmin
            word[:, 1] = word[:, 1] - ymin

            for j in range(4):
                k =  (j + 1) % 4
                absolute_distance = Distance(xs, ys, word[j], word[k])
                #min_distance = absolute_distance.min()
                #max_distance = absolute_distance.max()
                #print (min_distance ,max_distance)
                distance_map[4*i+j] = absolute_distance

                #distance_map[4*i+j] = np.clip(absolute_distance / (2*min_distance), 0.5, 1)
        distance_map = distance_map.min(axis=0)
        #min_distance = np.mean(word_distances)
        #min_distance = distance_map.min()
        #print ('min-distance',min_distance)

        distance_map = np.clip(distance_map / (4*min_distance), 0.5, 1)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        this_canvas = np.zeros(np.shape(canvas),dtype=np.float32)
        this_canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
             distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            this_canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

        mask = np.zeros(np.shape(canvas),dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(padded_polygon)], 1)
        this_canvas = cv2.add(this_canvas,np.zeros(np.shape(canvas),dtype=np.float32),mask = mask)

        canvas = cv2.add(canvas,this_canvas)
        return canvas


    def generate_region(self, image_size, bboxes):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        for i in range(len(bboxes)):
            character_bbox = np.array(bboxes[i].copy())
            for j in range(bboxes[i].shape[0]):
                target = self.add_region_character(target, character_bbox[j])

        return target

    def generate_box(self, image_size, bboxes, words):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.float32)
        for i in range(len(words)):
            word_bbox = np.array(bboxes[i])
            target = self.add_box(target, word_bbox)
#        for i in range(height):
#            for j in range(width):
#                target[i,j] = np.clip(target[i,j] * 255, 0, 255)
        target = np.clip(target * 255, 0, 255)
        target.astype(np.uint8)
        return target, []

    def generate_mesh(self, image_size, bboxes,directions):

        height, width = image_size[0], image_size[1]
        #print ('canvas',height,width)
        target1 = np.zeros([height, width], dtype=np.float32)
        target2 = np.zeros([height, width], dtype=np.float32)
        for i in range(len(bboxes)):
            word_bbox = np.array(bboxes[i])
            #w = word_bbox[1][0] - word_bbox[0][0]
            #h = word_bbox[2][1] - word_bbox[0][1]
            #if min(w,h)<4:
            #    print ('error')
            #print (word_bbox)
            if directions[i] == 'hori':
                try:
                    target1 = self.add_mesh(target1, word_bbox,directions[i])
                except:
                    ...
            if directions[i] == 'vertical':
                try:
                    target2 = self.add_mesh(target2, word_bbox,directions[i])
                except:
                    ...
        #for i in range(height):
        #    for j in range(width):
        target1 = np.clip(target1 * 255, 0, 255)
        target2 = np.clip(target2 * 255, 0, 255)
        #target[i,j] = 255 - target[i,j]
        target1.astype(np.uint8)
        target2.astype(np.uint8)

        return target1,target2, []


    def generate_affinity(self, image_size, bboxes, words):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        affinities = []
        for i in range(len(words)):
            character_bbox = np.array(bboxes[i])
            total_letters = 0
            for char_num in range(character_bbox.shape[0] - 1):
                target, affinity = self.add_affinity(target, character_bbox[total_letters],
                                                     character_bbox[total_letters + 1])
                #if affinity is None:
                #    continue

                affinities.append(affinity)
                total_letters += 1
        if len(affinities) > 0:
            affinities = np.concatenate(affinities, axis=0)
        return target, affinities

    def saveGaussianHeat(self):
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
        warped_color = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
        cv2.polylines(warped_color, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
        standardGaussianHeat1 = self.standardGaussianHeat.copy()
        threshhold = self.region_threshold * 255
        standardGaussianHeat1[standardGaussianHeat1 > 0] = 255
        threshhold_guassian = cv2.applyColorMap(standardGaussianHeat1, cv2.COLORMAP_JET)
        cv2.polylines(threshhold_guassian, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)


if __name__ == '__main__':
    gaussian = GaussianTransformer(512, 0.4, 0.2)
    gaussian.saveGaussianHeat()
    gaussian._test()
    bbox0 = np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]])
    image = np.zeros((500, 500), np.uint8)
    # image = gaussian.add_region_character(image, bbox)
    bbox1 = np.array([[[100, 0], [200, 0], [200, 100], [100, 100]]])
    bbox2 = np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]])
    bbox3 = np.array([[[0, 100], [100, 100], [100, 200], [0, 200]]])

    bbox4 = np.array([[[96, 0], [151, 9], [139, 64], [83, 58]]])
    # image = gaussian.add_region_character(image, bbox)
    # print(image.max())
    image = gaussian.generate_region((500, 500, 1), [bbox4])
    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(image.copy() / 255)
    cv2.imshow("test", target_gaussian_heatmap_color)
    cv2.imwrite("test.jpg", target_gaussian_heatmap_color)
    cv2.waitKey()
    # weight, target = gaussian.generate_target((1024, 1024, 3), bbox.copy())
    # target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(weight.copy() / 255)
    # cv2.imshow('test', target_gaussian_heatmap_color)
    # cv2.waitKey()
    # cv2.imwrite("test.jpg", target_gaussian_heatmap_color)




# # coding=utf-8
# from math import exp
# import numpy as np
# import cv2
# import os
# import imgproc
#
#
# class GaussianTransformer(object):
#
#     def __init__(self, imgSize=512, distanceRatio=1.70):
#         scaledGaussian = lambda x: exp(-(1 / 2) * (x ** 2))
#
#         self.standardGaussianHeat = np.zeros((imgSize, imgSize), np.uint8)
#
#         for i in range(imgSize):
#             for j in range(imgSize):
#                 distanceFromCenter = np.linalg.norm(np.array([i - imgSize / 2, j - imgSize / 2]))
#                 distanceFromCenter = distanceRatio * distanceFromCenter / (imgSize / 2)
#                 scaledGaussianProb = scaledGaussian(distanceFromCenter)
#
#                 self.standardGaussianHeat[i, j] = np.clip(scaledGaussianProb * 255, 0, 255)
#         #print("gaussian heatmap min pixel is", self.standardGaussianHeat.min() / 255)
#         # self.standardGaussianHeat[self.standardGaussianHeat < (0.4 * 255)] = 255
#         self._test()
#
#     def _test(self):
#         sigma = 10
#         spread = 3
#         extent = int(spread * sigma)
#         center = spread * sigma / 2
#         gaussian_heatmap = np.zeros([extent, extent], dtype=np.float32)
#
#         for i_ in range(extent):
#             for j_ in range(extent):
#                 gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
#                     -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))
#
#         gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)
#         images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
#         threshhold_guassian = cv2.applyColorMap(gaussian_heatmap, cv2.COLORMAP_JET)
#         cv2.imwrite(os.path.join(images_folder, 'test_guassian.jpg'), threshhold_guassian)
#
#     def four_point_transform(self, target_bbox, save_dir=None):
#         '''
#
#         :param target_bbox:目标bbox
#         :param save_dir:如果不是None，则保存图片到save_dir中
#         :return:
#         '''
#         width, height = np.max(target_bbox[:, 0]).astype(np.int32), np.max(target_bbox[:, 1]).astype(np.int32)
#
#         right = self.standardGaussianHeat.shape[1] - 1
#         bottom = self.standardGaussianHeat.shape[0] - 1
#         ori = np.array([[0, 0], [right, 0],
#                         [right, bottom],
#                         [0, bottom]], dtype="float32")
#         M = cv2.getPerspectiveTransform(ori, target_bbox)
#         warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (width, height))
#         warped = np.array(warped, np.uint8)
#         if save_dir:
#             warped_color = cv2.applyColorMap(warped, cv2.COLORMAP_JET)
#             cv2.imwrite(os.path.join(save_dir, 'warped.jpg'), warped_color)
#         #print(warped.shape,(width, height))
#
#         return warped, width, height
#
#     def add_character(self, image, bbox):
#         if np.any(bbox < 0) or np.any(bbox[:, 0] > image.shape[1]) or np.any(bbox[:, 1] > image.shape[0]):
#             return image
#         top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
#         bbox -= top_left[None, :]
#         transformed, width, height = self.four_point_transform(bbox.astype(np.float32))
#         if width * height < 10:
#             return image
#
#         try:
#             score_map = image[top_left[1]:top_left[1] + transformed.shape[0],
#                         top_left[0]:top_left[0] + transformed.shape[1]]
#             score_map = np.where(transformed > score_map, transformed, score_map)
#             image[top_left[1]:top_left[1] + transformed.shape[0],
#             top_left[0]:top_left[0] + transformed.shape[1]] = score_map
#         except Exception as e:
#             print(e)
#         return image
#
#     def add_affinity(self, image, bbox_1, bbox_2):
#         center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
#         tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
#         bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
#         tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
#         br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)
#
#         affinity = np.array([tl, tr, br, bl])
#
#         return self.add_character(image, affinity.copy()), np.expand_dims(affinity, axis=0)
#
#     def generate_region(self, image_size, bboxes):
#         height, width, channel = image_size
#         target = np.zeros([height, width], dtype=np.uint8)
#         for i in range(len(bboxes)):
#             character_bbox = np.array(bboxes[i])
#             for j in range(bboxes[i].shape[0]):
#                 target = self.add_character(target, character_bbox[j])
#
#         return target
#
#     def saveGaussianHeat(self):
#         images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
#         cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
#         warped_color = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
#         cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
#         standardGaussianHeat1 = self.standardGaussianHeat.copy()
#         standardGaussianHeat1[standardGaussianHeat1 < (0.4 * 255)] = 255
#         threshhold_guassian = cv2.applyColorMap(standardGaussianHeat1, cv2.COLORMAP_JET)
#         cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)
#
#     def generate_affinity(self, image_size, bboxes, words):
#         height, width, channel = image_size
#
#         target = np.zeros([height, width], dtype=np.uint8)
#         affinities = []
#         for i in range(len(words)):
#             character_bbox = np.array(bboxes[i])
#             total_letters = 0
#             for char_num in range(character_bbox.shape[0] - 1):
#                 target, affinity = self.add_affinity(target, character_bbox[total_letters],
#                                                      character_bbox[total_letters + 1])
#                 affinities.append(affinity)
#                 total_letters += 1
#         if len(affinities) > 0:
#             affinities = np.concatenate(affinities, axis=0)
#         return target, affinities
#
#
# if __name__ == '__main__':
#     gaussian = GaussianTransformer(1024, 1.5)
#     gaussian.saveGaussianHeat()
#
#     bbox = np.array([[[1, 200], [510, 200], [510, 510], [1, 510]]])
#     print(bbox.shape)
#     bbox = bbox.transpose((2, 1, 0))
#     print(bbox.shape)
#     weight, target = gaussian.generate_target((1024, 1024, 3), bbox.copy())
#     target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(weight.copy() / 255)
#     cv2.imshow('test', target_gaussian_heatmap_color)
#     cv2.waitKey()
#     cv2.imwrite("test.jpg", target_gaussian_heatmap_color)



# coding=utf-8
# coding=utf-8
