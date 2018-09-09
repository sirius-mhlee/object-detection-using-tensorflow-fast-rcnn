import cv2

import numpy as np

import RectOperator as rto

class Region:
    def __init__(self, label, x, y, point_color, point_angle):
        self.label = label
        self.rect = rto.Rect(x, y, x, y)
        self.size = 1
        self.point_color = point_color
        self.point_angle = point_angle
        self.colour_hist = np.array([])
        self.texture_hist = np.array([])

    def add_point(self, x, y, point_color, point_angle):
        self.rect.expand(x, y)
        self.size += 1
        self.point_color = np.vstack((self.point_color, point_color))
        self.point_angle = np.vstack((self.point_angle, point_angle))

    def calc_colour_hist(self):
        BINS = 25
        RANGES = [(0.0, 180.0), (0.0, 255.0), (0.0, 255.0)]
        point_color_num, channel = self.point_color.shape

        colour_hist = np.array([])

        for c in range(channel):
            hist = np.histogram(self.point_color[:, c], BINS, RANGES[c])[0]
            colour_hist = np.hstack((colour_hist, hist))

        colour_hist_l1norm = np.linalg.norm(colour_hist, 1)

        self.colour_hist = colour_hist / colour_hist_l1norm

    def calc_texture_hist(self):
        BINS = 10
        RANGES = [(0.0, 180.0), (0.0, 255.0), (0.0, 255.0)]
        DIR_CONVERT_NUM = 22.5
        ORIENTATION = 8
        point_angle_num, channel = self.point_angle.shape

        angle_idx = (self.point_angle // DIR_CONVERT_NUM) % ORIENTATION
        
        angle_hist = []
        for o in range(ORIENTATION):
            angle_hist.append([])
            for c in range(channel):
                angle_hist[o].append([])

        for n in range(point_angle_num):
            for c in range(channel):
                idx = int(angle_idx[n, c])
                angle_hist[idx][c].append(self.point_color[n, c])

        texture_hist = np.array([])

        for o in range(ORIENTATION):
            for c in range(channel):
                hist = np.histogram(angle_hist[o][c], BINS, RANGES[c])[0]
                texture_hist = np.hstack((texture_hist, hist))

        texture_hist_l1norm = np.linalg.norm(texture_hist, 1)

        self.texture_hist = texture_hist / texture_hist_l1norm

def find_same_label_region(rgset, label):
    for idx in range(len(rgset)):
        if rgset[idx].label == label:
            return idx

    return -1

def has_same_rect_region(rgset, rect):
    for rg in rgset:
        if rto.is_same(rg.rect, rect):
            return True

    return False

def merge_region(rg1, rg2):
    new_label = rg1.label + rg2.label
    new_rect = rto.Rect(rg1.rect.left, rg1.rect.top, rg1.rect.right, rg1.rect.bottom)
    new_rect.expand(rg2.rect.left, rg2.rect.top)
    new_rect.expand(rg2.rect.right, rg2.rect.bottom)
    new_size = rg1.size + rg2.size
    new_point_color = np.vstack((rg1.point_color, rg2.point_color))
    new_point_angle = np.vstack((rg1.point_angle, rg2.point_angle))
    new_colour_hist = (rg1.colour_hist * rg1.size + rg2.colour_hist * rg2.size) / new_size
    new_texture_hist = (rg1.texture_hist * rg1.size + rg2.texture_hist * rg2.size) / new_size

    new_region = Region(new_label, new_rect.left, new_rect.right, new_point_color, new_point_angle)
    new_region.rect = new_rect
    new_region.size = new_size
    new_region.colour_hist = new_colour_hist
    new_region.texture_hist = new_texture_hist
    return new_region

def extract_region(img, ufset):
    height, width, channel = img.shape

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    gradient_magnitude, gradient_angle = cv2.cartToPolar(sobel_x, sobel_y, None, None, True)

    rgset = []

    for y in range(height):
        for x in range(width):
            label = ufset.find(y * width + x)
            idx = find_same_label_region(rgset, label)
            if idx == -1:
                rgset.append(Region(label, x, y, hsv_img[y, x, :], gradient_angle[y, x, :]))
            else:
                rgset[idx].add_point(x, y, hsv_img[y, x, :], gradient_angle[y, x, :])

    for rg in rgset:
        rg.calc_colour_hist()
        rg.calc_texture_hist()

    return rgset

def extract_neighbour(rgset):
    nbset = []

    for i in range(len(rgset)):
        for j in range(i + 1, len(rgset)):
            if rto.is_intersect(rgset[i].rect, rgset[j].rect):
                nbset.append((i, j))

    return nbset