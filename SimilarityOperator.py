import numpy as np

import RectOperator as rto

def calc_colour_similarity(rg1, rg2):
    colour_hist_min = np.minimum(rg1.colour_hist, rg2.colour_hist)
    return np.sum(colour_hist_min)

def calc_texture_similarity(rg1, rg2):
    texture_hist_min = np.minimum(rg1.texture_hist, rg2.texture_hist)
    return np.sum(texture_hist_min)

def calc_size_similarity(rg1, rg2, im_size):
    return 1.0 - ((rg1.size + rg2.size) / im_size)

def calc_fill_similarity(rg1, rg2, im_size):
    rect = rto.Rect(rg1.rect.left, rg1.rect.top, rg1.rect.right, rg1.rect.bottom)
    rect.expand(rg2.rect.left, rg2.rect.top)
    rect.expand(rg2.rect.right, rg2.rect.bottom)
    rect_area = rect.get_area()
    return 1.0 - ((rect_area  - (rg1.size + rg2.size)) / im_size)

def calc_similarity(rg1, rg2, im_size):
    colour_similarity = calc_colour_similarity(rg1, rg2)
    texture_similarity = calc_texture_similarity(rg1, rg2)
    size_similarity = calc_size_similarity(rg1, rg2, im_size)
    fill_similarity = calc_fill_similarity(rg1, rg2, im_size)
    return colour_similarity + texture_similarity + size_similarity + fill_similarity

def calc_init_similarity(rgset, nbset, im_size):
    simset = []

    for a, b in nbset:
        simset.append((calc_similarity(rgset[a], rgset[b], im_size), a, b))

    return simset