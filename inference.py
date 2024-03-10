import numpy as np
from matplotlib import pyplot as plt

from utils_ import swap_axes


def show_image(normal, top, m):
    top_layer = swap_axes(top.cpu().detach().numpy()[0])
    m = swap_axes(m.cpu().detach().numpy()[0])

    from_ = swap_axes(normal.cpu().detach().numpy()[0])
    gen_img = from_ * (1 - m) + top_layer * m
    # plt.imshow(((gen_img + 1) * 127.5).astype(np.uint8))
    plt.imsave("./official_visual.jpg", ((gen_img + 1) * 127.5).astype(np.uint8))
