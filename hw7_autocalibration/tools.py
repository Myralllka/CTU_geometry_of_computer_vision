# tools.py - helper functions that can be used in solutions of GVG homeworks
#
# (c) 2020-02-20 Martin Matousek
# Last change: $Date$
#              $Revision$

import matplotlib.pyplot as plt
import numpy as np


## p3p
def p3p_polynom(d12, d23, d31, c12, c23, c31):
    #  a0, a1, a2, a3, a4 = p3p_polynom( d12, d23, d31, c12, c23, c31 )

    a4 = -4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c23 ** 2 + d23 ** 8 - 2 * d23 ** 6 * d12 ** 2 - 2 * d23 ** 6 * d31 ** 2 + d23 ** 4 * d12 ** 4 + 2 * d23 ** 4 * d12 ** 2 * d31 ** 2 + d23 ** 4 * d31 ** 4

    a3 = 8 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c12 * c23 ** 2 + 4 * d23 ** 6 * d12 ** 2 * c31 * c23 - 4 * d23 ** 4 * d12 ** 4 * c31 * c23 + 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c31 * c23 - 4 * d23 ** 8 * c12 + 4 * d23 ** 6 * d12 ** 2 * c12 + 8 * d23 ** 6 * d31 ** 2 * c12 - 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c12 - 4 * d23 ** 4 * d31 ** 4 * c12

    a2 = -8 * d23 ** 6 * d12 ** 2 * c31 * c12 * c23 - 8 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c31 * c12 * c23 + 4 * d23 ** 8 * c12 ** 2 - 4 * d23 ** 6 * d12 ** 2 * c31 ** 2 - 8 * d23 ** 6 * d31 ** 2 * c12 ** 2 + 4 * d23 ** 4 * d12 ** 4 * c31 ** 2 + 4 * d23 ** 4 * d12 ** 4 * c23 ** 2 - 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c23 ** 2 + 4 * d23 ** 4 * d31 ** 4 * c12 ** 2 + 2 * d23 ** 8 - 4 * d23 ** 6 * d31 ** 2 - 2 * d23 ** 4 * d12 ** 4 + 2 * d23 ** 4 * d31 ** 4

    a1 = 8 * d23 ** 6 * d12 ** 2 * c31 ** 2 * c12 + 4 * d23 ** 6 * d12 ** 2 * c31 * c23 - 4 * d23 ** 4 * d12 ** 4 * c31 * c23 + 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c31 * c23 - 4 * d23 ** 8 * c12 - 4 * d23 ** 6 * d12 ** 2 * c12 + 8 * d23 ** 6 * d31 ** 2 * c12 + 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c12 - 4 * d23 ** 4 * d31 ** 4 * c12

    a0 = -4 * d23 ** 6 * d12 ** 2 * c31 ** 2 + d23 ** 8 - 2 * d23 ** 4 * d12 ** 2 * d31 ** 2 + 2 * d23 ** 6 * d12 ** 2 + d23 ** 4 * d31 ** 4 + d23 ** 4 * d12 ** 4 - 2 * d23 ** 6 * d31 ** 2

    return a0, a1, a2, a3, a4


## u2f

def u2F_polynom(G1, G2):
    a3 = np.linalg.det(G2)

    a2 = (G2[1, 0] * G2[2, 1] * G1[0, 2]
          - G2[1, 0] * G2[0, 1] * G1[2, 2]
          + G2[0, 0] * G2[1, 1] * G1[2, 2]
          + G2[2, 0] * G1[0, 1] * G2[1, 2]
          + G2[2, 0] * G2[0, 1] * G1[1, 2]
          - G2[0, 0] * G1[2, 1] * G2[1, 2]
          - G2[2, 0] * G1[1, 1] * G2[0, 2]
          - G2[2, 0] * G2[1, 1] * G1[0, 2]
          - G2[0, 0] * G2[2, 1] * G1[1, 2]
          + G1[1, 0] * G2[2, 1] * G2[0, 2]
          + G2[1, 0] * G1[2, 1] * G2[0, 2]
          + G1[2, 0] * G2[0, 1] * G2[1, 2]
          - G1[1, 0] * G2[0, 1] * G2[2, 2]
          - G1[0, 0] * G2[2, 1] * G2[1, 2]
          - G2[1, 0] * G1[0, 1] * G2[2, 2]
          + G2[0, 0] * G1[1, 1] * G2[2, 2]
          + G1[0, 0] * G2[1, 1] * G2[2, 2]
          - G1[2, 0] * G2[1, 1] * G2[0, 2])

    a1 = (G1[0, 0] * G1[1, 1] * G2[2, 2]
          + G1[0, 0] * G2[1, 1] * G1[2, 2]
          + G2[2, 0] * G1[0, 1] * G1[1, 2]
          - G1[1, 0] * G1[0, 1] * G2[2, 2]
          - G2[0, 0] * G1[2, 1] * G1[1, 2]
          - G2[1, 0] * G1[0, 1] * G1[2, 2]
          - G2[2, 0] * G1[1, 1] * G1[0, 2]
          + G2[0, 0] * G1[1, 1] * G1[2, 2]
          + G1[1, 0] * G1[2, 1] * G2[0, 2]
          + G1[1, 0] * G2[2, 1] * G1[0, 2]
          + G1[2, 0] * G2[0, 1] * G1[1, 2]
          - G1[1, 0] * G2[0, 1] * G1[2, 2]
          - G1[2, 0] * G2[1, 1] * G1[0, 2]
          + G2[1, 0] * G1[2, 1] * G1[0, 2]
          - G1[0, 0] * G2[2, 1] * G1[1, 2]
          - G1[2, 0] * G1[1, 1] * G2[0, 2]
          + G1[2, 0] * G1[0, 1] * G2[1, 2]
          - G1[0, 0] * G1[2, 1] * G2[1, 2])

    a0 = np.linalg.det(G1)

    return a0, a1, a2, a3


## drawing
SCREEN_SIZE = [1920, 1080]
SCREEN_SIZE_OK = False


def prepare_figure(num=None, geom=None, title=None):
    if num == None:
        fig = plt.figure()
    else:
        fig = plt.figure(num)

    fig.clf()

    if geom != None:
        geom = list(geom)
        if type(geom[0]) is complex:
            geom[0] = geom[0].real + geom[0].imag * SCREEN_SIZE[0]
        if type(geom[1]) is complex:
            geom[1] = geom[1].real + geom[1].imag * SCREEN_SIZE[1]
        if type(geom[2]) is complex:
            geom[2] = geom[2].real + geom[2].imag * SCREEN_SIZE[0]
        if type(geom[3]) is complex:
            geom[3] = geom[3].real + geom[3].imag * SCREEN_SIZE[1]

        d = fig.get_dpi()
        fig.set_size_inches((geom[2] / d, geom[3] / d))

        m = plt.get_current_fig_manager()
        if hasattr(m, 'window'):
            if hasattr(m.window, 'setGeometry'):
                m.window.setGeometry(geom[0], geom[1], geom[2], geom[3])

    if title != None:
        plt.title(title)

    return fig


def getframe(fig, bbox=None):
    if bbox is None:
        bbox = fig.bbox

    b = fig.canvas.copy_from_bbox(bbox)
    e = b.get_extents()
    f = np.frombuffer(b, dtype=np.uint8)
    im = f.reshape((e[3] - e[1], e[2] - e[0], 4))
    im = im[:, :, 0:3]
    return im


def drawnow():
    figManager = plt.get_current_fig_manager()
    if figManager is not None:
        canvas = figManager.canvas
        canvas.draw()
        #        plt.show(block=False)
        canvas.flush_events()

# misc
# def wait_if_batchmode():
#     do_wait = True;
#     try:
#         if __IPYTHON__:
#             do_wait = False
#     except NameError:
#         pass
#
#     if do_wait:
#         input("Press Enter to continue...")
