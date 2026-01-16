from functools import partialmethod

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from vectoring import *


class view:
    def __init__(self, video, start, /, step=1, default_vid=None):
        self.vid = video
        self.start = start
        self.pos = self.start
        self.step = step
        self.default_vid = default_vid

    def get(self, **kwargs):
        self.control(**kwargs)

        if success := self.pos < len(self.vid):
            image = self.vid[self.pos]
            self.pos += self.step
            return success, image
        else:
            return False, None

    def set_vid(self, vid=None):
        self.vid = vid or self.default_vid()

    def control(self, **kwargs):
        if "n" in kwargs:
            self.pos = kwargs["n"]

    def __len__(self):
        return len(self.vid)



def init_test_file():
    base = r"C:\Users\danie\code\work\shadow_segmentation\\"
    # da video
    vid = r"CaMKII_DJ-Gi_CNO_3-21-22-Phase_3-new_cropped_part1_0000.nii.gz"

    # da annotations, labeled [0...5].
    seg = r"CaMKII_DJ-Gi_CNO_3-21-22-Phase_3-new_cropped_part1.nii.gz"

    # load the nifti images
    img = nib.load(base + vid)
    header = img.header
    jmg = nib.load(base + seg)

    # nii file as an array
    # set lim = -1 for all data
    lim = 100
    data = img.get_fdata()[..., :lim]
    segm = jmg.get_fdata()[..., :lim]
    t = 0

    # image_shape = img.shape               # Y x X x T
    data = np.flip(data, 1)  # reverse X for my viewing pleasure
    data = data[..., None] @ row(1, 1, 1)  # Y x X x ((T, 1) @ (1, 3)) = Y x X x T x 3
    data = np.swapaxes(data, 0, 2)  # T x X x Y x 3 (images, indexed by frame# first!)

    return data, segm


highlight = [1, 2, 3, 4, 5]


def make_segments(data, segm):
    global highlight

    # color of segments
    colors = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1]
    ])
    # image_shape = img.shape       # Y x X x T
    segm = np.flip(segm, 1)  # reverse X for my viewing pleasure
    segm = np.swapaxes(segm, 0, 2)  # T x X x Y (images, indexed by frame# first!)
    segm = segm.astype(int)

    ndxs = np.isin(segm, highlight)
    segn = np.where(ndxs, segm, 0)

    frames = data + 100 * colors[segn]
    segments = frames.astype(int).clip(0, 255)
    # segments = frames.astype(float).clip(0, 1)

    for seg in highlight:
        ndxs_ = np.isin(segm, [seg])
        segn_ = np.where(ndxs_, segm, 0)

        print(segn)

    return segments


def in_player(vid, n=0, draw=lambda *x: (None, None)):
    figure, axes = plt.subplots()
    figure.canvas.manager.set_window_title('')

    # hide the axes ticks and labels
    axes.set_xticks([])
    axes.set_yticks([])

    # initial frame
    success, frame = vid.get(n=n)
    if not success:
        quit()

    im = axes.imshow(frame)

    def on_key_press(event):
        nonlocal n, frame

        # keyboard controls
        if event.key == 'd':
            n += 1
        elif event.key == 'a':
            n -= 1
        elif event.key == 'D':
            n += 50
        elif event.key == 'A':
            n -= 50
        elif event.key == 'j':
            inp = input("Jump to frame #: ")
            try:
                n = int(inp)
            except Exception as e:
                print(f"Invalid input: {inp}")
                return
        elif (k := event.key).isnumeric():
            if (g := int(k)) in highlight:
                highlight.remove(g)
            else:
                highlight.append(g)
            vid.set_vid()
        else:
            return

        # get frame n
        n = np.clip(n, 0, len(vid) - 1)

        try:
            success, frame = vid.get(n=n)
        except:
            success, frame = False, None
            return

        #frame = vid[n]
        if frame is not None:
            im.set_array(frame)
            figure.canvas.draw()

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()


def main():
    bounds = array([200, 200])
    img = np.zeros((*bounds, 3))
    dot_5 = c2z2(2 * udisc(4, 15))
    # plt.scatter(*x.reshape((2, -1)))
    # plt.scatter(*round_(c2r2(udisc(9, 15))).reshape((2, -1)))
    # plt.scatter(*round_(disc).reshape((2, -1)))
    # plt.imshow(img)
    # plt.show()

    data, segm = init_test_file()
    segments = make_segments(data, segm)
    view_segments = view(segments, 0, default_vid=lambda: make_segments(data, segm))
    in_player(view_segments)


main()
