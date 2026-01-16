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
    base = r"C:\Users\danie\code\work\shadow_segmentation\evaluation\data\\"
    # video data, in grayscale
    vid = r"CaMKII_DJ-Gi_CNO_3-21-22-Phase_3-new_cropped_part1_0000.nii.gz"
    # annotations, labeled [0...5].
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

    # DATA SHAPE IS INITIALLY Y x X x Time
    data = np.flip(data, 1)  # reverse X for my viewing pleasure
    # turn data into a displayable imagee
    data = data[..., None] @ row(1, 1, 1)  # Y x X x ((T, 1) @ (1, 3)) = Y x X x T x 3
    data = np.swapaxes(data, 0, 2)  # T x X x Y x 3 (images, indexed by frame# first!)

    # SEGMENTATION SHAPE IS INITIALLY Y x X x Time
    segm = np.flip(segm, 1).T  # reverse X, then .T so shape is T x X x Y

    return data, segm


segmentation_ndx = [0, 1, 2, 3, 4, 5]
highlights = []
# TODO: make faster. toggling parts is painful
def make_segments(data, segm):
    global segmentation_ndx

    # color of segments
    colors = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1]
    ])

    segm = segm.astype(int)

    # show corresponding color on each value
    # of the segmentations. this will need changed
    # if files aren't only 0,1,...n
    ndxs = np.isin(segm, segmentation_ndx)
    segn = np.where(ndxs, segm, 0)

    # display colors on video data
    frames = data + 100 * colors[segn]

    # format into a showable image
    segments = frames.astype(int).clip(0, 255)

    # for seg in segmentation_ndx:
    #     ndxs_ = np.isin(segm, [seg])
    #     segn_ = np.where(ndxs_, segm, 0)

    return segments


def in_player(vid, n=0, draw=lambda *x: (None, None)):
    """
    takes an array formatted like a video/images and adds keyboard controls:
        a/d - move 1 frame forward/back
        A/D - move 50 frames froward/back
        j# -  jump to frame #, where # is a string of digits
        # -   toggles the visibility of segment #

    :param vid:
    :type vid:
    :param n:
    :type n:
    :param draw:
    :type draw:
    :return:
    :rtype:
    """
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
            g = int(k)
            if g in segmentation_ndx:
                segmentation_ndx.remove(g)
            else:
                segmentation_ndx.append(g)

            # update segments array with colors if we need.
            # this is a very inefficient way to do this
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

        # display
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
