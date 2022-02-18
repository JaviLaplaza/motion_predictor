from __future__ import print_function

import sys
#sys.path.insert(0, "/home/jlaplaza/DL/IRI-DL")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg

from pylab import *

import io
import PIL.Image
from torchvision.transforms import ToTensor
from mpl_toolkits.mplot3d import Axes3D

from utils.data_utils import expmap2rotmat, normalization_stats, normalize_data

from itertools import product, combinations


def plot_estim(img, estim, target):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    # display img
    ax.imshow(img)

    # add estim and target
    ax.text(0.5, 0.1, f"trg:{target}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='r', fontsize=20)
    ax.text(0.5, 0.04, f"est:{estim}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='r', fontsize=20)

    ser_fig = serialize_fig(fig)
    plt.close(fig)
    return ser_fig


def serialize_fig(fig):
    canvas = plt_backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    return image_chw




def animate_sequence(sequence, keep=False, fig=None, ax=None, ob=None, video_buf=None, color='black', show=False, frame=0, pause=0.005):

    xyz = compute_sequence((sequence))
    sequence_length = xyz.shape[0]

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ob = Ax3DPose(ax)
        video_buf = []

    if color == 'green':
        rcolor = "#039e00"
        lcolor = "#eaf202"
    elif color == 'red':
        rcolor = "#fa3a00"
        lcolor = "#0226f2"
    else:
        rcolor = "#272629"
        lcolor = "#798077"

    # Plot the sequence
    for i in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)
        frame += 1

        buf = io.BytesIO()
        ob.update(xyz[:i+1, :], lcolor=lcolor, rcolor=rcolor, pcolor="#b0b0b0", keep=keep)
        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        if show: plt.pause(pause)
        #plt.waitforbuttonpress()

    return video_buf, fig, ax, ob, frame


def animate_handover_sequence(joint_xyz, keep=False, fig=None, ax=None, video_buf=None, color='input', show=False, frame=0, pause=0.005):
    sequence_length = joint_xyz.shape[0]

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'yellow'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'red'

    joint_xyz = np.reshape(joint_xyz, (sequence_length, 15, -1))

    X = joint_xyz[:, :, 0]
    Y = joint_xyz[:, :, 1]
    Z = joint_xyz[:, :, 2]

    left = [0, 1, 2, 12, 5, 4, 3]
    right = [6, 7, 8, 12, 11, 10, 9]
    head = [12, 13, 14]
    hips = [2, 8]
    shoulders = [5, 11]

    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)


        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        for index in range(len(left) - 1):
            ax.plot((X[frame, left[index]], X[frame, left[index + 1]]),
                    (Y[frame, left[index]], Y[frame, left[index + 1]]),
                    (Z[frame, left[index]], Z[frame, left[index + 1]]), color_l)

        for index in range(len(right) - 1):
            ax.plot((X[frame, right[index]], X[frame, right[index + 1]]),
                    (Y[frame, right[index]], Y[frame, right[index + 1]]),
                    (Z[frame, right[index]], Z[frame, right[index + 1]]), color_r)

        for index in range(len(head) - 1):
            ax.plot((X[frame, head[index]], X[frame, head[index + 1]]),
                    (Y[frame, head[index]], Y[frame, head[index + 1]]),
                    (Z[frame, head[index]], Z[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((X[frame, hips[index]], X[frame, hips[index + 1]]),
                    (Y[frame, hips[index]], Y[frame, hips[index + 1]]),
                    (Z[frame, hips[index]], Z[frame, hips[index + 1]]), 'black')

        for index in range(len(shoulders) - 1):
            ax.plot((X[frame, shoulders[index]], X[frame, shoulders[index + 1]]),
                    (Y[frame, shoulders[index]], Y[frame, shoulders[index + 1]]),
                    (Z[frame, shoulders[index]], Z[frame, shoulders[index + 1]]), 'black')

        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        if show: plt.pause(pause)
        #if show: plt.waitforbuttonpress()
        frame += 1

    #plt.close()
    return video_buf, fig, ax, frame


def animate_iri_handover_sequence(joint_xyz, end_effector=None, keep=False, fig=None, ax=None, video_buf=None,
                                  color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                                  epoch=0, train=0, heatmap=None):
    sequence_length = joint_xyz.shape[0]

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        ax = plt.gca(projection='3d')#.set_aspect('equal')
        ax.view_init(elev=30., azim=-140)

        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'red'

    joint_xyz = np.reshape(joint_xyz, (sequence_length, 11, -1))

    X = joint_xyz[:, :, 0]
    Y = joint_xyz[:, :, 1]
    Z = joint_xyz[:, :, 2]


    #head = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #left = [11, 13, 15, 17, 19, 21, 23]
    #right = [12, 14, 16 , 18, 20, 22, 24]
    left = [10, 8, 1, 5,  6, 7]
    right = [9, 8, 1, 2, 3, 4]
    head = [0, 1]

    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)


        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        frame_keep = frame
        if keep:
            while frame_keep > 0:
                if frame_keep % 2 == 0:
                    for index in range(len(left) - 1):
                        ax.plot((X[frame_keep, left[index]], X[frame_keep, left[index + 1]]),
                                (Y[frame_keep, left[index]], Y[frame_keep, left[index + 1]]),
                                (Z[frame_keep, left[index]], Z[frame_keep, left[index + 1]]), 'grey')

                    for index in range(len(right) - 1):
                        ax.plot((X[frame_keep, right[index]], X[frame_keep, right[index + 1]]),
                                (Y[frame_keep, right[index]], Y[frame_keep, right[index + 1]]),
                                (Z[frame_keep, right[index]], Z[frame_keep, right[index + 1]]), 'grey')

                    for index in range(len(head) - 1):
                        ax.plot((X[frame_keep, head[index]], X[frame_keep, head[index + 1]]),
                                (Y[frame_keep, head[index]], Y[frame_keep, head[index + 1]]),
                                (Z[frame_keep, head[index]], Z[frame_keep, head[index + 1]]), 'grey')

                frame_keep -= 1

        for index in range(len(left) - 1):
            ax.plot((X[frame, left[index]], X[frame, left[index + 1]]),
                    (Y[frame, left[index]], Y[frame, left[index + 1]]),
                    (Z[frame, left[index]], Z[frame, left[index + 1]]), color_l)

        for index in range(len(right) - 1):
            ax.plot((X[frame, right[index]], X[frame, right[index + 1]]),
                    (Y[frame, right[index]], Y[frame, right[index + 1]]),
                    (Z[frame, right[index]], Z[frame, right[index + 1]]), color_r)

        for index in range(len(head) - 1):
            ax.plot((X[frame, head[index]], X[frame, head[index + 1]]),
                    (Y[frame, head[index]], Y[frame, head[index + 1]]),
                    (Z[frame, head[index]], Z[frame, head[index + 1]]), 'black')

        if end_effector == None:
            pass
        elif len(end_effector.shape) == 2:
            x, y, z = end_effector[frame]
            ax.scatter(x, y, z)

        elif len(end_effector.shape) == 1:
            x, y, z = end_effector
            ax.scatter(x, y, z)

        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            #plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/ICRA')

        #ax.auto_scale_xyz([0, 6], [-1, 1], [0, 2])
        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        #if show: plt.pause(pause)
        if show: plt.waitforbuttonpress()
        frame += 1

    #plt.close()

    return video_buf, fig, ax, frame

def plot_distribution(poses, probs):
    sequence_length = poses.shape[0]

    fig = plt.figure()
    ax = plt.gca(projection='3d')

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

    joint_xyz = np.reshape(poses[0, :, 0], (11, -1))
    possible_joint_xyz = np.reshape(poses[1], (11, 3, -1))

    X = joint_xyz[:, 0]
    Y = joint_xyz[:, 1]
    Z = joint_xyz[:, 2]

    X_probs = possible_joint_xyz[:, 0]
    Y_probs = possible_joint_xyz[:, 1]
    Z_probs = possible_joint_xyz[:, 2]

    probs_value = np.reshape(probs[0], (11, 3, -1))
    probs_value_x = probs_value[:, 0]
    #probs_value_x = (probs_value_x - np.min(probs_value_x)) / (np.max(probs_value_x) - np.min(probs_value_x))
    #print(probs_value_x)
    probs_value_y = probs_value[:, 1]
    probs_value_z = probs_value[:, 2]
    probs_value_total = (probs_value_x + probs_value_y + probs_value_z)/3

    print(probs_value_total.shape)
    #probs_value = np.mean(probs_value.numpy(), 1)[4]


    left = [10, 8, 1, 5, 6, 7]
    right = [9, 8, 1, 2, 3, 4]
    head = [0, 1]

    colors = cm.hsv(probs_value_total[4] / max(probs_value_total[4]))

    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(probs_value_total[4])

    for index in range(len(left) - 1):
        ax.plot((X[left[index]], X[left[index + 1]]),
                (Y[left[index]], Y[left[index + 1]]),
                (Z[left[index]], Z[left[index + 1]]), 'orange')

    for index in range(len(right) - 1):
        ax.plot((X[right[index]], X[right[index + 1]]),
                (Y[right[index]], Y[right[index + 1]]),
                (Z[right[index]], Z[right[index + 1]]), 'green')

        if index == 4:
            data = np.empty(4)
            for i in range(35, 65, 2):
                for j in range(35, 65, 2):
                    for k in range(25, 75, 2):
                        #print(probs_value[4, :, [i, j, k]])
                        #value = np.mean(probs_value[4, :, [i, j, k]].numpy())
                        #value_x = np.max(probs_value[4, 0, i].numpy())
                        #value_y = np.max(probs_value[4, 1, j].numpy())
                        #value_z = np.max(probs_value[4, 2, k].numpy())

                        if np.sqrt((X_probs[4, i] - X_probs[4, 50])**2 + (Y_probs[4, j] - Y_probs[4, 50])**2 + (Z_probs[4, k] - Z_probs[4, 50])**2) <= 0.15:

                            value = (probs_value_x[4, i] + probs_value_y[4, j] + probs_value_z[4, k])/3

                            data_ = np.array([X_probs[4, i], Y_probs[4, j], Z_probs[4, k], value])

                            data = np.vstack((data, data_))

                            #ax.scatter(X_probs[4, i], Y_probs[4, j], Z_probs[4, k], c=[value.item()],  lw=0, s=10)


            data = data[1:]

            colors = cm.hsv(data[:, 3] / max(data[:, 3]))
            #colors = [str(1 - item/np.max(data[:, 3])) for item in data[:, 3]]

            colmap = cm.ScalarMappable(cmap=cm.hsv)
            colmap.set_array(data[:, 3])

            #ax.scatter(X_probs[4], Y_probs[4], Z_probs[4], c=probs_value_total[4], lw=0, s=10)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, lw=0, s=5)
            #cb = fig.colorbar(colmap)



    for index in range(len(head) - 1):
        ax.plot((X[head[index]], X[head[index + 1]]),
                (Y[head[index]], Y[head[index + 1]]),
                (Z[head[index]], Z[head[index + 1]]), 'black')

    plt.show()
    #plt.close()

def animate_target_and_prediction(target, prediction, end_effector=None, keep=False, fig=None, ax=None, video_buf=None,
                                  color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                                  epoch=0, train=0):
    plt.ion()

    sequence_length = target.shape[0]

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ax.view_init(elev=30., azim=-140)
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'red'

    target = np.reshape(target, (sequence_length, 11, -1))
    prediction = np.reshape(prediction, (sequence_length, 11, -1))


    Xt = target[:, :, 0]
    Yt = target[:, :, 1]
    Zt = target[:, :, 2]

    Xp = prediction[:, :, 0]
    Yp = prediction[:, :, 1]
    Zp = prediction[:, :, 2]

    left = [10, 8, 1, 5,  6, 7]
    right = [9, 8, 1, 2, 3, 4]
    head = [0, 1]

    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)


        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        for index in range(len(left) - 1):
            ax.plot((Xt[frame, left[index]], Xt[frame, left[index + 1]]),
                    (Yt[frame, left[index]], Yt[frame, left[index + 1]]),
                    (Zt[frame, left[index]], Zt[frame, left[index + 1]]), 'red')

        for index in range(len(right) - 1):
            ax.plot((Xt[frame, right[index]], Xt[frame, right[index + 1]]),
                    (Yt[frame, right[index]], Yt[frame, right[index + 1]]),
                    (Zt[frame, right[index]], Zt[frame, right[index + 1]]), 'blue')

        for index in range(len(head) - 1):
            ax.plot((Xt[frame, head[index]], Xt[frame, head[index + 1]]),
                    (Yt[frame, head[index]], Yt[frame, head[index + 1]]),
                    (Zt[frame, head[index]], Zt[frame, head[index + 1]]), 'black')

        for index in range(len(left) - 1):
            ax.plot((Xp[frame, left[index]], Xp[frame, left[index + 1]]),
                    (Yp[frame, left[index]], Yp[frame, left[index + 1]]),
                    (Zp[frame, left[index]], Zp[frame, left[index + 1]]), 'yellow')

        for index in range(len(right) - 1):
            ax.plot((Xp[frame, right[index]], Xp[frame, right[index + 1]]),
                    (Yp[frame, right[index]], Yp[frame, right[index + 1]]),
                    (Zp[frame, right[index]], Zp[frame, right[index + 1]]), 'green')

        for index in range(len(head) - 1):
            ax.plot((Xp[frame, head[index]], Xp[frame, head[index + 1]]),
                    (Yp[frame, head[index]], Yp[frame, head[index + 1]]),
                    (Zp[frame, head[index]], Zp[frame, head[index + 1]]), 'black')

        if end_effector == None:
            pass

        elif end_effector is not None and len(end_effector.shape)>1:
            x, y, z = end_effector[frame]
            ax.scatter(x, y, z)

        else:
            x, y, z = end_effector
            ax.scatter(x, y, z)

        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        #if show: plt.pause(pause)
        if show: plt.waitforbuttonpress()
        frame += 1

    plt.close()

    return video_buf, fig, ax, frame


def animate_all_sequence(input_sequence, target_sequence, predicted_sequence=np.empty((0))):
    """
    Animate sequence of input, target and prediction.
    Input sequences must be numpy arrays onehot encoded. Shapes: [L, J, C]
    :param input_sequence:
    :param target_sequence:
    :param predicted_sequence:
    :return:
    """
    # Load all the data
    parent, offset, rotInd, expmapInd = _some_variables()

    # ==== Input sequence === #
    input_sequence_length = input_sequence.shape[0]

    #xyz_input_aux = np.zeros((input_sequence_length, 99))
    xyz_input = np.zeros((input_sequence_length, 96))

    for i in range(input_sequence_length):
        xyz_input[i, :] = fkl(input_sequence[i, :], parent, offset, rotInd, expmapInd)

    # ==== Target sequence === #
    target_sequence_length = target_sequence.shape[0]

    #xyz_target_aux = np.zeros((target_sequence_length, 99))
    xyz_target = np.zeros((target_sequence_length, 96))

    for i in range(target_sequence_length):
        xyz_target[i, :] = fkl(target_sequence[i, :], parent, offset, rotInd, expmapInd)

    #if predicted_sequence.size != 0:
    if predicted_sequence is not None:
        # ==== Predicted sequence === #
        predicted_sequence_length = predicted_sequence.shape[0]

        #xyz_predicted_aux = np.zeros((predicted_sequence_length, 99))
        xyz_predicted = np.zeros((predicted_sequence_length, 96))

        for i in range(predicted_sequence_length):
            xyz_predicted[i, :] = fkl(predicted_sequence[i, :], parent, offset, rotInd, expmapInd)

    # === Plot and animate ===
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = Ax3DPose(ax)
    video_buf = []

    # Plot the input sequence
    if input_sequence is not None:
        for i in range(input_sequence_length):
            buf = io.BytesIO()
            ob.update(xyz_input[:i+1, :], lcolor="#272629", rcolor="#4f4e52", pcolor="#b0b0b0", keep=True)
            #plt.show(block=False)
            fig.canvas.draw()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image).unsqueeze(0)
            video_buf.append(image)
            buf.close()
            plt.pause(0.005)
            #plt.waitforbuttonpress()

    # Plot the predicted sequence
    #if predicted_sequence.size != 0:
    if predicted_sequence is not None:
        for i in range(predicted_sequence_length):
            buf = io.BytesIO()
            ob.update(xyz_predicted[:i + 1, :], lcolor="#4dff00", rcolor="#73e841", pcolor="#aded9d",
                      offset=ob.offset)
            #plt.show(block=False)
            fig.canvas.draw()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image).unsqueeze(0)
            video_buf.append(image)
            buf.close()
            plt.pause(0.005)
            #plt.waitforbuttonpress()

    # Plot the target sequence
    if target_sequence is not None:
        for i in range(target_sequence_length):
            buf = io.BytesIO()
            ob.update(xyz_target[:i+1, :], lcolor="#fa3a00", rcolor="#f56c42", pcolor="#f2b4ae", offset=ob.offset, side_offset=300)
            #plt.show(block=False)
            fig.canvas.draw()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image).unsqueeze(0)
            video_buf.append(image)
            buf.close()
            plt.pause(0.005)
            #plt.waitforbuttonpress()

    return video_buf





def show_unidimensional_hist(x, bins):
    """
    Shows the histogram of a given distribution
    :param x: unidimensional vector of data
    :return:
    """
    plt.hist(x, bins=bins)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    plt.show()

def update_handover(figure, plot, X, Y, Z):
    plot.set_data(X, Y, Z)
    figure.gca().relim()
    figure.gca().autoscale_view()
    return plot

def cuboid_data(center, size):
    """
    Create a data array for cuboid plotting.


    ============= ================================================
       Argument      Description
       ============= ================================================
       center        center of the cuboid, triple
       size          size of the cuboid, triple, (x_length,y_width,z_height)
       :type size: tuple, numpy.array, list
       :param size: size of the cuboid, triple, (x_length,y_width,z_height)
       :type center: tuple, numpy.array, list
       :param center: center of the cuboid, triple, (x,y,z)


    """


    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return x, y, z


def animate_mediapipe_sequence(joint_xyz, end_effector=[], obstacles=[], keep=False, fig=None, ax=None, video_buf=None,
                               color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                               epoch=0, train=0, heatmap=None):

    sequence_length, num_joints = joint_xyz.shape
    num_joints = int(num_joints/3)

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        ax = plt.gca(projection='3d')#.set_aspect('equal')
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'orange'

    joint_xyz = np.reshape(joint_xyz, (sequence_length, num_joints, -1))
    chest = np.expand_dims(joint_xyz[:, 1] + (joint_xyz[:, 2] - joint_xyz[:, 1])/2, axis=1)
    joint_xyz = np.concatenate((joint_xyz, chest), axis=1)

    pelvis = np.expand_dims(joint_xyz[:, 7] + (joint_xyz[:, 8] - joint_xyz[:, 7])/2, axis=1)
    joint_xyz = np.concatenate((joint_xyz, pelvis), axis=1)

    X = joint_xyz[:, :, 0]
    Y = joint_xyz[:, :, 1]
    Z = joint_xyz[:, :, 2]

    head = [0, 9]
    left = [7, 10, 9, 1, 3, 5]
    right = [8, 10, 9, 2, 4, 6]
    hips = [7, 8]


    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)

        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        frame_keep = frame
        if keep:
            while frame_keep > 0:
                if frame_keep % 2 == 0:
                    for index in range(len(left) - 1):
                        ax.plot((X[frame_keep, left[index]], X[frame_keep, left[index + 1]]),
                                (Y[frame_keep, left[index]], Y[frame_keep, left[index + 1]]),
                                (Z[frame_keep, left[index]], Z[frame_keep, left[index + 1]]), 'grey')

                    for index in range(len(right) - 1):
                        ax.plot((X[frame_keep, right[index]], X[frame_keep, right[index + 1]]),
                                (Y[frame_keep, right[index]], Y[frame_keep, right[index + 1]]),
                                (Z[frame_keep, right[index]], Z[frame_keep, right[index + 1]]), 'grey')

                    for index in range(len(head) - 1):
                        ax.plot((X[frame_keep, head[index]], X[frame_keep, head[index + 1]]),
                                (Y[frame_keep, head[index]], Y[frame_keep, head[index + 1]]),
                                (Z[frame_keep, head[index]], Z[frame_keep, head[index + 1]]), 'grey')

                    for index in range(len(hips) - 1):
                        ax.plot((X[frame_keep, hips[index]], X[frame_keep, hips[index + 1]]),
                                (Y[frame_keep, hips[index]], Y[frame_keep, hips[index + 1]]),
                                (Z[frame_keep, hips[index]], Z[frame_keep, hips[index + 1]]), 'grey')

                frame_keep -= 1

        for index in range(len(left) - 1):
            ax.plot((X[frame, left[index]], X[frame, left[index + 1]]),
                    (Y[frame, left[index]], Y[frame, left[index + 1]]),
                    (Z[frame, left[index]], Z[frame, left[index + 1]]), color_l)

        for index in range(len(right) - 1):
            ax.plot((X[frame, right[index]], X[frame, right[index + 1]]),
                    (Y[frame, right[index]], Y[frame, right[index + 1]]),
                    (Z[frame, right[index]], Z[frame, right[index + 1]]), color_r)

        for index in range(len(head) - 1):
            ax.plot((X[frame, head[index]], X[frame, head[index + 1]]),
                    (Y[frame, head[index]], Y[frame, head[index + 1]]),
                    (Z[frame, head[index]], Z[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((X[frame, hips[index]], X[frame, hips[index + 1]]),
                    (Y[frame, hips[index]], Y[frame, hips[index + 1]]),
                    (Z[frame, hips[index]], Z[frame, hips[index + 1]]), 'black')

        if len(obstacles) > 0:
            for obstacle in obstacles[frame]:
                if np.sum(obstacle) == 0:
                    pass
                else:
                    # if frame == 0:
                    #center = obstacle[frame]
                    center = obstacle

                    # center[0] = -center[0]
                    # center[2] = -center[2]
                    Xo, Yo, Zo = cuboid_data(center, np.array((0.1, 0.1, 0.1)))
                    ax.plot_surface(np.array(Xo), np.array(Yo), np.array(Zo), color='r', rstride=1, cstride=1, alpha=0.1)

                    # r0 = [center[0] - side, center[0] + side]
                    # r1 = [center[1] - side, center[1] + side]
                    # r2 = [center[2] - side, center[2] + side]

                    # for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    #    if np.sum(np.abs(s-e)) == r0[1]-r0[0]:
                    #        ax.plot3D(*zip(s, e), color="r")

        if len(end_effector) > 0:
            if len(end_effector.shape) == 2:
                x, y, z = end_effector[frame]
                ax.scatter(x, y, z)

            elif len(end_effector.shape) == 1:
                x, y, z = end_effector
                ax.scatter(x, y, z)

        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        #ax.auto_scale_xyz([0, 6], [-1, 1], [0, 2])
        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        #if show: plt.pause(pause)
        if show: plt.waitforbuttonpress()
        frame += 1

    #plt.close()

    return video_buf, fig, ax, frame


def animate_mediapipe_target_and_prediction(target, prediction, end_effector=[], obstacles=[], keep=False, fig=None, ax=None, video_buf=None,
                                  color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                                  epoch=0, train=0):
    plt.ion()
    sequence_length, num_joints = target.shape
    num_joints = int(num_joints / 3)

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ax.view_init(elev=30., azim=-140)
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'red'

    target_xyz = np.reshape(target, (sequence_length, num_joints, -1))
    target_chest = np.expand_dims(target_xyz[:, 1] + (target_xyz[:, 2] - target_xyz[:, 1]) / 2, axis=1)
    target_xyz = np.concatenate((target_xyz, target_chest), axis=1)

    target_pelvis = np.expand_dims(target_xyz[:, 7] + (target_xyz[:, 8] - target_xyz[:, 7]) / 2, axis=1)
    target = np.concatenate((target_xyz, target_pelvis), axis=1)

    prediction_xyz = np.reshape(prediction, (sequence_length, num_joints, -1))
    prediction_chest = np.expand_dims(prediction_xyz[:, 1] + (prediction_xyz[:, 2] - prediction_xyz[:, 1]) / 2, axis=1)
    prediction_xyz = np.concatenate((prediction_xyz, prediction_chest), axis=1)

    prediction_pelvis = np.expand_dims(prediction_xyz[:, 7] + (prediction_xyz[:, 8] - prediction_xyz[:, 7]) / 2, axis=1)
    prediction = np.concatenate((prediction_xyz, prediction_pelvis), axis=1)


    target = np.reshape(target, (sequence_length, 11, -1))
    prediction = np.reshape(prediction, (sequence_length, 11, -1))


    Xt = target[:, :, 0]
    Yt = target[:, :, 1]
    Zt = target[:, :, 2]

    Xp = prediction[:, :, 0]
    Yp = prediction[:, :, 1]
    Zp = prediction[:, :, 2]

    head = [0, 9]
    left = [7, 10, 9, 1, 3, 5]
    right = [8, 10, 9, 2, 4, 6]
    hips = [7, 8]

    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)


        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        for index in range(len(left) - 1):
            ax.plot((Xt[frame, left[index]], Xt[frame, left[index + 1]]),
                    (Yt[frame, left[index]], Yt[frame, left[index + 1]]),
                    (Zt[frame, left[index]], Zt[frame, left[index + 1]]), 'red')

        for index in range(len(right) - 1):
            ax.plot((Xt[frame, right[index]], Xt[frame, right[index + 1]]),
                    (Yt[frame, right[index]], Yt[frame, right[index + 1]]),
                    (Zt[frame, right[index]], Zt[frame, right[index + 1]]), 'blue')

        for index in range(len(head) - 1):
            ax.plot((Xt[frame, head[index]], Xt[frame, head[index + 1]]),
                    (Yt[frame, head[index]], Yt[frame, head[index + 1]]),
                    (Zt[frame, head[index]], Zt[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((Xt[frame, head[index]], Xt[frame, head[index + 1]]),
                    (Yt[frame, head[index]], Yt[frame, head[index + 1]]),
                    (Zt[frame, head[index]], Zt[frame, head[index + 1]]), 'black')

        for index in range(len(left) - 1):
            ax.plot((Xp[frame, left[index]], Xp[frame, left[index + 1]]),
                    (Yp[frame, left[index]], Yp[frame, left[index + 1]]),
                    (Zp[frame, left[index]], Zp[frame, left[index + 1]]), 'yellow')

        for index in range(len(right) - 1):
            ax.plot((Xp[frame, right[index]], Xp[frame, right[index + 1]]),
                    (Yp[frame, right[index]], Yp[frame, right[index + 1]]),
                    (Zp[frame, right[index]], Zp[frame, right[index + 1]]), 'green')

        for index in range(len(head) - 1):
            ax.plot((Xp[frame, head[index]], Xp[frame, head[index + 1]]),
                    (Yp[frame, head[index]], Yp[frame, head[index + 1]]),
                    (Zp[frame, head[index]], Zp[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((Xp[frame, head[index]], Xp[frame, head[index + 1]]),
                    (Yp[frame, head[index]], Yp[frame, head[index + 1]]),
                    (Zp[frame, head[index]], Zp[frame, head[index + 1]]), 'black')

        if end_effector == []:
            pass

        elif end_effector is not None and len(end_effector.shape)>1:
            x, y, z = end_effector[frame]
            ax.scatter(x, y, z)

        else:
            x, y, z = end_effector
            ax.scatter(x, y, z)

        if len(obstacles) > 0:
            for obstacle in obstacles[frame]:
                if np.sum(obstacle) == 0:
                    pass
                else:
                    # if frame == 0:
                    # center = obstacle[frame]
                    center = obstacle

                    # center[0] = -center[0]
                    # center[2] = -center[2]
                    Xo, Yo, Zo = cuboid_data(center, np.array((0.1, 0.1, 0.1)))
                    ax.plot_surface(np.array(Xo), np.array(Yo), np.array(Zo), color='r', rstride=1, cstride=1, alpha=0.1)

                    # r0 = [center[0] - side, center[0] + side]
                    # r1 = [center[1] - side, center[1] + side]
                    # r2 = [center[2] - side, center[2] + side]

                    # for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    #    if np.sum(np.abs(s-e)) == r0[1]-r0[0]:
                    #        ax.plot3D(*zip(s, e), color="r")

        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        if show: plt.pause(pause)
        # if show: plt.waitforbuttonpress()
        frame += 1

    plt.close()

    return video_buf, fig, ax, frame


if __name__ == "__main__":
    from src.data.h36m import H36MDataset
    from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
    import torch
    """
    opt = ConfigParser().get_config()

    #train_dataset = H36MDataset(opt, "train", None, None, None)
    val_dataset = H36MDataset(opt, "val", None, None, None)

    #data_loader_train = CustomDatasetDataLoader(opt, is_for="train")
    data_loader_val = CustomDatasetDataLoader(opt, is_for="val")
    #dataset_val = data_loader_val.load_data()


    #iterator = iter(dataset_val)
    #train_batch = next(iter(data_loader_train._dataloader))
    val_index = 100
    for i in range(val_index):
        val_batch = next(iter(data_loader_val._dataloader))
    #for i in range(1):
    #    val_batch = next(iterator)

    #input_train_sequence = train_batch['input'][0].numpy()
    #target_train_sequence = train_batch['target'][0].numpy()

    input_val_sequence = val_batch['input'][0].numpy()
    target_val_sequence = val_batch['target'][0].numpy()

    #input_train_sequence = data_loader_train.get_dataset().postprocess_input(input_train_sequence)
    #data_loader_train.get_dataset().get_dataset_stats()
    input_val_sequence = data_loader_val.get_dataset().postprocess_input(input_val_sequence)
    data_loader_val.get_dataset().get_dataset_stats()

    input_val_sequence_smoothed = smooth_sequence(input_val_sequence)

    #target_train_sequence = data_loader_train._dataset.postprocess_target(target_train_sequence)
    target_val_sequence = data_loader_val._dataset.postprocess_target(target_val_sequence)

    #predicted_sequence = train_dataset[1]['target']
    predicted_sequence = val_dataset[1]['target']



    show = True
    video, fig, ax, ob, frame = animate_sequence(input_val_sequence, keep=False, color='black', show=show)
    animate_sequence(input_val_sequence_smoothed, keep=False, fig=fig, ax=ax, ob=ob, color='red', video_buf=video, frame=frame, show=show)
    while False:

        animate_sequence(target_val_sequence, keep=False, fig=fig, ax=ax, ob=ob, color='red', video_buf=video, frame=frame, show=show)
        animate_sequence(input_val_sequence, keep=False, fig=fig, ax=ax, ob=ob, color='black', video_buf=video,
                         frame=frame, show=show)

    #video, fig, ax, ob, frame = animate_sequence(input_val_sequence, keep=False, color='black', show=show)
    #animate_sequence(target_val_sequence, keep=False, fig=fig, ax=ax, ob=ob, color='red', video_buf=video, frame=frame, show=show)

    """


    data = np.array([[0.188338,-0.831972,2.924,0.880296],
                     [0.179086,-0.678787,3.007,0.911494],
                     [0.0432035,-0.687715,3.067,0.820593],
                     [0.0189614,-0.473878,3.187,0.797717],
                     [0.0246787,-0.250512,3.257,0.883722],
                     [0.352048,-0.809332,3.47,0.801273],
                     [0.38055,-0.524537,3.285,0.812598],
                     [0.460427,-0.333067,3.581,0.907636],
                     [0.194094,-0.282527,3.091,0.775655],
                     [0.114456,-0.283898,3.106,0.734109],
                     [0.143862,0.0193752,3.319,0.863516],
                     [0.154447,0.309262,3.315,0.774905],
                     [0.294524,-0.283258,3.099,0.799503],
                     [0.269953,0.0304658,3.351,0.859388],
                     [0.254497,0.318949,3.361,0.877799],
                     [0.164672,-0.865325,2.924,0.876407],
                     [0.233901,-0.922904,3.089,0.913489],
                     [0.111361,-0.859856,3.022,0.822964],
                     [0.281707,-0.948117,3.299,0.45568],
                     [0.229287,0.415451,3.235,0.819968],
                     [0.263911,0.415516,3.276,0.820442],
                     [0.235099,0.336051,3.317,0.605474],
                     [0.177711,0.412091,3.249,0.736942],
                     [0.13668,0.405105,3.276,0.658353],
                     [0.164802,0.329738,3.307,0.504784],
                     [-0.268168, -0.877255, 0.99826, 0]])

    xs = data[:, 2]
    ys = -data[:, 0]
    zs = -data[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    data_ = np.array([[0.0253648, -0.291487, 0.936, 0.838853],
                      [0.0280206, -0.14233, 1.034, 0.731615],
                      [-0.107585, -0.132759, 1.049, 0.633176],
                      [-0.153826, 0.0465061, 1.118, 0.718431],
                      [-0.208253, 0.0884405, 0.948, 0.782415],
                      [0.162624, -0.155439, 1.068, 0.625039],
                      [0.216201, 0.111846, 1.159, 0.325232],
                      [0.243476, 0.286891, 1.162, 0.70658],
                      [0.0434262, 0.249637, 1.083, 0.404431],
                      [-0.0473387, 0.264614, 1.105, 0.434722],
                      [0.136639, 0.267727, 1.118, 0.34333],
                      [-0.00665411, -0.309962, 0.941, 0.919284],
                      [0.0547505, -0.31128, 0.945, 0.914358],
                      [-0.0549208, -0.300924, 0.984, 0.647673],
                      [0.0967744, -0.309493, 0.985, 0.622171],
                      [-0.268367, -0.877263,  0.997972, 0]])

    xs_ = data_[:, 2]
    ys_ = -data_[:, 0]
    zs_ = -data_[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs_, ys_, zs_)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    data__ = np.array([[0.0253648, -0.291487, 0.936, 0.840021],
                       [0.0281019, -0.144382, 1.037, 0.743221],
                       [-0.107585, -0.132759, 1.049, 0.633715],
                       [-0.15355, 0.0464229, 1.116, 0.699642],
                       [-0.206383, 0.0897727, 0.946, 0.738374],
                       [0.160493, -0.155002, 1.065, 0.628123],
                       [0.206701, 0.0761761, 1.156, 0.280646],
                       [0, 0, 0, 0],
                       [0.0452263, 0.253114, 1.084, 0.398296],
                       [-0.0473815, 0.264853, 1.106, 0.419442],
                       [0.138418, 0.267727, 1.118, 0.347455],
                       [-0.268364, -0.877263,  0.997972, 0]])

    xs__ = data__[:, 2]
    ys__ = -data__[:, 0]
    zs__ = -data__[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs__, ys__, zs__)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    _data = np.array([[-0.00670672,-0.261347,1.232,0.791065],
                      [-0.0114563,-0.122493,1.317,0.815811],
                      [-0.169975,-0.120447,1.295,0.61795],
                      [-0.229587,0.0793392,1.204,0.604195],
                      [-0.224509,0.16187,1.022,0.760085],
                      [0.151599,-0.131042,1.385,0.687723],
                      [0.20609,0.116049,1.443,0.54981],
                      [0.211955,0.341212,1.452,0.741384],
                      [-0.0110649,0.30839,1.272,0.449974],
                      [-0.0991909,0.31874,1.291,0.429357],
                      [0.0794051,0.322389,1.298,0.446328],
                      [-0.27101,-0.880283,0.975641,0]])

    _xs = _data[:, 2]
    _ys = -_data[:, 0]
    _zs = -_data[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(_xs, _ys, _zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

