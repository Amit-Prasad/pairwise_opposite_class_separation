import pandas as pd
import os
import numpy as np
from fpdf import FPDF
from helpers import *
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from sklearn import tree
from collections import Counter

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

color_map = {}
dir_points = {}
j=0
i=0

for k in range(0, 150):
    color_map[k] = cm.rainbow(k*0.5)
    if i*0.2 == 1.8:
        j += 0.1
        i=0
    dir_points[k] = [0.8 + j, i*0.2]
    i+=1


def min_distance_point(point, data_x, data_y, label):
    label_indices = np.where(data_y==label)[0]
    dist = np.zeros(data_y.shape) + np.Inf
    data_s = data_x[:, label_indices]
    dist[label_indices] = np.linalg.norm(data_s - point.reshape(-1, 1), axis = 0)
    min_index = np.argmin(dist)
    return min_index

# def line_at_planes_intersection(plane1, plane2):
#     '''r = a+lambda b'''
#     b = np.cross(plane1[0:-1], plane2[0:-1])
#     a = np.zeros(plane1.shape[0] - 1)
#     a[0:2] = np.dot(np.linalg.inv(np.array([plane1[0:2], plane2[0:2]])), np.array([-1*plane1[-1], -1*plane2[-1]]))
#     b = b/np.linalg.norm(b)
#     return a, b

# def project_line_in_plane_intersection(a, b, plane1, plane2):
#     grain = 0.001
#     lambs = np.arange(-100000, 100000, 100)
#     points = np.tile(a.reshape(-1, 1), (1, lambs.shape[0])) + np.tile(lambs, (b.shape[0], 1)) * np.tile(b.reshape(-1, 1), (1, lambs.shape[0]))
#     print(points.shape)
#     plane1_proj = np.dot(plane1[0:-1], points)
#     plane2_proj = np.dot(plane2[0:-1], points)
#     print('The max')
#     print(np.amax(plane1_proj))
#     print(np.amax(plane2_proj))
#     print(np.amin(plane1_proj))
#     print(np.amin(plane2_proj))
#     return plane1_proj, plane2_proj

def plot_line(x1, y1, deg):
    slope = np.deg2rad(deg)
    x = np.linspace(-1, 1, 100)

    y = y1 + (x - x1) * np.tan(slope)

    plt.plot(x, y)

    slope = np.deg2rad(deg*-1)
    x = np.linspace(-1, 1, 100)

    y = y1 + (x - x1) * np.tan(slope)

    plt.plot(x, y)

    slope = np.deg2rad(180-deg)
    x = np.linspace(-1, 1, 100)

    y = y1 + (x - x1) * np.tan(slope)

    plt.plot(x, y)

    slope = np.deg2rad((180-deg)*-1)
    x = np.linspace(-1, 1, 100)

    y = y1 + (x - x1) * np.tan(slope)

    plt.plot(x, y)

def find_point_projections_along_axis(points, vec_p, vec_n):
    n_proj=[]
    p_proj=[]
    for i in range(0, vec_n.shape[1]):
        n_proj.append(np.dot(vec_n[:, i].T, points[0:-1, ]))
        p_proj.append(np.dot(vec_p[:, i].T, points[0:-1, ]))
    n_proj = np.asarray(n_proj)
    p_proj = np.asarray(p_proj)
    return n_proj, p_proj

def axis_parallel_view(w_init, orig_hyp, data_x, data_y, candidate_points_pos, candidate_points_neg, iter, pdf):
    normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    parallels = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    orthogonal_data_x, parallel_data_x = find_point_projections_along_axis(data_x, parallels, normals)
    orthogonal_data_y, parallel_data_y = find_point_projections_along_axis(data_y, parallels, normals)
    orthogonal_candidate_points_pos, parallel_candidate_points_pos = find_point_projections_along_axis(candidate_points_pos, parallels, normals)
    orthogonal_candidate_points_neg, parallel_candidate_points_neg = find_point_projections_along_axis(candidate_points_neg, parallels, normals)
    midpoints = (candidate_points_pos + candidate_points_neg) / 2
    orthogonal_midpoints, parallel_midpoints = find_point_projections_along_axis(midpoints, parallels, normals)
    for i in range(0, parallel_data_x.shape[0]):
        #plot_line(orthogonal_midpoint_proj[i], parallel_midpoint_proj[i], 90 - 2)

        # plt.scatter(orthogonal_data_x, parallel_data_x[i], c='red', s=1)
        #plt.scatter(orthogonal_midpoint_proj[i], parallel_midpoint_proj[i], c='red', s=1)

        plt.scatter(orthogonal_candidate_points_pos[i, :], parallel_candidate_points_pos[i, :], c='lime', s=1)

        plt.scatter(orthogonal_candidate_points_neg[i, :], parallel_candidate_points_neg[i, :], c='pink', s=1)

        plt.scatter(orthogonal_midpoints[i, :], parallel_midpoints[i, :], c='red', s=1)
        plt.scatter(orthogonal_data_y[i, :], parallel_data_y[i, :], c='blue', s=1)
        x_min, x_max = orthogonal_candidate_points_pos[i].min() - 1, orthogonal_candidate_points_pos[i].max() + 1
        y_min, y_max = parallel_candidate_points_pos[i].min() - 1, parallel_candidate_points_pos[i].max() + 1
        plt.axis([x_min, x_max, y_min, y_max])
        plt.legend(loc='upper right')
        plt.savefig('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png')
        if i == 0:
            pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=0, y=0, w=300, h=300)
            #pdf.text(0, 310, 'The parallel line = ' + str(vectors[:, i]))
        elif i == 1:
            pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=310, y=0, w=300, h=300)
            #pdf.text(340, 310, 'The parallel line = ' + str(vectors[:, i]))
        elif i == 2:
            pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=0, y=310, w=300, h=300)
            #pdf.text(0, 330, 'The parallel line = ' + str(vectors[:, i]))
        else:
            pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=310, y=310, w=300, h=300)
            #pdf.text(340, 330, 'The parallel line = ' + str(vectors[:, i]))
        plt.gca().cla()
        pdf.text(20, 26, 'The hyperplane = ' + str(w_init))
        # pdf.text(340, 26, 'angle= ' + str((180/np.pi)*np.arccos(np.dot(w_init[0:-1], orig_hyp[0:-1])/(np.linalg.norm(w_init[0:-1]) * np.linalg.norm(orig_hyp[0:-1])))))
        # pdf.text(550, 26, 'Original= ' + str(orig_hyp))


def draw_projections_in_planes(plane1, plane2, data_x, data_y, candidate_points_pos, candidate_points_neg, name, pdf):
    # a, b = line_at_planes_intersection(plane1, plane2)
    # plane1_proj, plane2_proj = project_line_in_plane_intersection(a, b, plane1, plane2)
    for i in range(0, plane1.shape[1]):
        orthogonal_data_x, parallel_data_x = find_point_projections(data_x, plane2[0:-1], plane1[:, i])
        orthogonal_data_y, parallel_data_y = find_point_projections(data_y, plane2[0:-1], plane1[:, i])
        orthogonal_candidate_points_pos, parallel_candidate_points_pos = find_point_projections(candidate_points_pos, plane2[0:-1], plane1[:, i])
        orthogonal_candidate_points_neg, parallel_candidate_points_neg = find_point_projections(candidate_points_neg, plane2[0:-1], plane1[:, i])
        side_candidate_points_pos = np.sign(np.dot(plane1[:, i], candidate_points_pos))
        side_candidate_points_neg = np.sign(np.dot(plane1[:, i], candidate_points_neg))
        correct_side_pos = np.intersect1d(np.where(np.ones(candidate_points_pos.shape[1]) == 1)[0], np.where(side_candidate_points_pos == 1)[0])
        incorrect_side_pos = np.intersect1d(np.where(np.ones(candidate_points_pos.shape[1]) == 1)[0], np.where(side_candidate_points_pos == -1)[0])
        correct_side_neg = np.intersect1d(np.where(-1*np.ones(candidate_points_neg.shape[1]) == -1)[0], np.where(side_candidate_points_neg == -1)[0])
        incorrect_side_neg = np.intersect1d(np.where(-1*np.ones(candidate_points_neg.shape[1]) == -1)[0], np.where(side_candidate_points_neg == 1)[0])
        plt.scatter(orthogonal_data_x, parallel_data_x, c='red', s=1)
        plt.scatter(orthogonal_data_y, parallel_data_y, c='blue', s=1)
        plt.scatter(orthogonal_candidate_points_pos[correct_side_pos], parallel_candidate_points_pos[correct_side_pos], c='lime', s=1)
        plt.scatter(orthogonal_candidate_points_neg[correct_side_neg], parallel_candidate_points_neg[correct_side_neg], c='pink', s=1)

        plt.scatter(orthogonal_candidate_points_pos[incorrect_side_pos], parallel_candidate_points_pos[incorrect_side_pos], c='brown', s=1)
        plt.scatter(orthogonal_candidate_points_neg[incorrect_side_neg], parallel_candidate_points_neg[incorrect_side_neg], c='purple', s=1)

        midpoints = (candidate_points_pos + candidate_points_neg) / 2
        covariance = np.cov(midpoints[0:-1, :], bias=False)
        vec_orig = plane1[0:-1, i] / np.linalg.norm(plane1[0:-1, i])
        vec_gen = plane2[0:-1] / np.linalg.norm(plane2[0:-1])
        var_orig = np.dot(vec_orig.T, np.dot(covariance, vec_orig.reshape(-1, 1)))
        var_gen = np.dot(vec_gen.T, np.dot(covariance, vec_gen.reshape(-1, 1)))
        angle = np.arccos(np.dot(vec_orig, vec_gen))*(180/np.pi)
        plt.axis([-1, 1, -1, 1])
        plt.savefig(str(name) + '_' +str(i) + '.png')
        if i==0:
            pdf.image(str(name) + '_' +str(i) + '.png', x=0, y=0, w=300, h=300)
            pdf.text(0, 290, 'variance original vs variance generated' + str(var_orig) + ' ' + str(var_gen))
            pdf.text(0, 310, 'angle in between ' + str(angle))
        elif i==1:
            pdf.image(str(name) + '_' +str(i) + '.png', x=310, y=0, w=300, h=300)
            pdf.text(340, 290, 'variance original vs variance generated' + str(var_orig) + ' ' + str(var_gen))
            pdf.text(340, 310, 'angle in between ' + str(angle))
        elif i==2:
            pdf.image(str(name) + '_' +str(i) + '.png', x=0, y=310, w=300, h=300)
            pdf.text(0, 330, 'variance original vs variance generated' + str(var_orig) + ' '+ str(var_gen))
            pdf.text(0, 340, 'angle in between ' + str(angle))
        else:
            pdf.image(str(name) + '_' +str(i) + '.png', x=310, y=310, w=300, h=300)
            pdf.text(340, 330, 'variance original vs variance generated' + str(var_orig) + ' ' + str(var_gen))
            pdf.text(340, 340, 'angle in between ' + str(angle))
        plt.gca().cla()

def find_vectors_in_plane(w, n_lines=4):
    point_in_plane_1 = -1 + 2*np.random.random((w.shape[0]-1, n_lines))
    point_in_plane_1[-1] = -1*(w[-1] + np.dot(w[0:-2], point_in_plane_1[0:-1]))/w[-2]
    point_in_plane_2 = -1 + 2 * np.random.random((w.shape[0] - 1, n_lines))
    point_in_plane_2[-1] = -1 * (w[-1] + np.dot(w[0:-2], point_in_plane_2[0:-1]))/w[-2]
    vectors_in_plane = point_in_plane_1 - point_in_plane_2
    # vectors_in_plane = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return vectors_in_plane

def find_point_projections(data_x, vectors, w):
    orthogonal = np.dot(w[0:-1], data_x[0:-1])
    parallel = np.dot(vectors.T, data_x[0:-1])
    return orthogonal, parallel

def plot_projections(hyp, orig_hyp, vectors, orthogonal, parallel, data_y, labels, pdf, axis_parallel = True):
    true_positive_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 1)[0])
    true_negative_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 0)[0])
    false_positive_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 1)[0])
    false_negative_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 0)[0])
    if axis_parallel == True:
        for i in range(0, parallel.shape[0]):
            plt.scatter(orthogonal[i, true_positive_indices], parallel[i, true_positive_indices], c='lime', s=1)
            plt.scatter(orthogonal[i, true_negative_indices], parallel[i, true_negative_indices], c='pink', s=1)
            plt.scatter(orthogonal[i, false_positive_indices], parallel[i, false_positive_indices], c='blue', s=1)
            plt.scatter(orthogonal[i, false_negative_indices], parallel[i, false_negative_indices], c='red', s=1)
            x_min, x_max = orthogonal.min() - 1, orthogonal.max() + 1
            y_min, y_max = parallel[i].min() - 1, parallel[i].max() + 1
            plt.axis([x_min, x_max, y_min, y_max])
            plt.legend(loc='upper right')
            plt.savefig('labels_' + str(parallel[i, 0]) + str(i) + '.png')
            if i == 0:
                pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=0, y=0, w=280, h=280)
                pdf.text(0, 310, 'The parallel line = ' + str(vectors[:, i]))
            elif i == 1:
                pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=310, y=0, w=280, h=280)
                pdf.text(340, 310, 'The parallel line = ' + str(vectors[:, i]))
            elif i == 2:
                pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=0, y=310, w=280, h=280)
                pdf.text(0, 330, 'The parallel line = ' + str(vectors[:, i]))
            else:
                pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=310, y=310, w=280, h=280)
                pdf.text(340, 330, 'The parallel line = ' + str(vectors[:, i]))
            plt.gca().cla()

    else:
        for i in range(0, parallel.shape[0]):
            plt.scatter(orthogonal[true_positive_indices], parallel[i, true_positive_indices], c='lime', s=1)
            plt.scatter(orthogonal[true_negative_indices], parallel[i, true_negative_indices], c='pink', s=1)
            plt.scatter(orthogonal[false_positive_indices], parallel[i, false_positive_indices], c='blue', s=1)
            plt.scatter(orthogonal[false_negative_indices], parallel[i, false_negative_indices], c='red', s=1)

            x_min, x_max = orthogonal.min() - 1, orthogonal.max() + 1
            y_min, y_max = parallel[i].min() - 1, parallel[i].max() + 1
            plt.axis([x_min, x_max, y_min, y_max])
            plt.legend(loc='upper right')
            plt.savefig('labels_' + str(parallel[i, 0]) + str(i) + '.png')
            if i==0:
                pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=0, y=0, w=280, h=280)
                pdf.text(0, 310, 'The parallel line = ' + str(vectors[:, i]))
            elif i==1:
                pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=310, y=0, w=280, h=280)
                pdf.text(340, 310, 'The parallel line = ' + str(vectors[:, i]))
            elif i==2:
                pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=0, y=310, w=280, h=280)
                pdf.text(0, 330, 'The parallel line = ' + str(vectors[:, i]))
            else:
                pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=310, y=310, w=280, h=280)
                pdf.text(340, 330, 'The parallel line = ' + str(vectors[:, i]))
            plt.gca().cla()
        pdf.text(20, 26, 'The hyperplane = ' + str(hyp))
        pdf.text(400, 16, 'angle= ' + str((180/np.pi)*np.arccos(np.dot(hyp[0:-1], orig_hyp[0:-1])/(np.linalg.norm(hyp[0:-1]) * np.linalg.norm(orig_hyp[0:-1])))))
        pdf.text(500, 26, 'Original= ' + str(orig_hyp))

def visualise_multi(hyperplanes, orig_hyp, data_x, data_y, labels, pdf, axis_parallel = True):
    if axis_parallel == True:
        normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        parallels = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        orthogonal, parallel = find_point_projections(data_x, parallels, normals)
        plot_projections(hyperplanes, orig_hyp, orig_hyp, orthogonal, parallel, data_y, labels, pdf, axis_parallel)
        pdf.add_page()
    else:
        for i in range(0, hyperplanes.shape[1]):
            hyp = hyperplanes[:, i]
            vectors = find_vectors_in_plane(hyp, 4)
            orthogonal, parallel = find_point_projections(data_x, vectors, hyp)
            dots = np.abs(np.dot(hyp[0:-1], orig_hyp[0:-1, ]/np.linalg.norm(orig_hyp[0:-1, ], axis=0)))
            prob_hyp_in = np.argmax(dots)
            plot_projections(hyp, orig_hyp[:, prob_hyp_in], vectors, orthogonal, parallel, data_y, labels, pdf, axis_parallel)
            pdf.add_page()


def draw_lines(W, data_x, data_y, pdf):
    positive_indices = np.where(data_y == 1)[0]
    negative_indices = np.where(data_y == 0)[0]

    plt.scatter(data_x[0, positive_indices], data_x[1, positive_indices], c='blue', s=1)
    plt.scatter(data_x[0, negative_indices], data_x[1, negative_indices], c='yellow', s=1)
    grain = 0.001
    x_min, x_max = data_x[0, :].min() - 1, data_x[0, :].max() + 1
    y_min, y_max = data_x[1, :].min() - 1, data_x[1, :].max() + 1
    x1 = np.arange(x_min, x_max, grain)

    for w in W:
        if w[1] == 0:
            x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
        else:
            x2 = -1 * (w[2] + w[0] * x1) / w[1]
        plt.plot(x1, x2, color='black')

    plt.axis([x_min, x_max, y_min, y_max])
    plt.legend(loc='upper right')
    plt.savefig('all_lines.png')
    pdf.image('all_lines.png', x=0, y=0, w=320,h=320)
    plt.gca().cla()

def draw_points(data_x, data_y, labels, pdf):
    true_positive_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 1)[0])
    true_negative_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 0)[0])
    false_positive_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 1)[0])
    false_negative_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 0)[0])

    plt.scatter(data_x[0, true_positive_indices], data_x[1, true_positive_indices], c='blue', s=1)
    plt.scatter(data_x[0, true_negative_indices], data_x[1, true_negative_indices], c='yellow', s=1)
    plt.scatter(data_x[0, false_positive_indices], data_x[1, false_positive_indices], c='lime', s=1)
    plt.scatter(data_x[0, false_negative_indices], data_x[1, false_negative_indices], c='pink', s=1)

    x_min, x_max = data_x[0, :].min() - 1, data_x[0, :].max() + 1
    y_min, y_max = data_x[1, :].min() - 1, data_x[1, :].max() + 1
    plt.axis([x_min, x_max, y_min, y_max])
    plt.legend(loc='upper right')
    plt.savefig('labels.png')
    pdf.image('labels.png', x=0, y=0, w=320, h=320)
    plt.gca().cla()

def draw_set_separator_angle_cut(w_init, orig_hyp, midpoint_proj, data_x, data_y, candidate_points_pos, candidate_points_neg, iter, pdf, axis_parallel = True):
    vectors = find_vectors_in_plane(w_init, 4)
    orthogonal_data_x, parallel_data_x = find_point_projections(data_x, vectors, w_init)
    orthogonal_data_y, parallel_data_y = find_point_projections(data_y, vectors, w_init)
    orthogonal_midpoint_proj, parallel_midpoint_proj = find_point_projections(midpoint_proj, vectors, w_init)
    orthogonal_candidate_points_pos, parallel_candidate_points_pos = find_point_projections(candidate_points_pos, vectors, w_init)
    orthogonal_candidate_points_neg, parallel_candidate_points_neg = find_point_projections(candidate_points_neg, vectors, w_init)
    midpoints = (candidate_points_pos + candidate_points_neg)/2
    orthogonal_midpoints, parallel_midpoints = find_point_projections(midpoints, vectors, w_init)
    for i in range(0, parallel_data_x.shape[0]):
        plot_line(orthogonal_midpoint_proj, parallel_midpoint_proj[i], 90-2)

        # plt.scatter(orthogonal_data_x, parallel_data_x[i], c='red', s=1)
        # plt.scatter(orthogonal_midpoint_proj, parallel_midpoint_proj[i], c='red', s=1)

        plt.scatter(orthogonal_candidate_points_pos, parallel_candidate_points_pos[i, :], c='lime', s=1)

        plt.scatter(orthogonal_candidate_points_neg, parallel_candidate_points_neg[i, :], c='pink', s=1)

        plt.scatter(orthogonal_midpoints, parallel_midpoints[i, :], c='red', s=1)
        plt.scatter(orthogonal_data_y, parallel_data_y[i, :], c='blue', s=1)
        x_min, x_max = orthogonal_candidate_points_pos.min() - 1, orthogonal_candidate_points_pos.max() + 1
        y_min, y_max = parallel_candidate_points_pos[i].min() - 1, parallel_candidate_points_pos[i].max() + 1
        plt.axis([x_min, x_max, y_min, y_max])
        plt.legend(loc='upper right')
        plt.savefig('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png')
        if i==0:
            pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=0, y=0, w=300, h=300)
            pdf.text(0, 310, 'The parallel line = ' + str(vectors[:, i]))
        elif i==1:
            pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=310, y=0, w=300, h=300)
            pdf.text(340, 310, 'The parallel line = ' + str(vectors[:, i]))
        elif i==2:
            pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=0, y=310, w=300, h=300)
            pdf.text(0, 330, 'The parallel line = ' + str(vectors[:, i]))
        else:
            pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=310, y=310, w=300, h=300)
            pdf.text(340, 330, 'The parallel line = ' + str(vectors[:, i]))
        plt.gca().cla()
        pdf.text(20, 26, 'The hyperplane = ' + str(w_init))
        #pdf.text(340, 26, 'angle= ' + str((180/np.pi)*np.arccos(np.dot(w_init[0:-1], orig_hyp[0:-1])/(np.linalg.norm(w_init[0:-1]) * np.linalg.norm(orig_hyp[0:-1])))))
        #pdf.text(550, 26, 'Original= ' + str(orig_hyp))
        plt.gca().cla()

def draw_set_separator(w_init, orig_hyp, vectors, data_x, data_y, candidate_points_pos, candidate_points_neg, iter, pdf, axis_parallel = True, return_parallel = False, use_parallel = False):
    if axis_parallel==True:
        normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        parallels = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        orthogonal_data_x, parallel_data_x = find_point_projections_along_axis(data_x, parallels, normals)
        orthogonal_data_y, parallel_data_y = find_point_projections_along_axis(data_y, parallels, normals)
        orthogonal_candidate_points_pos, parallel_candidate_points_pos = find_point_projections_along_axis(candidate_points_pos, parallels, normals)
        orthogonal_candidate_points_neg, parallel_candidate_points_neg = find_point_projections_along_axis(candidate_points_neg, parallels, normals)
        for i in range(0, parallel_data_x.shape[0]):
            plt.scatter(orthogonal_data_x[i], parallel_data_x[i], c='red', s=1)

            plt.scatter(orthogonal_data_y[i], parallel_data_y[i, :], c='blue', s=1)

            plt.scatter(orthogonal_candidate_points_pos[i, :], parallel_candidate_points_pos[i, :], c='lime', s=1)

            plt.scatter(orthogonal_candidate_points_neg[i, :], parallel_candidate_points_neg[i, :], c='pink', s=1)

            # plt.scatter(orthogonal_midpoints, parallel_midpoints[i, :], c='red', s=1)
            x_min, x_max = orthogonal_candidate_points_pos[i].min() - 1, orthogonal_candidate_points_pos[i].max() + 1
            y_min, y_max = parallel_candidate_points_pos[i].min() - 1, parallel_candidate_points_pos[i].max() + 1
            plt.axis([-1.5, 2.0, y_min, y_max])
            plt.legend(loc='upper right')
            plt.savefig('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png')
            if i == 0:
                pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=0, y=0, w=300, h=300)
            elif i == 1:
                pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=310, y=0, w=300, h=300)
            elif i == 2:
                pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=0, y=310, w=300, h=300)
            else:
                pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=310, y=310, w=300, h=300)
            plt.gca().cla()
            pdf.text(20, 26, 'The hyperplane = ' + str(w_init))
            pdf.text(400, 16, 'angle= ' + str((180/np.pi)*np.arccos(np.dot(w_init[0:-1], orig_hyp[0:-1])/(np.linalg.norm(w_init[0:-1]) * np.linalg.norm(orig_hyp[0:-1])))))
            pdf.text(500, 26, 'Original= ' + str(orig_hyp))
            plt.gca().cla()
    else:
        if use_parallel == False:
            vectors = find_vectors_in_plane(w_init, 4)
        orthogonal_data_x, parallel_data_x = find_point_projections(data_x, vectors, w_init)
        orthogonal_data_y, parallel_data_y = find_point_projections(data_y, vectors, w_init)
        orthogonal_candidate_points_pos, parallel_candidate_points_pos = find_point_projections(candidate_points_pos, vectors, w_init)
        orthogonal_candidate_points_neg, parallel_candidate_points_neg = find_point_projections(candidate_points_neg, vectors, w_init)
        midpoints = (candidate_points_pos + candidate_points_neg)/2
        orthogonal_midpoints, parallel_midpoints = find_point_projections(midpoints, vectors, w_init)
        for i in range(0, parallel_data_x.shape[0]):
            plt.scatter(orthogonal_data_x, parallel_data_x[i], c='red', s=1)

            plt.scatter(orthogonal_data_y, parallel_data_y[i, :], c='blue', s=1)

            plt.scatter(orthogonal_candidate_points_pos, parallel_candidate_points_pos[i, :], c='lime', s=1)

            plt.scatter(orthogonal_candidate_points_neg, parallel_candidate_points_neg[i, :], c='pink', s=1)

            #plt.scatter(orthogonal_midpoints, parallel_midpoints[i, :], c='red', s=1)
            x_min, x_max = orthogonal_candidate_points_pos.min() - 1, orthogonal_candidate_points_pos.max() + 1
            y_min, y_max = parallel_candidate_points_pos[i].min() - 1, parallel_candidate_points_pos[i].max() + 1
            plt.axis([-2, 2, y_min, y_max])
            plt.legend(loc='upper right')
            plt.savefig('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png')
            if i==0:
                pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=0, y=0, w=300, h=300)
                pdf.text(0, 310, 'The parallel line = ' + str(vectors[:, i]))
            elif i==1:
                pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=310, y=0, w=300, h=300)
                pdf.text(340, 310, 'The parallel line = ' + str(vectors[:, i]))
            elif i==2:
                pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=0, y=310, w=300, h=300)
                pdf.text(0, 330, 'The parallel line = ' + str(vectors[:, i]))
            else:
                pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + str(i) + '.png', x=310, y=310, w=300, h=300)
                pdf.text(340, 330, 'The parallel line = ' + str(vectors[:, i]))
            plt.gca().cla()
            pdf.text(20, 26, 'The hyperplane = ' + str(w_init))
            pdf.text(400, 16, 'angle= ' + str((180/np.pi)*np.arccos(np.dot(w_init[0:-1], orig_hyp[0:-1])/(np.linalg.norm(w_init[0:-1]) * np.linalg.norm(orig_hyp[0:-1])))))
            pdf.text(550, 26, 'Original= ' + str(orig_hyp))
            plt.gca().cla()
    if return_parallel==True:
        return vectors


def local_search(S_pos, S_neg, uni_pos, uni_neg, ori_hyp, pdf):
    w_sol = np.zeros(S_pos.shape[0])
    x_indices = np.arange(0, uni_pos.shape[1])
    y_indices = np.arange(0, S_pos.shape[1])
    for x_index in x_indices:
        x_pos = uni_pos[:, x_index]
        x_neg = uni_neg[:, x_index]
        for y_index in y_indices:
            y_ind = y_index
            T_pos = np.zeros_like(S_pos)
            T_neg = np.zeros_like(S_neg)
            T_pos[:, 0:y_ind] = S_pos[:, 0:y_ind]
            T_pos[:, y_ind+1:] = S_pos[:, y_ind+1:]
            T_pos[:, y_ind] = x_pos
            T_neg[:, 0:y_ind] = S_neg[:, 0:y_ind]
            T_neg[:, y_ind + 1:] = S_neg[:, y_ind + 1:]
            T_neg[:, y_ind] = x_neg

            pca = PCA(n_components=S_pos.shape[0] - 1)
            midpoint_1 = (S_pos + S_neg) / 2
            pca.fit(midpoint_1[0:-1, :].T)
            w_s = np.zeros(S_pos.shape[0])
            w_s[0:-1] = pca.components_[S_pos.shape[0] - 2]
            mean = np.sum(midpoint_1[0:-1, :], axis=1) / midpoint_1.shape[1]
            w_s[-1] = np.dot(w_s[0:-1], mean) * -1
            dots = np.abs(np.dot(w_s[0:-1], ori_hyp[0:-1, ] / np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
            prob_hyp_in_s = np.argmax(dots)


            pca = PCA(n_components=T_pos.shape[0] - 1)
            midpoint_1 = (T_pos + T_neg) / 2
            pca.fit(midpoint_1[0:-1, :].T)
            w_t = np.zeros(T_pos.shape[0])
            w_t[0:-1] = pca.components_[T_pos.shape[0] - 2]
            mean = np.sum(midpoint_1[0:-1, :], axis=1) / midpoint_1.shape[1]
            w_t[-1] = np.dot(w_t[0:-1], mean) * -1
            dots = np.abs(np.dot(w_t[0:-1], ori_hyp[0:-1, ] / np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
            prob_hyp_in_t = np.argmax(dots)


            cost_t = u_shaped_x_by_one_plus_x(w_t, np.hstack((S_pos, uni_pos)), np.hstack((S_neg, uni_neg)))
            cost_s = u_shaped_x_by_one_plus_x(w_s, np.hstack((S_pos, uni_pos)), np.hstack((S_neg, uni_neg)))

            if cost_t < cost_s:
                pdf.text(0, 310, 'Hyperplane S: ' + str(w_s))
                pdf.text(10, 326, 'Closest to: ' + str(ori_hyp[:, prob_hyp_in_s]))
                angle = (180 / np.pi) * np.arccos(np.dot(w_s[0:-1], ori_hyp[:, prob_hyp_in_s][0:-1]) / (np.linalg.norm(w_s[0:-1]) * np.linalg.norm(ori_hyp[:, prob_hyp_in_s][0:-1])))
                pdf.text(10, 342, 'angle: ' + str(angle))
                pdf.text(10, 358, 'index: ' + str(prob_hyp_in_s + 1))
                pdf.text(10, 374, 'Hyperplane T: ' + str(w_t))
                pdf.text(10, 390, 'Closest to: ' + str(ori_hyp[:, prob_hyp_in_t]))
                angle = (180 / np.pi) * np.arccos(np.dot(w_t[0:-1], ori_hyp[:, prob_hyp_in_t][0:-1]) / (np.linalg.norm(w_t[0:-1]) * np.linalg.norm(ori_hyp[:, prob_hyp_in_t][0:-1])))
                pdf.text(10, 406, 'angle: ' + str(angle))
                pdf.text(10, 422, 'index: ' + str(prob_hyp_in_t + 1))
                pdf.text(10, 438, 'Cost T: ' + str(cost_t))
                pdf.text(10, 454, 'Cost S: ' + str(cost_s))
                pdf.add_page()
                uni_pos[:, x_index] = S_pos[:, y_ind]
                uni_neg[:, x_index] = S_neg[:, y_ind]
                S_pos = T_pos
                S_neg = T_neg
            else:
                w_sol = w_s
    return w_sol, S_pos, S_neg, uni_pos, uni_neg

def great_pick_algo_3(candidate_points_pos, candidate_points_neg, data_x, data_y, removed, ori_hyp, p, pdf, axis_parallel):
    angles = [10*np.pi/180]#, (9*np.pi)/180, (8*np.pi)/180, (7*np.pi)/180]
    w_f = np.zeros(candidate_points_pos.shape[0])
    rand_init_point = np.zeros(candidate_points_pos.shape[0])
    rand_closest_point = np.zeros(candidate_points_pos.shape[0])
    vectors = np.zeros((3, 5))
    candidate_points_pos_1 = np.copy(candidate_points_pos)
    candidate_points_neg_1 = np.copy(candidate_points_neg)
    for ang in angles:
        w, rand_init_point, rand_closest_point = pick_init_hyp_2(candidate_points_pos, candidate_points_neg, data_x, data_y, removed, ori_hyp, pdf)
        _, _, _ = pick_init_hyp_1(candidate_points_pos, candidate_points_neg, ori_hyp, pdf)
        if w is None:
            return None, None, None, None, None
        midpoints = (candidate_points_pos + candidate_points_neg)/2
        dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
        prob_hyp_in = np.argmax(dots)
        pdf.text(610, 42, 'PCA first')
        c = (candidate_points_pos + candidate_points_neg) / 2
        d = (candidate_points_pos - candidate_points_neg) / 2
        points_pos = c + 5 * d
        points_neg = c - 5 * d
        draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 10 + 0), pdf, axis_parallel)
        pdf.add_page()
        # # draw_projections_in_planes(ori_hyp, w, rand_init_point, rand_closest_point, points_pos, points_neg, 'line_in_plane' + str(p * 4 * 4 + 0), pdf)
        # # pdf.add_page()
        # # draw_set_separator(w, ori_hyp, rand_init_point, rand_closest_point, points_pos, points_neg, 'line_in_plane_' + str(p * 4 * 4 + 1), pdf, axis_parallel)
        # # pdf.add_page()

        angles_1 = np.pi/6
        w_prime = w
        for i in [1]:
            #w = w_prime
            master_midpoint = np.sum(rand_closest_point, axis=1) / rand_closest_point.shape[1]
            master_midpoint_proj = np.zeros(w.shape)
            master_midpoint_proj[0:-1] = master_midpoint[0:-1] - (np.dot(w, master_midpoint) / np.sqrt(np.dot(w[0:-1], w[0:-1]))) * w[0:-1]
            master_midpoint_proj[-1] = 1
            n_points_ = candidate_points_pos.shape[0]
            hypotenuse = np.linalg.norm(midpoints - master_midpoint_proj.reshape(-1, 1), axis=0)
            perpendicular = np.abs(np.dot(w, midpoints)/np.sqrt(np.dot(w[0:-1], w[0:-1])))
            dist_order_prom_proj = np.argsort(np.sqrt(hypotenuse ** 2 - perpendicular ** 2))
            angles_ = np.abs(np.arcsin(perpendicular/hypotenuse))
            again_taken = dist_order_prom_proj[0:n_points_]
            side_pos = np.sign(np.dot(ori_hyp.T, candidate_points_pos[:, again_taken]))
            side_neg = np.sign(np.dot(ori_hyp.T, candidate_points_neg[:, again_taken]))
            hyp_in = np.where(side_pos != side_neg, 1, 0)
            indexes = np.union1d(np.where(angles_ <= ang)[0], dist_order_prom_proj[0:n_points_])
            candidate_points_pos_1 = candidate_points_pos[:, indexes]
            candidate_points_neg_1 = candidate_points_neg[:, indexes]
            S_pos = candidate_points_pos[:, 0:n_points_]
            S_neg = candidate_points_neg[:, 0:n_points_]
            uni_pos = candidate_points_pos[:, n_points_:]
            uni_neg = candidate_points_neg[:, n_points_:]
            points_pos_ = S_pos
            points_neg_ = S_neg
            side_pos = np.sign(np.dot(ori_hyp.T, points_pos_))
            side_neg = np.sign(np.dot(ori_hyp.T, points_neg_))
            hyp_in = np.where(side_pos != side_neg, 1, 0)
            for j in range(0, ori_hyp.shape[1]):
                pdf.text(10, 16 * (j + 4), str(hyp_in[j, :]))
                pdf.text(200, 16 * (j + 4), str(np.sum(hyp_in[j, :])))
                pdf.text(230, 16 * (j + 4), str(j+1))
            pdf.add_page()
            w, S_pos, S_neg, uni_pos, uni_neg = local_search(S_pos, S_neg, uni_pos, uni_neg, ori_hyp, pdf)
            points_pos_ = S_pos
            points_neg_ = S_neg
            side_pos = np.sign(np.dot(ori_hyp.T, points_pos_))
            side_neg = np.sign(np.dot(ori_hyp.T, points_neg_))
            hyp_in = np.where(side_pos != side_neg, 1, 0)
            for j in range(0, ori_hyp.shape[1]):
                pdf.text(10, 16 * (j + 4), str(hyp_in[j, :]))
                pdf.text(200, 16 * (j + 4), str(np.sum(hyp_in[j, :])))
                pdf.text(230, 16 * (j + 4), str(j+1))
            side_pos = np.sign(np.dot(w.T, points_pos_))
            side_neg = np.sign(np.dot(w.T, points_neg_))
            hyp_in = np.where(side_pos != side_neg, 1, 0)
            pdf.text(100, 400, str(hyp_in))
            pdf.add_page()

            # side_pos = np.sign(np.dot(ori_hyp.T, candidate_points_pos_1))
            # side_neg = np.sign(np.dot(ori_hyp.T, candidate_points_neg_1))
            # hyp_in_ = np.where(side_pos != side_neg, 1, 0)
            # oppo_per_hyp = np.sum(hyp_in_, axis=1)
            # hyp_max_oppo = ori_hyp[:, np.argmax(oppo_per_hyp)]
            # new_indexes = np.where(np.sign(np.dot(hyp_max_oppo, candidate_points_pos_1)) != np.sign(np.dot(hyp_max_oppo, candidate_points_neg_1)))[0]
            # candidate_points_pos_1 = candidate_points_pos_1[:, new_indexes]
            # candidate_points_neg_1 = candidate_points_neg_1[:, new_indexes]
            #
            # side_pos = np.sign(np.dot(ori_hyp.T, candidate_points_pos_1))
            # side_neg = np.sign(np.dot(ori_hyp.T, candidate_points_neg_1))
            # hyp_in_ = np.where(side_pos != side_neg, 1, 0)
            # pointing_vectors = candidate_points_neg[:, again_taken] - candidate_points_pos[:, again_taken]
            # normed_pointing_vectors = pointing_vectors / np.linalg.norm(pointing_vectors, axis=0)
            # avg_pointing_vector = np.sum(normed_pointing_vectors, axis=1) / normed_pointing_vectors.shape[1]
            # avg_pointing_vector = avg_pointing_vector / np.linalg.norm(avg_pointing_vector)
            # normed_pointing_vectors = np.hstack((normed_pointing_vectors, avg_pointing_vector.reshape(-1, 1)))
            # pdf.set_font('Helvetica', 'I', 7)
            # for j in range(0, ori_hyp.shape[1]):
            #     printed = np.zeros((pointing_vectors.shape[1], 3),dtype = int)
            #     angs = (np.arccos(np.dot(ori_hyp[:, j], normed_pointing_vectors)/ np.linalg.norm(ori_hyp[:, j]))*(180/np.pi)).astype(int)
            #     printed[:, 0] = hyp_in[j, :]
            #     printed[:, 1] = angs[0:-1]
            #     angs_2 = (np.arccos(np.dot(avg_pointing_vector, normed_pointing_vectors[:, 0:-1]))*(180/np.pi)).astype(int)
            #     printed[:, 2] = angs_2
            #     printed = printed[printed[:, 0].argsort()]
            #     pdf.text(10, 16 * (2 * j + 4), str(printed[:, 0:-1]))
            #     pdf.text(10, 16 * (2*j + 1 + 4), str(printed[:, [0, 2]]))
            #     pdf.text(750, 16 * (2*j + 4), str(j + 1))
            #     pdf.text(770, 16 * (2*j + 4), str(np.sum(hyp_in[j, :])))
            #     pdf.text(790, 16 * (2*j + 4), str(angs[-1]))
            # pdf.set_font('Helvetica', 'I', 10)
            # pdf.add_page()

            # pointing_vectors = candidate_points_neg_1 - candidate_points_pos_1
            # normed_pointing_vectors = pointing_vectors / np.linalg.norm(pointing_vectors, axis=0)
            # avg_pointing_vector = np.sum(normed_pointing_vectors, axis=1) / normed_pointing_vectors.shape[1]
            # avg_pointing_vector = avg_pointing_vector / np.linalg.norm(avg_pointing_vector)
            # normed_pointing_vectors = np.hstack((normed_pointing_vectors, avg_pointing_vector.reshape(-1, 1)))
            # pdf.set_font('Helvetica', 'I', 3)
            # for j in range(0, ori_hyp.shape[1]):
            #     printed = np.zeros((pointing_vectors.shape[1], 3), dtype=int)
            #     angs = (np.arccos(np.dot(ori_hyp[:, j], normed_pointing_vectors) / np.linalg.norm(ori_hyp[:, j])) * (180 / np.pi)).astype(int)
            #     printed[:, 0] = hyp_in_[j, :]
            #     printed[:, 1] = angs[0:-1]
            #     angs_2 = (np.arccos(np.dot(avg_pointing_vector, normed_pointing_vectors[:, 0:-1])) * (180 / np.pi)).astype(int)
            #     printed[:, 2] = angs_2
            #     printed = printed[printed[:, 0].argsort()]
            #     pdf.text(10, 16 * (2 * j + 4), str(printed[:, 0:-1]))
            #     pdf.text(10, 16 * (2 * j + 1 + 4), str(printed[:, [0, 2]]))
            #     pdf.text(750, 16 * (2 * j + 4), str(j + 1))
            #     pdf.text(770, 16 * (2 * j + 4), str(np.sum(hyp_in_[j, :])))
            #     pdf.text(790, 16 * (2 * j + 4), str(angs[-1]))
            # pdf.set_font('Helvetica', 'I', 10)
            # pdf.add_page()
            # coshci = np.arccos(np.dot(avg_pointing_vectors, normed_pointing_vectors)/ np.linalg.norm(avg_pointing_vectors))
            # l = ["{:0.2f}".format(a) for a in coshci * 180/np.pi]
            # pdf.text(10, 16 * (ori_hyp.shape[1] + 4), str(l))
            # pdf.add_page()
            c = (candidate_points_pos_1 + candidate_points_neg_1) / 2
            d = (candidate_points_pos_1 - candidate_points_neg_1) / 2
            points_pos = c + 5 * d
            points_neg = c - 5 * d
            pdf.text(610, 42, 'PCA first discard ' + str(i) + 'd points ' + str(angles_1))
            dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
            prob_hyp_in = np.argmax(dots)
            draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(i) + str(p * 10 + 1), pdf, axis_parallel)
            pdf.add_page()
            w_f = w
            # if candidate_points_pos_1.shape[1] < candidate_points_pos_1.shape[0]-1:
            #     return None, None, None, None, None
            # pca = PCA(n_components = candidate_points_pos_1.shape[0] - 1)
            # midpoint_1 = np.hstack(((candidate_points_pos_1 + candidate_points_neg_1)/2, rand_closest_point))
            # pca.fit(midpoint_1[0:-1, :].T)
            # w = np.zeros(candidate_points_pos_1.shape[0])
            # w[0:-1] = pca.components_[candidate_points_pos_1.shape[0] - 2]
            # mean = np.sum(midpoint_1[0:-1, :], axis=1)/midpoint_1.shape[1]
            # w[-1] = np.dot(w[0:-1], mean)*-1
            # c = (candidate_points_pos_1 + candidate_points_neg_1) / 2
            # d = (candidate_points_pos_1 - candidate_points_neg_1) / 2
            # points_pos = c + 5 * d
            # points_neg = c - 5 * d
            # pdf.text(610, 32, 'PCA second '+ str(i) + 'd points ' + str(angles_1))
            # dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
            # prob_hyp_in = np.argmax(dots)
            # draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(i) + str(p * 10 + 2), pdf, axis_parallel)
            # pdf.add_page()
            # w_f = w
            # angles_1 = angles_1/2
    print("Here is one")
    print(w_f)
    return w_f, rand_init_point, rand_closest_point, candidate_points_pos_1, candidate_points_neg_1


def great_pick_algo_2(candidate_points_pos, candidate_points_neg, data_x, data_y, removed, ori_hyp, p, pdf, axis_parallel):
    angles=[10*np.pi/180]#, (9*np.pi)/180, (8*np.pi)/180, (7*np.pi)/180]
    w_f = np.zeros(candidate_points_pos.shape[0])
    rand_init_point = np.zeros(candidate_points_pos.shape[0])
    rand_closest_point = np.zeros(candidate_points_pos.shape[0])
    vectors = np.zeros((3, 5))
    candidate_points_pos_1 = np.copy(candidate_points_pos)
    candidate_points_neg_1 = np.copy(candidate_points_neg)
    for ang in angles:
        w, rand_init_point, rand_closest_point = pick_init_hyp_2(candidate_points_pos, candidate_points_neg, data_x, data_y, removed, ori_hyp, pdf)
        _, _, _ = pick_init_hyp_1(candidate_points_pos, candidate_points_neg, ori_hyp, pdf)
        if w is None:
            return None, None, None, None, None
        midpoints = (candidate_points_pos + candidate_points_neg)/2
        dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
        prob_hyp_in = np.argmax(dots)
        pdf.text(610, 42, 'PCA first')
        c = (candidate_points_pos + candidate_points_neg) / 2
        d = (candidate_points_pos - candidate_points_neg) / 2
        points_pos = c + 5 * d
        points_neg = c - 5 * d
        draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 10 + 0), pdf, axis_parallel)
        pdf.add_page()
        # # draw_projections_in_planes(ori_hyp, w, rand_init_point, rand_closest_point, points_pos, points_neg, 'line_in_plane' + str(p * 4 * 4 + 0), pdf)
        # # pdf.add_page()
        # # draw_set_separator(w, ori_hyp, rand_init_point, rand_closest_point, points_pos, points_neg, 'line_in_plane_' + str(p * 4 * 4 + 1), pdf, axis_parallel)
        # # pdf.add_page()

        angles_1 = np.pi/6
        w_prime = w
        for i in [2, 3, 5, 7]:
            #w = w_prime
            master_midpoint = np.sum(rand_closest_point, axis=1) / rand_closest_point.shape[1]
            master_midpoint_proj = np.zeros(w.shape)
            master_midpoint_proj[0:-1] = master_midpoint[0:-1] - (np.dot(w, master_midpoint) / np.sqrt(np.dot(w[0:-1], w[0:-1]))) * w[0:-1]
            master_midpoint_proj[-1] = 1
            n_points_ = i*(candidate_points_pos.shape[0]-1)
            hypotenuse = np.linalg.norm(midpoints - master_midpoint_proj.reshape(-1, 1), axis=0)
            dist_order_prom_proj = np.argsort(hypotenuse)
            perpendicular = np.abs(np.dot(w, midpoints)/np.sqrt(np.dot(w[0:-1], w[0:-1])))
            angles_ = np.abs(np.arcsin(perpendicular/hypotenuse))
            again_taken = dist_order_prom_proj[0:n_points_]
            side_pos = np.sign(np.dot(ori_hyp.T, candidate_points_pos[:, again_taken]))
            side_neg = np.sign(np.dot(ori_hyp.T, candidate_points_neg[:, again_taken]))
            hyp_in = np.where(side_pos != side_neg, 1, 0)
            indexes = np.union1d(np.where(angles_ <= ang)[0], dist_order_prom_proj[0:n_points_])
            candidate_points_pos_1 = candidate_points_pos[:, indexes]
            candidate_points_neg_1 = candidate_points_neg[:, indexes]

            side_pos = np.sign(np.dot(ori_hyp.T, candidate_points_pos_1))
            side_neg = np.sign(np.dot(ori_hyp.T, candidate_points_neg_1))
            hyp_in_ = np.where(side_pos != side_neg, 1, 0)
            oppo_per_hyp = np.sum(hyp_in_, axis=1)
            hyp_max_oppo = ori_hyp[:, np.argmax(oppo_per_hyp)]
            new_indexes = np.where(np.sign(np.dot(hyp_max_oppo, candidate_points_pos_1)) != np.sign(np.dot(hyp_max_oppo, candidate_points_neg_1)))[0]
            candidate_points_pos_1 = candidate_points_pos_1[:, new_indexes]
            candidate_points_neg_1 = candidate_points_neg_1[:, new_indexes]

            side_pos = np.sign(np.dot(ori_hyp.T, candidate_points_pos_1))
            side_neg = np.sign(np.dot(ori_hyp.T, candidate_points_neg_1))
            hyp_in_ = np.where(side_pos != side_neg, 1, 0)
            pointing_vectors = candidate_points_neg[:, again_taken] - candidate_points_pos[:, again_taken]
            normed_pointing_vectors = pointing_vectors / np.linalg.norm(pointing_vectors, axis=0)
            avg_pointing_vector = np.sum(normed_pointing_vectors, axis=1) / normed_pointing_vectors.shape[1]
            avg_pointing_vector = avg_pointing_vector / np.linalg.norm(avg_pointing_vector)
            normed_pointing_vectors = np.hstack((normed_pointing_vectors, avg_pointing_vector.reshape(-1, 1)))
            pdf.set_font('Helvetica', 'I', 7)
            for j in range(0, ori_hyp.shape[1]):
                printed = np.zeros((pointing_vectors.shape[1], 3),dtype = int)
                angs = (np.arccos(np.dot(ori_hyp[:, j], normed_pointing_vectors)/ np.linalg.norm(ori_hyp[:, j]))*(180/np.pi)).astype(int)
                printed[:, 0] = hyp_in[j, :]
                printed[:, 1] = angs[0:-1]
                angs_2 = (np.arccos(np.dot(avg_pointing_vector, normed_pointing_vectors[:, 0:-1]))*(180/np.pi)).astype(int)
                printed[:, 2] = angs_2
                printed = printed[printed[:, 0].argsort()]
                pdf.text(10, 16 * (2 * j + 4), str(printed[:, 0:-1]))
                pdf.text(10, 16 * (2*j + 1 + 4), str(printed[:, [0, 2]]))
                pdf.text(750, 16 * (2*j + 4), str(j + 1))
                pdf.text(770, 16 * (2*j + 4), str(np.sum(hyp_in[j, :])))
                pdf.text(790, 16 * (2*j + 4), str(angs[-1]))
            pdf.set_font('Helvetica', 'I', 10)
            pdf.add_page()

            pointing_vectors = candidate_points_neg_1 - candidate_points_pos_1
            normed_pointing_vectors = pointing_vectors / np.linalg.norm(pointing_vectors, axis=0)
            avg_pointing_vector = np.sum(normed_pointing_vectors, axis=1) / normed_pointing_vectors.shape[1]
            avg_pointing_vector = avg_pointing_vector / np.linalg.norm(avg_pointing_vector)
            normed_pointing_vectors = np.hstack((normed_pointing_vectors, avg_pointing_vector.reshape(-1, 1)))
            pdf.set_font('Helvetica', 'I', 3)
            for j in range(0, ori_hyp.shape[1]):
                printed = np.zeros((pointing_vectors.shape[1], 3), dtype=int)
                angs = (np.arccos(np.dot(ori_hyp[:, j], normed_pointing_vectors) / np.linalg.norm(ori_hyp[:, j])) * (180 / np.pi)).astype(int)
                printed[:, 0] = hyp_in_[j, :]
                printed[:, 1] = angs[0:-1]
                angs_2 = (np.arccos(np.dot(avg_pointing_vector, normed_pointing_vectors[:, 0:-1])) * (180 / np.pi)).astype(int)
                printed[:, 2] = angs_2
                printed = printed[printed[:, 0].argsort()]
                pdf.text(10, 16 * (2 * j + 4), str(printed[:, 0:-1]))
                pdf.text(10, 16 * (2 * j + 1 + 4), str(printed[:, [0, 2]]))
                pdf.text(750, 16 * (2 * j + 4), str(j + 1))
                pdf.text(770, 16 * (2 * j + 4), str(np.sum(hyp_in_[j, :])))
                pdf.text(790, 16 * (2 * j + 4), str(angs[-1]))
            pdf.set_font('Helvetica', 'I', 10)
            pdf.add_page()
            # coshci = np.arccos(np.dot(avg_pointing_vectors, normed_pointing_vectors)/ np.linalg.norm(avg_pointing_vectors))
            # l = ["{:0.2f}".format(a) for a in coshci * 180/np.pi]
            # pdf.text(10, 16 * (ori_hyp.shape[1] + 4), str(l))
            # pdf.add_page()
            c = (candidate_points_pos_1 + candidate_points_neg_1) / 2
            d = (candidate_points_pos_1 - candidate_points_neg_1) / 2
            points_pos = c + 5 * d
            points_neg = c - 5 * d
            pdf.text(610, 42, 'PCA first discard ' + str(i) + 'd points ' + str(angles_1))
            dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
            prob_hyp_in = np.argmax(dots)
            draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(i) + str(p * 10 + 1), pdf, axis_parallel)
            pdf.add_page()
            if candidate_points_pos_1.shape[1] < candidate_points_pos_1.shape[0]-1:
                return None, None, None, None, None
            pca = PCA(n_components = candidate_points_pos_1.shape[0] - 1)
            midpoint_1 = np.hstack(((candidate_points_pos_1 + candidate_points_neg_1)/2, rand_closest_point))
            pca.fit(midpoint_1[0:-1, :].T)
            w = np.zeros(candidate_points_pos_1.shape[0])
            w[0:-1] = pca.components_[candidate_points_pos_1.shape[0] - 2]
            mean = np.sum(midpoint_1[0:-1, :], axis=1)/midpoint_1.shape[1]
            w[-1] = np.dot(w[0:-1], mean)*-1
            c = (candidate_points_pos_1 + candidate_points_neg_1) / 2
            d = (candidate_points_pos_1 - candidate_points_neg_1) / 2
            points_pos = c + 5 * d
            points_neg = c - 5 * d
            pdf.text(610, 32, 'PCA second '+ str(i) + 'd points ' + str(angles_1))
            dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
            prob_hyp_in = np.argmax(dots)
            draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(i) + str(p * 10 + 2), pdf, axis_parallel)
            pdf.add_page()
            w_f = w
            angles_1 = angles_1/2
    return w_f, rand_init_point, rand_closest_point, candidate_points_pos_1, candidate_points_neg_1

def great_pick_algo_1(candidate_points_pos, candidate_points_neg, data_x, data_y, removed, ori_hyp, p, pdf, axis_parallel):
    angles=[10*np.pi/180]#, (9*np.pi)/180, (8*np.pi)/180, (7*np.pi)/180]
    w_f = np.zeros(candidate_points_pos.shape[0])
    rand_init_point = np.zeros(candidate_points_pos.shape[0])
    rand_closest_point = np.zeros(candidate_points_pos.shape[0])
    vectors = np.zeros((3, 5))
    for ang in angles:
        w, rand_init_point, rand_closest_point = pick_init_hyp_2(candidate_points_pos, candidate_points_neg, data_x, data_y, removed, ori_hyp, pdf)
        _, _, _ = pick_init_hyp_1(candidate_points_pos, candidate_points_neg, ori_hyp, pdf)
        if w is None:
            return None, None, None, None, None
        midpoints = (candidate_points_pos + candidate_points_neg)/2
        dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
        prob_hyp_in = np.argmax(dots)
        pdf.text(610, 42, 'PCA first')
        c = (candidate_points_pos + candidate_points_neg) / 2
        d = (candidate_points_pos - candidate_points_neg) / 2
        points_pos = c + 5 * d
        points_neg = c - 5 * d
        draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 10 + 0), pdf, axis_parallel)
        pdf.add_page()
        # # draw_projections_in_planes(ori_hyp, w, rand_init_point, rand_closest_point, points_pos, points_neg, 'line_in_plane' + str(p * 4 * 4 + 0), pdf)
        # # pdf.add_page()
        # # draw_set_separator(w, ori_hyp, rand_init_point, rand_closest_point, points_pos, points_neg, 'line_in_plane_' + str(p * 4 * 4 + 1), pdf, axis_parallel)
        # # pdf.add_page()
        master_midpoint = np.sum(rand_closest_point, axis=1)/rand_closest_point.shape[1]
        master_midpoint_proj = np.zeros(w.shape)
        master_midpoint_proj[0:-1] = master_midpoint[0:-1]-(np.dot(w, master_midpoint)/np.sqrt(np.dot(w[0:-1], w[0:-1])))*w[0:-1]
        master_midpoint_proj[-1] = 1
        hypotenuse = np.linalg.norm(midpoints - master_midpoint_proj.reshape(-1, 1), axis=0)
        perpendicular = np.abs(np.dot(w, midpoints)/np.sqrt(np.dot(w[0:-1], w[0:-1])))
        angles_ = np.abs(np.arcsin(perpendicular/hypotenuse))
        indexes = np.where(angles_<=ang)[0]
        candidate_points_pos = candidate_points_pos[:, indexes]
        candidate_points_neg = candidate_points_neg[:, indexes]
        c = (candidate_points_pos + candidate_points_neg) / 2
        d = (candidate_points_pos - candidate_points_neg) / 2
        points_pos = c + 5 * d
        points_neg = c - 5 * d
        pdf.text(610, 42, 'PCA first discard')
        dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
        prob_hyp_in = np.argmax(dots)
        draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 10 + 1), pdf, axis_parallel)
        pdf.add_page()
        if candidate_points_pos.shape[1] < candidate_points_pos.shape[0]-1:
            return None, None, None, None, None
        pca = PCA(n_components = candidate_points_pos.shape[0] - 1)
        pca.fit((candidate_points_pos + candidate_points_neg/2)[0:-1, :].T)
        w = np.zeros(candidate_points_pos.shape[0])
        w[0:-1] = pca.components_[candidate_points_pos.shape[0] - 2]
        mean = np.sum(((candidate_points_pos + candidate_points_neg)/2)[0:-1, :], axis=1)/candidate_points_pos.shape[1]
        w[-1] = np.dot(w[0:-1], mean)*-1
        c = (candidate_points_pos + candidate_points_neg) / 2
        d = (candidate_points_pos - candidate_points_neg) / 2
        points_pos = c + 5 * d
        points_neg = c - 5 * d
        pdf.text(610, 32, 'PCA second')
        dots = np.abs(np.dot(w[0:-1], ori_hyp[0:-1, ]/np.linalg.norm(ori_hyp[0:-1, ], axis=0)))
        prob_hyp_in = np.argmax(dots)
        draw_set_separator(w, ori_hyp[:, prob_hyp_in], ori_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 10 + 2), pdf, axis_parallel)
        pdf.add_page()
        w_f=w
    return w_f, rand_init_point, rand_closest_point, candidate_points_pos, candidate_points_neg

def great_pick_algo(candidate_points_pos, candidate_points_neg, ori_hyp, p, pdf):
    c = (candidate_points_pos + candidate_points_neg) / 2
    d = (candidate_points_pos - candidate_points_neg) / 2
    points_pos = c + 5 * d
    points_neg = c - 5 * d
    midpoints = c
    w, rand_init_point, rand_closest_point = pick_init_hyp_1(candidate_points_pos, candidate_points_neg)
    if w is None:
        return None, None, None, None, None
    pdf.text(610, 32, 'The hyperplane picked through PCA')
    # draw_set_separator(w, ori_hyp[:, 0], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 6 + 0), pdf)
    # pdf.add_page()
    # draw_set_separator(ori_hyp[:, 0], ori_hyp[:, 0], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 6 + 1), pdf)
    # pdf.add_page()
    draw_projections_in_planes(ori_hyp, w, rand_init_point, rand_closest_point, points_pos, points_neg, 'line_in_plane' + str(p * 6 + 0), pdf)
    pdf.add_page()
    master_midpoint = np.sum(rand_closest_point, axis=1)/rand_closest_point.shape[1]
    master_midpoint_proj = np.zeros(w.shape)
    master_midpoint_proj[0:-1] = master_midpoint[0:-1] - (np.dot(w, master_midpoint)/np.sqrt(np.dot(w[0:-1], w[0:-1])))*w[0:-1]
    master_midpoint_proj[-1] = 1
    hypotenuse = np.linalg.norm(midpoints - master_midpoint_proj.reshape(-1, 1), axis=0)
    perpendicular = np.dot(w, midpoints)/np.sqrt(np.dot(w[0:-1], w[0:-1]))
    angles = np.abs(np.arcsin(perpendicular/hypotenuse))
    indexes = np.where((angles<=np.pi/18))[0]
    candidate_points_pos = candidate_points_pos[:, indexes]
    candidate_points_neg = candidate_points_neg[:, indexes]
    pca = PCA(n_components = candidate_points_pos.shape[0] - 2)
    pca.fit((candidate_points_pos + candidate_points_neg/2)[0:-1, :].T)
    w = np.zeros(candidate_points_pos.shape[0])
    w[0:-1] = pca.components_[candidate_points_pos.shape[0] - 3]
    mean = np.sum(((candidate_points_pos + candidate_points_neg)/2)[0:-1, :], axis=1)/candidate_points_pos.shape[1]
    w[-1] = np.dot(w[0:-1], mean)*-1
    c = (candidate_points_pos + candidate_points_neg) / 2
    d = (candidate_points_pos - candidate_points_neg) / 2
    points_pos = c + 5 * d
    points_neg = c - 5 * d
    pdf.text(610, 32, 'The hyperplane picked through PCA after angle based discard')
    # draw_set_separator(w, ori_hyp[:, 0], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 6 + 2), pdf)
    # pdf.add_page()
    # draw_set_separator(ori_hyp[:, 0], ori_hyp[:, 0], rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 6 + 3), pdf)
    # pdf.add_page()
    # draw_projections_in_planes(ori_hyp, w, rand_init_point, rand_closest_point, points_pos, points_neg, 'line_in_plane' + str(p * 6 + 1), pdf)
    # pdf.add_page()
    return w, rand_init_point, rand_closest_point, candidate_points_pos, candidate_points_neg

def resample(n_sample, lines, data_s, data_s_y, points_pos, points_neg, upsilon):
    candidate_points_pos = np.zeros((points_pos.shape[0], points_pos.shape[1] + n_sample))
    candidate_points_neg = np.zeros((points_pos.shape[0], points_pos.shape[1] + n_sample))
    candidate_points_pos[:, 0:points_pos.shape[1]] = points_pos
    candidate_points_neg[:, 0:points_neg.shape[1]] = points_neg
    count = 0
    while True:
        p2_old = -1
        p3_old = -1
        index = np.random.choice(np.where(data_s_y == 0)[0], 1, replace=True)
        x_fp = data_s[:, index]
        while True:
            p2 = min_distance_point(x_fp, data_s, data_s_y, 1)
            p3 = min_distance_point(data_s[:, p2], data_s, data_s_y, 0)
            if p2 == p2_old and p3 == p3_old:
                break
            p2_old = p2
            p3_old = p3
            x_fp = data_s[:, p3]
        midpoint = (data_s[:, p2_old] + data_s[:, p3_old])/2
        lines_vec = np.asarray(lines).T
        distances = np.abs(np.dot(lines_vec.T, midpoint)/np.linalg.norm(lines_vec[0:-1, :], axis=0))
        if np.any(distances<upsilon):
            data_s = np.delete(data_s, [p2_old, p3_old], 1)
            data_s_y = np.delete(data_s_y, [p2_old, p3_old])
            continue
        candidate_points_pos[:, points_pos.shape[1] + count] = data_s[:, p2_old]
        # print(candidate_points_pos[:, points_pos.shape[1] + count])
        candidate_points_neg[:, points_neg.shape[1] + count] = data_s[:, p3_old]
        data_s = np.delete(data_s, [p2_old, p3_old], 1)
        data_s_y = np.delete(data_s_y, [p2_old, p3_old])
        count += 1
        if count == n_sample:
            break
    return candidate_points_pos, candidate_points_neg

def get_init_hyperplanes(data_x, data_y, orig_hyp, pdf, visualise=False, axis_parallel = True):
    data_s = data_x
    data_s_y = data_y
    n_sample = [332]
    lines = []
    upsilon = 0.1

    for i in n_sample:
        candidate_points_pos = []
        candidate_points_neg = []
        for j in range(i):
            p2_old = -1
            p3_old = -1
            index = np.random.choice(np.where(data_s_y == 0)[0], 1, replace=True)
            x_fp = data_s[:, index]
            while True:
                p2 = min_distance_point(x_fp, data_s, data_s_y, 1)
                p3 = min_distance_point(data_s[:, p2], data_s, data_s_y, 0)
                if p2 == p2_old and p3 == p3_old:
                    break
                point1 = data_s[:, p2]
                point2 = data_s[:, p3]

                p2_old = p2
                p3_old = p3
                x_fp = data_s[:, p3]
            candidate_points_pos.append(data_s[:, p2_old])
            candidate_points_neg.append(data_s[:, p3_old])
            data_s = np.delete(data_s, [p2_old, p3_old], 1)
            data_s_y = np.delete(data_s_y, [p2_old, p3_old])
        candidate_points_pos = np.asarray(candidate_points_pos).T
        candidate_points_neg = np.asarray(candidate_points_neg).T
        candidate_points_pos_orig = np.copy(candidate_points_pos)
        candidate_points_neg_orig = np.copy(candidate_points_neg)
        # w = w/np.linalg.norm(w[0:-1])

        #find associated original hyperplane
        association_indices = []
        for j in range(0, orig_hyp.shape[1]):
            side_pos = np.sign(np.dot(orig_hyp[:, j], candidate_points_pos))
            side_neg = np.sign(np.dot(orig_hyp[:, j], candidate_points_neg))
            association_indices.append(np.where(side_pos != side_neg, 1, 0))

        epochs = 300
        count_k = 5
        k = count_k
        alpha = 5
        threshold_alpha = 10
        p = 0
        start_print = 436
        lines = []
        score = []
        line_count=0
        removed = []
        for j in range(0, 100):
            learn_rate = 0.01
            w, rand_init_point, rand_closest_point, candidate_points_pos_1, candidate_points_neg_1 = great_pick_algo_3(candidate_points_pos, candidate_points_neg, data_x, data_y, removed, orig_hyp, j, pdf, axis_parallel)
            if w is None:
                pdf.text(20, start_print, 'Resampling points = ' + str(i - candidate_points_pos.shape[1]))
                pdf.add_page()
                candidate_points_pos, candidate_points_neg = resample(i - candidate_points_pos.shape[1], lines, data_s, data_s_y, candidate_points_pos, candidate_points_neg, upsilon)
                continue
            # # w = great_pick_algo_1(candidate_points_pos, candidate_points_neg, orig_hyp, j, pdf)
            # c = (candidate_points_pos_1 + candidate_points_neg_1) / 2
            # d = (candidate_points_pos_1 - candidate_points_neg_1) / 2
            # # d = d/np.linalg.norm(d)
            # points_pos = c + alpha * d
            # points_neg = c - alpha * d
            # # w = -1 + 2 * np.random.random(data_s.shape[0])
            # # w = w/np.linalg.norm(w)
            # end_final_obj = 0
            # for l in range(0, epochs):
            #     start_obj = u_shaped_x_by_one_plus_x(w, candidate_points_pos_1, candidate_points_neg_1)
            #     end_final_obj = start_obj
            #     gradient = u_shaped_x_by_one_plus_x_der(w, candidate_points_pos_1, candidate_points_neg_1)
            #     w_new = w - learn_rate * gradient
            #     end_obj = u_shaped_x_by_one_plus_x(w_new, candidate_points_pos_1, candidate_points_neg_1)
            #     if (l % 10 == 0) and (visualise == True):
            #         dots = np.abs(np.dot(w[0:-1], orig_hyp[0:-1, ] / np.linalg.norm(orig_hyp[0:-1, ], axis=0)))
            #         prob_hyp_in = np.argmax(dots)
            #         pdf.text(20, start_print - 16*2, 'Most probable hyperplane = ' + str(orig_hyp[:, prob_hyp_in]))
            #         pdf.text(20, start_print - 16, 'angle= ' + str((180/np.pi)*np.arccos(np.dot(w[0:-1], orig_hyp[0:-1, prob_hyp_in])/(np.linalg.norm(w[0:-1]) * np.linalg.norm(orig_hyp[0:-1, prob_hyp_in])))))
            #         pdf.text(20, start_print, 'learn = ' + str(learn_rate))
            #         pdf.text(20, start_print + 16, 'Start objective value = ' + str(start_obj))
            #         pdf.text(20, start_print + 16 * 2, 'Hyperplane' + str(w))
            #         pdf.text(20, start_print + 16 * 3, 'derivative' + str(gradient))
            #         pdf.text(20, start_print + 16 * 4, 'End objective value = ' + str(end_obj))
            #         com = np.sum((points_pos + points_neg) / 2, axis=1) / points_pos.shape[1]
            #         pdf.text(20, start_print + 16 * 5, 'Com = ' + str(com))
            #         pdf.text(20, start_print + 16 * 6, 'n_points = ' + str(points_pos.shape[1]))
            #         pdf.add_page()
            #         W_temp = []
            #         W_temp.append(w.reshape(-1, 1))
            #         # draw_set_separator(orig_hyp[:, 0], orig_hyp, rand_init_point, rand_closest_point, candidate_points_pos, candidate_points_neg, 3 * p * epochs + i + 1, pdf)
            #         # pdf.add_page()
            #         # draw_projections_in_planes(orig_hyp, w, rand_init_point, rand_closest_point, candidate_points_pos, candidate_points_neg, 'line_in_plane' + str(3 * p * epochs + i + 2), pdf)
            #         # pdf.add_page()
            #         c = (candidate_points_pos_1 + candidate_points_neg_1) / 2
            #         d = (candidate_points_pos_1 - candidate_points_neg_1) / 2
            #         points_pos = c + alpha * d
            #         points_neg = c - alpha * d
            #         dots = np.abs(np.dot(w[0:-1], orig_hyp[0:-1, ]/np.linalg.norm(orig_hyp[0:-1, ], axis=0)))
            #         prob_hyp_in = np.argmax(dots)
            #         #1draw_set_separator(w, orig_hyp[:, prob_hyp_in], orig_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, candidate_points_pos, candidate_points_neg, 'line_in_plane' + str(p * epochs + i), pdf, axis_parallel)
            #         #1pdf.add_page()
            #         # draw_set_separator(w, orig_hyp[:, prob_hyp_in], orig_hyp[:, prob_hyp_in], rand_init_point, rand_closest_point, candidate_points_pos, candidate_points_neg, 'line_in_orig_plane' + str(p * epochs + i), pdf, axis_parallel)
            #         # pdf.add_page()
            #     if end_obj >= start_obj:
            #         k = count_k
            #         learn_rate = learn_rate / 2
            #     else:
            #         w = w_new
            #         k = k - 1
            #         if k <= 0:
            #             k = count_k
            #             learn_rate = learn_rate * 1.5
            distances = np.abs(np.dot(w, (candidate_points_pos + candidate_points_neg) / 2) / np.linalg.norm(w[0:-1]))
            new_indices = np.where(distances > upsilon)[0]
            removed = list(set(removed) | set(np.where((np.abs(np.dot(w, data_x)) / np.linalg.norm(w[0:-1]))<=upsilon)[0]))

            #calculating associations
            distances_asso = np.abs(np.dot(w, (candidate_points_pos_orig + candidate_points_neg_orig) / 2) / np.linalg.norm(w[0:-1]))
            new_indices_asso = np.where(distances_asso <= upsilon, 1, 0)

            #printing overlap info
            for l in range(0, orig_hyp.shape[1]):
                overlap_indices = np.intersect1d(np.where(new_indices_asso == 1)[0], np.where(association_indices[l] == 1)[0])
                pdf.text(20, 16 + 48*l, 'overlap count = ' + str(len(overlap_indices)))
                pdf.text(20, 16 + 48*l + 16, 'Original association count = ' + str(np.sum(association_indices[l])))
                pdf.text(20, 16 + 48*l + 32, 'Generated association count = ' + str(np.sum(new_indices_asso)))
            pdf.add_page()
            if len(new_indices) == 0:
                lines.append(w)
                break
            candidate_points_pos = candidate_points_pos[:, new_indices]
            candidate_points_neg = candidate_points_neg[:, new_indices]
            p += 1
            #score.append(end_final_obj)
            line_count += 1
            lines.append(w)
            if line_count == 20:
                break
        pdf.add_page()
    return lines

def main():
    for a in range(3, 4):
        np.random.seed(a)
        data = pd.read_csv('../union_of_convex_datasets/ten_5D_hyperplanes_million.csv', header=None).values
        data_x = np.hstack((data[:, 0:-1], np.ones(data.shape[0]).reshape(-1, 1))).T
        data_y = data[:, -1]
        # asso_hyp = data[:, -1]
        #print(Counter(asso_hyp))
        orig_hyp = pd.read_csv('../union_of_convex_datasets/ten_5D_hyperplanes_hyperplanes_million.csv', header=None).values
        data_y = np.where(data_y == -1, 0, 1)
        print(np.sum(np.where(data_y == 1, 1, 0)))
        print(np.sum(np.where(data_y == 0, 1, 0)))
        current_dir = os.getcwd()
        if os.path.exists(os.path.join(current_dir, 'log')) is False:
            os.makedirs(os.path.join(current_dir, 'log'))
        os.chdir(os.path.join(current_dir, 'log'))

        pdf = FPDF(orientation='L', unit='pt', format='A4')
        pdf.add_page()
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(0, 0, 0)
        lines = get_init_hyperplanes(data_x, data_y, orig_hyp, pdf, visualise=False, axis_parallel=False)
        #draw_lines(lines, data_x, data_y, pdf)
        lines = np.asarray(lines).T
        print(lines)
        #pdf.add_page()
        dist_lines = np.dot(lines.T, data_x) * np.tile((1/np.linalg.norm(lines[0:-1, :], axis=0)).reshape(-1, 1), (1, data_x.shape[1]))
        #  asso_hyp_new = np.argmin(dist_lines, axis=0)
        # print(Counter(asso_hyp_new))
        # overlap_matrix = np.zeros((orig_hyp.shape[1], lines.shape[1]))
        # common_count_matrix = np.zeros((orig_hyp.shape[1], lines.shape[1]))
        # total_count_matrix = np.zeros((orig_hyp.shape[1], lines.shape[1]))
        # pairwise_distance_matrix = np.zeros((orig_hyp.shape[1], lines.shape[1]))
        # for i in range(0, orig_hyp.shape[1]):
        #     for j in range(0, lines.shape[1]):
        #         point_orig_ind = np.where(asso_hyp == i, 1, 0)
        #         point_new_hyp_ind = np.where(asso_hyp_new == j, 1, 0)
        #         total = np.sum(point_orig_ind) + np.sum(point_new_hyp_ind)
        #         common = len(np.intersect1d(np.where(point_orig_ind == 1)[0], np.where(point_new_hyp_ind == 1)[0]))
        #         overlap_matrix[i, j] = common/total
        #         common_count_matrix[i, j] = common
        #         total_count_matrix[i, j] = total
        #         pairwise_distance_matrix[i, j] = np.linalg.norm(((orig_hyp[:, i]/np.linalg.norm(orig_hyp[0:-1, i])) - (lines[:, j]/np.linalg.norm(lines[0:-1, j]))))
        # overlap_matrix = np.vstack((overlap_matrix, common_count_matrix))
        # overlap_matrix = np.vstack((overlap_matrix, total_count_matrix))
        # df = pd.DataFrame(overlap_matrix, columns = np.arange(1, lines.shape[1]+1))
        # df.to_csv('overlap_matrix.csv', header=True, index=True)
        # df = pd.DataFrame(pairwise_distance_matrix, columns=np.arange(1, lines.shape[1]+1))
        # df.to_csv('distance_matrix.csv', header=True, index=True)
        feature_vector = np.sign(np.dot(lines.T, data_x))
        clf = tree.DecisionTreeClassifier()
        clf.fit(feature_vector.T, data_y)
        labels = clf.predict(feature_vector.T)
        true_positive_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 1)[0])
        true_negative_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 0)[0])
        false_positive_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 1)[0])
        false_negative_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 0)[0])
        print(len(true_positive_indices))
        print(len(true_negative_indices))
        print(len(false_positive_indices))
        print(len(false_negative_indices))
        visualise_multi(lines, orig_hyp, data_x, data_y, labels, pdf, axis_parallel = False)
        pdf.text(10, 310, 'True positives = ' + str(len(true_positive_indices)))
        pdf.text(10, 328, 'True negatives = ' + str(len(true_negative_indices)))
        pdf.text(10, 346, 'False positives = ' + str(len(false_positive_indices)))
        pdf.text(10, 362, 'False negatives = ' + str(len(false_negative_indices)))
        pdf.text(10, 378, 'n_lines = ' + str(lines.shape[1]))
        pdf.add_page()

        #draw_points(data_x, data_y, labels, pdf)
        #pdf.add_page()
        tree.plot_tree(clf)
        #plt.savefig('tree.png')
        #pdf.image('tree.png', x=0, y=0, w=640, h=640)
        print(clf.tree_.node_count)
        pdf.output('local_d_plus_1_unsorted' + str(a) + '.pdf')
        os.chdir(current_dir)

if __name__ == '__main__':
    main()