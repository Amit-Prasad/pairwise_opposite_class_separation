import pandas as pd
import os
import numpy as np
from fpdf import FPDF
from helpers import *
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from regression_based_optimisation import get_hyp_1

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

def regression_format(candidate_points_pos, candidate_points_neg):
    data = np.zeros((candidate_points_pos.shape[0] + 1, candidate_points_pos.shape[1] + candidate_points_neg.shape[1]))
    data[0:-1, :] = np.hstack((candidate_points_pos, candidate_points_neg))
    data[-1, :] = np.zeros(candidate_points_pos.shape[1] + candidate_points_neg.shape[1])
    data[-1, 0:candidate_points_pos.shape[1]] = 1
    data = data.T
    np.random.shuffle(data)
    X = data[:, 0:-1]
    Y = data[:, -1]
    return X, Y

def min_distance_point(point, data_x, data_y, label):
    label_indices = np.where(data_y==label)[0]
    dist = np.zeros(data_y.shape) + np.Inf
    data_s = data_x[:, label_indices]
    dist[label_indices] = np.linalg.norm(data_s - point.reshape(-1, 1), axis = 0)
    min_index = np.argmin(dist)
    return min_index

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
    true_positive_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 1))
    true_negative_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 0))
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    false_negative_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 0))

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

def draw_set_separator(w_init, data_x, data_y, candidate_points_pos, candidate_points_neg, iter, pdf):
    plt.scatter(data_x[0], data_x[1], c='red', s=5)
    plt.scatter(data_y[0, :], data_y[1, :], c='blue', s=5)
    plt.scatter(candidate_points_pos[0, :], candidate_points_pos[1, :], c='lime', s=3)
    plt.scatter(candidate_points_neg[0, :], candidate_points_neg[1, :], c='pink', s=3)
    grain = 0.001
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    x1 = np.arange(x_min, x_max, grain)
    k = 0
    p = 0
    i = p+1
    w = w_init
    if w[1] == 0:
        x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
    else:
        x2 = -1 * (w[2] + w[0] * x1) / w[1]
    plt.plot(x1, x2, color='black')
    plt.axis([x_min, x_max, y_min, y_max])
    plt.savefig('separator_hyp' + str(w_init[0]) + str(iter) + '.png')
    pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + '.png', x=0, y=0, w=700,h=420)
    plt.gca().cla()

def great_pick_algo(candidate_points_pos, candidate_points_neg, p, pdf):
    c = (candidate_points_pos + candidate_points_neg) / 2
    d = (candidate_points_pos - candidate_points_neg) / 2
    points_pos = c + 5 * d
    points_neg = c - 5 * d
    midpoints = c
    w, rand_init_point, rand_closest_point = pick_init_hyp_1_2D(candidate_points_pos, candidate_points_neg)
    draw_set_separator(w, rand_init_point, rand_closest_point, points_pos, points_neg, 'init_'+str(p*6+0), pdf)
    start_print = 436
    pdf.text(20, start_print, 'Hyperplane: ' + str(w))
    pdf.add_page()
    master_midpoint = np.sum(rand_closest_point, axis=1)/rand_closest_point.shape[1]
    master_midpoint_proj = np.zeros(w.shape)
    master_midpoint_proj[0:-1] = master_midpoint[0:-1]-(np.dot(w, master_midpoint)/np.sqrt(np.dot(w[0:-1], w[0:-1])))*w[0:-1]
    master_midpoint_proj[-1] = 1
    hypotenuse = np.linalg.norm(midpoints - master_midpoint_proj.reshape(-1, 1), axis=0)
    perpendicular = np.dot(w, midpoints)/np.sqrt(np.dot(w[0:-1], w[0:-1]))
    angles = np.abs(np.arcsin(perpendicular/hypotenuse))
    indexes = np.where((angles<=np.pi/18))[0]
    candidate_points_pos = candidate_points_pos[:, indexes]
    candidate_points_neg = candidate_points_neg[:, indexes]
    pca = PCA(n_components=2)
    pca.fit((candidate_points_pos + candidate_points_neg/2)[0:-1, :].T)
    w = np.zeros(candidate_points_pos.shape[0])
    w[0:-1] = pca.components_[1]
    mean = np.sum(((candidate_points_pos + candidate_points_neg)/2)[0:-1, :], axis=1)/candidate_points_pos.shape[1]
    w[-1] = np.dot(w[0:-1], mean)*-1
    print(mean)
    c = (candidate_points_pos + candidate_points_neg) / 2
    d = (candidate_points_pos - candidate_points_neg) / 2
    points_pos = c + 5 * d
    points_neg = c - 5 * d
    draw_set_separator(w, rand_init_point, rand_closest_point, points_pos, points_neg, 'init_' + str(p * 6 + 1), pdf)
    pdf.text(20, start_print, 'Hyperplane: ' + str(w))
    pdf.add_page()
    return w, rand_init_point, rand_closest_point, candidate_points_pos, candidate_points_neg

def get_init_hyperplanes(data_x, data_y, pdf, visualise=False):
    data_s = data_x
    data_s_y = data_y
    n_sample = [200]
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

        # w = w/np.linalg.norm(w[0:-1])

        epochs = 100
        count_k = 5
        k = count_k
        alpha = 5
        threshold_alpha = 10
        p = 0
        start_print = 436
        lines = []
        score = []
        for j in range(0, 1):
            learn_rate = 0.01
            w, rand_init_point, rand_closest_point, candidate_points_pos_1, candidate_points_neg_1 = great_pick_algo(candidate_points_pos, candidate_points_neg, j, pdf)
            X,Y = regression_format(candidate_points_pos_1, candidate_points_neg_1)
            # clf = LogisticRegression(random_state=42).fit(X, Y)
            # w[0:-1] = clf.coef_[0, 0:-1]
            # w[-1] = clf.intercept_[0]*-1
            w = get_hyp_1(X, Y, 1.1*np.log(2), pdf).reshape(-1,)
            draw_set_separator(w, rand_init_point, rand_closest_point, candidate_points_pos_1, candidate_points_neg_1, p * epochs + i, pdf)
            pdf.text(20, start_print + 16 * 2, 'logistic_Hyperplane' + str(w))
            pdf.add_page()
            # c = (candidate_points_pos_1 + candidate_points_neg_1) / 2
            # d = (candidate_points_pos_1 - candidate_points_neg_1) / 2
            # # d = d/np.linalg.norm(d)
            # points_pos = c + alpha * d
            # points_neg = c - alpha * d
            # # w = -1 + 2 * np.random.random(data_s.shape[0])
            # # w = w/np.linalg.norm(w)
            # end_final_obj = 0
            # for i in range(0, epochs):
            #     start_obj = u_shaped_x_by_one_plus_x(w, candidate_points_pos_1, candidate_points_neg_1)
            #     end_final_obj = start_obj
            #     gradient = u_shaped_x_by_one_plus_x_der(w, candidate_points_pos_1, candidate_points_neg_1)
            #     w_new = w - learn_rate * gradient
            #     end_obj = u_shaped_x_by_one_plus_x(w_new, candidate_points_pos_1, candidate_points_neg_1)
            #     if (i % 10 == 0) and (visualise == True):
            #         pdf.text(20, start_print, 'learn = ' + str(learn_rate))
            #         pdf.text(20, start_print + 16, 'Start objective value = ' + str(start_obj))
            #         pdf.text(20, start_print + 16 * 2, 'Hyperplane' + str(w))
            #         pdf.text(20, start_print + 16 * 3, 'derivative' + str(gradient))
            #         pdf.text(20, start_print + 16 * 4, 'End objective value = ' + str(end_obj))
            #         com = np.sum((points_pos + points_neg) / 2, axis=1) / points_pos.shape[1]
            #         pdf.text(20, start_print + 16 * 5, 'Com = ' + str(com))
            #         pdf.text(20, start_print + 16 * 6, 'n_points = ' + str(points_pos.shape[1]))
            #         W_temp = []
            #         W_temp.append(w.reshape(-1, 1))
            #         draw_set_separator(w, rand_init_point, rand_closest_point, candidate_points_pos, candidate_points_neg, p * epochs + i, pdf)
            #         pdf.add_page()
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
            if len(new_indices) == 0:
                lines.append(w)
                break
            candidate_points_pos = candidate_points_pos[:, new_indices]
            candidate_points_neg = candidate_points_neg[:, new_indices]
            p += 1
            #score.append(end_final_obj)
            lines.append(w)
        pdf.add_page()
    return lines

def main():
    for a in range(1, 2):
        np.random.seed(a)
        data = pd.read_csv('../union_of_convex_datasets/checkerboard.csv', header=None).values
        data_x = np.hstack((data[:, 0:-1], np.ones(data.shape[0]).reshape(-1, 1))).T
        data_y = data[:, -1]
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
        lines = get_init_hyperplanes(data_x, data_y, pdf, visualise=True)
        draw_lines(lines, data_x, data_y, pdf)
        pdf.add_page()

        feature_vector = np.zeros((len(lines), data_x.shape[1]))
        for line, i in zip(lines, range(0, len(lines))):
            feature_vector[i, :] = np.sign(np.dot(line, data_x))

        clf = tree.DecisionTreeClassifier()
        clf.fit(feature_vector.T, data_y)
        labels = clf.predict(feature_vector.T)
        draw_points(data_x, data_y, labels, pdf)
        pdf.add_page()
        tree.plot_tree(clf)
        plt.savefig('tree.png')
        pdf.image('tree.png', x=0, y=0, w=640, h=640)
        pdf.output('pca_tree_logistic' + str(a) + '.pdf')
        os.chdir(current_dir)

if __name__ == '__main__':
    main()