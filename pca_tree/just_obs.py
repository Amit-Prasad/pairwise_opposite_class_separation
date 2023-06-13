import numpy as np
import pandas as pd
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

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

def cost_function(X, y, z, theta, beta = 0):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1/m) * (np.sum((y * np.log(h) + (1-y) * np.log(1-h))*z.reshape(-1, 1)) + beta*(np.sum(z)))
    return J

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    pred = sigmoid(np.dot(X, theta))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred

def min_distance_point(point, data_x, data_y, label):
    label_indices = np.where(data_y==label)[0]
    dist = np.zeros(data_y.shape) + np.Inf
    data_s = data_x[:, label_indices]
    dist[label_indices] = np.linalg.norm(data_s - point.reshape(-1, 1), axis = 0)
    min_index = np.argmin(dist)
    return min_index

def draw_set_separator(w_init, data_x, data_y, iter, pdf):
    labels = predict(data_x, w_init)
    true_positive_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 1)[0])
    true_negative_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 0)[0])
    false_positive_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 1)[0])
    false_negative_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 0)[0])
    plt.scatter(data_x[true_negative_indices, 0], data_x[true_negative_indices, 1], c='yellow', s=1)
    plt.scatter(data_x[false_positive_indices, 0], data_x[false_positive_indices, 1], c='lime', s=1)
    plt.scatter(data_x[false_negative_indices, 0], data_x[false_negative_indices, 1], c='pink', s=1)
    plt.scatter(data_x[true_positive_indices, 0], data_x[true_positive_indices, 1], c='blue', s=1)

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
    sgn = (w[0] * dir_points[i][0] + w[1] * dir_points[i][1] + w[2])
    if sgn > 0:
        plt.scatter(dir_points[i][0], dir_points[i][1], marker='+',
                    color='black')
    if sgn < 0:
        plt.scatter(dir_points[i][0], dir_points[i][1], marker='o',
                    color='black')

    plt.axis([x_min, x_max, y_min, y_max])
    plt.savefig('separator_hyp' + str(w_init[0]) + str(iter) + '.png')
    pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + '.png', x=0, y=0, w=700,h=420)
    plt.gca().cla()

def sample_pairs(data_x, data_y):
    data_s = data_x
    data_s_y = data_y
    n_sample = 200
    candidate_points_pos = []
    candidate_points_neg = []
    for j in range(n_sample):
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
        candidate_points_pos.append(data_s[:, p2_old])
        candidate_points_neg.append(data_s[:, p3_old])
        data_s = np.delete(data_s, [p2_old, p3_old], 1)
        data_s_y = np.delete(data_s_y, [p2_old, p3_old])
    candidate_points_pos = np.asarray(candidate_points_pos).T
    candidate_points_neg = np.asarray(candidate_points_neg).T
    data = np.zeros((candidate_points_pos.shape[0] + 1, candidate_points_pos.shape[1] + candidate_points_neg.shape[1]))
    data[0:-1, :] = np.hstack((candidate_points_pos, candidate_points_neg))
    data[-1, :] = np.zeros(candidate_points_pos.shape[1] + candidate_points_neg.shape[1])
    data[-1, 0:candidate_points_pos.shape[1]] = 1
    data = data.T
    np.random.shuffle(data)
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y


def main():
    data = pd.read_csv('../union_of_convex_datasets/stripes.csv', header=None).values
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
    np.random.seed(2)

    # w = np.array([1, 0, 0])
    X, y = sample_pairs(data_x, data_y)
    indices = np.intersect1d(np.where(X[:, 0]>-0.25)[0], np.where(X[:, 0]<0.25)[0])
    X = X[indices, :]
    y = y[indices]
    # y = y.reshape(-1, 1)
    # draw_set_separator(w.reshape(-1, 1), X, y, 0, pdf)
    # labels = predict(X, w.reshape(-1,1))
    # misclassified = np.where(y!=labels)[0]
    # classified = np.where(y==labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    #
    # w = np.array([-1, 1, 0])
    # draw_set_separator(w.reshape(-1, 1), X, y, 1, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    #
    # w = np.array([1, 1, 0])
    # draw_set_separator(w.reshape(-1, 1), X, y, 2, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    #
    # w = np.array([0, 1, 0])
    # draw_set_separator(w.reshape(-1, 1), X, y, 3, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    # draw_set_separator(w.reshape(-1, 1), X, y, 2, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    #
    # w = np.array([1.515, 1, 0])
    # draw_set_separator(w.reshape(-1, 1), X, y, 3, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    # draw_set_separator(w.reshape(-1, 1), X, y, 2, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    #
    # w = np.array([1.50, 1, 0])
    # draw_set_separator(w.reshape(-1, 1), X, y, 3, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    # draw_set_separator(w.reshape(-1, 1), X, y, 2, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    #
    # w = np.array([1.52, 1, 0])
    # draw_set_separator(w.reshape(-1, 1), X, y, 3, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    # draw_set_separator(w.reshape(-1, 1), X, y, 2, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    # w = np.array([0.00591239, -0.0036142, 0.00014389])
    # draw_set_separator(w.reshape(-1, 1), X, y, 3, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    # draw_set_separator(w.reshape(-1, 1), X, y, 2, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels)[0]
    # classified = np.where(y == labels)[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    coeff_x = 1
    coeff_y = 0
    bias = -1
    correct = []
    objective = []
    max_cor = 0
    hyp_max = np.array([0, 0, 0])
    for i in np.arange(0, 2.01, 0.01):
        w = np.array([coeff_x, coeff_y, bias+i])
        w = w/np.linalg.norm(w)
        labels = predict(X, w.reshape(-1, 1))
        objective.append(cost_function(X, y, np.ones(X.shape[0]), w.reshape(-1, 1), beta=0))
        classified = np.sum(np.where(y == labels.reshape(-1,), 1, 0))
        if classified > max_cor:
            max_cor = classified
            hyp_max = w
        correct.append(classified)
    w = hyp_max
    draw_set_separator(w.reshape(-1, 1), X, y, 1, pdf)
    labels = predict(X, w.reshape(-1, 1))
    misclassified = np.where(y != labels.reshape(-1,))[0]
    classified = np.where(y == labels.reshape(-1,))[0]
    start_print = 436
    pdf.text(20, start_print - 16, 'w = ' + str(w))
    pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    pdf.text(20, start_print + 32, 'objective = ' + str(cost_function(X, y, np.ones(X.shape[0]), w.reshape(-1, 1), beta=0)))
    pdf.add_page()
    plt.plot(-1*(bias + np.arange(0, 2.01, 0.01)), correct, color='black')
    plt.savefig('cor_vs_val.png')
    pdf.image('cor_vs_val.png', x=0, y=0, w=700, h=420)
    plt.plot(-1 * (bias + np.arange(0, 2.01, 0.01)), objective, color='black')
    plt.savefig('obj_vs_val.png')
    pdf.image('obj_vs_val.png', x=0, y=0, w=700, h=420)

    w = np.array([0.01942864, -0.00150808, 0.00080504])
    w = w/np.linalg.norm(w)
    draw_set_separator(w.reshape(-1, 1), X, y, 2, pdf)
    labels = predict(X, w.reshape(-1, 1))
    misclassified = np.where(y != labels.reshape(-1, ))[0]
    classified = np.where(y == labels.reshape(-1, ))[0]
    start_print = 436
    pdf.text(20, start_print - 16, 'w = ' + str(w))
    pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    pdf.text(20, start_print + 32, 'objective = ' + str(cost_function(X, y, np.ones(X.shape[0]), w.reshape(-1,1), beta = 0)))
    pdf.add_page()
    plt.plot(-1 * (bias + np.arange(0, 2.01, 0.01)), correct, color='black')
    plt.savefig('cor_vs_val_a.png')
    pdf.image('cor_vs_val_a.png', x=0, y=0, w=700, h=420)
    plt.plot(-1 * (bias + np.arange(0, 2.01, 0.01)), objective, color='black')
    plt.savefig('obj_vs_val_a.png')
    pdf.image('obj_vs_val_a.png', x=0, y=0, w=700, h=420)

    # coeff_x = 1
    # coeff_y = 0
    # bias = 0.66
    # correct = []
    # objective = []
    # max_cor = 0
    # hyp_max = np.array([0, 0, 0])
    # for i in np.arange(0.5, 1.5, 0.1):
    #     w = np.array([coeff_x, coeff_y, bias]) * i
    #     labels = predict(X, w.reshape(-1, 1))
    #     objective.append(cost_function(X, y, np.ones(X.shape[0]), w.reshape(-1, 1), beta=0))
    #     classified = np.sum(np.where(y == labels.reshape(-1,), 1, 0))
    #     if classified > max_cor:
    #         max_cor = classified
    #         hyp_max = w
    #     correct.append(classified)
    # w = hyp_max
    # draw_set_separator(w.reshape(-1, 1), X, y, 1, pdf)
    # labels = predict(X, w.reshape(-1, 1))
    # misclassified = np.where(y != labels.reshape(-1,))[0]
    # classified = np.where(y == labels.reshape(-1,))[0]
    # start_print = 436
    # pdf.text(20, start_print - 16, 'w = ' + str(w))
    # pdf.text(20, start_print, 'misclassified = ' + str(len(misclassified)))
    # pdf.text(20, start_print + 16, 'classified = ' + str(len(classified)))
    # pdf.add_page()
    # plt.plot(np.arange(0.5, 1.5, 0.1), correct, color='black')
    # plt.savefig('cor_vs_val.png')
    # pdf.image('cor_vs_val.png', x=0, y=0, w=700, h=420)
    # plt.plot(np.arange(0.5, 1.5, 0.1), objective, color='black')
    # plt.savefig('obj_vs_val.png')
    # pdf.image('obj_vs_val.png', x=0, y=0, w=700, h=420)

    pdf.output('disa_y.pdf')

if __name__ == '__main__':
    main()