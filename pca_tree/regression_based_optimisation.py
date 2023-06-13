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

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(X, y, z, theta, beta = 0):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1/m) * (np.sum((y * np.log(h) + (1-y) * np.log(1-h))*z.reshape(-1, 1)) + beta*(np.sum(z)))
    return J

def min_distance_point(point, data_x, data_y, label):
    label_indices = np.where(data_y==label)[0]
    dist = np.zeros(data_y.shape) + np.Inf
    data_s = data_x[:, label_indices]
    dist[label_indices] = np.linalg.norm(data_s - point.reshape(-1, 1), axis = 0)
    min_index = np.argmin(dist)
    return min_index

# Gradient descent function
def gradient_descent(X, y, z, theta, lr, num_iters, p, beta, pdf):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    #working_index = np.where(z==1)[0]
    start_print = 436
    #print(len(working_index))
    count_k = 5
    k = count_k
    for i in range(num_iters):
        start_obj = cost_function(X, y, z, theta, beta)
        h = sigmoid(np.dot(X, theta))
        gradient = (lr / m) * np.dot(X.T, (h - y) * z.reshape(-1, 1))
        theta_hat = theta - gradient
        end_obj = cost_function(X, y, z, theta_hat, beta)
        if ((i % 100 == 0) and (beta == 0)) or ((i % 1 == 0) and (beta>0)):
            draw_set_separator(theta.reshape(-1, ), X, y, 'gd' + str(p * (num_iters+1) + i), pdf)
            if beta==0:
                pdf.text(20, start_print - 16, 'Logistic')
            else:
                pdf.text(20, start_print - 16, 'Modified logistic')
            pdf.text(20, start_print, 'learn = ' + str(lr))
            pdf.text(20, start_print + 16, 'Start objective value = ' + str(start_obj))
            pdf.text(20, start_print + 16 * 2, 'Hyperplane' + str(theta))
            pdf.text(20, start_print + 16 * 3, 'derivative' + str(gradient))
            pdf.text(20, start_print + 16 * 4, 'End objective value = ' + str(end_obj))
            pdf.text(20, start_print + 16 * 5, 'Norm = ' + str(np.linalg.norm(theta[0:-1, :], axis=0)))
            pdf.text(20, start_print + 16 * 6, 'New Hyperplane = ' + str(theta_hat))
            pdf.text(20, start_print + 16 * 7, 'Beta = ' + str(beta))
            pdf.add_page()
        if end_obj>=start_obj:
            k = count_k
            lr = lr / 2
        else:
            theta = theta_hat
            k = k - 1
            if k <= 0:
                k = count_k
                lr = lr * 1.5
        J_history[i] = cost_function(X, y, z, theta, beta)
    draw_set_separator(theta.reshape(-1, ), X, y, 'gd' + str(p * (num_iters+1) + num_iters), pdf)
    pdf.text(20, start_print + 16 * 2, 'Learnt Hyperplane ' + str(theta))
    pdf.text(20, start_print + 16 * 3, 'Learnt objective ' + str(cost_function(X, y, z, theta, beta)))
    pdf.add_page()
    return theta, J_history, lr

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

    # w[-1] += threshold * np.linalg.norm(w[0:-1])
    # if w[1] == 0:
    #     x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
    # else:
    #     x2 = -1 * (w[2] + w[0] * x1) / w[1]
    # plt.plot(x1, x2, color='black', linestyle='dashed')
    #
    # w[-1] += -2*threshold * np.linalg.norm(w[0:-1])
    # if w[1] == 0:
    #     x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
    # else:
    #     x2 = -1 * (w[2] + w[0] * x1) / w[1]
    # w[-1] += threshold * np.linalg.norm(w[0:-1])
    # plt.plot(x1, x2, color='black', linestyle='dashed')
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

def predict(X, theta):
    pred = sigmoid(np.dot(X, theta))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred

def get_hyp(X, y, threshold, beta, pdf):
    y = y.reshape(-1, 1)
    z = np.ones(X.shape[0])

    # Initialize theta and hyperparameters
    theta = -1 + 2 * np.random.random((X.shape[1], 1))
    lr = 0.01
    num_iters = 10
    p = 0
    threshold_1 = threshold
    while p < 10:
        threshold_1 = threshold_1 / 2
        # Run gradient descent
        theta, J_history, lr = gradient_descent(X, y, z, theta, lr, num_iters, p, beta, pdf)

        # Make predictions
        pred = sigmoid(np.dot(X, theta))
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        distances = np.abs(np.dot(X, theta)) / np.linalg.norm(theta[0:-1, :])
        grossly_misclassified = np.intersect1d(
            np.intersect1d(np.where(pred != y)[0], np.where(distances > threshold_1)[0]), np.where(z == 1)[0])
        not_grossly_misclassified = np.intersect1d(
            np.union1d(np.where(pred == y)[0], np.where(distances <= threshold_1)[0]), np.where(z == 1)[0])
        start_print = 436
        draw_set_separator(theta.reshape(-1, ), X[not_grossly_misclassified, :], y[not_grossly_misclassified, :], 'sepa_1_1' + str(i * 10 + p), threshold_1, z, pdf)
        pdf.text(20, start_print, 'points_retained')
        pdf.text(20, start_print + 16, 'points_retained = ' + str(np.sum(z) - len(grossly_misclassified)))
        pdf.add_page()
        draw_set_separator(theta.reshape(-1, ), X[grossly_misclassified, :], y[grossly_misclassified, :], 'sepa_1_2' + str(i * 10 + p), threshold_1, z, pdf)
        pdf.text(20, start_print, 'points_removed')
        pdf.text(20, start_print + 16, 'points_removed = ' + str(len(grossly_misclassified)))
        pdf.add_page()
        if len(grossly_misclassified) == 0:
            break
        z[grossly_misclassified] = 0
        p += 1
    return theta

def simple_regression(X, y, pdf):
    y = y.reshape(-1, 1)
    num_iters = 1000
    z = np.ones(X.shape[0])
    theta = -1 + 2 * np.random.random((X.shape[1], 1))
    theta, J_history, _ = gradient_descent(X, y, z, theta, 0.01, num_iters, 0, 0, pdf)
    return theta.reshape(-1,)

def get_hyp_1(X, y, beta, pdf):
    y = y.reshape(-1, 1)
    z = np.ones(X.shape[0])

    # Initialize theta and hyperparameters
    theta = -1 + 2 * np.random.random((X.shape[1], 1))
    alpha = 0.1
    lr = 0.01
    num_iters = 10
    p = 0
    threshold_1 = 0
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    first = True
    while p < 10:
        if beta<np.log(2):
            break
        # w = theta.reshape(-1, )
        theta, J_history, _ = gradient_descent(X, y, z, theta, lr, num_iters, p, beta, pdf)
        # w_1 = theta.reshape(-1, )
        # Make predictions
        # if not first:
        #     x = np.argmin((y * np.log(sigmoid(np.dot(X, w.reshape(-1, 1)))) + (1 - y) * np.log(
        #         1 - sigmoid(np.dot(X, w.reshape(-1, 1))))))
        #     if y[x] == 1:
        #         beta = np.log(
        #             np.exp(beta) * (1 + np.exp(-1 * np.dot(w_1.T, X[x]))) / (1 + np.exp(-1 * np.dot(w.T, X[x]))))
        #     else:
        #         beta = np.log(
        #             np.exp(beta) * ((1 + np.exp(-1 * np.dot(w_1.T, X[x]))) / (1 + np.exp(-1 * np.dot(w.T, X[x])))) * (
        #                         np.exp(-1 * np.dot(w.T, X[x])) / np.exp(-1 * np.dot(w_1.T, X[x]))))

        y_i_hat = sigmoid(np.dot(X, theta))
        should_decrease_positive = np.intersect1d(np.where((np.log(y_i_hat[:, 0]) + beta) < 0)[0], pos_indices)
        should_decrease_negative = np.intersect1d(np.where((np.log(1 - y_i_hat[:, 0]) + beta) < 0)[0], neg_indices)
        should_increase_positive = np.intersect1d(np.where((np.log(y_i_hat[:, 0]) + beta) > 0)[0], pos_indices)
        should_increase_negative = np.intersect1d(np.where((np.log(1 - y_i_hat[:, 0]) + beta) > 0)[0], neg_indices)
        z_prime = np.copy(z)
        # z_prime = z
        z[should_decrease_positive] = z[should_decrease_positive] / (1 + alpha)
        z[should_decrease_negative] = z[should_decrease_negative] / (1 + alpha)
        z[should_increase_positive] = z[should_increase_positive] * (1 + alpha)
        z[should_increase_negative] = z[should_increase_negative] * (1 + alpha)
        z = np.where(z > 1, 1, z)
        # increased = np.where(z >= z_prime)[0]
        # decreased = np.where(z < z_prime)[0]
        # print(np.intersect1d(increased, pos_indices))
        # print(np.intersect1d(increased, neg_indices))
        # print(np.intersect1d(decreased, pos_indices))
        # print(np.intersect1d(decreased, neg_indices))
        increased = np.union1d(should_increase_positive, should_increase_negative)
        decreased = np.union1d(should_decrease_positive, should_decrease_negative)

        draw_set_separator(theta.reshape(-1, ), X[increased, :], y[increased, :], 'sepa_1_1' + str(i * 10 + p), pdf)
        start_print = 436
        pdf.text(20, start_print - 16, 'z = ' + str(theta))
        pdf.text(20, start_print, 'z incremented')
        pdf.text(20, start_print + 16, 'z incremented = ' + str(len(increased)))
        pdf.text(20, start_print + 32, 'beta = ' + str(beta))
        pdf.text(20, start_print + 48, 'alpha = ' + str(alpha))
        pdf.add_page()

        draw_set_separator(theta.reshape(-1, ), X[decreased, :], y[decreased, :], 'sepa_1_2' + str(i * 10 + p), pdf)
        start_print = 436
        pdf.text(20, start_print - 16, 'z = ' + str(theta))
        pdf.text(20, start_print, 'z decreased')
        pdf.text(20, start_print + 16, 'z decreased = ' + str(len(decreased)))
        pdf.text(20, start_print + 32, 'beta = ' + str(beta))
        pdf.text(20, start_print + 48, 'alpha = ' + str(alpha))
        pdf.add_page()
        # beta = beta/(1+alpha)
        p += 1
        first = False

    return theta


def modified_regression(X, y, theta, beta, z, pdf):
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    y_i_hat = sigmoid(np.dot(X, theta))
    alpha = 0.1
    lr = 0.001
    num_iters = 10
    print("books")
    for p in range(0, num_iters):
        should_decrease_positive = np.intersect1d(np.where((np.log(y_i_hat[:, 0]) + beta) < 0)[0], pos_indices)
        should_decrease_negative = np.intersect1d(np.where((np.log(1 - y_i_hat[:, 0]) + beta) < 0)[0], neg_indices)
        should_increase_positive = np.intersect1d(np.where((np.log(y_i_hat[:, 0]) + beta) > 0)[0], pos_indices)
        should_increase_negative = np.intersect1d(np.where((np.log(1 - y_i_hat[:, 0]) + beta) > 0)[0], neg_indices)

        z[should_decrease_positive] = z[should_decrease_positive] / (1 + alpha)
        z[should_decrease_negative] = z[should_decrease_negative] / (1 + alpha)
        z[should_increase_positive] = z[should_increase_positive] * (1 + alpha)
        z[should_increase_negative] = z[should_increase_negative] * (1 + alpha)
        z = np.where(z > 1, 1, z)
        increased = np.union1d(should_increase_positive, should_increase_negative)
        decreased = np.union1d(should_decrease_positive, should_decrease_negative)
        draw_set_separator(theta.reshape(-1, ), X[increased, :], y[increased, :], 'sepa_1_1' + str(i * 10 + p), pdf)
        start_print = 436
        pdf.text(20, start_print - 16, 'z = ' + str(theta))
        pdf.text(20, start_print, 'z incremented')
        pdf.text(20, start_print + 16, 'z incremented = ' + str(len(increased)))
        pdf.text(20, start_print + 32, 'beta = ' + str(beta))
        pdf.text(20, start_print + 48, 'alpha = ' + str(alpha))
        pdf.add_page()

        draw_set_separator(theta.reshape(-1, ), X[decreased, :], y[decreased, :], 'sepa_1_2' + str(i * 10 + p), pdf)
        start_print = 436
        pdf.text(20, start_print - 16, 'z = ' + str(theta))
        pdf.text(20, start_print, 'z decreased')
        pdf.text(20, start_print + 16, 'z decreased = ' + str(len(decreased)))
        pdf.text(20, start_print + 32, 'beta = ' + str(beta))
        pdf.text(20, start_print + 48, 'alpha = ' + str(alpha))
        pdf.add_page()
        theta, J_history, lr = gradient_descent(X, y, z, theta, lr, 20, p, beta, pdf)

    return theta, z


def get_hyp_2(X, y, pdf):
    y = y.reshape(-1, 1)
    # Initialize theta and hyperparameters
    theta = -1 + 2 * np.random.random((X.shape[1], 1))
    p = 0
    max_dist = 0
    z = np.ones(X.shape[0])
    first = True
    while p < 5:
        X_sub, y_sub = X[np.where(z==1)[0], :], y[np.where(z==1)[0], :]
        theta = simple_regression(X_sub, y_sub, pdf)
        max_y_i_hat_index = np.argmin((y * np.log(sigmoid(np.dot(X, theta.reshape(-1, 1)))) + (1-y) * np.log(1-sigmoid(np.dot(X, theta.reshape(-1, 1))))))
        if y[max_y_i_hat_index] == 1:
            max_dist = -1 * np.log(sigmoid(np.dot(X[max_y_i_hat_index, :], theta.reshape(-1, 1))))
        else:
            max_dist = -1 * np.log(1 - sigmoid(np.dot(X[max_y_i_hat_index, :], theta.reshape(-1, 1))))
        beta = (max_dist + np.log(2)) / 2
        start_print = 436
        pdf.text(20, start_print, 'Logistic regression finished')
        pdf.text(20, start_print + 16, 'Max dist = ' + str(max_dist))
        pdf.text(20, start_print + 32, 'beta = ' + str(beta))
        pdf.add_page()
        if first:
            theta = theta
        else:
            theta = theta_prev
        theta_prev, z = modified_regression(X, y, theta.reshape(-1, 1), beta, z, pdf)
        p += 1
        first = False
    return theta


# def main():
#     # Load data
#     data = pd.read_csv('../union_of_convex_datasets/checkerboard.csv', header=None).values
#     data_x = np.hstack((data[:, 0:-1], np.ones(data.shape[0]).reshape(-1, 1))).T
#     data_y = data[:, -1]
#     # asso_hyp = data[:, -1]
#     #print(Counter(asso_hyp))
#     orig_hyp = pd.read_csv('../union_of_convex_datasets/checkerboard.csv', header=None).values
#     data_y = np.where(data_y == -1, 0, 1)
#     print(np.sum(np.where(data_y == 1, 1, 0)))
#     print(np.sum(np.where(data_y == 0, 1, 0)))
#     current_dir = os.getcwd()
#     if os.path.exists(os.path.join(current_dir, 'log')) is False:
#         os.makedirs(os.path.join(current_dir, 'log'))
#     os.chdir(os.path.join(current_dir, 'log'))
#
#     for threshold, i in zip([0.0001, 0.001, 0.01, 0.1, 1, 10], [1, 2, 3, 4, 5, 6]):
#         np.random.seed(i)
#         pdf = FPDF(orientation='L', unit='pt', format='A4')
#         pdf.add_page()
#         pdf.set_font('Helvetica', 'I', 10)
#         pdf.set_text_color(0, 0, 0)
#
#         X, y = sample_pairs(data_x, data_y)
#         y = y.reshape(-1, 1)
#         z = np.ones(X.shape[0])
#
#
#         # Initialize theta and hyperparameters
#         theta = -1 + 2*np.random.random((X.shape[1], 1))
#         alpha = 0.01
#         num_iters = 1000
#         p = 0
#         threshold_1 = threshold
#         while p<10:
#             threshold_1 = threshold_1/2
#             # Run gradient descent
#             theta, J_history = gradient_descent(X, y, z, theta, alpha, num_iters, p, threshold_1, pdf)
#
#             # Make predictions
#             pred = sigmoid(np.dot(X, theta))
#             pred[pred >= 0.5] = 1
#             pred[pred < 0.5] = 0
#
#             distances = np.abs(np.dot(X, theta))/np.linalg.norm(theta[0:-1, :])
#             misclassified = np.where(pred!=y)[0]
#             grossly_misclassified = np.intersect1d(np.intersect1d(np.where(pred!=y)[0], np.where(distances>threshold_1)[0]), np.where(z==1)[0])
#             not_grossly_misclassified = np.intersect1d(np.union1d(np.where(pred==y)[0], np.where(distances<=threshold_1)[0]), np.where(z==1)[0])
#             start_print = 436
#             draw_set_separator(theta.reshape(-1,), X[not_grossly_misclassified, :], y[not_grossly_misclassified, :], 'sepa_1_1' + str(i*10 + p), threshold_1, pdf)
#             pdf.text(20, start_print, 'points_retained')
#             pdf.text(20, start_print + 16, 'points_retained = ' + str(np.sum(z) - len(grossly_misclassified)))
#             pdf.add_page()
#             draw_set_separator(theta.reshape(-1,), X[grossly_misclassified, :], y[grossly_misclassified, :], 'sepa_1_2' + str(i*10 + p), threshold_1, pdf)
#             pdf.text(20, start_print, 'points_removed')
#             pdf.text(20, start_print + 16, 'points_removed = ' + str(len(grossly_misclassified)))
#             pdf.add_page()
#             if len(grossly_misclassified) == 0:
#                 break
#             z[grossly_misclassified] = 0
#             p+=1
#
#         pdf.output('logistic_regression_variable_' + str(threshold) + '.pdf')
#     os.chdir(current_dir)

def main():
    data = pd.read_csv('../union_of_convex_datasets/stripes.csv', header=None).values
    data_x = np.hstack((data[:, 0:-1], np.ones(data.shape[0]).reshape(-1, 1))).T
    data_y = data[:, -1]
    # asso_hyp = data[:, -1]
    #print(Counter(asso_hyp))
    orig_hyp = pd.read_csv('../union_of_convex_datasets/stripes.csv', header=None).values
    data_y = np.where(data_y == -1, 0, 1)
    print(np.sum(np.where(data_y == 1, 1, 0)))
    print(np.sum(np.where(data_y == 0, 1, 0)))
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, 'log')) is False:
        os.makedirs(os.path.join(current_dir, 'log'))
    os.chdir(os.path.join(current_dir, 'log'))

    for threshold, i in zip([0.0001], [1]):
        np.random.seed(i+1)
        pdf = FPDF(orientation='L', unit='pt', format='A4')
        pdf.add_page()
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(0, 0, 0)

        X, y = sample_pairs(data_x, data_y)
        y = y.reshape(-1, 1)
        z = np.ones(X.shape[0])
        theta = get_hyp_2(X, y, pdf)

        # # Initialize theta and hyperparameters
        # theta = -1 + 2*np.random.random((X.shape[1], 1))
        # alpha = 0.1
        # num_iters = 1000
        # p = 0
        # threshold_1 = threshold
        # beta = 1.2*np.log(2)
        # pos_indices = np.where(y == 1)[0]
        # neg_indices = np.where(y == 0)[0]
        # c = 0
        # lr = 0.01
        # first = True
        # while p < 5:
        #     if beta<np.log(2):
        #        beta = beta*(1+alpha)
        #     # Run gradient descent
        #     # w = theta.reshape(-1,)
        #     theta, J_history, lr = gradient_descent(X, y, z, theta, lr, num_iters, p, threshold_1, beta, z, pdf)
        #     # w_1 = theta.reshape(-1,)
        #     # if not first:
        #     #     x = np.argmin((y * np.log(sigmoid(np.dot(X, w.reshape(-1, 1)))) + (1-y) * np.log(1-sigmoid(np.dot(X, w.reshape(-1, 1))))))
        #     #     if y[x]==1:
        #     #         beta = np.log(np.exp(beta)*(1+np.exp(-1*np.dot(w_1.T, X[x])))/(1+np.exp(-1*np.dot(w.T, X[x]))))
        #     #     else:
        #     #         beta = np.log(np.exp(beta) * ((1 + np.exp(-1 * np.dot(w_1.T, X[x]))) / (1 + np.exp(-1 * np.dot(w.T, X[x]))))*(np.exp(-1*np.dot(w.T, X[x]))/np.exp(-1*np.dot(w_1.T, X[x]))))
        #     # Make predictions
        #     y_i_hat = sigmoid(np.dot(X, theta))
        #     should_decrease_positive = np.intersect1d(np.where((np.log(y_i_hat[:, 0]) + beta) < 0)[0], pos_indices)
        #     should_decrease_negative = np.intersect1d(np.where((np.log(1-y_i_hat[:, 0]) + beta) < 0)[0], neg_indices)
        #     should_increase_positive = np.intersect1d(np.where((np.log(y_i_hat[:, 0]) + beta) > 0)[0], pos_indices)
        #     should_increase_negative = np.intersect1d(np.where((np.log(1-y_i_hat[:, 0]) + beta) > 0)[0], neg_indices)
        #     z_prime = np.copy(z)
        #     #z_prime = z
        #     z[should_decrease_positive] = z[should_decrease_positive]/(1+alpha)
        #     z[should_decrease_negative] = z[should_decrease_negative]/(1+alpha)
        #     z[should_increase_positive] = z[should_increase_positive]*(1+alpha)
        #     z[should_increase_negative] = z[should_increase_negative]*(1+alpha)
        #     z = np.where(z > 1, 1, z)
        #     # increased = np.where(z >= z_prime)[0]
        #     # decreased = np.where(z < z_prime)[0]
        #     # print(np.intersect1d(increased, pos_indices))
        #     # print(np.intersect1d(increased, neg_indices))
        #     # print(np.intersect1d(decreased, pos_indices))
        #     # print(np.intersect1d(decreased, neg_indices))
        #     increased = np.union1d(should_increase_positive, should_increase_negative)
        #     decreased = np.union1d(should_decrease_positive, should_decrease_negative)
        #
        #
        #     draw_set_separator(theta.reshape(-1, ), X[increased, :], y[increased, :], 'sepa_1_1' + str(i * 10 + p), threshold_1, z, pdf)
        #     start_print = 436
        #     pdf.text(20, start_print - 16, 'z = ' + str(theta))
        #     pdf.text(20, start_print, 'z incremented')
        #     pdf.text(20, start_print + 16, 'z incremented = ' + str(len(increased)))
        #     pdf.text(20, start_print + 32, 'beta = ' + str(beta))
        #     pdf.text(20, start_print + 48, 'alpha = ' + str(alpha))
        #     pdf.add_page()
        #
        #     draw_set_separator(theta.reshape(-1, ), X[decreased, :], y[decreased, :], 'sepa_1_2' + str(i * 10 + p), threshold_1, z, pdf)
        #     start_print = 436
        #     pdf.text(20, start_print - 16, 'z = ' + str(theta))
        #     pdf.text(20, start_print, 'z decreased')
        #     pdf.text(20, start_print + 16, 'z decreased = ' + str(len(decreased)))
        #     pdf.text(20, start_print + 32, 'beta = ' + str(beta))
        #     pdf.text(20, start_print + 48, 'alpha = ' + str(alpha))
        #     pdf.add_page()
        #     #beta = beta/(1+alpha)
        #     p += 1
        #     first = False
        #     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        #     #print(z)
        pdf.output('algo_on_stripes_1.pdf')
    os.chdir(current_dir)

if __name__ == '__main__':
    main()