import numpy as np
import pandas as pd
import os
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from fpdf import FPDF

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

def draw_set_separator(w_init, data_x, data_y, iter, pdf):
    #true_positive_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 1)[0])
    #true_negative_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 0)[0])
    #false_positive_indices = np.intersect1d(np.where(data_y == 0)[0], np.where(labels == 1)[0])
    #false_negative_indices = np.intersect1d(np.where(data_y == 1)[0], np.where(labels == 0)[0])
    #plt.scatter(data_x[true_negative_indices, 0], data_x[true_negative_indices, 1], c='yellow', s=1)
    #plt.scatter(data_x[false_positive_indices, 0], data_x[false_positive_indices, 1], c='lime', s=1)
    #plt.scatter(data_x[false_negative_indices, 0], data_x[false_negative_indices, 1], c='pink', s=1)
    #plt.scatter(data_x[true_positive_indices, 0], data_x[true_positive_indices, 1], c='blue', s=1)

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

    plt.plot(x1, x2, color='black', linestyle='dashed')
    plt.axis([x_min, x_max, y_min, y_max])
    plt.savefig('separator_hyp' + str(w_init[0]) + str(iter) + '.png')
    pdf.image('separator_hyp' + str(w_init[0]) + str(iter) + '.png', x=0, y=0, w=700,h=420)
    plt.gca().cla()

data = pd.read_csv('../union_of_convex_datasets/checkerboard.csv', header=None).values
#data_x = np.hstack((data[:, 0:-1], np.ones(data.shape[0]).reshape(-1, 1))).T
data_x = data.T
data_y = data[:, -1]
orig_hyp = pd.read_csv('../union_of_convex_datasets/checkerboard.csv', header=None).values
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

transformer = FastICA(random_state=0, whiten=False)
X_transformed = transformer.fit_transform(data_x.T)
print(X_transformed.shape)
print(transformer.mixing_)
for i in range(0, transformer.mixing_.shape[0]):
    draw_set_separator(transformer.mixing_[:, i], data_x, data_y, i, pdf)
    pdf.add_page()
pdf.output('just_things.pdf')
