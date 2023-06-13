import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

np.random.seed(0)
def assign(hyp, index, point):
    if index >= hyp.shape[1]:
        return 1
    if index<hyp.shape[1] and 2*index+1>=hyp.shape[1] and 2*index+2>=hyp.shape[1]:
        return np.sign(np.dot(hyp[:, index], point))
    if np.sign(np.dot(hyp[:, index], point)).all() == -1:
        return -1*assign(hyp, 2*index+1, point)
    else:
        return assign(hyp, 2*index+2, point)


def find_point_projections(data_x, vectors, w):
    orthogonal = np.dot(w, data_x)
    parallel = np.dot(vectors.T, data_x)
    return orthogonal, parallel

def plot_projections(hyp, vectors, orthogonal, parallel, data_y, labels, pdf):
    true_positive_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 1))
    true_negative_indices = np.intersect1d(np.where(data_y == -1), np.where(labels == -1))
    false_positive_indices = np.intersect1d(np.where(data_y == -1), np.where(labels == 1))
    false_negative_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == -1))

    for i in range(0, parallel.shape[0]):
        plt.scatter(orthogonal[true_positive_indices], parallel[i, true_positive_indices], c='blue', s=1)
        plt.scatter(orthogonal[true_negative_indices], parallel[i, true_negative_indices], c='yellow', s=1)
        plt.scatter(orthogonal[false_positive_indices], parallel[i, false_positive_indices], c='lime', s=1)
        plt.scatter(orthogonal[false_negative_indices], parallel[i, false_negative_indices], c='pink', s=1)

        x_min, x_max = orthogonal.min() - 1, orthogonal.max() + 1
        y_min, y_max = parallel[i].min() - 1, parallel[i].max() + 1
        plt.axis([x_min, x_max, y_min, y_max])
        plt.legend(loc='upper right')
        plt.savefig('labels_' + str(parallel[i, 0]) + str(i) + '.png')
        if i == 0:
            pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=0, y=0, w=300, h=300)
            pdf.text(0, 310, 'The parallel line = ' + str(vectors[:, i]))
            pdf.text(0, 322, 'positive vs negative counts = ' + str(len(true_positive_indices)) + ' ' + str(len(true_negative_indices)))
        elif i == 1:
            pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=310, y=0, w=300, h=300)
            pdf.text(340, 310, 'The parallel line = ' + str(vectors[:, i]))
        elif i == 2:
            pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=0, y=310, w=300, h=300)
            pdf.text(0, 330, 'The parallel line = ' + str(vectors[:, i]))
        else:
            pdf.image('labels_' + str(parallel[i, 0]) + str(i) + '.png', x=310, y=310, w=300, h=300)
            pdf.text(340, 330, 'The parallel line = ' + str(vectors[:, i]))
        plt.gca().cla()
    pdf.text(20, 26, 'The hyperplane = ' + str(hyp))

dim = 3
N = 10000
data_x = -1 + 2 * np.random.random((dim, N))
asso_hyp = np.zeros(N, dtype=int)


hyp_count = 3
hyp = -1 + 2 * np.random.random((dim+1, hyp_count))
hyp[:, 0] = np.array([1, 0, 0, 0])
#hyp[-1, 0] = -1 * np.dot(hyp[0:-1, 0], np.sum(data_x, axis=1)/data_x.shape[1])
side_root = np.sign(np.dot(hyp[0:-1, 0], data_x) + hyp[-1, 0])
pos_indices = np.where(side_root == 1)[0]
neg_indices = np.where(side_root == -1)[0]
print(len(pos_indices))
print(len(neg_indices))

data_x_left = data_x[:, neg_indices]
hyp[:, 1] = np.array([0, 1, 0, -0.5])
#hyp[-1, 1] = -1 * np.dot(hyp[0:-1, 1], np.sum(data_x_left, axis=1)/data_x_left.shape[1])
side_left = np.zeros_like(side_root)
side_left[neg_indices] = np.sign(np.dot(hyp[0:-1, 1], data_x_left) + hyp[-1, 1])
data_y_left = side_left[neg_indices]
print(np.sum(np.where(side_left == 1, 1, 0)))
print(np.sum(np.where(side_left == -1, 1, 0)))
side_left = np.where(side_left == 0, 1, side_left)

data_x_right = data_x[:, pos_indices]
hyp[:, 2] = np.array([0, 0, 1, 0.5])
#hyp[-1, 2] = -1 * np.dot(hyp[0:-1, 2], np.sum(data_x_right, axis=1)/data_x_right.shape[1])
side_right = np.zeros_like(side_root)
side_right[pos_indices] = np.sign(np.dot(hyp[0:-1, 2], data_x_right) + hyp[-1, 2])
data_y_right = side_right[pos_indices]
print(np.sum(np.where(side_right == 1, 1, 0)))
print(np.sum(np.where(side_right == -1, 1, 0)))
side_right = np.where(side_right == 0, 1, side_right)

data_y = side_root * (-1 * side_left) * side_right


pdf = FPDF(orientation='L', unit='pt', format='A4')
pdf.add_page()
pdf.set_font('Helvetica', 'I', 10)
pdf.set_text_color(0, 0, 0)

#left child y axis
orthogonal, parallel = find_point_projections(data_x_left, hyp[0:-1, 1], hyp[0:-1, 0])
plot_projections(hyp[0:-1, 0], hyp[0:-1, 1].reshape(-1, 1), orthogonal, parallel.reshape(1, -1), data_y_left, data_y_left, pdf)
pdf.add_page()

#right child y axis
orthogonal, parallel = find_point_projections(data_x_right, hyp[0:-1, 2], hyp[0:-1, 0])
plot_projections(hyp[0:-1, 0], hyp[0:-1, 2].reshape(-1, 1), orthogonal, parallel.reshape(1, -1), data_y_right, data_y_right, pdf)
pdf.add_page()

#left child y axis all points
orthogonal, parallel = find_point_projections(data_x, hyp[0:-1, 1], hyp[0:-1, 0])
plot_projections(hyp[0:-1, 0], hyp[0:-1, 1].reshape(-1, 1), orthogonal, parallel.reshape(1, -1), data_y, data_y, pdf)
pdf.add_page()

#right child y axis all points
orthogonal, parallel = find_point_projections(data_x, hyp[0:-1, 2], hyp[0:-1, 0])
plot_projections(hyp[0:-1, 0], hyp[0:-1, 2].reshape(-1, 1), orthogonal, parallel.reshape(1, -1), data_y, data_y, pdf)
pdf.add_page()
pdf.output('three_3D_axis_parallel_hyperplanes_viz' + '.pdf')

# for i in range(N):
#     point = -1 + 2*np.random.random(dim+1)
#     point[-1] = 1
#     dist = np.dot(hyp.T, point)/np.linalg.norm(hyp[0:-1, :], axis=0)
#     asso_hyp[i] = np.argmin(dist)
#     label = assign(hyp, 0, point)
#     data_x[i, :] = point[0:-1]
#     data_y[i] = label
# pos_indices = np.where(data_y == 1)[0]
# neg_indices = np.where(data_y == -1)[0]

#left_child
# orthogonal, parallel = find_point_projections(data_x.T, hyp[0:-1, 1], hyp[0:-1, 0])
# plot_projections(hyp[0:-1, 0], hyp[0:-1, 1].reshape(-1, 1), orthogonal, parallel.reshape(1, -1), data_y, data_y, pdf)
# pdf.add_page()
#right_child
# orthogonal, parallel = find_point_projections(data_x.T, hyp[0:-1, 2], hyp[0:-1, 0])
# plot_projections(hyp[0:-1, 0], hyp[0:-1, 2].reshape(-1, 1), orthogonal, parallel.reshape(1, -1), data_y, data_y, pdf)
# pdf.add_page()
# pdf.output('pca_tree_multi.pdf')
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(data_x[0, pos_indices], data_x[1, pos_indices], data_x[2, pos_indices], c='blue', s=1)
# ax.scatter3D(data_x[0, neg_indices], data_x[1, neg_indices], data_x[2, neg_indices], c='yellow', s=1)
# plt.show()

data = np.hstack((data_x.T, data_y.reshape(-1, 1)))
data = np.hstack((data, asso_hyp.reshape(-1, 1)))
df = pd.DataFrame(data)
df.to_csv('three_3D_axis_parallel_hyperplanes.csv', header = False, index = False)
df = pd.DataFrame(hyp)
df.to_csv('three_3D_axis_parallel_hyperplanes_hyperplanes.csv', header = False, index = False)


