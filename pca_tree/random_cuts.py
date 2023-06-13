import pandas as pd
import numpy as np
from fpdf import FPDF
import os
from sklearn import tree
import matplotlib.pyplot as plt

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
        lines = -1 + 2 * np.random.random((data_x.shape[0], 20))
        print(lines)
        rand_data = data_x[:, np.random.randint(0, data_x.shape[1], 20)]
        print(rand_data[0:-1, :].shape)
        print(lines[0:-1, :].shape)
        lines[-1, :] = -1 * np.sum(lines[0:-1, :] * rand_data[0:-1, :], axis=0)
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
        print(clf.tree_.node_count)
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
        pdf.output('random_technique_30' + str(a) + '.pdf')
        os.chdir(current_dir)


if __name__ == '__main__':
    main()