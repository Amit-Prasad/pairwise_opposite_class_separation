import numpy as np
from sklearn.decomposition import PCA

def min_distance_point(point, data_x, data_y, label):
    label_indices = np.where(data_y==label)[0]
    dist = np.zeros(data_y.shape) + np.Inf
    data_s = data_x[:, label_indices]
    dist[label_indices] = np.linalg.norm(data_s - point.reshape(-1, 1), axis = 0)
    min_index = np.argmin(dist)
    return min_index

def pick_init_hyp_1_2D(points_pos, points_neg):
    midpoints = (points_pos + points_neg) / 2

    points_pos_copy = np.copy(points_pos)
    points_neg_copy = np.copy(points_neg)

    random_index = np.random.choice(np.arange(0, midpoints.shape[1]))
    random_point = midpoints[:, random_index]
    midpoints = np.delete(midpoints, random_index, 1)

    points_pos_copy = np.delete(points_pos_copy, random_index, 1)
    points_neg_copy = np.delete(points_neg_copy, random_index, 1)

    closest_points = []
    closest_points_pos = []
    closest_points_neg = []
    n_points = 25
    if midpoints.shape[1]<n_points:
        return None, None, None
    #if midpoints.shape[1] < n_points:
    #    n_points = int((0.1)*n_points)

    for i in range(0, n_points):
        if midpoints.shape[1] == 0:
            break
        dist = np.linalg.norm(midpoints - random_point.reshape(-1, 1), axis=0)
        min_index = np.argmin(dist)
        closest_points.append(midpoints[:, min_index])
        closest_points_pos.append(points_pos_copy[:, min_index])
        closest_points_neg.append(points_neg_copy[:, min_index])
        midpoints = np.delete(midpoints, min_index, 1)
        points_pos_copy = np.delete(points_pos_copy, min_index, 1)
        points_neg_copy = np.delete(points_neg_copy, min_index, 1)

    closest_points = np.asarray(closest_points).T
    closest_points_pos = np.asarray(closest_points_pos).T
    closest_points_neg = np.asarray(closest_points_neg).T
    regressor = np.zeros((closest_points.shape[0]-1, closest_points.shape[1]))
    regressor[0, :] = closest_points[0, :]
    regressor[1, :] = closest_points[-1, :]
    pca = PCA(n_components=points_pos.shape[0]-1)
    pca.fit(closest_points[0:-1, :].T)
    line = pca.components_[points_pos.shape[0]-2]
    #line = LinearRegression(fit_intercept=False).fit(regressor.T, closest_points[1, :])
    w = np.zeros(points_pos.shape[0])
    # w[0] = line.coef_[0]*-1
    # w[1] = 1
    # w[-1] = line.coef_[1]*-1
    w[0:-1] = line
    mean = np.sum(closest_points, axis=1)/closest_points.shape[1]
    w[-1] = np.dot(w[0:-1], mean[0:-1]) * -1
    #line = LinearRegression().fit(closest_points[0, :].reshape(-1, 1), closest_points[1, :])
    #print(np.matmul(np.matmul(np.linalg.inv(np.matmul(regressor, regressor.T)), regressor), closest_points[1, :]))
    #print(closest_points)
    return w, random_point, closest_points

def pick_init_hyp_1(points_pos, points_neg, orig_hyp, pdf):
    midpoints = (points_pos + points_neg) / 2

    points_pos_copy = np.copy(points_pos)
    points_neg_copy = np.copy(points_neg)

    random_index = np.random.choice(np.arange(0, midpoints.shape[1]))
    random_point = midpoints[:, random_index]
    midpoints = np.delete(midpoints, random_index, 1)

    points_pos_copy = np.delete(points_pos_copy, random_index, 1)
    points_neg_copy = np.delete(points_neg_copy, random_index, 1)

    closest_points = []
    closest_points_pos = []
    closest_points_neg = []
    n_points = 25
    if midpoints.shape[1]<n_points:
        return None, None, None
    #if midpoints.shape[1] < n_points:
    #    n_points = int((0.1)*n_points)

    for i in range(0, n_points):
        if midpoints.shape[1] == 0:
            break
        dist = np.linalg.norm(midpoints - random_point.reshape(-1, 1), axis=0)
        min_index = np.argmin(dist)
        closest_points.append(midpoints[:, min_index])
        closest_points_pos.append(points_pos_copy[:, min_index])
        closest_points_neg.append(points_neg_copy[:, min_index])
        midpoints = np.delete(midpoints, min_index, 1)
        points_pos_copy = np.delete(points_pos_copy, min_index, 1)
        points_neg_copy = np.delete(points_neg_copy, min_index, 1)

    closest_points = np.asarray(closest_points).T
    closest_points_pos = np.asarray(closest_points_pos).T
    closest_points_neg = np.asarray(closest_points_neg).T
    regressor = np.zeros((closest_points.shape[0]-1, closest_points.shape[1]))
    regressor[0, :] = closest_points[0, :]
    regressor[1, :] = closest_points[-1, :]
    pca = PCA(n_components=points_pos.shape[0]-1)
    pca.fit(closest_points[0:-1, :].T)
    line = pca.components_[points_pos.shape[0]-2]
    #line = LinearRegression(fit_intercept=False).fit(regressor.T, closest_points[1, :])
    w = np.zeros(points_pos.shape[0])
    # w[0] = line.coef_[0]*-1
    # w[1] = 1
    # w[-1] = line.coef_[1]*-1
    w[0:-1] = line
    mean = np.sum(closest_points, axis=1)/closest_points.shape[1]
    w[-1] = np.dot(w[0:-1], mean[0:-1]) * -1
    side_pos = np.sign(np.dot(orig_hyp.T, closest_points_pos))
    side_neg = np.sign(np.dot(orig_hyp.T, closest_points_neg))
    hyp_in = np.where(side_pos != side_neg, 1, 0)
    #for i in range(0, orig_hyp.shape[1]):
    #    pdf.text(10, 16*(i+1), str(hyp_in[i,:]))
    #pdf.add_page()
    #line = LinearRegression().fit(closest_points[0, :].reshape(-1, 1), closest_points[1, :])
    #print(np.matmul(np.matmul(np.linalg.inv(np.matmul(regressor, regressor.T)), regressor), closest_points[1, :]))
    #print(closest_points)
    return w, random_point, closest_points

def pick_init_hyp_2(points_pos, points_neg, data_x, data_y, removed, orig_hyp, pdf):
    midpoints = (points_pos + points_neg) / 2
    random_index = np.random.choice(np.arange(0, midpoints.shape[1]))
    random_point = midpoints[:, random_index]
    random_pos = points_pos[:, random_index]
    random_neg = points_neg[:, random_index]
    midpoints = np.delete(midpoints, random_index, 1)
    n_points = 25
    if midpoints.shape[1] < n_points:
        return None, None, None

    dist = np.linalg.norm(data_x - random_point.reshape(-1, 1), axis=0)
    # dist[removed] = 1000000.0
    min_indices = np.argsort(dist)[0:n_points]
    candidate_points_pos = []
    candidate_points_neg = []
    data_s = np.copy(data_x)
    data_s_y = np.copy(data_y)
    # data_s = np.delete(data_s, removed, 1)
    # data_s_y = np.delete(data_s_y, removed)
    for i in range(0, len(min_indices)):
        p2_old = -1
        p3_old = -1
        index = min_indices[i]
        x_fp = data_x[:, index]
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
    closest_points = (candidate_points_pos + candidate_points_neg)/2
    regressor = np.zeros((closest_points.shape[0]-1, closest_points.shape[1]))
    regressor[0, :] = closest_points[0, :]
    regressor[1, :] = closest_points[-1, :]
    pca = PCA(n_components=points_pos.shape[0]-1)
    pca.fit(closest_points[0:-1, :].T)
    line = pca.components_[points_pos.shape[0]-2]
    #line = LinearRegression(fit_intercept=False).fit(regressor.T, closest_points[1, :])
    w = np.zeros(points_pos.shape[0])
    # w[0] = line.coef_[0]*-1
    # w[1] = 1
    # w[-1] = line.coef_[1]*-1
    w[0:-1] = line
    mean = np.sum(closest_points, axis=1)/closest_points.shape[1]
    w[-1] = np.dot(w[0:-1], mean[0:-1]) * -1
    side_pos = np.sign(np.dot(orig_hyp.T, candidate_points_pos))
    side_neg = np.sign(np.dot(orig_hyp.T, candidate_points_neg))
    rand_side_pos = np.sign(np.dot(orig_hyp.T, random_pos))
    rand_side_neg = np.sign(np.dot(orig_hyp.T, random_neg))
    hyp_in = np.where(rand_side_pos != rand_side_neg, 1, 0)
    pdf.text(10, 16, str(hyp_in))
    pdf.text(10, 32, str(np.arange(1, len(hyp_in)+1)))
    hyp_in = np.where(side_pos != side_neg, 1, 0)
    pointing_vectors = candidate_points_neg - candidate_points_pos
    normed_pointing_vectors = pointing_vectors / np.linalg.norm(pointing_vectors, axis=0)
    avg_pointing_vector = np.sum(normed_pointing_vectors, axis=1) / normed_pointing_vectors.shape[1]
    avg_pointing_vector = avg_pointing_vector/np.linalg.norm(avg_pointing_vector)
    normed_pointing_vectors = np.hstack((normed_pointing_vectors, avg_pointing_vector.reshape(-1,1)))
    pdf.set_font('Helvetica', 'I', 7)
    for i in range(0, orig_hyp.shape[1]):
        printed = np.zeros((pointing_vectors.shape[1], 3), dtype = int)
        angs = (np.arccos(np.dot(orig_hyp[:, i], normed_pointing_vectors) / np.linalg.norm(orig_hyp[:, i])) * (180 / np.pi)).astype(int)
        printed[:, 0] = hyp_in[i, :]
        printed[:, 1] = angs[0:-1]
        angs_2 = (np.arccos(np.dot(avg_pointing_vector, normed_pointing_vectors[:, 0:-1])) * (180 / np.pi)).astype(int)
        printed[:, 2] = angs_2
        printed = printed[printed[:, 0].argsort()]
        pdf.text(10, 16 * (2 * i + 4), str(printed[:, 0:-1]))
        pdf.text(10, 16*(2*i +1+4), str(printed[:, [0, 2]]))
        pdf.text(770, 16*(2*i+4), str(i+1))
        pdf.text(790, 16*(2*i+4), str(np.sum(hyp_in[i, :])))
        pdf.text(810, 16 * (2*i+4), str(angs[-1]))
    pdf.set_font('Helvetica', 'I', 10)
    pdf.add_page()
    #line = LinearRegression().fit(closest_points[0, :].reshape(-1, 1), closest_points[1, :])
    #print(np.matmul(np.matmul(np.linalg.inv(np.matmul(regressor, regressor.T)), regressor), closest_points[1, :]))
    #print(closest_points)
    return w, random_point, closest_points

def x_by_one_plus_x_der(x):
    return 1/((1+x)**2)

def x_by_one_plus_x(x):
    return x/(1+x)

def u_shaped_x_by_one_plus_x(w, points_pos, points_neg):
    temp_1 = points_pos + points_neg
    temp_2 = points_pos - points_neg
    num = np.dot(w, temp_1)
    deno = np.dot(w[0:-1], temp_2[0:-1, :])
    obj_temp = np.zeros((2, points_pos.shape[1]))
    obj_temp[0, :] = np.abs(num/deno) - 1
    weights = np.linalg.norm(temp_2, axis=0)
    ex = weights*np.amax(obj_temp, axis=0)
    return np.sum(x_by_one_plus_x(ex))

def u_shaped_x_by_one_plus_x_der(w, points_pos, points_neg):
    temp_1 = points_pos + points_neg
    temp_2 = points_pos - points_neg
    num = np.dot(w, temp_1)
    deno = np.dot(w[0:-1], temp_2[0:-1, :])
    signs = np.sign(num/deno)
    indicator = np.where(np.abs(num/deno)-1>0, 1, 0)
    gradient_deno = deno * deno
    gradient_num = np.tile(np.dot(w[0:-1], temp_2[0:-1, :]), (points_pos.shape[0]-1, 1))*(points_pos[0:-1, :]+points_neg[0:-1, :]) - np.tile((np.dot(w, (points_pos+points_neg))), (points_pos.shape[0]-1, 1)) * (points_pos[0:-1, :] - points_neg[0:-1, :])
    #weights = np.linalg.norm(temp_2, axis=0)
    gradient = np.zeros(w.shape)
    obj_temp = np.zeros((2, points_pos.shape[1]))
    obj_temp[0, :] = np.abs(num / deno) - 1
    weights = np.linalg.norm(temp_2, axis=0)
    ex = weights * np.amax(obj_temp, axis=0)
    temp = x_by_one_plus_x_der(ex)
    gradient[0:-1] = np.sum(weights*indicator*signs*temp*(gradient_num/gradient_deno), axis=1)
    gradient[-1] = np.sum((2/deno)*weights*indicator*signs*temp)
    return gradient


































