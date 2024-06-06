import numpy as np
import scipy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# Dataset
data_points = np.array([[1, 9], [2, 7], [2, 6], [3, 6], [3, 5], [4, 5],
                        [6, 7], [7, 7], [6, 6], [7, 6], [1, 2], [2, 3], [3, 2]])

# DBSCAN parameters
epsilon = 1
minpts = 4


# DBSCAN functions
def classify_points(data_points, epsilon, minpts):

    labels_inf = np.full(shape=data_points.shape[0], fill_value=-1)

    dist_matrix_inf = np.array([[max(abs(a - b)) for a in data_points] for b in data_points])

    for i, point_distances in enumerate(dist_matrix_inf):
        if len(point_distances[point_distances <= epsilon]) >= minpts:
            labels_inf[i] = 0

    for i, point_distances in enumerate(dist_matrix_inf):
        if labels_inf[i] == -1:
            if np.any(labels_inf[point_distances <= epsilon] == 0):
                labels_inf[i] = 1

    return labels_inf


def format_results(labels, data):
    core = data[labels == 0]
    border = data[labels == 1]
    noise = data[labels == -1]

    return {"core": core.tolist(),
            "border": border.tolist(),
            "noise": noise.tolist()}


# Run DBSCAN
labels_inf = classify_points(data_points, epsilon, minpts)
results = format_results(labels_inf, data_points)

print("DBSCAN Results:")
print(results)

# Rest of code
A = np.array([[0, 1, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 1],
              [0, 1, 1, 0]])

X = np.array([[1, 3], [1, 2], [2, 2], [4, 2], [4, 1], [4, 3]])
true_labels = [0, 0, 0, 1, 1, 1]

model = AgglomerativeClustering(n_clusters=2, linkage='average')
labels = model.fit_predict(X)
ari = adjusted_rand_score(true_labels, labels)

# print("Agglomerative Clustering ARI:", ari)