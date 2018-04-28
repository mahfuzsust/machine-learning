import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans

style.use("ggplot")

colors = ["b.", "g.", "r.", "c.", "m.", "k.", "y.", "w."]

high_value = 10000
total_points = 1000
number_of_cluster = 5


def rand_generator(n):
    return [random.randint(1, high_value) for i in range(n)]


x = rand_generator(total_points)
y = rand_generator(total_points)

plt.scatter(x, y)

# plt.show()
plt.savefig("random.png")

p = list(zip(x, y))

# p = [[x[i], y[i]] for i in range(total_points)]

X = np.array(p)
k_means = KMeans(n_clusters=number_of_cluster)
k_means.fit(X)
centroids = k_means.cluster_centers_
labels = k_means.labels_

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i] % 8], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
# plt.show()
plt.savefig("clustered.png")
