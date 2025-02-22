# %%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint

# %%
np.set_printoptions(threshold=sys.maxsize)

# %%
data = pd.read_csv("Family-Income-and-Expenditure.csv")
source_data = np.array(data)
print(source_data)


mini_data = data.iloc[:, :2]
mini_data = (mini_data - mini_data.min()) / (mini_data.max() - mini_data.min())
mini_data = np.array(mini_data)

data = (data - data.min()) / (data.max() - data.min())
data = np.array(data)

print(f"data_shape->{np.shape(data)}")


def generate_random_light_color():
    def random_light_value():
        return randint(75, 255)

    r = random_light_value()
    g = random_light_value()
    b = random_light_value()

    return f"#{r:02x}{g:02x}{b:02x}"


def distance(data, center):
    return np.sqrt(np.sum(np.pow((data - center), 2), axis=1))


def fit(data, centers):
    rows = np.shape(data)[0]
    res = -1 * np.ones([rows, len(centers)])

    i = 0
    for center in centers:
        res[:, i] = distance(data, center)
        i += 1

    dist_res = np.min(res, axis=1)
    res = np.argmin(res, axis=1)
    return [dist_res, res]


def update_centers(data, assigns):
    unique_clusters = np.unique(assigns)
    new_centers = -1 * np.ones([len(unique_clusters), np.shape(data)[1]])
    i = 0
    for cluster in unique_clusters:
        new_centers[i, :] = np.mean(data[assigns == cluster], axis=0)
        i += 1
    return new_centers


def sse(clusters):
    return np.sum(np.pow(clusters, 2))


def show_plot(sse_array):
    plt.plot(range(1, len(sse_array) + 1), sse_array)
    plt.xticks(range(1, len(sse_array) + 1))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()


def kmeans(data, k):
    centers = np.random.permutation(data)[:k]
    dist_clusters, clusters = fit(data, centers)
    new_centers = update_centers(data, clusters)
    sub = np.sum(centers - new_centers)

    i = 0
    while sub != 0:
        centers = new_centers
        dist_clusters, clusters = fit(data, centers)
        new_centers = update_centers(data, clusters)
        sub = np.sum(centers - new_centers)
        i += 1
    return [sse(np.array(dist_clusters)), clusters]


def best_kmeans(data, k, max_iter, n_init):
    centers = np.random.permutation(data)[:k]
    clusters = fit(data, centers)[1]
    new_centers = update_centers(data, clusters)
    sub = np.sum(centers - new_centers)

    i = 0
    while sub != 0 and n_init < i < max_iter:
        centers = new_centers
        clusters = fit(data, centers)[1]
        new_centers = update_centers(data, clusters)
        sub = np.sum(centers - new_centers)
        i += 1
    return clusters


def show_scatter_plot(assigns, x_index, y_index, x_text, y_text):
    unique_clusters = np.unique(assigns)
    for cluster in unique_clusters:
        plt.scatter(
            source_data[cluster == assigns, x_index],
            source_data[cluster == assigns, y_index],
            c=generate_random_light_color(),
        )
    plt.ylabel(y_text)
    plt.xlabel(x_text)
    plt.show()


# sse_array = []
# for i in range(1, 16):
#     sse_res = kmeans(data, i, 100)[0]
#     sse_array.append(sse_res)
# show_plot(np.array(sse_array))

optimal_k = 6
sse_res, clusters = kmeans(data, optimal_k)
print(f"SSE -> {sse_res}")
show_scatter_plot(clusters, 0, 1, "Total Household Income", "Total Food Expenditure")
show_scatter_plot(
    clusters, 0, 9, "Total Household Income", "Restaurant and hotels Expenditure"
)
show_scatter_plot(
    clusters, 0, 15, "Total Household Income", "Alcoholic Beverages Expenditure"
)

# %%
