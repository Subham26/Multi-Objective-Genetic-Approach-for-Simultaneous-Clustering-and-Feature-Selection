import math
import numpy as np
import cat_dist
import mutual_info


# Objective-I: Within-Cluster-Variance

def within_cluster_variance(chromosome, data):
    # Number of clusters
    K = len(chromosome[1])

    # Number of data points
    instance_count = data.shape[0]

    # Distances between cluster medoids should be > 0

    for m1 in range(K):
        index1 = chromosome[1][m1]

        for m2 in range(m1 + 1, K):
            index2 = chromosome[1][m2]

            if cat_dist.distance(np.multiply(np.array(data[index1]),
                                             np.array(chromosome[0])),
                                 np.multiply(np.array(data[index2]), np.array(chromosome[0]))) == 0:
                return math.inf

    # Within cluster variance
    wcv = 0

    for i in range(instance_count):
        wcv += min([cat_dist.distance(np.multiply(np.array(data[i]), np.array(chromosome[0])),
                                      np.multiply(np.array(data[chromosome[1][j]]), np.array(chromosome[0])))
                    for j in range(K)])

    # Normalized wcv
    return wcv / chromosome[0].count(1)


# Objective-II: Mutual Information-based Redundancy

def compute_redundancy(feature_set, data):
    # data [instance_count X feature_count]
    instance_count, feature_count = data.shape
    selected_feature_count = feature_set.count(1)

    redundancy = 0

    for counter1 in range(feature_count):
        if feature_set[counter1] == 1:
            vector1 = [int(data[i][counter1]) for i in range(instance_count)]
            for counter2 in range(counter1 + 1, feature_count):
                if feature_set[counter2] == 1:
                    vector2 = [int(data[i][counter2]) for i in range(instance_count)]
                    redundancy += mutual_info.compute_mutual_info(vector1, vector2)

    redundancy /= (selected_feature_count * (selected_feature_count - 1) / 2)

    return redundancy
