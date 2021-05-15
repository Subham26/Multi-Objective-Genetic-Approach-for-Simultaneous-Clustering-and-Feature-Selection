import cat_dist
import mutual_info


# Objective-I: Within-Cluster-Variance
def within_cluster_variance(data, chromosome, minimum_feature_count):

    # Number of clusters
    K = len(chromosome[1])

    # Number of data points
    instance_count = data.shape[0]

    # Choromosome should select the minimum number of features

    while chromosome[0].count(1) < minimum_feature_count:
        chromosome[0][random.randint(0, len(chromosome[0]) - 1)] = 1

    # Distances between cluster medoids should be > 0

    for m1 in range(K):
        index1 = chromosome[1][m1]

        for m2 in range(m1+1, K):
            index2 = chromosome[1][m2]

            if cat_dist.distance(data, index1, index2, chromosome[0]) == 0:
                return -INFINITY

    # Within cluster variance
    wcv = 0

    for i in range(instance_count):
        wcv += min([cat_dist.distance(data, i, chromosome[1][j], chromosome[0]) for j in range(K)])

    # Normalized wcv
    return -wcv / chromosome[0].count(1)

  
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

