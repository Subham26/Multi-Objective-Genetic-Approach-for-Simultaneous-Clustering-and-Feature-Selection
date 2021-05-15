import cat_dist


def dunn_index(Chromosome, data, instance_count):
    K = len(Chromosome[1])

    d = [[] for i in range(K)]
    labels = []
    for i in range(instance_count):
        # dK contains distances from K medoids
        dK = [cat_dist.distance(data, i, Chromosome[1][j], Chromosome[0]) for j in range(K)]
        d[dK.index(min(dK))].append(min(dK))
        labels.append(dK.index(min(dK)))

    # d2 contains distance between 2 medoids
    d2 = []
    for i in range(K):
        for j in range(i + 1, K):
            d2.append(cat_dist.distance(data,
                                        Chromosome[1][i],
                                        Chromosome[1][j],
                                        Chromosome[0]))

    avg_within_cluster_distances = [sum(d[i]) / labels.count(i) for i in range(K)]

    if max(avg_within_cluster_distances) > 0:
        return min(d2) / max(avg_within_cluster_distances)
    else:
        return -1
