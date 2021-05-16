# Computing distance between two data points
def distance(point1, point2):
    feature_count = len(point1)

    # "cat_dist" stands for distance metric for Categorical data
    cat_dist = 0
    
    # Computing Hamming distance
    for counter in range(feature_count):
        cat_dist += (point1[counter] != point2[counter])

    return cat_dist
