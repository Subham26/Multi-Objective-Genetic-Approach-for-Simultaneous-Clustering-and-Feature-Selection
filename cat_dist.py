# Computing distance between two data points
def distance(data, index1, index2, feature_vector):
    feature_count = len(feature_vector)

    # "cat_dist" stands for distance metric for Categorical data
    cat_dist = 0
    
    for counter in range(feature_count):
        cat_dist += (data[index1][counter] != data[index2][counter]) \
              * feature_vector[counter]
    
    # Distance normalization 
    # cat_dist = cat_dist * 1.0 / float(feature_vector.count(1))

    return cat_dist
