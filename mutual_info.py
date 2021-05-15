def compute_mutual_info(vector1, vector2):

    # vector1 and vector2 are feature vectors
    dimension1 = len(set(vector1))
    dimension2 = len(set(vector2))

    joint_probability_distribution = []

    # Initialize Joint Probability Distribution table
    for counter1 in range(dimension1):
        joint_probability_distribution.append([])
        for counter2 in range(dimension2):
            joint_probability_distribution[counter1].append(0.00)

    # Compute joint probability

    for counter in range(len(vector1)):
        joint_probability_distribution[vector1[counter]][vector2[counter]] += 1

    for counter1 in range(dimension1):
        for counter2 in range(dimension2):
            joint_probability_distribution[counter1][counter2] /= (len(vector1))

    # Calculate marginal probability
    marginal_probability1 = []
    marginal_probability2 = []
    for counter1 in range(dimension1):
        marginal_probability1.append(sum(joint_probability_distribution[counter1]))
    for counter2 in range(dimension2):
        marginal_probability2.append(
            sum([joint_probability_distribution[counter1][counter2] for counter1 in range(dimension1)]))

    # Calculate Mutual Information
    mutual_info = 0.00
    
    for counter1 in range(dimension1):
        for counter2 in range(dimension2):
            if joint_probability_distribution[counter1][counter2] != 0 and marginal_probability1[counter1] != 0 and \
                    marginal_probability2[counter2] != 0:
                mutual_info = mutual_info + \
                              joint_probability_distribution[counter1][counter2] * \
                              math.log((joint_probability_distribution[counter1][counter2] / (
                                          marginal_probability1[counter1] * marginal_probability2[counter2])), 2)

    return mutual_info
