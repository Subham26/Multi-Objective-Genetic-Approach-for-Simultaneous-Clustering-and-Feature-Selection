import sys
import numpy as np
import random
import math
import copy
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import time
from datetime import datetime
import objective_functions
from cluster_validation import dunn_index
import cat_dist


# Create Chromosome/Solution
def create_chromosome(instance_count, feature_count, K):
    random.seed(time.time())
    return [[random.randint(0, 1) for count1 in range(feature_count)],
            [random.randint(0, instance_count - 1) for count2 in range(K)]]


def objective_1(chromosome, data, minimum_feature_count):
    # Choromosome should have the minimum number of features

    while chromosome[0].count(1) < minimum_feature_count:
        chromosome[0][random.randint(0, len(chromosome[0]) - 1)] = 1

    return -1 * objective_functions.compute_redundancy(chromosome[0], data)


def objective_2(chromosome, data, minimum_feature_count):
    # Choromosome should have the minimum number of features

    while chromosome[0].count(1) < minimum_feature_count:
        chromosome[0][random.randint(0, len(chromosome[0]) - 1)] = 1

    return -1 * objective_functions.within_cluster_variance(chromosome, data)

  
def crossover(Q, crossover_probability, population_size, feature_count, K):
    # participating parents list
    parents = []

    for i in range(population_size):
        if random.random() <= crossover_probability:
            parents.append(i)

    random.shuffle(parents)

    j = 0

    for i in range(int(len(parents) / 2)):
        temp = copy.deepcopy(Q[parents[j + 1]])

        # Uniform Crossover
        # Feature Selection part

        mask = [random.randint(0, 1) for m in range(feature_count)]

        for m in range(feature_count):
            if mask[m] == 1:
                Q[parents[j + 1]][0][m] = Q[parents[j]][0][m]
                Q[parents[j]][0][m] = temp[0][m]

        # Random Respectful Crossover
        # Clustering part

        intersection_set = set.intersection(set(Q[parents[j]][1]), set(Q[parents[j + 1]][1]))
        intersection_set = list(intersection_set)
        remaining_set1 = set(Q[parents[j]][1]).difference(Q[parents[j + 1]][1])
        remaining_set2 = set(Q[parents[j + 1]][1]).difference(Q[parents[j]][1])
        remaining_set = list(remaining_set1) + list(remaining_set2)

        if len(intersection_set) != K:
            set1 = list(intersection_set) + random.sample(remaining_set, K - len(list(intersection_set)))
            set2 = list(intersection_set) + random.sample(remaining_set, K - len(list(intersection_set)))

            for counter in range(K):
                Q[parents[j]][1][counter] = set1[counter]
                Q[parents[j + 1]][1][counter] = set2[counter]

        j += 2


def mutation(Q, mutation_probability, population_size, instance_count, feature_count, K):
    for i in range(population_size):

        # Bit-flip mutation

        for j in range(feature_count):
            if random.random() <= mutation_probability:
                Q[i][0][j] = 1 - Q[i][0][j]

        # Random Replacement

        for j in range(K):

            if random.random() <= mutation_probability:
                rand_index = random.choice([count for count in range(instance_count)])

                while rand_index in Q[i][1]:
                    rand_index = random.choice([count for count in range(instance_count)])

                Q[i][1][j] = rand_index


def fast_non_dominated_sorting(values1, values2, values3):
    S = [[] for value in values1]
    front = [[]]
    n = [0 for value in values1]
    rank = [0 for value in values1]
    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q] and values3[p] > values3[q]) or \
                    (values1[p] >= values1[q] and values2[p] > values2[q] and values3[p] > values3[q]) or \
                    (values1[p] > values1[q] and values2[p] >= values2[q] and values3[p] > values3[q]) or \
                    (values1[p] > values1[q] and values2[p] > values2[q] and values3[p] >= values3[q]) or \
                    (values1[p] >= values1[q] and values2[p] >= values2[q] and values3[p] > values3[q]) or \
                    (values1[p] >= values1[q] and values2[p] > values2[q] and values3[p] >= values3[q]) or \
                    (values1[p] > values1[q] and values2[p] >= values2[q] and values3[p] >= values3[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p] and values3[q] > values3[p]) or \
                    (values1[q] >= values1[p] and values2[q] > values2[p] and values3[q] > values3[p]) or \
                    (values1[q] > values1[p] and values2[q] >= values2[p] and values3[q] > values3[p]) or \
                    (values1[q] > values1[p] and values2[q] > values2[p] and values3[q] >= values3[p]) or \
                    (values1[q] >= values1[p] and values2[q] >= values2[p] and values3[q] > values3[p]) or \
                    (values1[q] >= values1[p] and values2[q] > values2[p] and values3[q] >= values3[p]) or \
                    (values1[q] > values1[p] and values2[q] >= values2[p] and values3[q] >= values3[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i] != []:
        next_front = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in next_front:
                        next_front.append(q)
        i = i + 1
        front.append(next_front)

    del front[len(front) - 1]
    return front


# Function to find index of list
def index_of(a, list2):
    for i in range(0, len(list2)):
        if list2[i] == a:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = 9999999999999
    return sorted_list


# Function to calculate crowding distance
def compute_crowding_distance(values1, values2, values3, front):
    distance = [0 for i in range(0, len(front))]

    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    sorted3 = sort_by_values(front, values3[:])

    distance[front.index(sorted1[0])] = math.inf
    distance[front.index(sorted2[0])] = math.inf
    distance[front.index(sorted3[0])] = math.inf

    distance[front.index(sorted1[len(front) - 1])] = math.inf
    distance[front.index(sorted2[len(front) - 1])] = math.inf
    distance[front.index(sorted3[len(front) - 1])] = math.inf

    if max(values1) - min(values1) != 0:
        for k in range(1, len(front) - 1):
            distance[front.index(sorted1[k])] = distance[front.index(sorted1[k])] + (
                    values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (
                                                        max(values1) - min(values1))
    if max(values2) - min(values2) != 0:
        for k in range(1, len(front) - 1):
            distance[front.index(sorted2[k])] = distance[front.index(sorted2[k])] + (
                    values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                                                        max(values2) - min(values2))
    if max(values3) - min(values3) != 0:
        for k in range(1, len(front) - 1):
            distance[front.index(sorted3[k])] = distance[front.index(sorted3[k])] + (
                    values2[sorted3[k + 1]] - values2[sorted3[k - 1]]) / (
                                                        max(values3) - min(values3))

    return distance


def nsga2(data, K):
    instance_count, feature_count = data.shape
    minimum_feature_count = 3
    population_size = 100
    number_of_generation = 50
    crossover_probability = 0.85
    mutation_probability = 0.001

    # Create initial population

    P0 = [create_chromosome(instance_count, feature_count, K) for i in range(population_size)]
    objective_1_values = []
    objective_2_values = []
    objective_3_values = []
    for chromosome in P0:
        objective_1_values.append(objective_1(chromosome, data, minimum_feature_count))
        objective_2_values.append(objective_2(chromosome, data, minimum_feature_count))
        objective_3_values.append(dunn_index(chromosome, data, instance_count))

    FRONTS = fast_non_dominated_sorting(objective_1_values, objective_2_values, objective_3_values)
    rank_based_fitness = [0 for i in range(population_size * 2)]
    j = 0
    for FRONT in FRONTS:
        for index in FRONT:
            rank_based_fitness[index] = j
        j = j + 1

    # Binary tournament with replacement
    Q0 = []
    for index in range(population_size):
        index1 = random.randint(0, population_size - 1)
        index2 = random.randint(0, population_size - 1)
        if rank_based_fitness[index1] <= rank_based_fitness[index2]:
            Q0.append(copy.deepcopy(P0[index1]))
        else:
            Q0.append(copy.deepcopy(P0[index2]))

    crossover(Q0, crossover_probability, population_size, feature_count, K)

    mutation(Q0, mutation_probability, population_size, instance_count, feature_count, K)

    # From the first generation onward, the procedure is different.
    # The elitism mechanism for generation >= 1 is shown below.

    R0 = P0 + Q0

    objective_1_values = []
    objective_2_values = []
    objective_3_values = []
    for chromosome in R0:
        objective_1_values.append(objective_1(chromosome, data, minimum_feature_count))
        objective_2_values.append(objective_2(chromosome, data, minimum_feature_count))
        objective_3_values.append(dunn_index(chromosome, data, instance_count))

    FRONTS = fast_non_dominated_sorting(objective_1_values, objective_2_values, objective_3_values)

    # Create next generations
    for g in range(number_of_generation):

        if g % 10 == 0:
            print("GENERATION IS AT {}".format(g))

        # Parent Population
        P1 = []

        crowding_distance = []

        j = 0

        while len(P1) < population_size:
            FRONT = FRONTS[j]
            crowding_distance.append(
                compute_crowding_distance(objective_1_values, objective_2_values, objective_3_values,
                                          FRONT))
            indices = sorted(range(len(FRONT)), key=lambda u: crowding_distance[j][u], reverse=True)
            FRONT2 = [FRONT[i] for i in indices]
            for i in FRONT2:
                P1.append(R0[i])
            j = j + 1

        P1 = P1[:population_size]

        # Create a Child Population, Q1, from P1
        Q1 = []
        # Binary tournament with replacement
        fitness = [0 for i in range(population_size)]

        for i in range(population_size):
            fitness[i] = random.random()
        for i in range(population_size):
            index1 = random.randint(0, population_size - 1)
            index2 = random.randint(0, population_size - 1)
            if fitness[index1] <= fitness[index2]:
                Q1.append(copy.deepcopy(P1[index1]))
            else:
                Q1.append(copy.deepcopy(P1[index2]))

        crossover(Q1, crossover_probability, population_size, feature_count, K)

        mutation(Q1, mutation_probability, population_size, instance_count, feature_count, K)

        P0 = copy.deepcopy(P1)
        Q0 = copy.deepcopy(Q1)

        R0 = P0 + Q0

        objective_1_values = []
        objective_2_values = []
        objective_3_values = []
        for chromosome in R0:
            objective_1_values.append(objective_1(chromosome, data, minimum_feature_count))
            objective_2_values.append(objective_2(chromosome, data, minimum_feature_count))
            objective_3_values.append(dunn_index(chromosome, data, instance_count))

        FRONTS = fast_non_dominated_sorting(objective_1_values, objective_2_values, objective_3_values)

    best_front = FRONTS[0]

    # contains chormosomes of front-1
    front1_Chromosomes = [R0[i] for i in best_front]

    # Unique non-dominated front1_Chromosomes
    n = len(front1_Chromosomes)

    # Sort the medoid indices
    for i in range(n):
        front1_Chromosomes[i][1].sort()

    marked = [0 for i in range(n)]
    unique_non_dominated = []
    for i in range(n):
        Chromosome1 = front1_Chromosomes[i]
        if marked[i] == 1:
            continue
        else:
            marked[i] = 1
            unique_non_dominated.append(Chromosome1)
        for j in range(i + 1, n):
            Chromosome2 = front1_Chromosomes[j]
            flag = 0
            for k in range(feature_count):
                if Chromosome1[0][k] != Chromosome2[0][k]:
                    flag = 1
                    break

            if flag == 0:
                marked[j] = 1

    print("UNIQUE NON-DOMINATED = {}".format(len(unique_non_dominated)))

    for Chromosome in unique_non_dominated:
        print(Chromosome)
        print(dunn_index(Chromosome, data, instance_count))
