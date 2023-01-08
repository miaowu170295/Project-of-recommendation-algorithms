import numpy as np
import pandas as pd
import os
import time
import math


'''
the set MovieLens 100K
set URL: https://grouplens.org/datasets/movielens/100k/
'''
'''
    R:User-item corresponding co-occurrence matrix m*n
    P:user-factor matrix m*k
    Q:Item factor matrix k*n
    K:dimension of the hidden vector 
    steps:maximum number of iterations
    alpha:learning rate
    Lambda:weighting factor for L2 regularization
'''

# Decompose the matrix R into P,Q
def matrix_factorization(R, P, Q, K, steps, alpha=0.05, Lambda=0.002):
    # total time
    sum_st = 0
    # Size of previous loss
    e_old = 0
    # End-of-program marker
    flag = 1
    # Gradient descent end condition 1: maximum number of iterations satisfied
    for step in range(steps):
        # Time of the start of each fall generation
        st = time.time()
        cnt = 0
        e_new = 0
        for u in range(1, len(R)):
            for i in range(1, len(R[u])):
                if R[u][i] > 0:
                    eui = R[u][i] - np.dot(P[u, :], Q[:, i])
                    for k in range(K):
                        temp = P[u][k]
                        P[u][k] = P[u][k] + alpha * eui * Q[k][i] - Lambda * P[u][k]
                        Q[k][i] = Q[k][i] + alpha * eui * temp - Lambda * Q[k][i]
        for u in range(1, len(R)):
            for i in range(1, len(R[u])):
                if R[u][i] > 0:
                    cnt += 1
                    e_new = e_new + pow(R[u][i] - np.dot(P[u, :], Q[:, i]), 2)
        e_new = e_new / cnt
        et = time.time()
        sum_st = sum_st + (et - st)
        # Difference in loss before and after the first iteration is not executed
        if step == 0:
            e_old = e_new
            continue
        # Gradient descent end condition 2: loss too small, jump out
        if e_new < 1e-3:
            flag = 2
            break
        # Gradient descent end condition 3:
        # The difference between the front and rear loses is too small, jump out
        if (e_old - e_new) < 1e-10:
            flag = 3
            break
        else:
            e_old = e_new
    print(f'--------Summary---------\nThe type of jump out:{flag}\nTotal steps:{step+1}\nTotal time:{sum_st}\n'
          f'Average time:{sum_st / (step+1)}\nThe e is :{e_new}')
    return P, Q

# view
def view_data():
    rtnames = ['user', 'item', 'score', 'time']
    data = pd.read_csv("D:\\YSA\\MovieLens\\ml-100k\\ML100K.txt", sep='\t', header=None, names=rtnames)
    # print(data.head())
    # Number of users
    u_cnt = len(np.unique(data[rtnames[0]]))
    # item number
    i_cnt = len(np.unique(data['item']))
    # record number
    r_cnt = len(data)
    # Minimum number of items to be accessed by a user
    u_cnt_min = min(data['user'].value_counts().values)
    # Average number of items visited by users
    u_cnt_avg = r_cnt / u_cnt
    # Minimum number of visits to the project
    i_cnt_min = min(data['item'].value_counts().values)
    # Average number of visits to the project
    i_cnt_avg = r_cnt / i_cnt


# Split data integration training set, test set
def split_data(dformat):
    # Read raw data
    rating = pd.read_csv("D:\\YSA\\MovieLens\\ml-100k\\ML100K.txt", sep='\t', header=None, names=dformat)
    # Sort by chronological order
    rating.sort_values(by=['time'], axis=0, inplace=True)
    # Determine the boundary line according to the chronological value 8:2
    boundary = rating['time'].quantile(0.8)
    # Slice and dice data by time cut-off point to generate training set
    train = rating[rating['time'] < boundary]
    # Training set sorted by user, chronological order
    train.sort_values(by=['user', 'time'], axis=0, inplace=True)
    # Slice and dice data by time cut-off point to generate test sets
    test = rating[rating['time'] >= boundary]
    # Test sets sorted by user, chronological order
    test.sort_values(by=['usr', 'time'], axis=0, inplace=True)
    data = pd.concat([train, test])
    # Check if the catalogue exists
    if os.path.exists("D:\\YSA\\MovieLens\\ml-100k"):
        pass
    else:
        os.mkdir("D:\\YSA\\MovieLens\\ml-100k")
    # Write the training set and test set to a file
    train.to_csv("D:\\YSA\\MovieLens\\ml-100k\\ML100K_Train.txt", sep=',', index=False, header=None)
    test.to_csv("D:\\YSA\\MovieLens\\ml-100k\\ML100K_test.txt", sep=',', index=False, header=None)
    print(f'split data complete!')


# Get local data
def getData(path, dformat):
    # Read user-coincidence matrix data
    rating = np.loadtxt(path+"\\Basic_MF\\rating.txt", delimiter=',', dtype=float)
    # Retrieve training set data
    trainData = pd.read_csv(path+"\\ML100K_Train.txt", sep=',', header=None, names=dformat)
    # Read test set data
    testData = pd.read_csv(path+"\\ML100K_test.txt", sep=',', header=None, names=dformat)
    data = pd.concat([trainData, testData])
    # Total number of users
    all_user = np.unique(data['user'])
    # Total number of items
    all_item = np.unique(data['item'])
    return rating, trainData, testData, all_user, all_item


# Generate a user-item matrix and save it to a local file
def getUserItem(path, train, all_user, all_item):
    train.sort_values(by=['user, item'], axis=0, inplace=True)
    # Number of rows of the user-project co-occurrence matrix
    num_user = len(all_user)+1
    # Number of columns of the user-item co-occurrence matrix
    num_item = len(all_item)+1
    # Initialisation of the user-project co-occurrence matrix
    rating_mat = np.zeros([num_user, num_item], dtype=int)
    # User-project co-occurrence matrix assignment
    for i in range(len(train)):
        user = train.iloc[i]['user']
        item = train.iloc[i]['item']
        score = train.iloc[i]['score']
        rating_mat[user][item] = score
    # Determine if a folder exists
    if os.path.exists(path+"\\BasicMF"):
        pass
    else:
        os.mkdir(path+"\\BasicMF")
    # Save user-project co-occurrence matrix to file
    np.savetxt(path+"\\BasicMF\\rating.txt", rating_mat, fmt='%d', delimiter=',', newline='\n')
    print(f'generate rating matrix complete!')


# train
def train(path, rating, K, steps, matrix_factorization_BiasSVD=None):
    R = rating
    M = len(R)
    N = len(R[0])
    # User matrix initialisation
    P = np.random.normal(loc=0, scale=0.01, size=(M, K))
    # Project matrix initialisation
    Q = np.random.normal(loc=0, scale=0.01, size=(K, N))
    P, Q = matrix_factorization_BiasSVD(R, P, Q, K, steps)
    # Determine if a folder exists
    if os.path.exists(path + "\\Basic_MF"):
        pass
    else:
        os.mkdir(path + "\\Basic_MF")
    # Save P, Q to file
    np.savetxt(path+"\\Basic_MF\\userMatrix.txt", P, fmt="%.6f", delimiter=',', newline='\n')
    np.savetxt(path+"\\Basic_MF\\itemMatrix.txt", Q, fmt="%.6f", delimiter=',', newline='\n')
    print("train complete!")


# Generate topk recommendation list
def topK(dic, k):
    keys = []
    values = []
    for i in range(k):
        key, value = max(dic.items(), key=lambda x: x[1])
        keys.append(key)
        values.append(value)
        dic.pop(key)
    return keys, values


# test
def test(path, trainData, testData, all_item, k):
    # Read the user matrix
    P = np.loadtxt(path+"\\Basic_MF\\userMatrix.txt", delimiter=',', dtype=float)
    # Read the project matrix
    Q = np.loadtxt(path+"\\Basic_MF\\itemMatrix.txt", delimiter=',', dtype=float)
    # Collection of users in the test set
    testUser = np.unique(testData['user'])
    # Length of test set
    test_lenght = len(testData)

    Hits = 0
    MRR = 0
    NDCG= 0
    # Start time
    st = time.time()
    for user_i in testUser:
        # Items that have been accessed by the ith user in the test set in the training set
        visited_list = list(trainData[trainData['user'] == user_i]['item'])
        # No training data, skipped
        if len(visited_list) == 0:
            continue
        # Test set of items accessed by the ith user and de-duplicated
        test_list = list(testData[testData['user'] == user_i]['item'].drop_duplicates())
        # The items accessed by the ith user in the test set are removed from the items
        # already accessed by that user in the training set
        test_list = list(set(test_list) - set(test_list).intersection(set(visited_list)))
        # Test set ith user's access item is empty, skipped
        if len(test_list) == 0:
            continue
        # Generate a test set of items not visited by the ith user:rating pair
        poss = {}
        for item in all_item:
            if item in visited_list:
                continue
            else:
                poss[item] = np.dot(P[user_i, :], Q[:, item])
        # Recommended list of the num i user of the test set
        ranked_list, test_score = topK(poss, k)
        # List of items accessed by the i-th user of the hit test set
        h = list(set(test_list).intersection(set(ranked_list)))
        Hits += len(h)
        for item in test_list:
            for i in range(len(ranked_list)):
                if item == ranked_list[i]:
                    MRR += 1 / (i+1)
                    NDCG += 1 / (math.log2(i+1+1))
                else:
                    continue
    HR = Hits / test_lenght
    MRR /= test_lenght
    NDCG /= test_lenght
    # End Time
    et = time.time()
    print("HR@10:%.4f\nMRR@10:%.4f\nNDCG@10:%.4f\nTotal time:%.4f" % (HR, MRR, NDCG, et-st))

if __name__ == '__main__':
    rtnames = ['user', 'item', 'score', 'time']
    path = "C:\\Desktop\\test01 of project\\ml-100k"
    rating, trainData, testData, all_user, all_item = getData(path, rtnames)
    # train(path, rating, 30, 10)
    test(path, trainData, testData, all_item, 10)
