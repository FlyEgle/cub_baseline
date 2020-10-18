from numpy import *
import matplotlib.pyplot as plt
import operator
import time
import numpy as np
from sklearn.cluster import KMeans
import math


def loadDataSet(fileName):
    """
        load dataset from feature.txt.
        Data example:
            train/0000/0000*.jpg 0.82 0.35 -0.48 ……
    """

    dataSet = []
    dataset_total = []
    paths = []
    paths_total = []
    id_pre = ''
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            path = line.strip().split(' ')[0]
            fts = line.strip().split(' ')[1:]
            # curline = line.strip().split(' ')
            fltline = list(map(float, fts))

            id_after = path[6:10]  # load by id
            if id_after != id_pre:
                # print("id_pre:", id_pre, " id_after:", id_after)
                id_pre = id_after
                if len(dataSet):
                    dataset_total.append(dataSet)
                    dataSet = []
                    paths_total.append(paths)
                    paths = []
            dataSet.append(fltline)
            paths.append(path)

    dataset_total.append(dataSet)
    paths_total.append(paths)

    dataset_total = np.array(dataset_total)
    paths_total = np.array(paths_total)
    return paths_total, dataset_total


def getCluster(paths, clusterAssment):
    pathsInCurCluster1 = np.array(paths)[nonzero(clusterAssment[:] == 0)[0]]
    pathsInCurCluster2 = np.array(paths)[nonzero(clusterAssment[:] == 1)[0]]

    proportion1 = len(pathsInCurCluster1) / len(paths)
    proportion2 = len(pathsInCurCluster2) / len(paths)
    # print(pathsInCurCluster1.shape, pathsInCurCluster2.shape)
    if proportion1 < proportion2:
        # if proportion2 < 0.9:
        #     pick_num = math.ceil((0.9 - proportion2) * len(paths))
        #     np.random.shuffle(pathsInCurCluster1)
        #     pathsInCurCluster2 = np.append(pathsInCurCluster2, pathsInCurCluster1[:pick_num])
        return pathsInCurCluster2
    else:
        # if proportion1 < 0.9:
        #     pick_num = math.ceil((0.9 - proportion1) * len(paths))
        #     np.random.shuffle(pathsInCurCluster2)
        #     pathsInCurCluster1 = np.append(pathsInCurCluster1, pathsInCurCluster2[:pick_num])
        return pathsInCurCluster1


def main():
    paths_total, dataset_total = loadDataSet('features_efnetb5.txt')
    f = open('label_clean_efnetb5.txt', 'a')
    for i in range(len(dataset_total)):
        print(len(dataset_total),dataset_total.shape)
        dataSet = mat(dataset_total[i])  # dataset is a matrix
        paths = paths_total[i]
        y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(dataSet)
        pathsInCurCluster = getCluster(paths, y_pred)
        for path in pathsInCurCluster:
            f.write(path + '\n')
        print('the %d class is processing: %d/%d = %.2f' % (
            i, len(pathsInCurCluster), len(y_pred), len(pathsInCurCluster) / len(y_pred)))


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))
