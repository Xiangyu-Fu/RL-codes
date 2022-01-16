import numpy as np
import matplotlib.pyplot as plt

# calculate Euclidean distance
def euclDistance(vector1, vector2):
     return np.sqrt(sum(np.power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
     numSamples, dim = dataSet.shape
     centroids = np.zeros((k, dim))
     for i in range(k):
          index = int(np.random.uniform(0, numSamples))
          centroids[i, :] = dataSet[index, :]
     return centroids


# k-means cluster
def kMeans(dataSet, k):
     numSamples = dataSet.shape[0]
     # first column stores which cluster this sample belongs to,
     # second column stores the error between this sample and its centroid
     clusterAssment = np.mat(np.zeros((numSamples, 2)))
     clusterChanged = True

     ## step 1: init centroids
     centroids = initCentroids(dataSet, k)

     while clusterChanged:
          clusterChanged = False
          ## for each sample
          for i in range(numSamples):
               minDist = 100000.0
               minIndex = 0
               ## for each centroid
               ## step 2: find the centroid who is closest
               for j in range(k):
                    distance = euclDistance(centroids[j, :], dataSet[i, :])
                    if distance < minDist:
                         minDist = distance
                         minIndex = j

               ## step 3: update its cluster
               if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    clusterAssment[i, :] = minIndex, minDist ** 2

          ## step 4: update centroids
          for j in range(k):
               pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
               centroids[j, :] = np.mean(pointsInCluster, axis=0)

     return centroids, clusterAssment


def main():
     # Create 2D Datapoints
     N = 50
     x = np.random.rand(N, 2) * 0.3
     y = np.random.rand(N, 2) * 0.2 + 0.5
     z = np.random.rand(N, 2) * 0.2 + [0.5, 0]
     points = np.concatenate((x, y, z))

     # Plot Points
     plt.scatter(points[:, 0], points[:, 1])
     plt.show()

     # Kmeans
     nrClasses = 3
     maxIter = 10
     centroids, clusterAssment = kMeans(points, nrClasses)
     print(centroids)


if __name__ == "__main__":
     main()
