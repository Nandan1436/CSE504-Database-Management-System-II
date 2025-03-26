import numpy as np
import matplotlib.pyplot as plt
import cv2
#from utils import *

def load_data():
    img = cv2.imread('./image/test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def find_closest_centroids(X, centroids):
    # m = X.shape[0]
    # k = centroids.shape[0]

    # idx = np.zeros(m,dtype=int)

    # for i in range(m):
    #     dist = []
    #     for j in range(k):
    #         norm_ij = np.linalg.norm(X[i] - centroids[j])
    #         dist.append(norm_ij)
    #     idx[i] = np.argmin(dist)

    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    idx = np.argmin(distances, axis=1)

    return idx

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k,n))

    for i in range(k):
        points = X[np.where(idx==i)]
        if points.shape[0]>0:
            centroids[i] = np.mean(points, axis=0)
    
    return centroids

def kMeans_init_centroids(X,k):
    randidx = np.random.permutation(X.shape[0])

    centroids = X[randidx[:k]]

    return centroids

def run_KMeans(X, initial_centroids, max_iters=10):

    m, n = X.shape
    k = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m, dtype=int)

    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))

        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    
    return centroids, idx

def main():
    img = load_data()

    X = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    k = 10
    initial_centroids = kMeans_init_centroids(X, k)
    centroids, idx = run_KMeans(X, initial_centroids, max_iters=10)

    X_recovered = centroids[idx.astype(int), :]
    X_recovered = np.reshape(X_recovered,img.shape)

    X_recovered = np.clip(X_recovered, 0, 255).astype(np.uint8)
    X_recovered_bgr = cv2.cvtColor(X_recovered, cv2.COLOR_RGB2BGR)

    cv2.imshow("K-Means Clustered Image", X_recovered_bgr)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()  
if __name__ == "__main__":
    main()