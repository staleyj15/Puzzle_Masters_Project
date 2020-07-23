from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from deco import synchronized, concurrent
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from sklearn.cluster import KMeans
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean
import pandas as pd


# @concurrent  # We add this for the concurrent function
# def process_single_picture(pic):
    
#     '''
#     check RGB values for single picture and return boolean array indicating which pixels are background
#     '''
    
#     def check_rgb(row):
#         return not ((row >= 210).all() and (row <= 235).all())

#     bool_array = np.apply_along_axis(check_rgb, 1, pic)
#     return pic[bool_array]


# @synchronized  # And we add this for the function which calls the concurrent function
# def process_all_pics(data):
    
#     '''
#     check RGB values for all pictures to identify background pixels
#     '''
    
#     filtered_pixels = []
#     for pic in data:
#         filtered_pixels.append(process_single_picture(pic))
#     return filtered_pixels


def cluster_pixels(pixels, n_clusters):
    
    '''
    cluster pixels of each picture into n_clusters
    '''
    
    def order_clusters(labels, n_clusters):
        unique, counts = np.unique(labels, return_counts = True)
        cluster_size = dict(zip(unique, counts))
        return [k for k,v in sorted(cluster_size.items(), key = lambda x: x[1], reverse = True)]
    
    pic_centers = np.zeros((len(pixels), n_clusters, 3))
    for indx, pic in enumerate(pixels):
        km_pic = KMeans(n_clusters=n_clusters).fit(pic)
        # sort clusters by size
        order_indx = order_clusters(km_pic.labels_, n_clusters)
        pic_centers[indx, :, :] = km_pic.cluster_centers_[order_indx]
    
    return pic_centers

def cluster_pieces(array, k_start = 3, k_stop = 15):
    
    '''
    cluster pieces of puzzle
    '''
    
    km = KMeans()
    param_grid_km = {'n_clusters':np.arange(k_start, k_stop), 'algorithm':['full','elkan']}
    km_cv = GridSearchCV(km, param_grid_km, cv = 5).fit(array)
    print(f'Best parameters for KMeans model: {km_cv.best_params_}')
    km.set_params(**km_cv.best_params_).fit(array)
    return km


def silhouette_analysis(X, n_clusters):

    '''
    perform silhouette analysis to determine optimal k for clustering
    '''
    
    # Initialize the clusterer with n_clusters value and a random generator
    clusterer = KMeans(n_clusters = n_clusters, random_state = 55)
    cluster_labels = clusterer.fit_predict(X)
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {round(silhouette_avg, 6)}")
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    return sample_silhouette_values, cluster_labels


def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal