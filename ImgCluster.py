from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from deco import synchronized, concurrent
import numpy as np
from sklearn.cluster import KMeans


@concurrent  # We add this for the concurrent function
def process_single_picture(pic):
    
    '''
    check RGB values for single picture and return boolean array indicating which pixels are background
    '''
    
    def check_rgb(row):
        return not ((row >= 210).all() and (row <= 235).all())

    bool_array = np.apply_along_axis(check_rgb, 1, pic)
    return pic[bool_array]


@synchronized  # And we add this for the function which calls the concurrent function
def process_all_pics(data):
    
    '''
    check RGB values for all pictures to identify background pixels
    '''
    
    filtered_pixels = []
    for pic in data:
        filtered_pixels.append(process_single_picture(pic))
    return filtered_pixels


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
