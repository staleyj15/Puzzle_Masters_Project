import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score

def display_img(array):

    """
    display image
    """

    arr = array.astype(dtype="uint8")
    img = Image.fromarray(arr, "RGB")
    plt.figure()
    plt.imshow(np.asarray(img))
    

def display_cluster_imgs(pictures, labels, cluster_k):
    
    '''
    display all images in cluster k
    '''
    
    print(f'\nImages in cluster {cluster_k}:\n')
    displayIndx = np.where(labels == cluster_k)[0]
    for indx in displayIndx:
        display_img(pictures[indx])
        

def img_scatter(Z, images, k, zoom=0.5):
    
    '''
    plot node images on top of scatter plot
    '''
    
    m = len(images)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title(
        "2D Components from Isomap of Puzzle Piece Images", fontsize=16, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xsize = (max(Z[:, 0]) - min(Z[:, 0])) * 0.04
    ysize = (max(Z[:, 1]) - min(Z[:, 1])) * 0.04

    # show 30 images on the plot
    for i in range(k):
        img_num = np.random.randint(0, m)
        x0 = Z[img_num, 0] - (xsize / 2)
        y0 = Z[img_num, 1] - (ysize / 2)
        x1 = Z[img_num, 0] + (xsize / 2)
        y1 = Z[img_num, 1] + (ysize / 2)

        arr = images[img_num].astype(dtype="uint8")
        img = Image.fromarray(arr, "RGB")

        ax.imshow(
            img,
            aspect="auto",
            zorder=100000,
            extent=(x0, x1, y0, y1),
        )

    ax.scatter(Z[:, 0], Z[:, 1])
    plt.show()
    

def plot_silhouette_analysis(X, sample_silhouette_values, cluster_labels, n_clusters):
    
    '''
    plot silhouette scores for n_clusters
    '''
    
    fig, ax = plt.subplots(figsize = (7,5))
    
    # The (n_clusters +1 ) * 10 is for inserting blank space between silhouette
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    ax.set_xlim([-0.1, 1])
    y_lower = 10
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # plot aesthetics
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor = color, edgecolor = color, alpha = 0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        

    # The vertical line for average silhouette score of all the values
    silhouette_avg = silhouette_score(X, cluster_labels)
    ax.axvline(x = silhouette_avg, color = "red", linestyle = "--")
    
    # plot aesthetics
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()
    

def plot_gap_stats(gapdf, k):
    
    '''
    plot gap-statistic for different numbers of clusters
    '''
    
    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    plt.show()





