import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from tensorflow.python.keras import utils as np_utils

def rearrange(cluster_centers, labels):
    centroids = cluster_centers.reshape(1,-1)[0]
    ix = np.argsort(centroids)
    #print(ix)
    lut= np.zeros_like(ix)
    lut[ix]=np.arange(1,len(centroids)+1)
    return (lut[labels])
#rearrang(clusterer.cluster_centers_,clusterer.labels_)


def cluster_series(clusterer, meter_series, on_threshold):
    meter_series = meter_series.reshape(-1,1)
    ix = meter_series >=on_threshold
    cluster_labels = clusterer.predict(meter_series[ix].reshape(-1,1))
    
    cluster_labels = rearrange(clusterer.cluster_centers_, cluster_labels )
    print(set(cluster_labels))
    meter_series[ix]= cluster_labels

    meter_series[ix==False] = 0
    meter_series = np_utils.to_categorical(meter_series)
    return meter_series
def lut(clusterer):
    c= np.sort(clusterer.cluster_centers_)
    c = np.append(np.zeros(1),c)
    return c
# if __name__ == '__main__':
#     app='dish_washer'
#     clf =joblib.load('ShortSeq2Point/appconf/{}_clf.pkl'.format(app))
#     y_train = np.load("ShortSeq2Point/dataset/trainsets/Y-{}.npy".format(app))
#     res = cluster_series(clf, y_train,10)
#     print(res)