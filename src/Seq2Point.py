
import logging
logger = logging.getLogger("Experiment.AED")
#import nilmtk.dataset_converters.dataport.download_dataport as dp
import sys
import gc
import  pandas   as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.externals import joblib
from sklearn.cluster import KMeans


import random
# To hanle HDF Data files
import h5py
 #Visualisation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Keras Modules
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dropout,Reshape,Dense,Flatten, Conv1D
from tensorflow.python.keras.callbacks import  ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam,RMSprop
from tensorflow.python.keras import optimizers 

#NILMTK Modules
from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore
from nilmtk import DataSet
from nilmtk.electric import align_two_meters

# Imported from src
from quantize import cluster_series, lut
from gen import  opends, gen_batch
import metrics

# Windows based on Neural NILM Paper
WINDOW_PER_BUILDING = {
    #4
    1: ("2013-04-12", "2013-12-15"),
    2: ("2013-05-22", "2013-10-03 06:16:00"),
    3: ("2013-02-27", "2013-04-01 06:15:05"),
    4: ("2013-03-09", "2013-09-24 06:15:14"),
    5: ("2014-06-29", "2014-09-01")
}
# this AE Class inherits the Disaggregator object from NILMTK which
# provides a common interface to all disaggregation classes.
class AE():
    '''
    Proof of Concept Densoising AutoEncoder based on
    Kelly & Knottenbelt (2015) Paper
    '''
    def __init__(self,  appliance, conf):
        '''
        Initializing the model
        '''
        self.MODEL_NAME = "AE"
        #self.data_path = '/projects/da33/ozeidi/Project/data/UKDALE/ukdale.h5'

        self.loss = None
        self.appliance = appliance
        self.input_window = conf['lookback']
        self.mamax = 5000
        self.memax = conf['memax']
        self.mean = conf['mean']
        self.std = conf['std']
        self.on_threshold = conf['on_threshold']
        self.train_buildings = conf['train_buildings']
        self.test_building = conf['test_building']
        self.meter_key = conf['nilmtk_key']
        self.save_path = 'output/{}/'.format(appliance)
        self.state_lut = None
        self.start_e= None
        self.end_e = None

    def _create_model(self, input_window, output_nodes):
        '''
        Creates and returns the ShortSeq2Point Network
        Based on: https://arxiv.org/pdf/1612.09106v3.pdf

        '''
        do=0.5
        model = Sequential()
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
        #RMSprop(clipvalue=0.5)
        #optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) #RMSprop(clipvalue=0.5)
        # ZHANG ARCHITECTURE
        # 1D Conv
        model.add(Conv1D(30, 10, activation='relu', input_shape=(input_window,1), padding="same", strides=1))
        model.add(Dropout(do))
        model.add(Conv1D(30, 8, activation='relu', padding="same", strides=1))
        model.add(Dropout(do))
        model.add(Conv1D(40, 6, activation='relu', padding="same", strides=1))
        model.add(Dropout(do))
        model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
        model.add(Dropout(do))
        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(do))

        model.add(Dense(output_nodes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt)

        # model.compile(loss='mse', optimizer='adam')
        model.summary()
        #plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def train(self, start_e, end_e):
        self.start_e = start_e
        self.end_e = end_e
        logger.info("Training for device: {}".format(self.appliance))
        logger.info("    train_buildings: {}".format(self.train_buildings))

        # Open train sets
        X_train = np.load("data/trainsets/X-{}.npy".format(self.appliance))
        X_train = np.nan_to_num(X_train)
        self.mamax = X_train.max()
        assert(np.count_nonzero(np.isnan(X_train)) == 0, " Array must not include Nan " )
        logger.info('Max value in X_train ={} '.format(np.max(X_train)))
        X_train = self._normalize(X_train)
        logger.info('Max value in X_train ={} '.format(np.max(X_train)))


        Y_train = np.load("data/trainsets/Y-{}.npy".format(self.appliance))
        assert(np.count_nonzero(np.isnan(Y_train)) == 0, " Array must not include Nan " )
        self.memax = Y_train.max()

        #X_train = X_train[:50000]
        #Y_train = Y_train[:50000]	
        logger.info ('Omar The shape for X_train ={} '.format(X_train.shape))
        logger.info ('Omar The shape for Y_train ={} '.format(Y_train.shape))
        #Y_train = normalize(Y_train, memax, mean, std)
        #transfom the Y_train into a hot vector for appliance states 
        clf = joblib.load('conf/{}_clf.pkl'.format(self.appliance))
        self.state_lut = lut(clf)
        Y_train= cluster_series(clf, Y_train,self.on_threshold)
        logger.info ('Omar The shape for Y_train after clustering={} '.format(Y_train.shape))
        logger.info ('Omar The max value of Y_train ={} '.format(Y_train.max))
        

        #model = create_model(input_window,output_nodes=Y_train.shape[1])
        self.model = self._create_model(input_window = self.input_window,output_nodes=Y_train.shape[1])
        # Train model and save checkpoints
        logger.info('Start of the training ')
        if start_e>0:
            self.import_model(self.save_path+"CHECKPOINT-{}-{}epochs.hdf5".format(self.appliance, start_e))

        if end_e > start_e:
            filepath = self.save_path+"CHECKPOINT-"+self.appliance+"-{epoch:01d}epochs.hdf5"
            checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
            history = self.model.fit(X_train, Y_train, batch_size=128, epochs=end_e, shuffle=True, initial_epoch=start_e, callbacks=[checkpoint])
            self.losses = history.history['loss']

            self.model.save("{}CHECKPOINT-{}-{}epochs.hdf5".format(self.save_path, self.appliance, end_e),self.model)

            #------Plotting loss------------
            fig, (ax1) = plt.subplots(1)
            fig.set_size_inches(7, 4)
            ax1.plot(self.losses, label='Mains',color='red')
            fig.legend(loc='best', fancybox=True, shadow=True, ncol=1)
            ax1.set_title('Loss Plot')
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Epoch')
            fig.savefig('logs/{}_loss.png'.format(self.meter_key))
            # Save training loss per epoch
            try:
                a = np.loadtxt("{}losses.csv".format(self.save_path))
                self.losses = np.append(a,self.losses)
            except:
                pass
            np.savetxt("{}losses.csv".format(self.save_path), self.losses, delimiter=",")
        logger.info('finished training')


        
        
    def disaggregate(self):
        '''
        
        '''
       	# ======= Disaggregation phase ==============
        logger.info('disaggregation phase')
        mains, meter = opends(self.test_building, self.meter_key)
        X_test = self._normalize(mains)
        y_test = meter

        # Predict data
        X_batch, Y_batch = gen_batch(X_test, y_test, len(X_test)-self.input_window, 0,self. input_window)
        pred = self.model.predict(X_batch)
        pred = np.array([ np.argmax(p) for p in pred])
        pred = np.array([self.state_lut[p] for p in pred])
        logger.info('Sum of pred: {} '.format(np.sum(pred)))
        #pred = denormalize(pred, memax, mean, std)
        #pred[pred<0] = 0
        #pred = np.transpose(pred)[0]
        # Save results
        np.save("{}pred-{}-epochs{}".format(self.save_path, self.meter_key, self.end_e), pred)

        rpaf = metrics.recall_precision_accuracy_f1(pred, Y_batch, self.on_threshold)
        rete = metrics.relative_error_total_energy(pred, Y_batch)
        mae = metrics.mean_absolute_error(pred, Y_batch)

        logger.info("============ Recall: {}".format(rpaf[0]))
        logger.info("============ Precision: {}".format(rpaf[1]))
        logger.info("============ Accuracy: {}".format(rpaf[2]))
        logger.info("============ F1 Score: {}".format(rpaf[3]))

        logger.info("============ Relative error in total energy: {}".format(rete))
        logger.info("============ Mean absolute error(in Watts): {}".format(mae))

        res_out = open("{}results-pred-{}-{}epochs".format(self.save_path, self.meter_key, self.end_e), 'w')
        for r in rpaf:
            res_out.write(str(r))
            res_out.write(',')
        res_out.write(str(rete))
        res_out.write(',')
        res_out.write(str(mae))
        res_out.close()


    def import_model(self, filename):
        '''Loads keras model from h5
        
        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''

        self.model = load_model(filename)
        # with h5py.File(filename, 'a') as hf:
        #     ds = hf.get('disaggregator-data').get('mmax')
        #     self.mmax = np.array(ds)[0]

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.model.save(filename)
        # with h5py.File(filename, 'a') as hf:
        #     gr = hf.create_group('disaggregator-data')
        #     gr.create_dataset('mmax', data = [self.mmax])

    def _normalize(self, chunk):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        #logger.info('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))

        tchunk = chunk / self.mamax
        logger.info('Normalized Chunk Shape: {}'.format(tchunk.shape))
        return tchunk

    def _denormalize(self, chunk):


        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * self.mamax
        return tchunk
  
