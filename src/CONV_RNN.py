
import logging
logger = logging.getLogger("Experiment.Model")

#from __future__ import print_function, division
from warnings import warn, filterwarnings

from matplotlib import rcParams
import matplotlib.pyplot as plt

import random
import sys
import pandas as pd
import numpy as np
import h5py
import os

# Keras Modules
#import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout, Flatten
from tensorflow.python.keras.callbacks import  ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import Adam,RMSprop
from tensorflow.python.keras import optimizers , metrics 
from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class CONV_RNN(Disaggregator):
    '''Attempt to create a RNN Disaggregator

    Attributes
    ----------
    model : keras Sequential model
    mamax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, appliance,MODEL_NAME, meta):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = MODEL_NAME        
        self.appliance  = appliance
        self.MIN_CHUNK_LENGTH = meta['seq_length']
        self.window_size = meta['seq_length']
        self.mamax = None
        self.memax = meta['max_power']
        self.mean = meta['mean_power'] 
        self.std = meta['std_power']
        self.save_path = 'output/{}/{}/'.format(self.MODEL_NAME,appliance)
        self.loss = ''
        self.model = self._create_model()
        self.size=0


    def train_across_buildings(self, mainlist, meterlist, start_e=1, end_e =5, batch_size=128, **load_kwargs):
        """Train using data from multiple buildings
        
        Arguments:
            mainlist {list} -- a list of nilmtk.ElecMeter objects for the aggregate data of each building
            meterlist {list} -- a list of nilmtk.ElecMeter objects for the meter data of each building
            **load_kwargs : keyword arguments passed to `meter.power_series()`
        Keyword Arguments:
            start_e {int} -- starting epoch (default: {1})
            end_e {int} -- end epoch (default: {5})
            batch_size {int} -- size of batch used for training (default: {128})
        """
        self.batch_size = batch_size
        train_gen = self.Data_Gen( mainlist, meterlist, batch_size=batch_size, **load_kwargs)
        next(train_gen)

        spe= self.size//(batch_size*500)
        print('Max Main value',self.mamax)
        print('Omar, Total Size of g= {}, spe = {}'.format(self.size, spe))
        #self.model.fit_generator(train_gen,epochs=epochs, steps_per_epoch=(self.size//batch_size**2) , verbose=1)

        # Train model and save checkpoints
        logger.info('Start of the training ')
        if start_e == 0:
            clean( self.save_path)
        if start_e>0:
            self.import_model(self.save_path+"CHECKPOINT-{}-{}epochs.h5".format(self.appliance, start_e))

        if end_e > start_e:
            filepath = self.save_path+"CHECKPOINT-"+self.appliance+"-{epoch:01d}epochs.h5"
            checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
            tb = TensorBoard(log_dir='logs/Graph', histogram_freq=0, write_graph=True, write_images=True)
            # model training
            history = self.model.fit_generator(train_gen, initial_epoch = start_e, epochs=end_e,
             steps_per_epoch=spe , verbose=1,  callbacks=[checkpoint,tb])
            self.losses = history.history['loss']
            self.model.save("{}CHECKPOINT-{}-{}epochs.h5".format(self.save_path, self.appliance, end_e),self.model)
            self.plot_loss()
    def plot_loss(self):
        #------Plotting loss------------
        fig, (ax1) = plt.subplots(1)
        fig.set_size_inches(7, 4)
        ax1.plot(self.losses, label='loss',color='red')
        fig.legend(loc='best', fancybox=True, shadow=True, ncol=1)
        ax1.set_title('Loss Plot')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        print('{}{}_loss.png'.format(self.save_path,self.appliance))
        fig.savefig('{}{}_loss.png'.format(self.save_path,self.appliance))


    def Data_Gen(self, mainlist, meterlist, batch_size=128, **load_kwargs):
        logger.info('Data Generator')
        assert len(mainlist) == len(meterlist), "Number of main and meter channels should be equal"
        num_meters = len(mainlist)

        mainps = [None] * num_meters
        meterps = [None] * num_meters
        mainchunks = [None] * num_meters
        meterchunks = [None] * num_meters

        # Get generators of timeseries
        for i,m in enumerate(mainlist):
            mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(meterlist):
            meterps[i] = m.power_series(**load_kwargs)

        # Get a chunk of data
        for i in range(num_meters):
            mainchunks[i] = next(mainps[i])
            meterchunks[i] = next(meterps[i])
        if self.mamax == None:
            self.mamax = max([m.max(skipna=True) for m in mainchunks])
        run = True
        while(run):
            # Normalize and train
            
            mainchunks = [self._normalize(m, self.mamax,) for m in mainchunks]
            meterchunks = [self._normalize(m, self.memax) for m in meterchunks]
            X_batch , Y_batch = self.gen_batch(mainchunks, meterchunks, batch_size)

            ix= range(0,len(X_batch)-batch_size, batch_size)
            ix = np.random.permutation(ix)
            for i  in ix:
                yield (X_batch[i:i+batch_size],Y_batch[i:i+batch_size])
            try:
                for i in range(num_meters):
                    mainchunks[i] = next(mainps[i])
                    meterchunks[i] = next(meterps[i])
                    print('NEW CHUCNK IN TRAIN')
            except:
                pass
                # # Get generators of timeseries
                for i,m in enumerate(mainlist):
                    mainps[i] = m.power_series(**load_kwargs)

                for i,m in enumerate(meterlist):
                    meterps[i] = m.power_series(**load_kwargs)

                # Get a chunk of data
                for i in range(num_meters):
                    mainchunks[i] = next(mainps[i])
                    meterchunks[i] = next(meterps[i])
                #run = False
    def gen_batch(self, mainchunks, meterchunks, batch_size):
        '''Train using only one chunk of data. This chunk consists of data from
        all buildings.

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        batch_size : size of batch used for training
        '''
        num_meters = len(mainchunks)
        #batch_size = int(batch_size/num_meters)
        num_of_batches = [None] * num_meters

        # Find common parts of timeseries
        for i in range(num_meters):
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            print(mainchunks[i].shape)
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            #print('OMAR {}'.format(ix))
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            mainchunks[i] = m1[ix]
            meterchunks[i] = m2[ix]

            indexer = np.arange(self.window_size)[None, :] + np.arange(len(mainchunks[i].values)-self.window_size+1)[:, None]
            mainchunks[i] = mainchunks[i].values[indexer]
            meterchunks[i] = meterchunks[i].values[self.window_size-1:]
            
            #num_of_batches[i] = int(len(ix)/batch_size) - 1  
        #print('Omar Num of Batches: {}'.format(num_of_batches))
        print('OMAR SHAPES: {}'.format([m.shape for m in mainchunks]))

        X_batch = np.concatenate(mainchunks)
        
        Y_batch = np.concatenate(meterchunks)
        X_batch = X_batch.reshape(len(X_batch), self.window_size, 1)
        #Y_batch =.reshape(len(X_batch),1 , 1)
        print('Omar X_batch:{}, Y_batch:{}'.format(X_batch.shape, Y_batch.shape))
        # Shuffle data
        p = np.random.permutation(len(X_batch))
        X_batch, Y_batch = X_batch[p], Y_batch[p]
        self.size = X_batch.shape[0]
        return (X_batch, Y_batch)
        # Train model
        #self.model.fit(X_batch, Y_batch, batch_size= batch_size,epochs= 1,shuffle=True)
        #print("\n")
    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : a nilmtk.ElecMeter of aggregate data
        meter_metadata: a nilmtk.ElecMeter of the observed meter used for storing the metadata
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mamax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power = self._denormalize(appliance_power, self.memax)
            appliance_power[appliance_power < 0] = 0
            

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series of aggregate data
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        X_batch = np.array(mains)
        Y_len = len(X_batch)
        indexer = np.arange(self.window_size)[None, :] + np.arange(len(X_batch)-self.window_size+1)[:, None]
        X_batch = X_batch[indexer]
        X_batch = np.reshape(X_batch, (X_batch.shape[0],X_batch.shape[1],1))

        pred = self.model.predict(X_batch, batch_size=self.batch_size,  verbose=1)
        pred = np.reshape(pred, (len(pred)))
        column = pd.Series(pred, index=mains.index[self.window_size-1:Y_len], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    def import_model(self, filename):
        '''Loads keras model from h5

        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''
        print(filename)
        self.model = load_model(filename)
        # with h5py.File(filename, 'a') as hf:
        #     ds = hf.get('disaggregator-data').get('mamax')
        #     self.mamax = np.array(ds)[0]

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        
        self.model.save(filename)
        # with h5py.File(filename, 'a') as hf:
        #     gr = hf.create_group('disaggregator-data')
        #     gr.create_dataset('mamax', data = [self.mamax])

    def _normalize(self, chunk, mamax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = (chunk -self.mean) / self.std #mamax
        return tchunk

    def _denormalize(self, chunk, mamax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = (chunk*self.std)+self.mean #* mamax
        return tchunk

    def _create_model(self):
        '''Creates and returns the ShortSeq2Point Network
        Based on: https://arxiv.org/pdf/1612.09106v3.pdf
        '''
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(16, 4, activation='relu', input_shape=(self.window_size,1), padding="same", strides=1))

        #Bi-directional GRUs
        model.add(Bidirectional(GRU(64, activation='relu', return_sequences=True), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Bidirectional(GRU(128, activation='relu', return_sequences=False), merge_mode='concat'))
        model.add(Dropout(0.5))

        # Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        print(model.summary())
        return model

def clean(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)