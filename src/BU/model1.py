from __future__ import print_function, division
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
from tensorflow.python.keras.callbacks import  ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam,RMSprop
from tensorflow.python.keras import optimizers 


from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class ShortSeq2PointDisaggregator(Disaggregator):
    '''Attempt to create a RNN Disaggregator

    Attributes
    ----------
    model : keras Sequential model
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, appliance, meta):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = "Seq2Point"
        self.appliance  = appliance
        self.mmax = None
        self.MIN_CHUNK_LENGTH = meta['seq_length']
        self.window_size = meta['seq_length']
        self.mmax = meta['max_power']
        self.save_path = 'output/{}/'.format(appliance)
        self.loss = ''
        self.model = self._create_model()



    def train_across_buildings(self, mainlist, meterlist, epochs=1, batch_size=128, **load_kwargs):
        '''Train using data from multiple buildings

        Parameters
        ----------
        mainlist : a list of nilmtk.ElecMeter objects for the aggregate data of each building
        meterlist : a list of nilmtk.ElecMeter objects for the meter data of each building
        batch_size : size of batch used for training
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''
        train_gen = self.Data_Gen( mainlist, meterlist, epochs=1, batch_size=batch_size, **load_kwargs)
        print('Omar',train_gen)
        self.model.fit_generator(train_gen,epochs=epochs, steps_per_epoch=100 , verbose=1)

    def Data_Gen(self, mainlist, meterlist, epochs=1, batch_size=128, **load_kwargs):
        
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
        if self.mmax == None:
            self.mmax = max([m.max() for m in mainchunks])
        run = True
        while(run):
            # Normalize and train
            mainchunks = [self._normalize(m, self.mmax) for m in mainchunks]
            meterchunks = [self._normalize(m, self.mmax) for m in meterchunks]
            X_batch , Y_batch = self.gen_batch(mainchunks, meterchunks, batch_size)
            ix= np.random.permutation(len(X_batch)-batch_size)
            for i  in ix:
                yield (X_batch[i:i+batch_size],Y_batch[i:i+batch_size])

            # If more chunks, repeat
            try:
                for i in range(num_meters):
                    mainchunks[i] = next(mainps[i])
                    meterchunks[i] = next(meterps[i])
                    print('NEW CHUCNK IN TRAIN')
            except:
                # Get generators of timeseries
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
            
            num_of_batches[i] = int(len(ix)/batch_size) - 1  
        print('Omar Num of Batches: {}'.format(num_of_batches))
        print('OMAR SHAPES: {}'.format([m.shape for m in mainchunks]))

        X_batch = np.concatenate(mainchunks)
        
        Y_batch = np.concatenate(meterchunks)
        X_batch = X_batch.reshape(len(X_batch), self.window_size, 1)
        #Y_batch =.reshape(len(X_batch),1 , 1)
        # Shuffle data
        p = np.random.permutation(len(X_batch))
        X_batch, Y_batch = X_batch[p], Y_batch[p]
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
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

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

        pred = self.model.predict(X_batch, batch_size=128)
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
        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

    def _create_model(self):
        '''Creates and returns the ShortSeq2Point Network
        Based on: https://arxiv.org/pdf/1612.09106v3.pdf
        '''
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(30, 10, activation='relu', input_shape=(self.window_size,1), padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(30, 8, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(40, 6, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))        
        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam', )
        print(model.summary())
        #plot_model(model, to_file='model.png', show_shapes=True)

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