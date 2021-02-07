
import logging
logger = logging.getLogger("Experiment.AED")
#import nilmtk.dataset_converters.dataport.download_dataport as dp
import sys
import gc
import  pandas   as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
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
# from keras.models import Sequential
# from keras.models import load_model
# from keras.utils import plot_model
# from keras.layers import Dropout,Reshape,Dense,Flatten, Conv1D

#NILMTK Modules
from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore
from nilmtk import DataSet

# this AE Class inherits the Disaggregator object from NILMTK which
# provides a common interface to all disaggregation classes.
class AE(Disaggregator):
    '''
    Proof of Concept Densoising AutoEncoder based on
    Kelly & Knottenbelt (2015) Paper
    '''
    def __init__(self,  appliance,meta):
        '''
        Initializing the model
        '''
        self.MODEL_NAME = "AE"
        self.data_path = '/projects/da33/ozeidi/Project/data/UKDALE/ukdale.h5'
        self.appliance  = appliance
        self.mmax = meta['max_power']
        self.sequence_length = meta['seq_length']
        self.MIN_CHUNK_LENGTH = self.sequence_length
        
        self.save_path = 'output/{}/'.format(appliance)
        self.loss = ''

        self.model = self._create_model(self.sequence_length)
        #self.train_set = self._create_train_set()
    def _create_model(self, sequence_len):
        '''Creates the Auto encoder model
        Below is the exact Architecture as described in the paper

        1. Input (length determined by appliance duration)
        2. 1D conv (filter size=4, stride=1, number of filters=8,
        activation function=linear, border mode=valid)
        3. Fully connected (N=(sequence length - 3) × 8,
        activation function=ReLU)
        4. Fully connected (N=128; activation function=ReLU)
        5. Fully connected (N=(sequence length - 3) × 8,
        activation function=ReLU)
        6. 1D conv (filter size=4, stride=1, number of filters=1,
        activation function=linear, border mode=valid)
        '''
        model = Sequential()

        #2. 1D Conv 
        # Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)    
        model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_len, 1), padding="same", strides=1))
        model.add(Flatten())

        #3. Fully Connected Layers
        model.add(Dropout(0.2))
        model.add(Dense((sequence_len-0)*8, activation='relu'))

        # 4. Fully connected (N=128; activation function=ReLU)
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))

        #5. Fully connected (N=(sequence length - 3) × 8,
        #   activation function=ReLU)
        model.add(Dropout(0.2))
        model.add(Dense((sequence_len-0)*8, activation='relu'))

        model.add(Dropout(0.2))

        #6. 1D conv (filter size=4, stride=1, number of filters=1,
        #   activation function=linear, border mode=valid)
        model.add(Reshape(((sequence_len-0), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

        model.compile(loss='mse', optimizer='adam')
        #Omar
        #plot_model(model, to_file='model.png', show_shapes=True)

        return model

    def _create_train_set(self):
        for building_no in train_buidlings:
            training_set.set_window(start=WINDOW_PER_BUILDING[building_no][0],end=WINDOW_PER_BUILDING[building_no][1])
            # This makes a list of references to the NILMTK main meters and submeter
            # no data is generated from them at this point
            mains_lst.append(training_set.buildings[building_no].elec.mains()) 
            meters_lst.append(training_set.buildings[building_no].elec.submeters()[k])
            logger.info('building {}'.format(building_no))
    def train_across_buildings(self, mainlist, meterlist, start_e, end_e , batch_size=128, synthetic_source =None, **load_kwargs):
        '''Train using data from multiple buildings

        Parameters
        ----------
        mainlist : a list of nilmtk.ElecMeter objects for the aggregate data of each building
        meterlist : a list of nilmtk.ElecMeter objects for the meter data of each building
        batch_size : size of batch used for training
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''
        self.start_e = start_e
        self.end_e = end_e
        self.batch_size = batch_size
        assert(len(mainlist) == len(meterlist), "Number of main and meter channels should be equal")
        num_meters = len(mainlist)

        mainps = [None] * num_meters
        meterps = [None] * num_meters
        mainchunks = [None] * num_meters
        meterchunks = [None] * num_meters

        # returns  generators of timeseries for each main meter
        for i,m in enumerate(mainlist):
            mainps[i] = m.power_series(**load_kwargs)

        # return generators of timeseries for each sub-meter
        for i,m in enumerate(meterlist):
            meterps[i] = m.power_series(**load_kwargs)


        # Get the first chunk of data from the main and submeters
        for i in range(num_meters):
            mainchunks[i] = next(mainps[i])
            meterchunks[i] = next(meterps[i])
        if self.mmax == None:
            self.mmax = max([m.max() for m in mainchunks])


        run = True
        while(run):
            logger.info('Number of points in each main chunk ={} '.format([len(m) for m in mainchunks]))
            logger.info('Number of points in each meter chunk ={} '.format([len(m) for m in meterchunks]))
            logger.info('sum Number of points in each main chunk after indexing = {}  '.format(sum([len(m) for m in mainchunks])))
            logger.info('sum Number of points in each meter chunk after indexing = {} '.format(sum([len(m) for m in meterchunks])))

            # Normalize and train
            #mainchunks = [self._normalize(m) for m in mainchunks]
            #meterchunks = [self._normalize(m) for m in meterchunks]

            self.train_across_buildings_chunk(mainchunks, meterchunks,synthetic_source)

            # If more chunks, repeat
            try:
                for i in range(num_meters):
                    mainchunks[i] = next(mainps[i])
                    meterchunks[i] = next(meterps[i])
                    
            except:
                run = False


    def train_across_buildings_chunk(self, mainchunks, meterchunks,synthetic_source):
        '''Train using only one chunk of data. This chunk consists of data from
        all buildings.

        Parameters
        ----------
        mainchunk : list of chunk of site meter
        meterchunk : list of chunk of appliance
        epochs : number of epochs for training
        batch_size : size of batch used for training
        '''
        seq_len = self.sequence_length
        num_meters = len(mainchunks)
        #batch_size = int(batch_size/num_meters)
        #num_of_batches = [None] * num_meters

        # Find common parts of timeseries
        for i in range(num_meters):
            logger.info('meter {}'.format(i))
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            mainchunks[i] = m1[ix]
            meterchunks[i] = m2[ix]
            del m1
            del m2
            gc.collect()
            assert(len(mainchunks[i]) == len(mainchunks[i]), "Number of main and meter channels should be equal")
            # pad the chunks to have multiple of window sequence length
            additional = seq_len - (len(ix)%seq_len)
            logger.info ('number of additional points: {}'.format(additional))
            mainchunks[i] = np.append(mainchunks[i], np.zeros(additional))
            meterchunks[i] = np.append(meterchunks[i], np.zeros(additional))           


        logger.info('Number of points in each main chunk after indexing = {} '.format([len(m) for m in mainchunks]))
        logger.info('Number of points in each meter chunk after indexing = {} '.format([len(m) for m in meterchunks]))

        logger.info('sum Number of points in each main chunk after indexing = {}  '.format(sum([len(m) for m in mainchunks])))
        logger.info('sum Number of points in each meter chunk after indexing = {} '.format(sum([len(m) for m in meterchunks])))
        #num_of_batches[i] = int(len(ix)/batch_size) - 1
        logger.info('Shape of mainchunks {}'.format([m.shape for m in mainchunks]))
        X_batch = np.concatenate(mainchunks)
        Y_batch = np.concatenate(meterchunks)
        X_batch, Y_batch = self.gen_batch(X_batch, Y_batch, X_batch.shape[0]-self.sequence_length, 0, self.sequence_length)
        logger.info('Shape of X_batch {}'.format(X_batch.shape))
        logger.info('Shape of Y_batch {}'.format(Y_batch.shape))     
        num_batches = int(len(X_batch) / seq_len)   
        # Reshape the batches into proper dimension to feed the network input
        #X_batch = np.reshape(X_batch, (int(len(X_batch) / seq_len), seq_len, 1))
        #Y_batch = np.reshape(Y_batch, (int(len(Y_batch) / seq_len), seq_len, 1))
        if synthetic_source:
            logger.info('Augmenting with synthetic set')
            logger.info('Shape of X_batch before augmentation {}'.format(X_batch.shape))
            logger.info('Shape of Y_batch before augmentation {}'.format(Y_batch.shape))        
            # augment the data with synthetic set--------
            X_batch_syn, Y_batch_syn = synthetic_source.gen_seq(target_appliance=self.appliance,
            seq_length=self.sequence_length, batch_size = num_batches)

           # X_batch_syn = self._normalize(X_batch_syn)
            #Y_batch_syn = self._normalize(Y_batch_syn)

            X_batch = np.concatenate([X_batch, X_batch_syn])
            Y_batch = np.concatenate([Y_batch, Y_batch_syn])

        del mainchunks
        del meterchunks
        gc.collect()
        X_batch = self._normalize(X_batch)
        Y_batch = (Y_batch/self.mmax)        
        #X_batch = np.reshape(X_batch, (int(len(X_batch) / seq_len), seq_len, 1))
        #Y_batch = np.reshape(Y_batch, (int(len(Y_batch) / seq_len), seq_len, 1))
        logger.info('Shape of X_batch {}'.format(X_batch.shape))
        logger.info('Shape of Y_batch {}'.format(Y_batch.shape))
        #for e in range(epochs): # Iterate for every epoch
        #logger.info(e)
        logger.info('Training Accross multiple house chunks')
        # self.history = self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)
        # logger.info("\n")

        #####################################################################
        #####################################################################    
        # Train model and save checkpoints
        if self.start_e>0:
            model = load_model(self.save_path+"CHECKPOINT-{}-{}epochs.hdf5".format(self.appliance, self.start_e))

        if self.end_e > self.start_e:
            filepath = self.save_path+"CHECKPOINT-"+self.appliance+"-{epoch:01d}epochs.hdf5"
            checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
            history = self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=self.end_e, shuffle=True, initial_epoch=self.start_e, callbacks=[checkpoint])
            losses = history.history['loss']

            self.model.save("{}CHECKPOINT-{}-{}epochs.hdf5".format(self.save_path, self.appliance, self.end_e),self.model)

            # Save training loss per epoch
            try:
                a = np.loadtxt("{}losses.csv".format(self.save_path))
                losses = np.append(a,losses)
            except:
                pass
            np.savetxt("{}losses.csv".format(self.save_path), losses, delimiter=",")

        
        self.loss = np.loadtxt("{}losses.csv".format(self.save_path))
        #plot the loss from the model training
        #loss.append(ae_disaggregator.history.history['loss'])
        #loss=list(itertools.chain(*loss))
        plt.plot(self.loss)
        #plt.plot(history.history['val_loss'])
        plt.title('{} model loss'.format(self.appliance))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('{}{}_loss.png'.format(self.save_path,self.appliance))

    def gen_batch(self, mainchunk, meterchunk, batch_size, index, window_size):
        '''Generates batches from dataset

        Parameters
        ----------
        index : the index of the batch
        '''
        w = window_size
        offset = index*batch_size
        X_batch = np.array([ mainchunk[i+offset:i+offset+w]
                            for i in range(batch_size) ])

        Y_batch =np.array( [ meterchunk[i+offset:i+offset+w]
                            for i in range(batch_size) ])
        X_batch = np.reshape(X_batch, (len(X_batch), w ,1))
        Y_batch = np.reshape(Y_batch, (len(Y_batch), w ,1))
        return X_batch, Y_batch   
    


    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        meter_metadata : metadata for the produced output
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
        # Omar
        # number of datapoints 
        n = 0
        # loss
        losss = 0
        for chunk in mains.power_series(**load_kwargs):
            # skip the chunk if its shorter the specified length
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk.fillna(0, inplace=True)
            chunk[:] = self._normalize(chunk.values).reshape(-1,)
            # use the model to disaggregate the nomralize chunck
            appliance_power = self.disaggregate_chunk(chunk)
            # if the model predict the appliance power to be negative then reset it to zero
            appliance_power = self._denormalize(appliance_power)
            appliance_power[appliance_power < 0] = 0
            
            # Omar
            # Calculate testing loss
            n = n + len(chunk.shape)
            loss = 0 #np.sum((appliance_power-chunk)**2)
            
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
        mains : pd.Series to disaggregate
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''

        # the belwo code checks if window length of mains is larger then sequence length
        # then reshape the main into a  MxN matrix with n= equal to seq. length
        s = self.sequence_length
        up_limit = len(mains)
        
        additional = s - (up_limit % s)
        X_batch = np.append(mains, np.zeros(additional))
        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s ,1))
        #np.reshape
        pred = self.model.predict(X_batch)
        pred = np.reshape(pred, (up_limit + additional))[:up_limit]
        column = pd.Series(pred, index=mains.index, name=0)

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

    def _normalize(self, chunk):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        # chunk = chunk.reshape(-1, 1)
        # train the standardization
        # scaler = StandardScaler()
        # scaler = scaler.fit(chunk)
        #print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
        # standardization the dataset and print the first 5 rows
        # tchunk = scaler.transform(chunk)
        tchunk = chunk / self.mmax
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
        tchunk = chunk * self.mmax
        return tchunk
  
