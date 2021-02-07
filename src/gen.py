#----------------------------------------------------------------------------------
import os
import argparse
import logging
import warnings; warnings.filterwarnings('ignore')
import math
#Visualisation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import metrics
#from summary_stat import dataset_summary, apps_summary, apps_power_hist

#Preprocessing
import itertools
from sklearn import preprocessing
import numpy as np
import pandas as pd
from datetime import datetime as dt
import json
#NILMTK
from nilmtk import DataSet
from nilmtk.utils import print_dict
from nilmtk.datastore import HDFDataStore
from nilmtk.electric import align_two_meters
#from summary_stat import *

#from model import AE
from SyntheticAggregateSource import SyntheticAggregateSource
######################
from datetime import datetime
now=datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
#initialise the logger
logger = logging.getLogger("Data Generator")
logger.setLevel(logging.INFO)
# create the logging file handler
fh = logging.FileHandler("logs/Data_Gen.log", 'a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add handler to logger object
logger.addHandler(fh)
logger.info("Experiment started")

allowed_key_names = ['fridge','microwave','dish_washer','kettle','washing_machine']
WINDOW_PER_BUILDING = {
    1: ("2013-04-12", "2014-12-15"),
    2: ("2013-05-22", "2013-10-03 06:16:00"),
    3: ("2013-02-27", "2013-04-01 06:15:05"),
    4: ("2013-03-09", "2013-09-24 06:15:14"),
    5: ("2014-06-29", "2014-09-01")}
parser = argparse.ArgumentParser()
#parser.add_argument('--train', help='train a model for appliance', action="store_true")
#parser.add_argument('--disag', help='disaggregate appliance consumption using previous model', action="store_true")
parser.add_argument('app', help='name of the appliance used for disaggregation')
#parser.add_argument('start_e', help='starting epoch', nargs='?', type=int, default = 0)
#parser.add_argument('end_e', help='ending  epoch', nargs='?', type=int, default = 7)

class DataGen():
    '''
    
    '''
    def __init__(self, key):   



        #slice of the appliances dict for testing
        #appliances = dict(itertools.islice(appliances.items(),2,3))
        # Preparing paths

        self.data_path = '/projects/da33/ozeidi/Project/data/UKDALE/ukdale.h5'
        self.ds = DataSet(self.data_path)

    def generate_dataset(self,key):
                # App Conf File
        filename = "conf/{}.json".format(key)
        with open(filename) as data_file:
            conf = json.load(data_file)
        self.train_buildings = conf['train_buildings']
        self.window_size = conf['lookback']
        self.appliance = conf['nilmtk_key']
        save_path = conf['save_path']
        os.makedirs(save_path, exist_ok=True)
        # validation_set = DataSet(data_path)
        sas = None #SyntheticAggregateSource(data_set= training_set)

        sp=6
        ep=2
        #k= appliances[app]['key']
        #v = appliances[k]
        #for k,v in appliances.items():
        # mains_lst = []
        # meters_lst = []
        # logger.info(v)
        # disag_filename = '{}/{}_disag-out.h5'.format(save_path,k) # The filename of the resulting datastore
        # output = HDFDataStore(disag_filename, 'w')
        # logger.info(v)
        #train_buidlings = v['train_buildings']
        
        #ae_disaggregator= AE(appliance = k , meta=v)
        self.X_train, self.Y_train = self._create_train_set()
        np.save('data/trainsets/X-{}'.format(conf['synth_key']),self.X_train)
        np.save('data/trainsets/Y-{}'.format(conf['synth_key']),self.Y_train)
            
    def _create_train_set(self):
        logger.info('creating the train set')
        train_buidlings = self.train_buildings
        all_x = np.zeros((1,self.window_size,1))
        all_y = np.zeros((1,self.window_size,1))
        X_batch = [None] *len(train_buidlings)
        Y_batch = [None]*len(train_buidlings)
        for i, building_no in enumerate(train_buidlings):
            self.ds.set_window(start=WINDOW_PER_BUILDING[building_no][0],end=WINDOW_PER_BUILDING[building_no][1])
            # This makes a list of references to the NILMTK main meters and submeter
            # no data is generated from them at this point
            mains = self.ds.buildings[building_no].elec.mains()
            meter = self.ds.buildings[building_no].elec.submeters()[self.appliance]
            logger.info('Start aligning')
            chunk = align_two_meters(meter, mains)
            chunk = pd.concat([a for a in chunk])
            chunk.fillna(0, inplace = True)
            mains_s = chunk.iloc[:,1]
            meter_s = chunk.iloc[:,0]
            logger.info('finished aligning')
            #pad the ending with zeros
            # size = chunk.shape[0]
            # additional = self.window_size - (size % self.window_size)
            # add_df = pd.DataFrame(np.zeros((additional, 2)))
            # chunk = chunk.append(add_df,  ignore_index=True)
            logger.info('Start batch generation')
            X_batch[i], Y_batch[i] = self.gen_batch(mains_s, meter_s, chunk.shape[0]-self.window_size, 0, self.window_size)
            logger.info('Finished batch generation')
            #print('Xbatch shape {}'.format(X_batch.shape))

            logger.info('building {}'.format(building_no))
        all_x = np.concatenate(X_batch)
        all_y = np.concatenate(Y_batch)
        return all_x, all_y

    def gen_batch(self, mainchunk, meterchunk, batch_size, index, window_size):
        '''Generates batches from dataset

        Parameters
        ----------
        index : the index of the batch
        '''
        w = window_size
        offset = index*batch_size
        logger.info('Start X batch generation')
        X_batch = np.array([ mainchunk[i+offset:i+offset+w]
                            for i in range(batch_size) ])
        logger.info('Finished X batch generation')
        logger.info('Start Y batch generation')
        # Y_batch =np.array( [ meterchunk[i+offset:i+offset+w]
        #                     for i in range(batch_size) ])

        Y_batch = meterchunk[w-1+offset:w-1+offset+batch_size]
        logger.info('Finished Y batch generation')
        X_batch = np.reshape(X_batch, (len(X_batch), w ,1))
        #Y_batch = np.reshape(Y_batch, (len(Y_batch), w ,1))
        return X_batch, Y_batch   


def opends(building, meter):
	'''Opens dataset of synthetic data from Neural NILM

	Parameters
	----------
	building : The integer id of the building
	meter : The string key of the meter

	Returns: np.arrays of data in the following order: main data, meter data
	'''

	path = "data/ground_truth_and_mains/"
	main_filename = "{}building_{}_mains.csv".format(path, building)
	meter_filename = "{}building_{}_{}.csv".format(path, building, meter)
	mains = np.genfromtxt(main_filename)
	meter = np.genfromtxt(meter_filename)
	mains = mains
	meter = meter
	up_limit = min(len(mains),len(meter))
	return mains[:up_limit], meter[:up_limit]

def gen_batch( mainchunk, meterchunk, batch_size, index, window_size):
        '''Generates batches from dataset

        Parameters
        ----------
        index : the index of the batch
        '''
        w = window_size
        offset = index*batch_size
        logger.info('Start X batch generation')
        X_batch = np.array([ mainchunk[i+offset:i+offset+w]
                            for i in range(batch_size) ])
        logger.info('Finished X batch generation')
        logger.info('Start Y batch generation')
        # Y_batch =np.array( [ meterchunk[i+offset:i+offset+w]
        #                     for i in range(batch_size) ])

        Y_batch = meterchunk[w-1+offset:w-1+offset+batch_size]
        logger.info('Finished Y batch generation')
        X_batch = np.reshape(X_batch, (len(X_batch), w ,1))
        #Y_batch = np.reshape(Y_batch, (len(Y_batch), w ,1))
        return X_batch, Y_batch 
if __name__ == '__main__':
    args = parser.parse_args()
    if args.app in allowed_key_names:
        DG = DataGen(args.app)  




