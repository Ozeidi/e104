
#----------------------------------------------------------------------------------
import os
import argparse
import logging
import warnings; warnings.filterwarnings('ignore')
import math
import json
import importlib
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

#NILMTK
from nilmtk import DataSet
from nilmtk.utils import print_dict
from nilmtk.datastore import HDFDataStore
from nilmtk.electric import align_two_meters
#from summary_stat import *

# from S2P import S2P 
# from conv_rnn_model import CONV_RNN
from SyntheticAggregateSource import SyntheticAggregateSource

#Windows based on Neural NILM Paper
# WINDOW_PER_BUILDING = {
#     1: ("2013-04-12", "2014-12-15"),
#     2: ("2013-05-22", "2013-10-03 06:16:00"),
#     3: ("2013-02-27", "2013-04-01 06:15:05"),
#     4: ("2013-03-09", "2013-09-24 06:15:14"),
#     5: ("2014-06-29", "2014-09-01")
# }
WINDOW_PER_BUILDING = {
    1: ("2013-04-12", "2013-06-15"),
    2: ("2013-05-22", "2013-10-03 06:16:00"),
    3: ("2013-02-27", "2013-03-27 06:15:05"),
    4: ("2013-03-09", "2013-04-24 06:15:14"),
    5: ("2014-06-29", "2014-09-01")
}
# Applicance details
#appliances = json.load(open('conf/apps.json'))
# appliances = {
#                 'washing_machine':{'key':'washing machine',
#                 'seq_length':1024,
#                 'train_buildings' : [1, 5],
#                 'validation_buildings' : [2],
#                 'max_power':2500,
#                 'on_power_threshould': 20,
#                 'min_on_duration': 1800,
#                 'min_off_duration': 160, 
#                 'appliances':['washe dryer', 'washing machine']},

#                 'fridge':{'key':'fridge',
#                 'seq_length' : 512,
#                 'train_buildings' : [1, 4], #2,
#                 'validation_buildings' : [5],
#                 'max_power':300,
#                 'on_power_threshould': 50,
#                 'min_on_duration': 60,
#                 'min_off_duration': 12, 
#                 'appliances':['fridge freezer', 'fridge', 'freezer']},

#                 'kettle':{'key':'kettle',
#                 'seq_length':100,
#                 'train_buildings' : [1,3,4],# 4,5],
#                 'validation_buildings' : [2],
#                 'max_power':3100,
#                 'on_power_threshould': 2000,
#                 'min_on_duration': 12,
#                 'min_off_duration': 0, 
#                 'appliances':['kettle']},
#                 #seq_length':1024 + 512
#                 'dish_washer':{'key':'dish washer',
#                 'seq_length':1024+512,
#                 'train_buildings' : [1, 2],
#                 'validation_buildings' : [5],
#                 'max_power':2500,
#                 'on_power_threshould': 10,
#                 'min_on_duration': 1800,
#                 'min_off_duration': 1800, 
#                 'appliances':['dish washer']},


#                 'microwave':{'key':'microwave',
#                 'seq_length':288,
#                 'train_buildings' : [1, 2],
#                 'validation_buildings' : [5],
#                 'max_power':3000,
#                 'on_power_threshould': 200,
#                 'min_on_duration': 12,
#                 'min_off_duration': 30, 
#                 'appliances':['microwave']}}

#####################Argument Parser###############################

parser = argparse.ArgumentParser()
#parser.add_argument('--train', help='train a model for appliance', action="store_true")
#parser.add_argument('--disag', help='disaggregate appliance consumption using previous model', action="store_true")
parser.add_argument('app', help='name of the appliance used in disaggregation')
parser.add_argument('model', help='name of the model used for disaggregation')
parser.add_argument('start_e', help='starting epoch', nargs='?', type=int, default = 0)
parser.add_argument('end_e', help='ending  epoch', nargs='?', type=int, default = 7)

#####################Experiment###############################

def Experiment(app, model_name, start_e, end_e):
    # =======  Logger settings  ===============
    #initialise the logger
    logger = logging.getLogger("Experiment")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    fh = logging.FileHandler("logs/Experiment_{}.log".format(app), 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    logger.info("Experiment started, Epochs: {} - {}".format(start_e, end_e))
    sp=6
    k= appliances[app]['key']
    v = appliances[app]

    module = importlib.import_module(model_name)
    class_ = getattr(module, model_name)
    nilm_model = class_(appliance = k ,MODEL_NAME = model_name, meta=v)

    # Preparing paths
    save_path = 'output/{}/{}/'.format(nilm_model.MODEL_NAME,k)
    os.makedirs(save_path, exist_ok=True)
    data_path= r'C:\Users\ozeidi\Desktop\MDS\00. Project\01. Coding\data\ukdale2015\ukdale.h5'  #'/projects/da33/ozeidi/Project/data/UKDALE/ukdale.h5'


    sas = None #SyntheticAggregateSource(data_set= training_set)




    logger.info(v)


    
    mains_lst = []
    meters_lst = []

    train_buidlings = v['train_buildings']
    training_set = [DataSet(data_path) for i in train_buidlings]
    for i, building_no in enumerate(train_buidlings):
        training_set[i].set_window(start=WINDOW_PER_BUILDING[building_no][0],end=WINDOW_PER_BUILDING[building_no][1])
        # This makes a list of references to the NILMTK main meters and submeter
        # no data is generated from them at this point
        mains_lst.append(training_set[i].buildings[building_no].elec.mains()) 
        meters_lst.append(training_set[i].buildings[building_no].elec.submeters()[k])
        logger.info('building {}'.format(building_no))
    nilm_model.train_across_buildings(mains_lst, meters_lst,start_e = start_e, end_e = end_e,sample_period =sp)
    logger.info ('Omar Training complete for {}'.format(k))


    ######################################
    ################ Testing #############
    ######################################
    validation_set = DataSet(data_path)
    validation_buidlings = v['validation_buildings'][0]
    validation_set.set_window(start=WINDOW_PER_BUILDING[validation_buidlings][0],end=WINDOW_PER_BUILDING[validation_buidlings][1])

    test_mains = validation_set.buildings[validation_buidlings].elec.mains()
    test_meter = validation_set.buildings[validation_buidlings].elec.submeters()[k]
    
    
    # test_mains: The aggregated signal meter
    # output: The output datastore
    # train_meter: This is used in order to copy the metadata of the train meter into the datastore
    disag_filename = '{}{}_disag-out_{}epochs.h5'.format(save_path,k,max(start_e,end_e)) # The filename of the resulting datastore
    print(disag_filename)
    output = HDFDataStore(disag_filename, 'w')
    nilm_model.disaggregate(test_mains, output, test_meter, sample_period=sp)
    print('Omar Disagregate complete for {}'.format(k))
    output.close()
        
    # #####################################
    # ############### Evaluation ##########
    # #####################################
    measures=['Recall','Precision', 'Accuracy', 'F1','RETE','MAE']

    performance = pd.DataFrame(index=appliances.keys(), columns=measures )

    result = DataSet(disag_filename)
    validation_buidlings = v['validation_buildings'][0]
    validation_set.set_window(start=WINDOW_PER_BUILDING[validation_buidlings][0],end=WINDOW_PER_BUILDING[validation_buidlings][1])

    #test_mains = validation_set.buildings[validation_buidlings].elec.mains().all_meters()[0]
    ground_truth = validation_set.buildings[validation_buidlings].elec.submeters()[k]
    predicted = result.buildings[validation_buidlings].elec.submeters()[k]
    # #------Plotting------------
    # fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    # fig.set_size_inches(18, 7)
    # #main plot
    # ax1.plot(validation_set.buildings[validation_buidlings].elec.mains().power_series_all_data(), label='Mains',color='red')
    # # submeter plot
    # aligned = align_two_meters(predicted, ground_truth)
    # aligned = pd.concat([a for a in aligned])
    # ax2.plot( aligned.slave, label='Ground Truth')
    # ax2.plot( aligned.master, label='Predicted')

    # #formating
    # ax1.set_title('Mains')
    # ax1.set_ylabel('Power (W)')
    # ax2.set_title('Submeter')
    # ax2.set_ylabel('Power (W)')
    
    # fig.legend(loc='best',fancybox=True, shadow=True, ncol=3)
    # fig.suptitle("{} Disaggregation Plot".format(k),fontsize=14, fontweight='bold')
    # fig.autofmt_xdate()
    # fig.savefig('{}/{}_disag_plot'.format(save_path,k))
    
    #---------Metrics-------------
    rpaf = list(metrics.recall_precision_accuracy_f1(predicted, ground_truth))
    rpaf.append( metrics.relative_error_total_energy(predicted, ground_truth))
    rpaf.append( metrics.mean_absolute_error(predicted, ground_truth))
    performance.at[k] = rpaf
    logger.info("============ Recall: {}".format(rpaf[0]))
    logger.info("============ Precision: {}".format(rpaf[1]))
    logger.info("============ Accuracy: {}".format(rpaf[2]))
    logger.info("============ F1 Score: {}".format(rpaf[3]))
    logger.info("============ Re. Error in Tot. Energy: {}".format(rpaf[4]))
    logger.info("============ MAE: {}".format(rpaf[5]))
    result.close()
    logger.info('\n {}'.format(performance))
    print(performance)
    performance.to_csv('{}{}_performance.csv'.format(save_path,k))
if __name__ == '__main__':
    args = parser.parse_args()
    if args.app in appliances.keys():
        Experiment(args.app,args.model, args.start_e, args.end_e)