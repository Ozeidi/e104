import logging
logger = logging.getLogger("Experiment.SAS")
import nilmtk
import numpy as np
class SyntheticAggregateSource():
    def __init__(self, data_set, rng_seed=1234,
                 sample_period=6,distractor_inclusion_prob=0.25,
                 target_inclusion_prob=0.5,
                 uniform_prob_of_selecting_each_building=True,
                 allow_incomplete_target=True,
                 allow_incomplete_distractors=True,
                 include_incomplete_target_in_output=True,):
        
        self.data = data_set
        self.appliances = ['kettle', 'microwave', 'dish washer', 'washing machine', 'fridge']
        self.sample_period = sample_period
        self.rng = np.random.RandomState(rng_seed)
        self.distractor_inclusion_prob = distractor_inclusion_prob
        self.target_inclusion_prob = target_inclusion_prob
        self.uniform_prob_of_selecting_each_building = (uniform_prob_of_selecting_each_building)
        self.allow_incomplete_target = allow_incomplete_target
        self.allow_incomplete_distractors = allow_incomplete_distractors
        self.include_incomplete_target_in_output = (include_incomplete_target_in_output)
        #Collect Activation data
        self._collect_activations()
        
    
    def gen_seq(self,  target_appliance=None, seq_length=None,
                distractor_inclusion_prob=0.25,
                target_inclusion_prob=0.5,
                uniform_prob_of_selecting_each_building=True,
                allow_incomplete_target=True,
                allow_incomplete_distractors=True,
                include_incomplete_target_in_output=True,
                batch_size=120
               ):
        
        self.target_appliance = target_appliance if target_appliance  else self.target_appliance
        self.seq_length = seq_length if seq_length  else self.seq_length
        self.distractor_inclusion_prob = distractor_inclusion_prob
        self.target_inclusion_prob = target_inclusion_prob
        self.uniform_prob_of_selecting_each_building = (uniform_prob_of_selecting_each_building)
        self.allow_incomplete_target = allow_incomplete_target
        self.allow_incomplete_distractors = allow_incomplete_distractors
        self.include_incomplete_target_in_output = (include_incomplete_target_in_output)
        self.batch_size = batch_size
        #---------------------------------
        main = np.zeros(shape=(self.batch_size,self.seq_length, 1))
        sub_meter = np.zeros(shape=(self.batch_size,self.seq_length, 1))
        for i in range(self.batch_size):
            main[i,:], sub_meter[i,:] = self._gen_seq()
        return (main, sub_meter)
            
    def _gen_seq(self):
        
        main = np.zeros(shape= (self.seq_length,1))
        sub_meter =  np.zeros(shape= (self.seq_length,1))
        # target appliance
        if self.rng.binomial(n=1, p =self.target_inclusion_prob):
            n_act_target= len(self.activation_dict[self.target_appliance])
            ix = np.random.randint(low=0,high=n_act_target)
            window = self.activation_dict[self.target_appliance][ix].values
            main[:,0] = self._pad(window[:], incomplete= self.allow_incomplete_target)
            sub_meter = main.copy()
#             n_window = len(window)
#             if len(window) <= self.seq_length:
#                 main[:n_window,0] = window
#                 sub_meter = main.copy()
#             else:
#                 main = window[:self.seq_length]
#                 sub_meter = main.copy()
                
        # Distractors
        distractors = self.appliances.copy()
        distractors.remove(self.target_appliance)
        for dis in distractors:
            if self.rng.binomial(n=1, p =self.distractor_inclusion_prob):
                n_act_dis= len(self.activation_dict[dis])
                ix = np.random.randint(low=0,high=n_act_dis)
                window = self.activation_dict[dis][ix].values
                
                main[:,0] += self._pad(window[:], incomplete=self.allow_incomplete_distractors)

        return (main, sub_meter)        
    def _collect_activations(self):
        logger.info('collecting activations')
        activation_dict={}
        for a in self.appliances:
            logger.info('Collecting activations for {}'.format(a))
            # remove building =1 used for testing only
            meter_group = nilmtk.global_meter_group.select_using_appliances(dataset = self.data.metadata['name'],
                                                                            building = 1, type=a)
            activation_dict[a]= meter_group.get_activations(sample_period= self.sample_period)
        self.activation_dict = activation_dict
    

    
    def _pad(self, Arr,incomplete = True):
        n, remainder = divmod(len(Arr), self.seq_length)
        n += bool(remainder)
        res = np.zeros(n * self.seq_length)

        if incomplete:
            #placment index
            start_ix= np.random.randint(low=0, high =len(Arr))
            end_ix = np.random.randint(low=start_ix, high =len(Arr))

            res[start_ix:end_ix] = Arr[start_ix:end_ix]

        else:
            #placment index
            ix= np.random.randint(low=0, high =len(res)-len(Arr)+1)
            res[ix:ix+len(Arr)] = Arr
            
        ix = np.random.randint(low=0, high =len(res)-self.seq_length+1)
        return res[ix:ix+self.seq_length]  
  