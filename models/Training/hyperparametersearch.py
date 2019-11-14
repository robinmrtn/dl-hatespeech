import random
import numpy as np
import datetime
import hashlib
import os
import csv
import tensorflow as tf
from collections import OrderedDict

class HyperparameterSearch:
  def __init__(self,
               trainer,
               hparams,
               create_model_func,
               weight_matrix,
               max_words,
               max_seq_len,
               dataset,                     
               log_file_loc):
    
    self.trainer = trainer
    self.hparams = hparams
    self.create_model_func = create_model_func
    self.weight_matrix = weight_matrix
    self.max_words = max_words
    self.max_seq_len = max_seq_len
    self.dataset = dataset    
    
    if os.path.exists(log_file_loc):
      self.log_file_loc = log_file_loc
    else:
      raise ValueError(f"'{log_file_loc}' is not a valid folder.")
                   
  def grid_search(self, n=None):

    #create log file with timestamp in name
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_file = os.path.join(self.log_file_loc, f'/content/log_{dt_string}.csv')
    open(log_file, 'a').close()    

    max_unique_combs = np.prod([len(i) for i in  list(self.hparams.values())])
    if n is None or n > max_unique_combs:
      n = max_unique_combs
    
    hparams_space = self._create_hparam_space(n)

    for i in range(n):
      self._grid_search_generator(hparams_space[i], log_file)
    
  def _grid_search_generator(self, hparams, log_file):
       
    model = self.create_model_func(self.max_words,
                                    self.max_seq_len,
                                    self.weight_matrix,
                                    hparams)
      
      
    self.trainer.load_model(model)

    optimizer = tf.keras.optimizers.SGD()
    if hparams['optimizer'] == 'sgd':
      lr = hparams['lr'] * 0.01
      optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif hparams['optimizer'] == 'adam':
      lr = hparams['lr'] * 0.001
      optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
      raise ValueError("No known optimizer found in hyperparameters.")

    self.trainer.load_optimizer(optimizer)

    hist = self.trainer.train(self.dataset, hparams['epochs'], hparams['batch_size'],
                                search_mode=True)  
    hparams.update({'hparams_hash': self._hparams_to_hashid(hparams.values())})
    hparams.update(hist)
    hparams.update({'model':model.name})
    
    with open(log_file) as f:
      line_nums = (sum(1 for line in f))

    with open(log_file,'a') as csv_file:
      writer = csv.DictWriter(csv_file, delimiter=';', fieldnames=list(hparams.keys()))                
      if line_nums == 0:
        writer.writeheader()
      writer.writerow(hparams)
            
  @staticmethod  
  def _hparams_to_hashid(hparms):
    s = '-'.join(str(c) for c in hparms)
    hash_string = hashlib.md5(s.encode('utf-8')).hexdigest()
    return hash_string

  def _create_hparam_space(self, n):
                  
    i = 0
    hparam_space = []
    hashed_hparam_space = []

    while i<n:
      choice_dict = {}
      
      for param in self.hparams:
        
        picked_val = random.choice(self.hparams[param])
        choice_dict[param] = picked_val

      hash_string = self._hparams_to_hashid(list(choice_dict.values()))
      if hash_string not in hashed_hparam_space:
        hashed_hparam_space.append(hash_string)
        hparam_space.append(choice_dict)
        i += 1 

    return hparam_space 