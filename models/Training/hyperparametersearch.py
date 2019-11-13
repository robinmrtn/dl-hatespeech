import random
import numpy as np
import hashlib
import csv
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
               epochs,
               batch_size,
               log_file=None):
    
    self.trainer = trainer
    self.hparams = hparams
    self.create_model_func = create_model_func
    self.weight_matrix = weight_matrix
    self.max_words = max_words
    self.max_seq_len = max_seq_len
    self.dataset = dataset
    self.epochs = epochs
    self.batch_size = batch_size
    
    if log_file is not None:
      self.log_file = log_file
            
  def grid_search(self, n=None):

    max_unique_combs = np.prod([len(i) for i in  list(self.hparams.values())])
    if n is None or n > max_unique_combs:
      n = max_unique_combs
    
    hparams_space = self._create_hparam_space(n)

    for i in range(n):
      self._grid_search_generator(hparams_space[i])
      pass


  def _grid_search_generator(self, hparams):
       
    model = self.create_model_func(self.max_words,
                                    self.max_seq_len,
                                    self.weight_matrix,
                                    hparams)
      
      
    self.trainer.load_model(model)

    hist = self.trainer.train(self.dataset, self.epochs, self.batch_size,
                                search_mode=True)  
    hparams.update(hist)
    
    if self.log_file is not None:

      with open(self.log_file) as f:
        line_nums = (sum(1 for line in f))

      with open(self.log_file,'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(hparams.keys()))                
        if line_nums == 0:
          writer.writeheader()
        writer.writerow(hparams)
        
    
  def _create_hparam_space(self, n):
                  
    i = 0
    hparam_space = []
    hashed_hparam_space = []

    while i<n:
      choice_dict = {}
      
      for param in self.hparams:
        
        picked_val = random.choice(self.hparams[param])
        choice_dict[param] = picked_val

      s = '-'.join(str(c) for c in list(choice_dict.values()))
      hash_string = hashlib.md5(s.encode('utf-8')).hexdigest()

      if hash_string not in hashed_hparam_space:
        hashed_hparam_space.append(hash_string)
        hparam_space.append(choice_dict)
        i += 1 

    return hparam_space