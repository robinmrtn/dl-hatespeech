import time
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import datetime
import hashlib
import os
import csv

from sklearn.model_selection import train_test_split

from collections import OrderedDict
class Trainer:
  def __init__(self, model=None, optimizer=None):

    self.optimizer = optimizer
    self.model = model

    self.loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    self.train_acc = tf.keras.metrics.BinaryAccuracy()
    self.train_acc = tf.keras.metrics.BinaryAccuracy()
    self.val_acc = tf.keras.metrics.BinaryAccuracy()
        
    self.val_precision = tf.keras.metrics.Precision()
    self.val_recall = tf.keras.metrics.Recall()
     

  def load_model(self, model):
    self.model = model

  def load_optimizer(self, optimizer):
    self.optimizer = optimizer
  
  def train(self,
            dataset,
            epochs,
            batch_size,
            search_mode=False,
            patience=5):
        
    
    if self.model is None:
      raise RuntimeError('No model loaded.')
    if self.optimizer is None:
      raise RuntimeError('No optimizer loaded.')

    X_train, X_val, y_train, y_val = train_test_split(dataset[0],
                                                      dataset[1], 
                                                        test_size=0.1 
                                                        , random_state=9)
    stopping_step = 0
    best_acc = tf.constant(0.0)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))    
    train_dataset = train_dataset.shuffle(buffer_size=1024,seed=SEED).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(32)
    history_columns = ["train_loss","train_acc","val_loss",
                       "val_acc","val_precision","val_recall",
                       "val_f1","runtime"]
    history_array = np.zeros((epochs,len(history_columns)))

    #create graph for every model/optimizer combination
    train_batch_fun = self.train_batch_fn()
    
    for epoch in range(epochs):
      epoch += 1     
      
      print(f'Start of epoch {epoch}:')
      start_time = time.time()
      for step, (X_batch_train, y_batch_train) in enumerate(train_dataset):
         loss_value, train_acc = train_batch_fun(X_batch_train, y_batch_train,
                                     self.model, self.optimizer, self.loss_fn,
                                      self.train_acc)
        
      print(f'Train Loss on epoch {epoch}: {loss_value}')
      print(f'Train Accuracy on epoch {epoch}: {train_acc}')

      for X_batch_val, y_batch_val in val_dataset:
       
        loss_value_val, val_acc, val_precision, val_recall = self.validate_batch(self.model,
                                                                                 X_batch_val, y_batch_val,
                                             self.loss_fn, self.val_acc,
                                             self.val_precision, self.val_recall)
      
      val_f1 = 2 * (self.val_recall.result() * self.val_precision.result()) / (self.val_recall.result() + self.val_precision.result())
      end_time = time.time()
      runtime = end_time-start_time
      history_array[epoch-1] = np.array([loss_value,
                                         train_acc,
                                         loss_value_val,
                                         val_acc,
                                         val_precision,
                                         val_recall,
                                         val_f1,
                                         runtime]).round(6)
      print(f"Val Loss: {loss_value_val}")
      print(f"Val Accuracy: {self.val_acc.result()}")
      print(f"Val Precision: {self.val_precision.result()}")
      print(f"Val Recall: {self.val_recall.result()}")
      
      '''early stopping'''
      
      if epoch == 1:
        best_history = history_array[epoch-1]
      print(val_acc.numpy().astype(float), " > ", best_acc.numpy().astype(float) )
      print(f"cond: {self.val_acc.result().numpy().astype(float) > best_acc.numpy().astype(float)}")
      if tf.math.greater(val_acc.numpy(), best_acc).numpy():
        best_acc = val_acc
        best_history = history_array[epoch-1]
        stopping_step = 0
        print(f"new best acc={best_acc}")
      else:
        stopping_step += 1
      
      if search_mode and patience <= stopping_step:
        print('Early stopping triggerd!')
        history_array[-1] = best_history
        break
 
      self.train_acc.reset_states()
      self.val_acc.reset_states()
      self.val_precision.reset_states()
      self.val_recall.reset_states()
      
    history_df = pd.DataFrame(data=history_array, columns=history_columns)

    if search_mode:
      '''with early stopping only return best hist element'''
      return dict(zip(history_columns,history_array[-1].tolist()))
    else:
      return history_df
  @staticmethod    
  def train_batch_fn():
    @tf.function
    def train_batch(X, y, model, optimizer, loss_fn, train_acc):
      with tf.GradientTape() as tape:

        logits = model(X)
          
        loss_value = loss_fn(y, logits)

      grads = tape.gradient(loss_value, model.trainable_weights)

      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      train_acc(y, logits)
      
      return loss_value, train_acc.result()

    return train_batch

  @staticmethod
  @tf.function
  def validate_batch(model, X, y, loss_fn, val_acc, val_precision, val_recall):

    val_logits = model(X)
    
    y = tf.reshape(y ,shape=val_logits.shape)
    val_acc(y, val_logits)
    
    val_precision(y, val_logits)
    val_recall(y, val_logits)

    loss_value = loss_fn(y, val_logits)

    return loss_value, val_acc.result(), val_precision.result(), val_recall.result() 


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