import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
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
            patience=10):
        
    
    if self.model is None:
      raise RuntimeError('No model loaded.')
    if self.optimizer is None:
      raise RuntimeError('No optimizer loaded.')

    X_train, X_val, y_train, y_val = train_test_split(dataset[0],
                                                      dataset[1], 
                                                        test_size=0.1 
                                                        , random_state=9)
    stopping_step = 0

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
      best_acc = 0
      
      print(f'Start of epoch {epoch}:')
      start_time = time.time()
      for step, (X_batch_train, y_batch_train) in enumerate(train_dataset):
        train_acc, loss_value = train_batch_fun(X_batch_train, y_batch_train,
                                     self.model, self.optimizer, self.loss_fn,
                                      self.train_acc)
        #loss_value = self.train_batch(X_batch_train, y_batch_train, self.optimizer)
        #self.train_acc = train_acc
      print(f'Train Loss on epoch {epoch}: {loss_value}')
      print(f'Train Accuracy on epoch {epoch}: {train_acc}')

      for X_batch_val, y_batch_val in val_dataset:
       
        loss_value_val = self.validate_batch(X_batch_val, y_batch_val)
      
      val_f1 = 2 * (self.val_recall.result() * self.val_precision.result()) / (self.val_recall.result() + self.val_precision.result())
      end_time = time.time()
      runtime = end_time-start_time
      history_array[epoch-1] = np.array([loss_value,
                                         self.train_acc.result(),
                                         loss_value_val,
                                         self.val_acc.result(),
                                         self.val_precision.result(),
                                         self.val_recall.result(),
                                         val_f1,
                                         runtime]).round(5)
      print(f"Val Loss: {loss_value_val}")
      print(f"Val Accuracy: {self.val_acc.result()}")
      print(f"Val Precision: {self.val_precision.result()}")
      print(f"Val Recall: {self.val_recall.result()}")
      
      '''early stopping'''
      
      if best_acc < val_acc.result():
        best_acc = val_acc.result()
        best_history = history_array[epoch-1]
        stopping_step = 0
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

  @tf.function
  def validate_batch(self, X, y):

    val_logits = self.model(X)
    
    y = tf.reshape(y ,shape=val_logits.shape)
    self.val_acc(y, val_logits)
    
    self.val_precision(y, val_logits)
    self.val_recall(y, val_logits)

    loss_value = self.loss_fn(y, val_logits)

    return loss_value  