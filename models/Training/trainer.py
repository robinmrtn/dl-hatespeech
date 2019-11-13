import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
class Trainer:
  def __init__(self, optimizer, seed=9):
    self.optimizer = optimizer
    self.seed = seed

    self.model = None

    self.loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    self.train_acc = tf.keras.metrics.BinaryAccuracy()
    self.train_acc = tf.keras.metrics.BinaryAccuracy()
    self.val_acc = tf.keras.metrics.BinaryAccuracy()
        
    self.val_precision = tf.keras.metrics.Precision()
    self.val_recall = tf.keras.metrics.Recall()
    

  def load_model(self, model):
    self.model = model
  
  def train(self,
            dataset,
            epochs,
            batch_size,
            search_mode=False):
    
    if self.model is None:
      raise RuntimeError('No model loaded.')

    X_train, X_val, y_train, y_val = train_test_split(dataset[0],
                                                      dataset[1], 
                                                        test_size=0.1, 
                                                        random_state=SEED)
    

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))    
    train_dataset = train_dataset.shuffle(buffer_size=1024,seed=SEED).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(32)
    history_columns = ["train_loss","train_acc","val_loss",
                       "val_acc","val_precision","val_recall",
                       "val_f1","runtime"]
    history_array = np.zeros((epochs,len(history_columns)))
    
    for epoch in range(epochs):
      epoch +=1
      print(f'Start of epoch {epoch}:')
      start_time = time.time()
      for step, (X_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = self.train_batch(X_batch_train, y_batch_train)
       
      print(f'Train Loss on epoch {epoch}: {loss_value}')
      print(f'Train Accuracy on epoch {epoch}: {self.train_acc.result()}')

      

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
                                         runtime])
      print(f"Val Loss: {loss_value_val}")
      print(f"Val Accuracy: {self.val_acc.result()}")
      print(f"Val Precision: {self.val_precision.result()}")
      print(f"Val Recall: {self.val_recall.result()}")
      
      self.train_acc.reset_states()
      self.val_acc.reset_states()
      self.val_precision.reset_states()
      self.val_recall.reset_states()
      
    history_df = pd.DataFrame(data=history_array, columns=history_columns)

    if search_mode:
      return dict(zip(history_columns,history_array[-1].tolist()))
    else:
      return history_df

  @tf.function
  def train_batch(self, X, y):
    with tf.GradientTape() as tape:

      logits = self.model(X)
         
      loss_value = self.loss_fn(y, logits)

    grads = tape.gradient(loss_value, self.model.trainable_weights)

    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    self.train_acc(y, logits)
    
    return loss_value
  
  @tf.function
  def validate_batch(self, X, y):

    val_logits = self.model(X)
    
    y = tf.reshape(y ,shape=val_logits.shape)
    self.val_acc(y, val_logits)
    
    self.val_precision(y, val_logits)
    self.val_recall(y, val_logits)

    loss_value = self.loss_fn(y, val_logits)

    return loss_value