from sklearn.model_selection import train_test_split

import tensorflow as tf

class Trainer:
  def __init__(self, model, optimizer, seed=9):
    self.optimizer = optimizer
    self.model = model
    self.seed = seed

    self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    self.train_acc = tf.keras.metrics.BinaryAccuracy()
    self.val_acc = tf.keras.metrics.BinaryAccuracy()
        
    self.val_precision = tf.keras.metrics.Precision()
    self.val_recall = tf.keras.metrics.Recall()


  def train(self,
            dataset,
            epochs,
            batch_size):
    X_train, X_val, y_train, y_val = train_test_split(dataset[0],
                                                      dataset[1], 
                                                        test_size=0.1, 
                                                        random_state=9)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))    
    train_dataset = train_dataset.shuffle(buffer_size=1024,seed=self.seed).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(32)

    

    for epoch in range(epochs):
      epoch +=1
      print(f'Start of epoch {epoch}:')
      
      for step, (X_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = self.train_batch(X_batch_train, y_batch_train)
       
      print(f'Train Loss on epoch {epoch}: {loss_value}')
      print(f'Train Accuracy on epoch {epoch}: {self.train_acc.result()}')

      self.train_acc.reset_states()

      for X_batch_val, y_batch_val in val_dataset:
       
        loss_value_val = self.validate_batch(X_batch_val, y_batch_val)
      
      print(f"Val Loss on epoch {epoch}: {loss_value_val}")
      print(f"Val Accuracy on epoch {epoch}: {self.val_acc.result()}")
      print(f"Val Precision on epoch {epoch}: {self.val_precision.result()}")
      print(f"Val Recall on epoch {epoch}: {self.val_recall.result()}")
      self.val_acc.reset_states()
      self.val_precision.reset_states()
      self.val_recall.reset_states()
      
  
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
    val_logits = tf.reshape(val_logits,shape=y.shape)
    self.val_acc(y, val_logits)
    self.val_precision(y, val_logits)
    self.val_recall(y, val_logits)

    loss_value = self.loss_fn(y, val_logits)

    return loss_value
