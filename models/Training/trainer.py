from sklearn.model_selection import train_test_split

import tensorflow as tf

class Trainer:
  def __init__(self, model, optimizer, seed=9):
    self.optimizer = optimizer
    self.model = model
    self.seed = seed
 
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
    val_dataset = val_dataset.batch(64)

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    train_acc = tf.keras.metrics.BinaryAccuracy()
    val_acc = tf.keras.metrics.BinaryAccuracy()

    for epoch in range(epochs):
      print(f'Start of epoch {epoch}:')
      epoch_loss_avg = tf.keras.metrics.Mean()
      epoch_accuracy = tf.keras.metrics.BinaryCrossentropy()
      for step, (X_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:

          logits = self.model(X_batch_train)
         
          loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        train_acc(y_batch_train, logits)
      
      print(f'Training Loss on epoch {epoch}: {loss_value}')
      print(f'Training ACC on epoch {epoch}: {train_acc.result()}')

      train_acc.reset_states()

      for X_batch_val, y_batch_val in val_dataset:
        val_logits = self.model(X_batch_val)

        val_acc(y_batch_val, val_logits)
      
      print(f"Val Acc on epoch {epoch}: {val_acc.result()}")
      val_acc.reset_states()
