import tensorflow as tf
#custom activation func import 
class MLPLayer(tf.keras.layers.Layer):
    def __init__(self, shape, neurons, layers, dropout,
                activation_func, l1_reg, l2_reg, name="MLPLayer"):

        super(MLPLayer, self).__init__(name=name)
        self.layers = layers
        self.dense_layers = []
        self.dropout_layers = []
        
        for _ in range(self.layers):
            self.dense_layers.append(tf.keras.layers.Dense(neurons,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg),
                                    activation=activation_func))
            
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout))
        
            if shape == "pyramid":
                neurons = int(neurons/2)

    def __call__(self, inputs):
        x = inputs
        for i in range(self.layers):
            x = self.dense_layers[i](x)
            x = self.dropout_layers[i](x)
        return x

