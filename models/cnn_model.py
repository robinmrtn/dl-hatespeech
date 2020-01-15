import tensorflow as tf
import MLPLayer
#custom activation func import 
class CNNModel(tf.keras.Model):

    def __init__(self, max_num_words, max_sequence_length, embedding_matrix, hparams):

        self.embedding_layer =tf.keras.layers.Embedding(max_num_words,
                                              embedding_matrix.shape[1],
                                              weights=[embedding_matrix],
                                              name="Embedding",
                                              trainable=False, 
                                              input_length=max_sequence_length)
        
        self.embedding_dropout = tf.keras.layers.Dropout(hparams['we_do'])
        self.cnn_dropout = tf.keras.layers.Dropout(hparams['cnn_do'])

        self.cnn_layers = []
        self.pooling_layers = []

        for filter_size in hparams['filter_sizes']:
            cnn_layer = tf.keras.layers.SeparableConv1D(hparams['n_filters'],
                                                kernel_size=filter_size,
                                                activation=hparams['cnn_activation_func'],
                                                 kernel_regularizer=tf.keras.regularizers.l1_l2(hparams['cnn_l1'],hparams['cnn_l2']))  
            pooling_layer = tf.keras.layers.GlobalMaxPooling1D()

            self.cnn_layers.append(cnn_layer)
            self.pooling_layers.append(pooling_layer)
            
        self.mlp_layer = MLPLayer(hparams['mlp_shape'], hparams['mlp_neurons'], hparams['mlp_layers'],
                                hparams['mlp_do'], hparams['mlp_activation_func'],hparams['mlp_l1'], hparams['mlp_l2'])
        
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid", name="Classifier")
    
    def __call__(self,inputs):
        x = self.embedding_layer(inputs)
        
        x_cnn = []
        for i,_ in enumerate(self.cnn_layers):
            x1 = self.cnn_layers[i](x)
            x1 = self.pooling_layers[i](x1)
            x_cnn.append(x1)

        if len(x_cnn) > 1:
            x = tf.keras.layers.Concatenate(axis=1)(x_cnn)
        else:
            x = x_cnn[0]
        
        x = self.mlp_layer(x)

        return self.classifier(x)
        
        

