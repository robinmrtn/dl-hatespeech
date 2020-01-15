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
        x = self.embedding_dropout(x)
        
        x_cnn = []
        for i,_ in enumerate(self.cnn_layers):
            x1 = self.cnn_layers[i](x)
            x1 = self.pooling_layers[i](x1)
            x_cnn.append(x1)

        if len(x_cnn) > 1:
            x = tf.keras.layers.Concatenate(axis=1)(x_cnn)
        else:
            x = x_cnn[0]
        
        x = self.cnn_dropout(x)
        x = self.mlp_layer(x)

        return self.classifier(x)

class RNNModel(tf.keras.Model):

    def __init__(self, max_num_words, max_sequence_length, embedding_matrix, hparams):

        rnn_neurons = hparams['rnn_neurons']
        self.embedding_layer =tf.keras.layers.Embedding(max_num_words,
                                              embedding_matrix.shape[1],
                                              weights=[embedding_matrix],
                                              name="Embedding",
                                              trainable=False, 
                                              input_length=max_sequence_length)
        
        self.embedding_dropout = tf.keras.layers.Dropout(hparams['we_do'])
        self.rnn_dropout = tf.keras.layers.Dropout(hparams['rnn_do'])
        self.pooling_layer = tf.keras.layers.GlobalMaxPool1D(name="GlobalMaxPool1D")
        self.mlp_layer = MLPLayer(hparams['mlp_shape'], hparams['mlp_neurons'], hparams['mlp_layers'],
                                hparams['mlp_do'], hparams['mlp_activation_func'],hparams['mlp_l1'], hparams['mlp_l2'])
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid", name="Classifier")
        self.rnn_layers = []

        for i in range(0, hparams['rnn_layers']):    

            if hparams['rnn_type'] == "gru":
                rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_neurons,
                                                                           return_sequences=True,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(hparams['rnn_l1'], 
                                                                                    hparams['rnn_l2'])))
            elif hparams['rnn_type'] == "lstm":
                rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_neurons,
                                                            return_sequences=True,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(hparams['rnn_l1'], 
                                                                                    hparams['rnn_l2'])))
            self.rnn_layers.append(rnn_layer)
            rnn_neurons = int(rnn_neurons/2)
        
    def __call__(self, inputs):
        
        x = self.embedding_layer(inputs)
        x = self.embedding_dropout(x)
        
        for i,_ in enumerate(self.rnn_layers):
            x = self.rnn_layers[i](x)
            
        x = self.rnn_dropout(x)
        x = self.mlp_layer(x)

        return self.classifier(x)

        
        
        
        
        

