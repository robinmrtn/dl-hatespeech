import numpy as np
import io, os
import spacy

from tensorflow.keras.preprocessing.sequence import pad_sequences


class Tokenizer:
  """"""
  def __init__(self, language='english', oov_token=None):
        
    if language == 'english':
      self.nlp = spacy.load('en_core_web_sm')
    elif language == 'german': 
      self.nlp = spacy.load('de_core_news_md')
    else:
      raise ValueError(f"Language '{language}' not found.")
    
    self.oov_token = oov_token
    
    if self.oov_token is not None:
      self.word_count = 1
    else:
      self.word_count = 0
      
    self.vocab = dict()
    self.reverse_vocab = dict()  

    
  def fit(self, texts):
    
    if self.oov_token is not None:
      self.vocab[oov_token] = 1
      self.reverse_vocab[1] = self.oov_token    
    
    for text in texts:
      doc = self.nlp(text)
      for token in doc:
        if token.text.lower() not in self.vocab:
          self.vocab[token.text.lower()] = self.word_count + 1
          self.reverse_vocab[self.word_count + 1] = token.text.lower()        
          self.word_count += 1       
        
  
  def texts_to_sequences(self, texts):
    
    sequences = list()        
    
    for text in texts:
      sequence = list()
      doc = self.nlp(text)
         
      for token in doc:
        if token.text.lower() in self.vocab:
          sequence.append(self.vocab[token.text.lower()])
        elif self.oov_token is not None:
          sequence.append(self.vocab[self.oov_token])
      
      sequences.append(sequence)                           
         
    return sequences
    
    
  def sequences_to_texts(self, sequences):
    
    texts = list()
    for sequence in sequences:
      text = list()
      for element in sequence:
       
        if element in self.reverse_vocab:
          text.append(self.reverse_vocab[element])
        elif oov_token is not None:
          text.append(self.oov_token)
      
      text = ' '.join(text)
      texts.append(text)
    
    return texts
    
    
  def create_weight_matrix(self, word_embedding):
    embedding_matrix = None
    if isinstance(word_embedding, WordEmbedding):
      
      if word_embedding.oov_init == 'rand':
        embedding_matrix = word_embedding.random_state.default_rng().uniform(-0.5,
                                                                                0.5,
                                                              size=(self.word_count+1,
                                                                    word_embedding.dimensions))
        embedding_matrix[0] = np.zeros(word_embedding.dimensions)
        
      elif word_embedding.oov_init == 'zero':
        embedding_matrix = np.zeros((self.word_count+1, word_embedding.dimensions))
        
      for word, index in self.vocab.items():
        
        embedding_matrix[index] = word_embedding.get(word)
    
    return embedding_matrix


class WordEmbedding:
  """docstring"""
  def __init__(self, dimensions, oov_init='rand', random_state=None):
    
    self.embedding = None
    self.vocab = None
    if oov_init in ('zero', 'rand'):
      self.oov_init = oov_init
    else:
      raise ValueError("Unknown Out-of-Vocabulary initializer. "+
                      "Value must be 'zero' or 'rand'.")
      
    if isinstance(random_state, int):
      self.random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
      self.random_state = random_state
    elif random_state is None:
      self.random_state = np.random
    else:
      raise ValueError("Variable 'random_state' must be an Integer or " +
                       "an instance of numpy.random.RandomState.")    
    
    if dimensions in range(50,301):
      self.dimensions = dimensions
    else:
      raise ValueError("Dimensions must be between 50 and 300.")  
          
          
  def load(self, filepath, encoding="utf-8"):    
    
    self.embedding = np.empty(shape=(self.__count_lines(filepath, encoding),
                                     self.dimensions))
    self.vocab = {}
    i = 0
    
    f = open(filepath, encoding=encoding)
    for line in f:
      splitLine = line.split()
      word = splitLine[0]
      
      embedding = np.array([float(val) for val in splitLine[1:]])      
        
      self.embedding[i] = embedding
      self.vocab[word] = i
      i += 1
    f.close()  
      

  def get(self, word, include_oov=True):
    
    if self.vocab is None or self.embedding is None:
      raise RuntimeError("No word embedding loaded.")
    
    if word in self.vocab:
      vocab_index = self.vocab[word]
      embedding_vec = self.embedding[vocab_index]
    elif self.oov_init == 'rand' and include_oov:
      embedding_vec = self.random_state.default_rng().uniform(-0.5,0.5,
                                                              self.dimensions)
    else:
      embedding_vec = np.zeros(self.dimensions)
    return embedding_vec
    
  
  def __count_lines(self, filepath, encoding):
    num_lines = 0
    
    f = open(filepath, encoding="utf-8")
    for line in f:
      num_lines += 1
    
    return num_lines