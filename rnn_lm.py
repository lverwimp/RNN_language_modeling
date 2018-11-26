#! /usr/bin/env python

import tensorflow as tf

class rnn_lm(object):
  '''
  This is a class to build and execute a recurrent neural network language model.
  '''
  
  def __init__(self,
              cell='LSTM',
              optimizer='SGD',
              lr=1,
              vocab_size=10000,
              embedding_size=64,
              hidden_size=128,
              dropout_rate=0.5,
              batch_size=BATCH_SIZE,
              num_steps = NUM_STEPS,
              is_training=True):
    # hyperparameters that can be changed
    self.which_cell = cell
    self.which_optimizer = optimizer
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    self.is_training = is_training
    self.lr = lr
    self.batch_size = batch_size
    self.num_steps = num_steps
    
    # hard-coded hyperparameters
    self.max_grad_norm = 5
    
    self.init_graph()
    
    self.output, self.state = self.feed_to_network()
    
    self.loss = self.calc_loss(self.output)
    
    if self.is_training:
      self.update_params(self.loss)
    
    
  def init_graph(self):
    '''
    This function initializes all elements of the network.
    '''
    
    self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps])
    self.targets = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps])
    
    # input embedding weights
    self.embedding = tf.get_variable("embedding", 
                                     [self.vocab_size, self.embedding_size], 
                                     dtype=tf.float32)
    
    # hidden layer
    if self.which_cell == 'LSTM':
      self.basic_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
    elif self.which_cell == 'RNN':
      self.basic_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size)
    else:
      raise ValueError("Specify which type of RNN you want to use: RNN or LSTM.")
      
    # apply dropout  
    self.cell = tf.contrib.rnn.DropoutWrapper(self.basic_cell, 
                                              output_keep_prob=self.dropout_rate)
    
    # initial state contains all zeros
    self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
    
    # output weight matrix and bias
    self.softmax_w = tf.get_variable("softmax_w",
                                     [self.hidden_size, self.vocab_size], 
                                     dtype=tf.float32)
    self.softmax_b = tf.get_variable("softmax_b",
                                     [self.vocab_size], 
                                     dtype=tf.float32)
    
    self.initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
    
    
  def feed_to_network(self):
    '''
    This function feeds the input to the network and returns the output and the state.
   
    '''
    
    # map input indices to continuous input vectors
    inputs = tf.nn.embedding_lookup(self.embedding, self.inputs)

	  # use dropout on the input embeddings
    inputs = tf.nn.dropout(inputs, self.dropout_rate)
    
    state = self.initial_state
    
    # feed inputs to network: outputs = predictions, state = new hidden state
    outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=None, initial_state=state)
    
    output = tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_size])
    
    return output, state
    
  
  def calc_loss(self, output):
    
    # calculate logits
    # shape of logits = [batch_size*num_steps, vocab_size]
    logits = tf.matmul(output, self.softmax_w) + self.softmax_b
    
    self.softmax = tf.nn.softmax(logits)
      
    # calculate cross entropy loss
    # reshape targets such that it has shape [batch_size*num_steps]
    # loss: contains loss for every time step in every batch
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.reshape(self.targets, [-1]), logits=logits)
      
    # average loss per batch
    avg_loss = tf.reduce_sum(loss) / self.batch_size
    
    return avg_loss
  
  def update_params(self, loss):
    
    # calculate gradients for all trainable variables 
    # + clip them if their global norm > 5 (prevents exploding gradients)
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, tf.trainable_variables()), self.max_grad_norm)
    
    if self.which_optimizer == 'SGD':
      optimizer = tf.train.GradientDescentOptimizer(self.lr)
    elif self.which_optimizer == 'Adam':
      optimizer = tf.train.AdamOptimizer(self.lr)
    else:
      raise ValueError("Specify which type of optimizer you want to use: SGD or Adam.")
    
    # update the weights
    self.train_op = optimizer.apply_gradients(
				zip(grads, tf.trainable_variables()))
