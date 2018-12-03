#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import batchGenerator, rnn_lm

def run_lm(name='LSTM', cell='LSTM', 
           optimizer='Adam', lr=0.01, 
           vocab_size = 10000, embedding_size=64, 
           hidden_size=128, dropout_rate=0.5, 
           num_steps=50, inspect_emb=False, 
           train_ids=None, valid_ids=None, 
           test_ids=None, test_log_prob=False):
  '''
  Creates training, validation and/or test models,
  trains, validates and/or tests the model.
  Arguments:
    name: name that will be used to save the model
    cell: type of RNN cell (only LSTM is currently implemented)
    optimizer: 'SGD' or 'Adam'
    lr: learning rate
    vocab_size: size of the vocabulary
    embedding_size: size of continuous embedding that will be input to the RNN
    hidden_size: size of the hidden layer
    dropout rate: value between 0 and 1, number of neurons that will be 
        kept (not dropped) during training, prevents overfitting
    num_steps
    inspect_emb: boolean, if True we want to return the embedding_matrix
    train_ids: training data
    valid_ids: validation data
    test_ids: test data
    test_log_prob: boolean, if True we only want to test the log probability for a test sentence
  '''
    
  with tf.Graph().as_default() as graph:

      # create the models
      if not test_log_prob:
      
        with tf.variable_scope("Model"):
          rnn_train = rnn_lm.rnn_lm(cell=cell,
                             optimizer=optimizer, 
                             lr=lr,
                             vocab_size=vocab_size,
                             embedding_size=embedding_size,
                             hidden_size=hidden_size,
                             dropout_rate=dropout_rate)

          saver = tf.train.Saver()

        with tf.variable_scope("Model", reuse=True):
          rnn_valid = rnn_lm.rnn_lm(cell=cell, 
                             optimizer=optimizer,
                             lr=lr,
                             vocab_size=vocab_size, 
                             embedding_size=embedding_size,
                             hidden_size=hidden_size,
                             dropout_rate=dropout_rate,
                             is_training=False)
          
        reuse = True
        
      else:
        reuse = False
               
      with tf.variable_scope("Model", reuse=reuse):
        rnn_test = rnn_lm.rnn_lm(cell=cell, 
                           optimizer=optimizer, 
                           lr=lr,
                           vocab_size=vocab_size,
                           embedding_size=embedding_size,
                           hidden_size=hidden_size,
                           dropout_rate=dropout_rate,
                           batch_size=1,
                           num_steps=1,
                           is_training=False)
      

      sv = tf.train.Supervisor(logdir=name)

      with sv.managed_session(config=tf.ConfigProto()) as session:
        
        if not test_log_prob:
        
          for i in xrange(5):

            print('Epoch {0}'.format(i+1))

            train_ppl = run_epoch(session, rnn_train, train_ids, num_steps=num_steps)
            print('Train perplexity: {0}'.format(train_ppl))

            valid_ppl = run_epoch(session, rnn_valid, valid_ids, num_steps=num_steps, is_training=False)
            print('Validation perplexity: {0}'.format(valid_ppl))

          save_path = saver.save(session, "{0}/rnn.ckpt".format(name))
          print('Saved the model to ',save_path)

        test_ppl = run_epoch(session, rnn_test, test_ids, num_steps=num_steps,
                             is_training=False, is_test=True, 
                             test_log_prob=test_log_prob)
        if not test_log_prob:
          print('Test perplexity: {0}'.format(test_ppl))
        
        if inspect_emb: 
          emb_matrix = tf.get_default_graph().get_tensor_by_name("Model/embedding:0")
          emb_matrix_np = emb_matrix.eval(session=session)

          return emb_matrix_np

        else:

          return None

def run_epoch(session, rnn, data, num_steps=50, is_training=True, is_test=False, test_log_prob=False):
    '''
    This function runs a single epoch (pass) over the data,
    updating the model parameters if we are training,
    and returns the perplexity.
    Input arguments:
      rnn: object of the rnn_lm class
      data: list of word indices
      num_steps
      is_training: boolean, True is we are training the model
      is_test: boolean, True is we are testing a trained model
      test_log_prob: boolean, True if we want the log probability
    Returns:
      ppl: float, perplexity of the dataset
    '''
  
    generator = batchGenerator.batchGenerator(data, test=is_test)
      
    state = session.run(rnn.initial_state)
    sum_loss = 0.0
    iters = 0
    
    if test_log_prob: 
      sum_log_prob = 0.0
      
    while True:

      input_batch, target_batch, end_reached = generator.generate()
        
      if end_reached:
        break

      feed_dict = {rnn.inputs: input_batch,
                  rnn.targets: target_batch,
                  rnn.initial_state : state}

      fetches = {'loss': rnn.loss,
                'state': rnn.state}
      
      if is_training:
        fetches['train_op'] = rnn.train_op
        
      if test_log_prob:
        fetches['softmax'] = rnn.softmax
        
      result = session.run(fetches, feed_dict)
        
      state = result['state']
      loss = result['loss']
      
      if test_log_prob:
        softmax = result['softmax']
        prob_target = softmax[0][target_batch[0][0]]
        sum_log_prob += np.log(prob_target)

      sum_loss += loss
      # the loss is an average over num_steps
      if is_test:
        iters += 1
      else:
        iters += num_steps
        
    # calculate perplexity    
    ppl = np.exp(sum_loss / iters)
    
    if test_log_prob:
      print('Log probability: {0}'.format(sum_log_prob))
    
    return ppl
