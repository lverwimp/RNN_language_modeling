#! /usr/bin/env python

def run_lm(cell='LSTM', optimizer='SGD', lr=1, 
           embedding_size=64, hidden_size=128, 
           dropout_rate=0.5, inspect_emb=False,
           train_ids=None, valid_ids=None, test_ids=None,
           test_log_prob=False):
  '''
  Creates training, validation and/or test models,
  trains, validates and/or tests the model.
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
          rnn_valid = rnn_lm(cell=cell, 
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
        rnn_test = rnn_lm(cell=cell, 
                           optimizer=optimizer, 
                           lr=lr,
                           vocab_size=vocab_size,
                           embedding_size=embedding_size,
                           hidden_size=hidden_size,
                           dropout_rate=dropout_rate,
                           batch_size=1,
                           num_steps=1,
                           is_training=False)
      

      sv = tf.train.Supervisor(logdir='models')

      with sv.managed_session(config=tf.ConfigProto()) as session:
        
        if not test_log_prob:
        
          for i in xrange(5):

            print('Epoch {0}'.format(i+1))

            train_ppl = run_epoch(session, rnn_train, train_ids)
            print('Train perplexity: {0}'.format(train_ppl))

            valid_ppl = run_epoch(session, rnn_valid, valid_ids, is_training=False)
            print('Validation perplexity: {0}'.format(valid_ppl))

          save_path = saver.save(session, "models/rnn.ckpt")
          print('Saved the model to ',save_path)

        test_ppl = run_epoch(session, rnn_test, test_ids, 
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
