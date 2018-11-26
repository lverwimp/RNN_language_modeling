#! /usr/bin/env python

import numpy as np

class batchGenerator(object):
  '''
  This class generates batches for a dataset.
  Input argument:
    data: list of indices (word ids)
  '''
  
  def __init__(self, data, test=False):
    '''
    Prepares a dataset.
    '''
  
    self.test = test 

    self.data_array = np.array(data)
  
    if not self.test:
      len_batch_instance = len(data) / BATCH_SIZE

      data_array = self.data_array[:BATCH_SIZE*len_batch_instance]

      # divide data in BATCH_SIZE parts
      self.data_reshaped = np.reshape(data_array, (BATCH_SIZE, len_batch_instance))

      # number of mini-batches that can be generated
      self.num_batches_in_data = len_batch_instance / NUM_STEPS - 1
    
    self.curr_idx = 0
  
  def generate(self):
    '''
    Generates
      input_batch: numpy array or None, if the end of the dataset is reached
      target_batch: numpy array or None, if the end of the dataset is reached
      end_reached: boolean, True is end of dataset is reached
    '''
    
    if self.test:
      if self.curr_idx+1 >= len(self.data_array):
        return None, None, True
      
      input_batch = [[self.data_array[self.curr_idx]]]
      target_batch = [[self.data_array[self.curr_idx+1]]]
      
    else:
      if self.curr_idx >= self.num_batches_in_data:
        return None, None, True

      # input: take slice of size BATCH_
      input_batch = self.data_reshaped[:,self.curr_idx*BATCH_SIZE:self.curr_idx*BATCH_SIZE+BATCH_SIZE]

      # target = input shifted 1 time step
      target_batch = self.data_reshaped[:,self.curr_idx*BATCH_SIZE+1:self.curr_idx*BATCH_SIZE+BATCH_SIZE+1]    

    self.curr_idx += 1
    
    return input_batch, target_batch, False
