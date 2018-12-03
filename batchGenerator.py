#! /usr/bin/env python

import numpy as np

class batchGenerator(object):
  '''
  This class generates batches for a dataset.
  Input arguments:
    data: list of indices (word ids)
    batch_size: number of sequences in a mini-batch
    num_steps: length of each sequence in the mini-batch
    test: boolean, is True if we are testing; in that case batch_size and num_steps are 1
  '''
  
  def __init__(self, data, batch_size=32, num_steps=50, test=False):
    '''
    Prepares a dataset.
    '''
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.test = test 

    self.data_array = np.array(data)
  
    if not self.test:
      len_batch_instance = len(data) / batch_size

      data_array = self.data_array[:batch_size*len_batch_instance]

      # divide data in batch_size parts
      self.data_reshaped = np.reshape(data_array, (batch_size, len_batch_instance))

      # number of mini-batches that can be generated
      self.num_batches_in_data = len_batch_instance / num_steps - 1
    
    self.curr_idx = 0
  
  def generate(self):
    '''
    Generates
      input_batch: numpy array (batch_size x num_steps) or None, if the end of the dataset is reached
      target_batch: numpy array (batch_size x num_steps) or None, if the end of the dataset is reached
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

      # input: take slice of size 
      input_batch = self.data_reshaped[:,self.curr_idx*self.num_steps:self.curr_idx*self.num_steps+self.num_steps]

      # target = input shifted 1 time step
      target_batch = self.data_reshaped[:,self.curr_idx*self.num_steps+1:self.curr_idx*self.num_steps+self.num_steps+1]

    self.curr_idx += 1
    
    return input_batch, target_batch, False
