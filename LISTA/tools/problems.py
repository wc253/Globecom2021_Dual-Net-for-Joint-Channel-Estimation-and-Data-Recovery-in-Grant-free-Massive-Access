#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math
import tensorflow as tf
import os
import scipy.io as io
from collections import OrderedDict

class Generator(object):
    def __init__(self,A,L,**kwargs):
        self.A = A
        self.L = L
        M,N = A.shape
        vars(self).update(kwargs)
        self.x_ = tf.placeholder( tf.float32,(N,L),name='x' )
        self.y_ = tf.placeholder( tf.float32,(M,L),name='y' )

class TFGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)
    def __call__(self,sess):
        'generates y,x pair for training'
        return sess.run( ( self.ygen_,self.xgen_ ) )
    def get_batch(self):
        dataset_info_path='train_info.txt'#train_info.txt
        with open(dataset_info_path,'r') as dataset_info:
            input_info=OrderedDict()
            for line in dataset_info.readlines():
                items=line.split(',')
                try:
                    input_info[items[0]]=[int(dim) for dim in items[1:]]
                except:
                    input_info[items[0]]=[]
        def _parse_tf_example(example_proto):
            features=dict([(key,tf.FixedLenFeature([],tf.string)) for key,_ in input_info.items()])
            parsed_features=tf.parse_single_example(example_proto,features=features)
            return [tf.reshape(tf.decode_raw(parsed_features[key],tf.float32),value) for key,value in input_info.items()]

        dataset_path='train.tfrecords'
        dataset=tf.data.TFRecordDataset(dataset_path)#[dataset_path1,dataset_path2]
        dataset=dataset.map(_parse_tf_example)
        dataset=dataset.repeat()
        dataset=dataset.batch(1)
        iterator=dataset.make_initializable_iterator()
        data_batch=iterator.get_next()
        keys=list(input_info.keys())
        data_batch=dict([(keys[i],data_batch[i]) for i in range(len(keys))])
        return data_batch,iterator.initializer

class NumpyGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)

    def __call__(self,sess):
        'generates y,x pair for training'
        return self.p.genYX(self.nbatches,self.nsubprocs)
        
    def get_batch(self):
        dataset_info_path='train_info.txt'#train_info.txt
        with open(dataset_info_path,'r') as dataset_info:
            input_info=OrderedDict()
            for line in dataset_info.readlines():
                items=line.split(',')
                try:
                    input_info[items[0]]=[int(dim) for dim in items[1:]]
                except:
                    input_info[items[0]]=[]
        def _parse_tf_example(example_proto):
            features=dict([(key,tf.FixedLenFeature([],tf.string)) for key,_ in input_info.items()])
            parsed_features=tf.parse_single_example(example_proto,features=features)
            return [tf.reshape(tf.decode_raw(parsed_features[key],tf.float32),value) for key,value in input_info.items()]

        dataset_path='train.tfrecords'
        dataset=tf.data.TFRecordDataset(dataset_path)#[dataset_path1,dataset_path2]
        dataset=dataset.map(_parse_tf_example)
        dataset=dataset.repeat()
        dataset=dataset.batch(1)
        iterator=dataset.make_initializable_iterator()
        data_batch=iterator.get_next()
        keys=list(input_info.keys())
        data_batch=dict([(keys[i],data_batch[i]) for i in range(len(keys))])
        return data_batch,iterator.initializer

def bernoulli_gaussian_trial(L):

    matrix = io.loadmat('para.mat')
	
    A=matrix['cwplt_new']
    M,N = A.shape
    
    A_ = tf.constant(A,name='A',dtype=tf.float32)
    prob = TFGenerator(A=A,L=L,A_=A_)
    prob.name = 'Bernoulli-Gaussian, random A'

    if not os.path.exists(os.path.join(os.getcwd(), 'train.tfrecords')):
        print('preparing training dataset\n')
        f1 = open('prepare_data.txt', 'w')
        f1.close()
        def bytes_feature(value):
          return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        writer=tf.python_io.TFRecordWriter(os.path.join(os.getcwd(),'train.tfrecords'))
        for i in range(500):
          feature={}
          if i%100==0:
              print(i)
              f1 = open('prepare_data.txt', 'a+')
              f1.write('%d\n'%(i))
              f1.close()
          np.random.seed()
          data=io.loadmat('/data/data'+str(i+1))
          prob.xval=np.transpose(data['x'].astype(np.float32))
          prob.yval = data['y'].astype(np.float32)
          feature['y']=bytes_feature(prob.yval.tostring())
          feature['x'] = bytes_feature(prob.xval.tostring())

          example=tf.train.Example(features=tf.train.Features(feature=feature))
          writer.write(example.SerializeToString())
        writer.close()
        with open(os.path.join(os.getcwd(),'train_info.txt'),'w') as dataset_info:
          dataset_info.write('y'+','+str(M)+','+str(L)+'\n')
          dataset_info.write('x' + ','+str(N)+',' + str(L) + '\n')
    if not os.path.exists(os.path.join(os.getcwd(), 'xval.npy')):
        print('preparing validating dataset\n')
        data=io.loadmat('/data/data0')
        prob.xval=np.transpose(data['x'].astype(np.float32))
        prob.yval =data['y'].astype(np.float32)
        np.save('xval.npy', prob.xval)
        np.save('yval.npy', prob.yval)


    data=io.loadmat('/data/data1')
    prob.xinit = np.transpose(data['x'].astype(np.float32))
    prob.yinit = np.transpose(data['y'].astype(np.float32))
    prob.xval = np.transpose(data['x'].astype(np.float32))
    prob.yval = np.transpose(data['y'].astype(np.float32))

    return prob
