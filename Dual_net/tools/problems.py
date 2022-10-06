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
    def __init__(self,A,L,A1,nd,**kwargs):
        self.A = A
        self.L = L
        lp,N = A.shape
        self.A1 = A1
        ls,N = A1.shape
        self.nd = nd
        vars(self).update(kwargs)
        self.x_ = tf.placeholder( tf.float32,(N,L),name='x' )
        self.y_ = tf.placeholder( tf.float32,(lp,L),name='y' )
        self.x1_ = tf.placeholder( tf.float32,(L,nd,N),name='x1' )
        self.y1_ = tf.placeholder( tf.float32,(L,nd,ls),name='y1' )		
		



class TFGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)
    def __call__(self,sess):
        'generates y,x pair for training'
        return sess.run( ( self.ygen_,self.xgen_ ,self.ygen1_,self.xgen1_ ) )
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







def bernoulli_gaussian_trial(L=2000,nd=18):

    matrix = io.loadmat('para.mat')
	
    A=matrix['cwplt_new']
    lp,N = A.shape

    A1=matrix['cwsbl_new']
    ls,N = A1.shape
    

    prob = TFGenerator(A=A,L=L,A1=A1,nd=nd)
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
          prob.xval1=np.transpose(data['x1'].astype(np.float32))
          prob.yval1 = data['y1'].astype(np.float32)         
          
          feature['y']=bytes_feature(prob.yval.tostring())
          feature['x'] = bytes_feature(prob.xval.tostring())
          feature['y1']=bytes_feature(prob.yval.tostring())
          feature['x1'] = bytes_feature(prob.xval.tostring())

          example=tf.train.Example(features=tf.train.Features(feature=feature))
          writer.write(example.SerializeToString())
        writer.close()
        with open(os.path.join(os.getcwd(),'train_info.txt'),'w') as dataset_info:
          dataset_info.write('y'+','+str(lp)+','+str(L)+'\n')
          dataset_info.write('x' + ','+str(N)+',' + str(L) + '\n')
          dataset_info.write('y1'+','+str(L)+','+str(nd)+','+str(N)+'\n')
          dataset_info.write('x1'+','+str(L)+','+str(nd)+','+str(ls)+'\n')   
                    
    if not os.path.exists(os.path.join(os.getcwd(), 'xval.npy')):
        print('preparing validating dataset\n')
        data=io.loadmat('/data/data0')
        prob.xval=np.transpose(data['x'].astype(np.float32))
        prob.yval =data['y'].astype(np.float32)
        np.save('xval.npy', prob.xval)
        np.save('yval.npy', prob.yval)
        prob.xval1=np.transpose(data['x1'].astype(np.float32))
        prob.yval1 =data['y1'].astype(np.float32)
        np.save('xval1.npy', prob.xval)
        np.save('yval1.npy', prob.yval)


    data=io.loadmat('/data/data1')
    prob.xinit = np.transpose(data['x'].astype(np.float32))
    prob.yinit = np.transpose(data['y'].astype(np.float32))
    prob.xval = np.transpose(data['x'].astype(np.float32))
    prob.yval = np.transpose(data['y'].astype(np.float32))
    prob.xinit1 = np.transpose(data['x1'].astype(np.float32))
    prob.yinit1 = np.transpose(data['y1'].astype(np.float32))
    prob.xval1 = np.transpose(data['x1'].astype(np.float32))
    prob.yval1 = np.transpose(data['y1'].astype(np.float32))
    return prob




