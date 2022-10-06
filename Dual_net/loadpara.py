# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:59:22 2020

@author: lenovo
"""


import numpy as np
import scipy.io 

para=np.load('LISTA_bg_giid.npz')

namelist=para.files  # name of variables

dic={}

for k,d in para.items():
    dic[k[:-2]]=d

scipy.io.savemat('dnetpara4.mat',dic)    
        