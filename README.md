# Globecom2021_Dual-Net-for-Joint-Channel-Estimation-and-Data-Recovery-in-Grant-free-Massive-Access
Code of Dual-Net for Joint Channel Estimation and Data Recovery in Grant-free Massive Access

  
# Folder structure  
Attention:  We upload all trained networks thus you can directly turn to Step 4 to gennerate the same figure in paper. Training your own networks, please follow Step 1-3.  
  
Step 1: generate the training data via matlab using generate_data.m  
Attention:   
1.We trained networks for each parameters (different number of active users/ length of pilot/SNRs)  
2.We generated 500*2000 pairs training data for each network. Increasing the amount of training data may influence the performance.  
  
Step2: train the LISTA/Dual_net for massive access via python using LISTA.py
Environment: Tensorflow 1.12 +Python3.6+GPU  
Attention: The main differnt between LISTA and Dual_net is tools/problems.py and train.py(related to traning data load), networks.py (network structure), shrinkage.py( the activation function)  
  
Step3: load the parameters  of the trained networks  as .mat for test  using loadpara.py
  
Step4: Test the trained network in Matlab  using figure_diff_acc_cs.m  
Attention:  The pilot matrix and user path loss used in Step 1 and Step 4 should keep constant  
  

# Citation
@INPROCEEDINGS{9685696,  
  author={Bai, Yanna and Chen, Wei and Ma, Yuan and Wang, Ning and Ai, Bo},  
  booktitle={2021 IEEE Global Communications Conference (GLOBECOM)},   
  title={Dual-Net for Joint Channel Estimation and Data Recovery in Grant-free Massive Access},   
  year={2021},  
  volume={},  
  number={},  
  pages={1-6},  
  doi={10.1109/GLOBECOM46510.2021.9685696}}    
