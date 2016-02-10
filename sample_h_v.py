import crbm as CO
import pickle
import scipy
import numpy as np
import motion
import theano
import theano.tensor as T

model = scipy.io.loadmat('crbmconfig_sean_100h_3p.mat');

mat_dict = scipy.io.loadmat('Motion-Sean-Pre1.mat')
Motion = mat_dict['Motion']

#batchdata, seqlen, data_mean, data_std = motion.preprocess_data_ms3(Motion);
#print Motion[0,0];
batchdata = Motion[0,0];
#batchdata, seqlen, data_mean, data_std = motion.load_data_ms3('Motion-Sean-Pre1.mat');
#print batchdata[0];
#orig_data = np.asarray(batchdata.get_value(borrow=True), dtype=theano.config.floatX);
#print orig_data.shape;

A = model['A'];
B = model['B'];
W = model['W'];
hbias = model['hbias'];
vbias = model['vbias'];

bjstar = np.zeros((150,));
delay = 3;
b = np.zeros((batchdata.shape[0]-delay,150));

for j in range(delay, batchdata.shape[0]): 
	bjstar = np.zeros((150,));
	for i in range(0, 150):
		for k in range(0, delay):
			bjstar[i] =  bjstar[i] + batchdata[j-k-1,:].dot(B[k*52:k*52+52, i]);


	b[j-delay] = hbias + bjstar.transpose();



bottomup = np.dot(batchdata[delay:,:], W);

p = np.divide(1 , (1 + np.exp(-b - bottomup)));
print p[0:5, 0:5];
scipy.io.savemat('python_feat1.mat',{'python_feat1':p, 'python_b':b, 'python_bottomup':bottomup});




