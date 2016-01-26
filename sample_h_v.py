import crbm as CO
import pickle
import scipy
import numpy as np

model = scipy.io.loadmat('crbmconfig_sean_100h_3p.mat');

A = model['A'];
B = model['B'];
W = model['W'];
hbias = model['hbias'];
vbias = model['vbias'];

bjstar = np.zeros((150, 52));
delay = 3;

for i in range(0, delay):
	if i == 0:
		bjstar = bjstar + np.dot(B[0:52, :].transpose(), A[0:52, :]);
	else:
		bjstar = bjstar + np.dot(B[i*delay+1:i*delay+52, :].transpose(), A[i*delay+1:i*delay+52, :]);

print bjstar
