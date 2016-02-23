import crbm as CO
import pickle
import scipy
import numpy as np
import motion
import theano
import theano.tensor as T

def sample_h_v(batchdata, B, hbias, delay, n_hidden, n_visible):
	#n_hidden = 150
	#n_visible = 52

	bjstar = np.zeros((n_hidden,));
	b = np.zeros((batchdata.shape[0]-delay,n_hidden));

	for j in range(delay, batchdata.shape[0]): 
		bjstar = np.zeros((n_hidden,));
		for i in range(0, n_hidden):
			for k in range(0, delay):
				bjstar[i] =  bjstar[i] + batchdata[j-k-1,:].dot(B[k*n_visible:k*n_visible+n_visible, i]);


		b[j-delay] = hbias + bjstar.transpose();

	bottomup = np.dot(batchdata[delay:,:], W);

	p = np.divide(1 , (1 + np.exp(-b - bottomup)));
	return p;


if __name__ == '__main__':
	#loading datasets
	dataset = 'Motion-Sean-Pre1.mat';
	mat_dict = scipy.io.loadmat('Motion-Sean-Pre1.mat');
	model = scipy.io.loadmat('crbmconfig_sean_100h_3p.mat');

	#extracting variables
	Motion = mat_dict['Motion'];
	data = Motion[0,0];

	A = model['A'];
	B = model['B'];
	W = model['W'];
	hbias = model['hbias'];
	vbias = model['vbias'];

	#setting CRBM parameters
	n_visible = data.shape[1];
	n_hidden = B.shape[1];
	delay = B.shape[0]/n_visible;

	learning_rate = 1e-3;
	training_epoch = 200;
	batchsize = 100;

	#train CRBM and sample hidden giving visible
	crbm, batchdata = CO.train_crbm(learning_rate, training_epoch, dataset, n_visible, batchsize, n_hidden, delay);
	p = sample_h_v(data, B, hbias, delay, n_hidden, n_visible);




