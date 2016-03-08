import crbm as CO
import pickle
import scipy
import numpy as np
import motion
import theano
import theano.tensor as T

def sample_h_v(batchdata, crbm, config):
	#n_hidden = 150
	#n_visible = 52

	B = crbm.B.get_value();
	hbias = crbm.hbias.get_value();
	delay = config['delay'];
	n_hidden = config['n_hidden'];
	n_visible = config['n_visible'];

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
	mat_dict = scipy.io.loadmat(dataset);

	batchdata1, seqlen, data_mean, data_std = motion.load_data_ms3(dataset);

	#extracting variables
	Motion = mat_dict['Motion'];
	data = Motion[0,0];

	#setting CRBM parameters
	L1Config = {};
	L1Config['n_visible'] = data.shape[1];
	L1Config['n_hidden'] = 150;
	L1Config['delay'] = 3;

	L1Config['learning_rate'] = 1e-3;
	L1Config['training_epochs'] = 200;
	L1Config['batch_size'] = 100;


	#train CRBM and sample hidden giving visible
	crbm, batchdata_l1 = CO.train_crbm(L1Config, batchdata1, seqlen, data_mean, data_std);
	p = sample_h_v(data, crbm, L1Config);




