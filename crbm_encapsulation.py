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
	W = crbm.W.get_value();	
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

	batchdata1, seqlen1, data_mean1, data_std1 = motion.load_data_ms3(dataset);

	#extracting variables
	Motion = mat_dict['Motion'];	

	#setting CRBM parameters
	L1Config = {};
	L1Config['n_visible'] = Motion[0,0].shape[1];
	L1Config['n_hidden'] = 150;
	L1Config['delay'] = 3;

	L1Config['learning_rate'] = 1e-3;
	L1Config['training_epochs'] = 200;
	L1Config['batch_size'] = 100;

	#train CRBM and sample hidden giving visible
	crbm1, batchdata_l1 = CO.train_crbm(L1Config, batchdata1, seqlen1, data_mean1, data_std1);	


	#save the layer 1 crbm1
	L1Config['data_mean'] = data_mean1;
	L1Config['data_std'] = data_std1;
	L1Config['model'] = {'A':crbm1.A.get_value(), 'B':crbm1.B.get_value(), 'W':crbm1.W.get_value(), 'hbias':crbm1.hbias.get_value(),'vbias':crbm1.vbias.get_value()};
	scipy.io.savemat('crbmconfig_1.mat',{'PyCRBMConfig':L1Config});

	# put together the features for each input sequence
	Feat1 = np.ndarray(shape=(1,9), dtype='O');
	for idx, data in enumerate(Motion[0]):
		Feat1[0,idx] = sample_h_v(data, crbm1, L1Config);


	#setting CRBM parameters
	L2Config = {};
	L2Config['n_visible'] = Feat1[0,0].shape[1];
	L2Config['n_hidden'] = 200;
	L2Config['delay'] = 3;

	L2Config['learning_rate'] = 1e-3;
	L2Config['training_epochs'] = 200;
	L2Config['batch_size'] = 100;


	batchdata2, seqlen2, data_mean2, data_std2 = motion.load_data_crbm(Feat1);


	#train CRBM and sample hidden giving visible
	crbm2, batchdata_l2 = CO.train_crbm(L2Config, batchdata2, seqlen2, data_mean2, data_std2);	
	l2 = sample_h_v(Feat1[0,0], crbm2, L2Config);	

	#save the layer 2 CRBM
	L2Config['data_mean'] = data_mean2;
	L2Config['data_std'] = data_std2;
	L2Config['model'] = {'A':crbm2.A.get_value(), 'B':crbm2.B.get_value(), 'W':crbm2.W.get_value(), 'hbias':crbm2.hbias.get_value(),'vbias':crbm2.vbias.get_value()};
	scipy.io.savemat('crbmconfig_2.mat',{'PyCRBMConfig':L2Config});