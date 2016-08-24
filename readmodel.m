%%
load 'crbmconfig_1.mat'

%%

CRBMConfig.numdims = PyCRBMConfig.n_visible;
CRBMConfig.numhid = PyCRBMConfig.n_hidden;
CRBMConfig.numepochs = PyCRBMConfig.training_epochs;
CRBMConfig.gsd = 1;
CRBMConfig.order = PyCRBMConfig.delay;
CRBMConfig.data_mean = PyCRBMConfig.data_mean;
CRBMConfig.data_std = PyCRBMConfig.dat_mean;
CRBMConfig.model.w = W';
CRBMConfig.model.A = reshape(A', 52,52,3);
CRBMConfig.model.B = reshape(B', 100,52,3);
CRBMConfig.model.bj = hbias';
CRBMConfig.model.bi = vbias';


%%

GenLog = gen_crbm(CRBMConfig,MotionData,200,10);

imagesc(GenLog.hidden{1}'); colormap gray;
GenLog = Post_AddEmpty(GenLog, MotionData,0);
GenLog.ppostvis = GenLog.postvis;
GenLog = postprocess2(GenLog, MotionData);

figure(2); expPlayData(skel, GenLog.postvis, 1/30)