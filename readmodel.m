load('crbmconfig_sean_100h_3p.mat')

% for now, just using another CRBM as a template, and to fill in the mean, std, and other variables that we didn't read from python
load('av_feat_crbm_4layer_exp17_sean_lay1_100hid_6ord_10cd_gaussian_crbm_ep100.mat')

CRBMConfig_p = CRBMConfig;
CRBMConfig_p.model.w = W';
CRBMConfig_p.model.A = reshape(A', 52,52,3);
CRBMConfig_p.model.B = reshape(B', 100,52,3);
CRBMConfig_p.model.bj = hbias';
CRBMConfig_p.model.bi = vbias';



GenLog = gen_crbm(CRBMConfig_p,MotionData,200,10);

imagesc(GenLog.hidden{1}'); colormap gray;
GenLog = Post_AddEmpty(GenLog, MotionData,0);
GenLog.ppostvis = GenLog.postvis;
GenLog = postprocess2(GenLog, MotionData);

figure(2); expPlayData(skel, GenLog.postvis, 1/30)