close all; clear all;

data_path='/space_lin1/hcp';

fstem={
'2_fsaverage_tfMRI_EMOTION_LR';
};




TR=0.72; %second

n_dummy=0;
flag_gavg=0;

subject='';
d=textread('subject_list_all.txt');
for d_idx=1:length(d)
        subject{d_idx}=num2str(d(d_idx));
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for f_idx=1:length(fstem)
    valid_subj_idx=[];
    for d_idx=1:length(subject)
	fprintf('[%s]...(%04d|%04d)....\r',subject{d_idx},d_idx,length(subject));

	roi=[];
	STC=[];
    	for hemi_idx=1:2
        	switch hemi_idx
            	case 1
                	hemi_str='lh';
             	case 2
                	hemi_str='rh';
        	end;

            	fn=sprintf('%s/%s/analysis/%s_%s-%s.stc',data_path,subject{d_idx},subject{d_idx},fstem{f_idx},hemi_str);
		if(exist(fn))
            	[stc{hemi_idx},v{hemi_idx},d0,d1,timeVec]=inverse_read_stc(fn);

		%remove dummy scans 
		stc{hemi_idx}(:,1:n_dummy)=[];
		stc{hemi_idx}(:,end-n_dummy+1:end)=[];

		STC=cat(1,STC,stc{hemi_idx});
		flag_fe=1;
		else
		flag_fe=0;
		end;
	end;
	if(flag_fe) 
	
	valid_subj_idx=cat(1,valid_subj_idx,d_idx);

        fn=sprintf('%s/analysis/%s_regressors_emotion.mat',data_path,subject{d_idx});
	if(exist(fn))
		D_reg=[];
		load(fn);
		D_reg(:,1)=regressor_ventricle(1:end-1);
		D_reg(:,2)=regressor_wm(1:end-1);
		D_reg(1:n_dummy,:)=[];
		D_reg(end-n_dummy+1:end,:)=[];
	else
		D_reg=[];
	end;

	%remove global mean
	D=ones(size(STC,2),1);
	if(~isempty(D_reg))
		D=cat(2,D,D_reg);
	end;
	if(flag_gavg);
		D=cat(2,D,mean(STC,1)');
	end;
	STC=(STC'-D*(inv(D'*D)*D'*STC')).';

	%%% do your analysis for each subject here ....








	%%% end of subject-wise analysis.....
	end;
    end;

    fprintf('\n');
end;
