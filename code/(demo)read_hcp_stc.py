import pandas as pd
import numpy as np
import os
import mne

data_path = '/space_lin1/hcp'

fstem = ['2_fsaverage_tfMRI_EMOTION_LR']

TR = 0.72
n_dummy = 0
flag_gavg = 0


subject = pd.read_table('subject_list_all.txt', sep = '\n', header=None).iloc[:,0].tolist()
for i in range(len(subject)):
    subject[i] = str(subject[i])

for f_idx in range(len(fstem)):
    valid_subj_idx = []
    for d_idx in range(len(subject)):
        print('[{subject}]...({d_idx}|{length})....\r'.format(subject=subject[d_idx], d_idx=d_idx, length=len(subject)))
        roi = []
        STC = []

        stc = {
            "lh": [],
            "rh": []
        }

        v = {
            "lh": [],
            "rh": []
        }

        for hemi_idx in range(2):
            if hemi_idx == 0:
                hemi_str = 'lh'
            elif hemi_idx == 1:
                hemi_str = 'rh'

            fn = data_path + "/" + subject[d_idx] + "/analysis/" + subject[d_idx] + "_" + fstem[f_idx] + "-" + hemi_str + ".stc"
            if os.path.exists(fn):
                stc_info = mne.read_source_estimate(fn)
                stc[hemi_idx] = stc_info.data
                v[hemi_idx] = stc_info.vertices
                d0 = stc_info.tmin
                d1 = stc_info.tstep
                timeVec = stc_info.times

                stc[hemi_idx][:, 0:n_dummy] = []
                stc[hemi_idx][:, -1-n_dummy:-1] = []

                STC.append(stc[hemi_idx])
                flag_fe = 1
            else:
                flag_fe = 0

        if flag_fe:
            valid_subj_idx.append(d_idx)

    print("number of valid_subj_idx in " + fstem[f_idx], ":" + str(len(valid_subj_idx)))

