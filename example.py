'''
Computation of motion artifaction correction metrics:

1. QC-FC correlation
2. QC-FC distance dependence
3. Modularity quality
4. Modularity quality-motion correlation
5. Recovery of canonical resting state networks (RSNs) via seed-based
   connectivity 

For computation of RSN recovery via Group ICA, we refer users to the [GIFT
toolbox](https://trendscenter.org/software/gift/).
'''

# User-specific paths to functional brain atlas and DMN template (binary mask).
# The functional brain atlas is used to compute functional connectivity for the
# QC-FC, QC-FC distance dependence, modularity quality, and modularity
# quality-mFD correlation metrics. 
ATLAS = '/project/bioinformatics/DLLab/Kevin/MotionCorrection/MotionCorrectionCode/atlases/Parcels/Parcels_MNI_333.nii'
# This particular DMN template was downloaded from
# https://brainnexus.com/resting-state-fmri-templates/. 
DMN = '/project/bioinformatics/DLLab/shared/Atlases/brainnexus templates/rsfmrinetwork_default.nii.gz'

import os
import glob
import nilearn.plotting

import seaborn as sns
sns.set_style('whitegrid')
from qctool import QCTool

# 1. Create lists of fMRI files and corresponding head motion parameter files.
#    We used FSL MCFLIRT to compute these head motion parameters. 

# # !!! Example only: !!!

# PROCESSING_DIR = '/project/bioinformatics/DLLab/Vyom/Pipelines/ABIDE1/NYU_041620_ExpandedPipelines'

# lsImages = glob.glob(os.path.join(PROCESSING_DIR, 'sub*', 'motion', '*hmpSaromaSphysioSfreq_bold.nii.gz'))
# lsImages.sort()

# lsHmp = glob.glob(os.path.join(PROCESSING_DIR, 'sub*', 'motion', 'hmp_exp.csv'))
# lsHmp.sort()

# #

# 2. Initialize QCTool with the lists of fMRI and head motion parameter files.
#    Processing can be parallelized using the n_procs argument.

qc = QCTool(lsImages, lsHmp, n_procs=4)
qc.set_atlas(ATLAS)

# 3. Compute QC-FC and QC-FC distance dependence metrics.
dfQcFc, dictQcFcDD = qc.compute_qcfc()
sns.histplot(qc.dfMfd) # mean framewise displacement
sns.histplot(dfQcFc['QC-FC'])

print('QC-FC distance dependence:')
print(dictQcFcDD)

# 4. Compute network modularity quality (using the Louvain algorithm) and
#    modularity quality-mFD correlation.

dfModularity, dictModularityFD = qc.compute_modularity()
sns.histplot(dfModularity)

print('Modularity-mFD correlation')
print(dictModularityFD)

# 5. Compute seed-based connectivity maps using a posterior cingulate seed (4,
#    -54, 26), then binarize at a Z-threshold of 0.4 and compute Dice similarity
#    with the DMN template.

dfSBCDice = qc.compute_sbc_template_dice(DMN, seed=(4, -54, 26), threshold=0.4)
sns.histplot(dfSBCDice)