"""
Copyright (c) 2021 The University of Texas Southwestern Medical Center.
"""
from functools import partial
import multiprocessing as mp
import os
from typing import List, Tuple
import nibabel

import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial
import tqdm
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker, NiftiMasker
from nilearn.plotting import find_parcellation_cut_coords
from nilearn.image import resample_to_img

from brainconn import modularity

def mean_framewise_displacement(fsl_hmp_path: str) -> float:
    """Computes mean framewise displacement as defined by Power et al., using
    backwards time differences.

    Args: fsl_hmp_path (str): path to FSL-format, tab-delimited head motion
        parameter file

    Returns: float: mean framewise displacement
    """    

    df = pd.read_csv(fsl_hmp_path, header=None, delim_whitespace=True)
    arr2dHmp = df.values[:, :6]
    
    # getting rotational displacement on surface of sphere of radius 50 mm
    arr2dHmp[:, 0:3] = arr2dHmp[:, 0:3] * 50

    # getting differences
    arr2dDiff = np.diff(arr2dHmp, axis=0)
    arr2dDiff = np.insert(arr2dDiff, 0, values=np.zeros(arr2dHmp.shape[1]), axis=0)
    arr2dDiff = np.abs(arr2dDiff)

    # summing up each col to get FD_power
    arr1dPower = np.sum(arr2dDiff, axis=1)

    mFD = np.mean(arr1dPower)
    
    return mFD

class QCTool(object):
    def __init__(self, images: List[str], head_motion_params: List[str], n_procs: int=1):
        """Computes QC-FC, QC-FC distance dependence, modularity quality,
        moduarity quality-motion correlation, and Dice similarity between a
        seed-based connectivity map and a template.

        Args: 
            images (List[str]): list of fMRI NIfTI files 
            head_motion_params (List[str]): list of corresponding FSL-format head motion parameter files 
            n_procs (int, optional): number of parallel processes. Defaults to 1.

        """        
        
        if len(images) != len(head_motion_params):
            raise ValueError('Length of images does not match length of head motion params')
        
        self.lsImages = images
        self.lsHmp = head_motion_params
        self.nProcs = int(n_procs)
        
        self.dfMfd = None
        self.arr3dFC = None
        self.arr2dFCVec = None
        self.strAtlasPath = None
        
        return
    
    def set_atlas(self, atlas_path: str):
        """Set atlas for computing functional connectivity.

        Args:
            atlas_path (str): path to atlas NIfTI file.
        """        
        self.strAtlasPath = atlas_path
        
        # Invalidate any previously computed FC values
        self.arr3dFC = None
        self.arr2dFCVec = None
    
    def get_atlas(self) -> str:
        """Get currently set atlas for computing functional connectivity.

        Returns:
            str: path to atlas
        """        
        if self.strAtlasPath is None:
            raise RuntimeError('Atlas has not be specified. Call .set_atlas() first.')
        else:
            return self.strAtlasPath
    
    def _compute_mfd(self):
        # Computes mean framewise displacment from each image's head motion parameters
        print('Computing mean framewise displacement')
        
        lsIndex = [os.path.basename(x) for x in self.lsImages]
        
        dfMfd = pd.Series(index=lsIndex, name='mFD')
        
        for i, strHmpPath in enumerate(self.lsHmp):
            dfMfd.iloc[i] = mean_framewise_displacement(strHmpPath)
            
        self.dfMfd = dfMfd
        return
    
    def _compute_fc(self) -> np.ndarray:
        # Compute functional connectivity using a given atlas
        print('Computing functional connectivity with the following atlas:', self.get_atlas(), flush=True)
        
        masker = NiftiLabelsMasker(labels_img=self.get_atlas())
        
        if self.nProcs > 1:
            with mp.Pool(self.nProcs) as pool:
                lsTimeseries = list(tqdm.tqdm(pool.imap(masker.fit_transform, self.lsImages), total=len(self.lsImages)))
                
        else:
            lsTimeseries = []
            for x in tqdm.tqdm(self.lsImages):
                lsTimeseries += [masker.fit_transform(x)]
        
        connectivity = ConnectivityMeasure(kind='correlation')
        self.arr3dFC = connectivity.fit_transform(lsTimeseries)
                
        connectivityVec = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
        self.arr2dFCVec = connectivityVec.fit_transform(lsTimeseries)
        
        return 
    
    def _compute_edge_distances(self) -> np.ndarray:
        # Compute physical Euclidean distances of each ROI-ROI connection
        arr2dCoords, lsLabels = find_parcellation_cut_coords(self.get_atlas(), return_label_names=True)
        
        nLabels = len(lsLabels)
        arr2dDistances = np.zeros((nLabels, nLabels))
        
        for i in range(nLabels):
            for j in range(nLabels):
                    arr2dDistances[i, j] = np.sqrt(np.sum((arr2dCoords[i, :] - arr2dCoords[j, :])**2))
                    
        return arr2dDistances
        
    
    def compute_qcfc(self) -> Tuple[pd.DataFrame, dict]:
        """Compute QC-FC and QC-FC distance dependence metrics.

        QC-FC is defined as the correlation between connectivity strength and
        mean framewise displacement (mFD) and is computed for each edge in the
        functional connectivity matrix.

        QC-FC distance dependence is the correlation between the QC-FC of each
        edge and its physical (Euclidean) length in the brain.

        Returns: 
            pd.DataFrame: dataframe containing the QC-FC value for each edge in the
            functional connectivity matrix. Columns are 'QC-FC' containing the
            Pearson's correlation and 'p_value" containing the two-sided p-value.
            
            dict: dictionary containing Spearman and Pearson correlations and 
            p-values for the QC-FC distance dependence metric
        """        
        
        # Compute mFD and FC if not done already
        if self.dfMfd is None:
            self._compute_mfd()
    
        if self.arr2dFCVec is None: 
            self._compute_fc() # n_images x n_edges
        
        nEdges = self.arr2dFCVec.shape[1]      
        
        arr1dCorr = np.zeros((nEdges,))
        arr1dPVal = np.zeros((nEdges,))
        
        for iEdge in range(nEdges):
            r, p = scipy.stats.pearsonr(self.arr2dFCVec[:, iEdge], self.dfMfd.values)
            arr1dCorr[iEdge] = r
            arr1dPVal[iEdge] = p
            
        dfQcFc = pd.DataFrame()
        dfQcFc['QC-FC'] = arr1dCorr
        dfQcFc['p_value'] = arr1dPVal
        
        arr2dDistances = self._compute_edge_distances()
        arr1dDistances = arr2dDistances[np.tril_indices_from(arr2dDistances, k=-1)]
        
        rSpearman, pSpearman = scipy.stats.spearmanr(arr1dDistances, dfQcFc['QC-FC'].values)
        rPearson, pPearson = scipy.stats.pearsonr(arr1dDistances, dfQcFc['QC-FC'].values)
        
        dictQcFcDD = {'Spearman_r': rSpearman,
                      'Spearman_p': pSpearman,
                      'Pearson_r': rPearson,
                      'Pearson_p': pPearson}
                
        return dfQcFc, dictQcFcDD

    def compute_modularity(self, random_seed=0) -> Tuple[pd.DataFrame, dict]:
        """Compute the modularity quality and modularity quality-motion correlation metrics.

        Args:
            random_seed (int, optional): Random seed for the Louvain algorithm. Defaults to 0.

        Returns:
            pd.Series: series containing the modularity Q value for each image, which is 
            computed from their respective functional connectivity matrices
            
            dict: dictionary containing Spearman and Pearson correlations and 
            p-values for the modularity quality-motion correlation metric
        """        
            
        # Compute mFD and FC if not done already
        if self.dfMfd is None:
            self._compute_mfd()
            
        if self.arr3dFC is None:
            self._compute_fc()
        
        lsIndex = [os.path.basename(x) for x in self.lsImages]
        dfModularity = pd.Series(index=lsIndex, name='modularity')
        
        # Compute modularity (Q) using Louvain algorithm    
        for idx in range(self.arr3dFC.shape[0]):
            _, Q = modularity.modularity_louvain_und_sign(self.arr3dFC[idx,], seed=random_seed)
            dfModularity.iloc[idx] = Q
            
        # Compute Q-mFD correlation
        rSpearman, pSpearman = scipy.stats.spearmanr(self.dfMfd, dfModularity.values)
        rPearson, pPearson = scipy.stats.pearsonr(self.dfMfd, dfModularity.values)

        dictQFDCorr = {'Spearman_r': rSpearman,
                       'Spearman_p': pSpearman,
                       'Pearson_r': rPearson,
                       'Pearson_p': pPearson}

        return dfModularity, dictQFDCorr
    
    @staticmethod
    def compute_sbc_map(image: str, seed: Tuple[int]=(4, -54, 26), radius: int=6, 
                        mask_strategy: str='template') -> nibabel.Nifti1Image:
        """Compute seed-based connectivity map.

        Args:
            image (str): path to fMRI image
            seed (Tuple[int], optional): MNI coordinates of seed. Defaults to (4, -54, 26).
            radius (int, optional): radius in mm of sphere around the seed. Defaults to 6.
            mask_strategy (str, optional): strategy for masking brain voxels in the fMRI (see 
            nilearn.input_data.NiftiMasker for details). Defaults to 'template'.

        Returns:
            nibabel.Nifti1Image: seed-based connectivity map
        """        
        
        # Compute seed mean timeseries
        seedMasker = NiftiSpheresMasker([seed], radius=radius, detrend=False, standardize=True)
        arrSeedTimeseries = seedMasker.fit_transform(image)

        # Compute brain voxel timeseries
        brainMasker = NiftiMasker(detrend=False, standardize=True, mask_strategy=mask_strategy)
        arrBrainTimeseries = brainMasker.fit_transform(image)
        
        # Compute correlations
        arr2dCorr = np.dot(arrBrainTimeseries.T, arrSeedTimeseries) / arrSeedTimeseries.shape[0]
        # Convert to Fisher z-transformed connectivity
        arr2dCorrZ = np.arctanh(arr2dCorr)
        # Clip negative values
        arr2dCorrZ[arr2dCorrZ < 0] = 0
        
        # Transform back into image
        imgCorrZ = brainMasker.inverse_transform(arr2dCorrZ.T)
        
        return imgCorrZ
    
    @staticmethod
    def _compute_sbc_template_dice_single(image: str, template: str, 
                                          seed: Tuple[int]=(4, -54, 26), radius: int=6, 
                                          threshold: float=0.4,
                                          mask_strategy: str='template') -> float:
        # Wrapper function for parallelization
        imgSBC = QCTool.compute_sbc_map(image, seed=seed, radius=radius, mask_strategy=mask_strategy)
        imgTemplate = resample_to_img(template, image, interpolation='nearest')
        dice = 1 - scipy.spatial.distance.dice(imgTemplate.get_data().flatten(),
                                               imgSBC.get_data().flatten() >= threshold)
        return dice

    def compute_sbc_template_dice(self, template: str, 
                                  seed: Tuple[int]=(4, -54, 26), 
                                  radius: int=6, 
                                  threshold: float=0.4, 
                                  mask_strategy: str='template') -> pd.Series:
        """Compute Dice similarity between seed-based connectivity maps derived from each image and a resting-state network template.

        Args:
            template (str): path to template
            seed (Tuple[int], optional): MNI coordinates of seed. Defaults to (4, -54, 26).
            radius (int, optional): radius in mm of sphere around the seed. Defaults to 6.
            threshold (float, optional): [description]. Defaults to 0.4.
            mask_strategy (str, optional): strategy for masking brain voxels in the fMRI (see 
            nilearn.input_data.NiftiMasker for details). Defaults to 'template'.

        Returns:
            pd.Series: series containing Dice similarity for each image
        """        
        
        fn = partial(self._compute_sbc_template_dice_single, 
                     template=template, 
                     seed=seed, 
                     radius=radius, 
                     threshold=threshold, 
                     mask_strategy=mask_strategy)
        
        # Compute SBC maps with parallelization
        if self.nProcs > 1:
            with mp.Pool(self.nProcs) as pool:
                lsDice = list(tqdm.tqdm(pool.imap(fn, self.lsImages), total=len(self.lsImages)))
        else:
            lsDice = []
            for x in tqdm.tqdm(self.lsImages):
                lsDice += [fn(x)]
            
        lsIndex = [os.path.basename(x) for x in self.lsImages]
        dfDice = pd.Series(index=lsIndex, data=lsDice, name='sbc_dice')
        
        return dfDice
    
    