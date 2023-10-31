'''
  function written by Alexander Schmidt, based on the paper "Sheintuch et al., ...", allowing 
  for complete registration

  TODO:
    * some issues with matching: e.g. cluster #2 and #1099 are almost the same (check 555wt
      vs 555wt_new), after two strongly overlapping neurons appear in one or two  - check 
      this and find solution!
    * write plotting procedure for cluster footprints (3D), to allow manual corrections
    * clean up plotting procedure and write separate plotting functions that can be used 
    to create subplots and show different aspects of results & method
    * check whether some procedures can be further outsourced to other files
    * save data-attributee / structure after model-building, not only after registration
  
  
  last updated on October 31st, 2023
'''


import os, cv2, copy, math, time, logging
from tqdm import *
from itertools import chain
import numpy as np

import scipy as sp
from scipy.io import loadmat
from scipy.optimize import curve_fit, linear_sum_assignment

import matplotlib.pyplot as plt
from matplotlib import rc
# from matplotlib import animation
# from matplotlib import cm
import matplotlib.colors as mcolors

from matplotlib.widgets import Slider


from .utils import pickleData, center_of_mass, calculate_img_correlation, get_shift_and_flow, fun_wrapper, load_dict_from_hdf5, normalize_sparse_array
from .parameters import matchingParams
from .fit_functions import functions

logging.basicConfig(level=logging.INFO)

## create paths that should be processed


class matching:

    def __init__(self,params=matchingParams,logLevel=logging.ERROR):

        self.log = logging.getLogger("matchinglogger")
        self.log.setLevel(logLevel)

        self.para = params

        self.update_bins(self.para['nbins'])

        ## initialize the main data dictionaries
        self.data_blueprint = {
            'nA':       np.zeros(2,'int'),
            'cm':       None,
            'p_same':   None,
            'idx_eval': None,
        }

        self.data = {}
        self.data_tmp = {}

        self.data_cross = {
            'D_ROIs':   [],
            'fp_corr':  [],
        }

        neighbor_dictionary = {
            'NN':    [],
            'nNN':   [],
            'all':   []
          }
        
        self.status = {
            'counts_calculated': False,
            'model_calculated': False,

            'neurons_matched': False,
        }

        self.model = {
            'counts':                 np.zeros((self.para['nbins'],self.para['nbins'],3)),
            'counts_unshifted':       np.zeros((self.para['nbins'],self.para['nbins'],3)),
            'counts_same':            np.zeros((self.para['nbins'],self.para['nbins'])),
            'counts_same_unshifted':  np.zeros((self.para['nbins'],self.para['nbins'])),
            
            'fit_function': {
               'single': {
                  'distance':         copy.deepcopy(neighbor_dictionary),
                  'correlation':      copy.deepcopy(neighbor_dictionary)
               },
               'joint': {
                  'distance':         copy.deepcopy(neighbor_dictionary),
                  'correlation':      copy.deepcopy(neighbor_dictionary)
               }
              },
            
            'fit_parameter': {
               'single': {
                  'distance':       copy.deepcopy(neighbor_dictionary),
                  'correlation': copy.deepcopy(neighbor_dictionary)
                  },
                'joint': {}
              },
            
            'pdf': {
               'single': {
                  'distance':       [],
                  'correlation': []
                  },
                'joint': []
              },
            
            'p_same': {
               'single': {},
                'joint': []
              },
            
            'f_same':         False,
            'kernel': {
                'idxes':   {},
                'kde':     {}
              },
          }

        # ## defines the functions to fit to the count histograms for calculation of p_same
        # self.model['fit_functions'] = {
           
        #     ## for the single model
        #     'single': {
        #         'distance': {
        #           'NN':   'lognorm',
        #           'nNN':  'linear_sigmoid',
        #         },
        #         'correlation': {
        #           'NN':   'lognorm_reverse',
        #           'nNN':  ['gauss','beta'], 
        #         }
        #       },

        #     ## for the joint model
        #     'joint': {
        #         'distance': {
        #           'NN':   'lognorm',
        #           'nNN':  'gauss'
        #         },
        #         'correlation': {
        #           'NN':   'lognorm_reverse',
        #           'nNN':  'gauss'
        #         }
        #       }
        #   }


    
    def update_bins(self,nbins):
        
        self.para['nbins'] = nbins
        
        ## create value arrays for distance and footprint correlation
        self.para['arrays'] = {}
        self.para['arrays']['distance_bounds'] = np.linspace(0,self.para['neighbor_distance'],nbins+1)#[:-1]
        self.para['arrays']['correlation_bounds'] = np.linspace(0,1,nbins+1)#[:-1]

        self.para['arrays']['distance_step'] = self.para['neighbor_distance']/nbins
        self.para['arrays']['correlation_step'] = 1./nbins


        self.para['arrays']['distance'] = self.para['arrays']['distance_bounds'][:-1] + self.para['arrays']['distance_step']/2
        self.para['arrays']['correlation'] = self.para['arrays']['correlation_bounds'][:-1] + self.para['arrays']['correlation_step']/2


        ## update histogram counts (if nbins_new < nbins), merge counts, otherwise split evenly
        self.log.warning('implement method to update count histograms!')


    
    def run_matching(self,p_thr=0.5):
        print('Now running matching procedue in %s'%self.para['pathData'])

        print('Building model for matching ...')
        self.build_model(save_results=True)
        
        print('Matching neurons ...')
        self.register_neurons(save_results=True,p_thr=p_thr)
        
        print('Done!')


    def build_model(self,save_results=False,suffix=''):

        '''
          Iterate through all sessions in chronological order to build a model for neuron matching.
          The model assumes that the closest neuron to a neuron in the reference session ("nearest neighbor", NN) has a high probability of being the same neuron. Neuron distance and footprint correlation are calculated and registered distinctly for NN and "non-nearest neighbours" (nNN, within the viscinity of each neuron) in a histogram, which is then used to build a 2D model of matching probability, assuming a characteristic function for NN and nNN.

          This function takes several steps applied to each session:
            - align data to reference session (rigid and non-rigid shift), or skip if not possible
            - remove highly correlated footprints within one session
            - calculate kernel density of neuron locations (if requested by "use_kde")
            - update count histogram
          Finally, the count histogram is used to create the model

        '''

        self.nS = len(self.para['paths'])
        self.progress = tqdm(enumerate(self.para['paths']),total=self.nS)

        s_ref = 0
        for (s,self.currentPath) in self.progress:
            self.data[s] = copy.deepcopy(self.data_blueprint)
            self.data[s].filePath = self.currentPath

            self.A = self.load_footprints(s,store_data=True)

            if not (self.A is None):
                
                self.progress.set_description('Aligning data from %s'%self.currentPath)
                
                ## prepare (and align) footprints
                if not self.prepare_footprints(align_to_reference=s>0): continue
                
                ## calculating various statistics
                self.data[s]['idx_eval'] &= (np.diff(self.A.indptr) != 0) ## finding non-empty rows in sparse array (https://mike.place/2015/sparse/)
                self.data[s]['nA'][0] = self.A.shape[1]    # number of neurons
                self.data[s]['cm'] = center_of_mass(self.A,self.para['dims'][0],self.para['dims'][1],convert=self.para['pxtomu'])

                self.progress.set_description('Calculate self-referencing statistics for %s'%self.currentPath)

                ## find and mark potential duplicates of neuron footprints in session
                self.data_cross['D_ROIs'],self.data_cross['fp_corr'] = self.calculate_statistics(s,s,self.A)
                self.data[s]['nA'][1] = self.data[s]['idx_eval'].sum()

                if self.para['use_kde']:
                    # self.progress.set_description('Calculate kernel density for Session %d'%s0)
                    self.position_kde(s)         # build kernel

                self.update_joint_model(s,s)

                if s>0:
                    self.progress.set_description('Calculate cross-statistics for %s'%self.currentPath)
                    self.data_cross['D_ROIs'],self.data_cross['fp_corr'] = self.calculate_statistics(s,s_ref,self.A_ref)       # calculating distances and footprint correlations
                    self.progress.set_description('Update model with data from %s'%self.currentPath)
                    self.update_joint_model(s,s_ref)

                self.A_ref = self.A.copy()
                s_ref = s

        self.fit_model()
        
        if save_results:
            self.save_model(suffix=suffix)


    def register_neurons(self, p_thr=0.5, save_results=False, save_suffix='',model='shifted'):
        
        '''
          This function iterates through sessions chronologically to create a set of all detected neurons, matched to one another based on the created model

        '''

        self.para['model'] = model
        
        self.nS = len(self.para['paths'])
        self.progress = tqdm(zip(range(1,self.nS),self.para['paths'][1:]),total=self.nS,leave=True)
        self.currentPath = self.para['paths'][0]

        ## load and prepare first set of footprints
        self.A = self.load_footprints(0)
        self.prepare_footprints(align_to_reference=False)
        self.A_ref = self.A[:,self.data[0]['idx_eval']]
        self.A0 = self.A.copy()   ## store initial footprints for common reference
        
        ## initialize reference session, containing the union of all neurons
        self.data['joint'] = copy.deepcopy(self.data_blueprint)
        self.data['joint']['nA'][0] = self.A_ref.shape[1]
        self.data['joint']['idx_eval'] = np.ones(self.data['joint']['nA'][0],'bool')
        self.data['joint']['cm'] = center_of_mass(self.A_ref,self.para['dims'][0],self.para['dims'][1],convert=self.para['pxtomu'])

        ## prepare and initialize assignment- and p_matched-arrays for storing results
        self.results = {
          'assignments': np.zeros((self.data[0]['nA'][1],self.nS))*np.NaN,
          'p_matched': np.zeros((self.data[0]['nA'][1],self.nS))*np.NaN
        }
        self.results['assignments'][:,0] = np.where(self.data[0]['idx_eval'])[0]
        # self.results['p_matched'][:,0] = np.NaN

        for (s,self.currentPath) in self.progress:
            self.A = self.load_footprints(s)

            if not (self.A is None):
                
                self.progress.set_description('A union size: %d, Preparing footprints from Session #%d'%(self.data['joint']['nA'][0],s))

                ## preparing data of current session
                if not self.prepare_footprints(A_ref=self.A0): continue

                self.data[s]['cm'] = center_of_mass(self.A,self.para['dims'][0],self.para['dims'][1],convert=self.para['pxtomu'])
                

                ## calculate matching probability between each pair of neurons 
                self.progress.set_description('A union size: %d, Calculate statistics for Session #%d'%(self.data['joint']['nA'][0],s))
                self.data_cross['D_ROIs'],self.data_cross['fp_corr'] = self.calculate_statistics(s,'joint',self.A_ref)
                self.data[s]['p_same'] = self.calculate_p(s,'joint')


                ## run hungarian algorithm (HA) with (1-p_same) as score
                self.progress.set_description('A union size: %d, Perform Hungarian matching on Session #%d'%(self.data['joint']['nA'][0],s))
                matches = linear_sum_assignment(1 - self.data[s]['p_same'].toarray())
                p_matched = self.data[s]['p_same'].toarray()[matches]
                

                idx_TP = np.where(p_matched > p_thr)[0] ## thresholding results (HA matches all pairs, but we only want matches above p_thr)
                if len(idx_TP) > 0:
                    matched_ref = matches[0][idx_TP]    # matched neurons in s_ref
                    matched = matches[1][idx_TP]        # matched neurons in s

                    ## find neurons which were not matched in current and reference session
                    non_matched_ref = np.setdiff1d(list(range(self.data['joint']['nA'][0])), matched_ref)
                    non_matched = np.setdiff1d(list(np.where(self.data[s]['idx_eval'])[0]), matches[1][idx_TP])
                    non_matched = non_matched[self.data[s]['idx_eval'][non_matched]]

                    ## calculate number of matches found
                    TP = np.sum(p_matched > p_thr).astype('float32')

                self.A_ref = self.A_ref.tolil()
                
                ## update footprint shapes of matched neurons with A_ref = (1-p/2)*A_ref + p/2*A to maintain part or all of original shape, depending on p_matched
                self.A_ref[:,matched_ref] = self.A_ref[:,matched_ref].multiply(1-p_matched[idx_TP]/2) + self.A[:,matched].multiply(p_matched[idx_TP]/2)
                
                ## append new neuron footprints to union
                self.A_ref = sp.sparse.hstack([self.A_ref, self.A[:,non_matched]]).asformat('csc')

                ## update union data
                self.data['joint']['nA'][0] = self.A_ref.shape[1]
                self.data['joint']['idx_eval'] = np.ones(self.data['joint']['nA'][0],'bool')
                self.data['joint']['cm'] = center_of_mass(self.A_ref,self.para['dims'][0],self.para['dims'][1],convert=self.para['pxtomu'])

                
                ## write neuron indices of neurons from this session
                self.results['assignments'][matched_ref,s] = matched   # ... matched neurons are added

                ## ... and non-matched (new) neurons are appended 
                N_add = len(non_matched)
                match_add = np.zeros((N_add,self.nS))*np.NaN
                match_add[:,s] = non_matched
                self.results['assignments'] = np.vstack([self.results['assignments'],match_add])

                ## write match probabilities to matched neurons and reshape array to new neuron number
                self.results['p_matched'][matched_ref,s] = p_matched[idx_TP]
                p_same_add = np.zeros((N_add,self.nS))*np.NaN
                self.results['p_matched'] = np.vstack([self.results['p_matched'],p_same_add])


        ## some post-processing to create cluster-structures values / statistics
        self.results['cm'] = np.zeros(self.results['assignments'].shape + (2,)) * np.NaN
        for key in ['SNR_comp','r_values','cnn_preds']:
           self.results[key] = np.zeros_like(self.results['assignments'])
        
        for s in range(self.nS):
          idx_c = np.where(~np.isnan(self.results['assignments'][:,s]))[0]
          idx_n = self.results['assignments'][idx_c,s].astype('int')

          for key in ['cm','SNR_comp','r_values','cnn_preds']:
            self.results[key][idx_c,s,...] = self.data[s][key][idx_n,...]

        # finally, save results
        if save_results:  
            self.save_registration(suffix=save_suffix)
    

    def calculate_p(self,s,s_ref,reset_fsame=False):
        
        '''
          evaluates the probability of neuron footprints belonging to the same neuron. It uses an interpolated version of p_same

          This function requires the successful building of a matching model first
        '''

        assert self.status['model_calculated'], 'Model not yet created - please run build_model first'

        ## create probability function from count-histogram, if not done yet
        ### why not moved to "fit function"?
        if reset_fsame or not self.model['f_same']:
          self.model['f_same'] = sp.interpolate.RectBivariateSpline(self.para['arrays']['distance'],self.para['arrays']['correlation'],self.model['p_same']['joint'])

        ## evaluate probability-function for each of a neurons neighbors
        neighbours = self.data_cross['D_ROIs'] < self.para['neighbor_distance']

        p_same = np.zeros_like(self.data_cross['D_ROIs'])
        p_same[neighbours] = self.model['f_same'].ev(
           self.data_cross['D_ROIs'][neighbours],
           self.data_cross['fp_corr'][1,...][neighbours] if self.para['model'] == 'shifted' else self.data_cross['fp_corr'][0,...][neighbours] ## switch between models
        )
        
        ## fill "bad" entries with zeros
        p_same[~self.data[s_ref]['idx_eval'],:] = 0
        p_same[:,~self.data[s]['idx_eval']] = 0

        ## cap values at 0 and 1 (function-shapes may allow for other values)
        p_same[p_same<0] = 0
        p_same[p_same>1] = 1

        return sp.sparse.csc_matrix(p_same)


    def load_footprints(self,s,store_data=False):
        '''
          function to load results from neuron detection (CaImAn, OnACID) and store in according dictionaries

          TODO:
            * implement min/max thresholding (maybe shift thresholding to other part of code?)
        '''

        if os.path.exists(self.currentPath):
            
            ext = os.path.splitext(self.currentPath)[-1]  # obtain extension
            if ext=='.hdf5':
                ld = load_dict_from_hdf5(self.currentPath)    # function from CaImAn
            elif ext=='.mat':
                ld = loadmat(self.currentPath,squeeze_me=True)
                # ld = loadmat(self.currentPath,variable_names=['A','C','SNR_comp','r_values','cnn_preds'],squeeze_me=True)
            else:
                self.log.error('File extension not yet implemented for loading data!')
                return False
            
            ## load some data necessary for further processing
            self.data_tmp['C'] = ld['C']
            if store_data & np.all([key in ld.keys() for key in ['SNR_comp','r_values','cnn_preds']]):
                ## if evaluation parameters are present, use them to define used neurons
                for key in ['SNR_comp','r_values','cnn_preds']:
                    self.data[s][key] = ld[key]

                ## threshold neurons according to evaluation parameters
                self.data[s]['idx_eval'] = (ld['SNR_comp']>self.para['SNR_thr']) & (ld['r_values']>self.para['r_thr']) & (ld['cnn_preds']>self.para['CNN_thr'])
            elif store_data:
                ## else, use all neurons
                self.data[s]['idx_eval'] = np.ones(ld['A'].shape[1],'bool')

            return ld['A']
                
        else:
            return False

    def prepare_footprints(self,A_ref=None,align_to_reference=True,use_opt_flow=True):
        '''
          Function to prepare footprints for calculation and matching:
            - casting to / ensuring sparse type
            - calculating and reverting rigid and non-rigid shift
            - normalizing
        '''

        ## ensure footprints are stored as sparse matrices to lower computational costs and RAM usage
        if 'csc_matrix' not in str(type(self.A)):
            self.A = sp.sparse.csc_matrix(self.A)

        if align_to_reference:  
            ## align footprints A to reference set A_ref
            
            # if no reference set of footprints is specified, use current reference set
            if A_ref is None: 
                A_ref = self.A_ref
            
            ## cast this one to sparse as well, if needed
            if 'csc_matrix' not in str(type(A_ref)):
                A_ref = sp.sparse.csc_matrix(A_ref)
            
            ## test whether images might be transposed (sometimes happens...) and flip accordingly
            Cn = np.array(self.A.sum(1).reshape(512,512))
            Cn_ref = np.array(A_ref.sum(1).reshape(512,512))
            c_max,_ = calculate_img_correlation(Cn_ref,Cn,plot_bool=False)
            c_max_T,_ = calculate_img_correlation(Cn_ref,Cn.T,plot_bool=False)

            ##  if no good alignment is found, don't include this session in the matching procedure (e.g. if imaging window is shifted too much)
            if (c_max > self.para['min_session_correlation']) & \
              (c_max_T > self.para['min_session_correlation']):
                print('skipping %s (correlation c=%5.3g too low)'%(self.currentPath,c_max))
                return False

            if (c_max_T > c_max) & (c_max_T > self.para['min_session_correlation']):
                print('Transposed image in %s'%self.currentPath)
                A = sp.sparse.hstack([img.reshape(self.para['dims']).transpose().reshape(-1,1) for img in A.transpose()])
            
            ## calculate rigid shift and optical flow from reduced (cumulative) footprint arrays
            (x_shift,y_shift),flow,(x_grid,y_grid),_ = get_shift_and_flow(A_ref,self.A,self.para['dims'],projection=1,plot_bool=False)

            ## use shift and flow to align footprints - define reverse mapping
            if use_opt_flow:
                x_remap = (x_grid - x_shift + flow[:,:,0])
                y_remap = (y_grid - y_shift + flow[:,:,1])
            else:
                x_remap = (x_grid - x_shift)
                y_remap = (y_grid - y_shift)

            ## use shift and flow to align footprints - apply reverse mapping
            self.A = sp.sparse.hstack([
               sp.sparse.csc_matrix(                    # cast results to sparse type
                  cv2.remap(
                      img.reshape(self.para['dims']),   # reshape image to original dimensions
                      x_remap, y_remap,                 # apply reverse identified shift and flow
                      cv2.INTER_CUBIC                 
                  ).reshape(-1,1)                       # reshape back to allow sparse storage
                ) for img in self.A.toarray().T        # loop through all footprints
            ])

        ## finally, apply normalization (watch out! changed normalization to /sum instead of /max)
        # self.A_ref = normalize_sparse_array(self.A_ref)
        self.A = normalize_sparse_array(self.A)
        return True


    def calculate_statistics(self,s,s_ref,A_ref,binary='half'):
      '''
          function to calculate footprint correlations used for the matching procedure for 
            - nearest neighbour (NN) - closest center of mass to reference footprint
            - non-nearest neighbour (nNN) - other neurons with center of mass distance below threshold
          
          footprint correlations are calculated as shifted and non-shifted (depending on model used)

          this method is used both for comparing statistics 
            - referencing previous session (used for matching)
            - referencing itself (used for removal of duplicated footprints AND something else - what?)

          TODO:
            * check what self-referencing is used for other than removing duplicates
      '''

      # calculate distance between footprints and identify NN
      com_distance = sp.spatial.distance.cdist(self.data[s_ref]['cm'],self.data[s]['cm'])

      ## prepare arrays to hold statistics
      footprint_correlation = np.zeros((2,self.data[s_ref]['nA'][0],self.data[s]['nA'][0]))*np.NaN
      
      c_rm = 0
      for i in tqdm(range(self.data[s_ref]['nA'][0]),desc='calculating footprint correlation of %d neurons'%self.data[s_ref]['idx_eval'].sum(),leave=False):
          if self.data[s_ref]['idx_eval'][i]:
            for j in np.where(com_distance[i,:]<self.para['neighbor_distance'])[0]:
                if self.data[s]['idx_eval'][j]:
                  ## calculate pairwise correlation between reference and current set of neuron footprints
                  if (self.para['model']=='both') | (self.para['model']=='unshifted'):
                      footprint_correlation[0,i,j],_ = calculate_img_correlation(self.A[:,j],A_ref[:,i],shift=False)

                  if (self.para['model']=='both') | (self.para['model']=='shifted'):
                      footprint_correlation[1,i,j],_ = calculate_img_correlation(self.A[:,j],A_ref[:,i],crop=True,shift=True,binary=binary)

                      ## tag footprints for removal when calculating statistics for self-matching and they pass some criteria:
                      if (s==s_ref) & \
                        (i!=j) & \
                        ('SNR_comp' in self.data[s].keys()) & \
                        (footprint_correlation[1,i,j] > 0.2): # 1) significant overlap with closeby neuron ("contestant")
                      
                          C_corr = np.corrcoef(self.data_tmp['C'][i,:],self.data_tmp['C'][j,:])[0,1]
                          if C_corr > 0.5:  # 2) high correlation of neuronal activity ("C")
                              c_rm += 1
                              idx_remove = j if self.data[s]['SNR_comp'][i]>self.data[s]['SNR_comp'][j] else i
                              self.data[s]['idx_eval'][idx_remove] = False  # 3) lower SNR than contestant

                              self.log.info('removing neuron %d (%d vs %d) from data (Acorr: %.3f, Ccorr: %.3f; SNR: %.2f vs %.2f)'%(idx_remove,i,j,footprint_correlation[1,i,j],C_corr,self.data[s]['SNR_comp'][j],self.data[s]['SNR_comp'][j]))
                              footprint_correlation[1,i,j] = np.NaN

                              if idx_remove==i: break   ## jump to next one
      
      self.log.info('%d neurons removed'%c_rm)

      return com_distance, footprint_correlation


    def update_joint_model(self,s,s_ref):
      
      '''
        Function to update counts in the joint model

        inputs:
          - s,s_ref: int / string
              key of current (s) and reference (s_ref) session
          - use_kde: bool
              defines, whether kde (kernel density estimation) is used to ...
        
      '''

      ## use all neurons or only those from "medium-dense regions", defined by kde
      idxes = self.model['kernel']['idxes'][s_ref] if self.para['use_kde'] else np.ones(self.data[s_ref]['nA'][0],'bool')

      ## find all neuron pairs below a distance threshold
      neighbors = self.data_cross['D_ROIs'][idxes,:] < self.para['neighbor_distance']

      if s!=s_ref:
          ## identifying next-neighbours
          idx_NN = np.nanargmin(self.data_cross['D_ROIs'][self.data[s_ref]['idx_eval'],:],axis=1)

          NN_idx = np.zeros((self.data[s_ref]['nA'][0],self.data[s]['nA'][0]),'bool')
          NN_idx[self.data[s_ref]['idx_eval'],idx_NN] = True
          NN_idx = NN_idx[idxes,:][neighbors]
        
      ## obtain distance and correlation values of close neighbors, only
      D_ROIs = self.data_cross['D_ROIs'][idxes,:][neighbors]
      fp_corr = self.data_cross['fp_corr'][0,idxes,:][neighbors]
      fp_corr_shifted = self.data_cross['fp_corr'][1,idxes,:][neighbors]

      ## update count histogram with data from current session pair
      for i in tqdm(range(self.para['nbins']),desc='updating joint model',leave=False):

        ## find distance indices of values falling into the current bin
        idx_dist = (D_ROIs >= self.para['arrays']['distance_bounds'][i]) & (D_ROIs < self.para['arrays']['distance_bounds'][i+1])

        for j in range(self.para['nbins']):

          ## differentiate between the two models for calculating footprint-correlation
          if (self.para['model']=='unshifted') | (self.para['model']=='both'):
            ## find correlation indices of values falling into the current bin
            idx_fp = (fp_corr > self.para['arrays']['correlation_bounds'][j]) & (self.para['arrays']['correlation_bounds'][j+1] > fp_corr)
            idx_vals = idx_dist & idx_fp

            if s==s_ref:  # for self-comparing
              self.model['counts_same_unshifted'][i,j] += np.count_nonzero(idx_vals)
            else:         # for cross-comparing
              self.model['counts_unshifted'][i,j,0] += np.count_nonzero(idx_vals)
              self.model['counts_unshifted'][i,j,1] += np.count_nonzero(idx_vals & NN_idx)
              self.model['counts_unshifted'][i,j,2] += np.count_nonzero(idx_vals & ~NN_idx)

          if (self.para['model']=='shifted') | (self.para['model']=='both'):
            idx_fp = (fp_corr_shifted > self.para['arrays']['correlation_bounds'][j]) & (self.para['arrays']['correlation_bounds'][j+1] > fp_corr_shifted)
            idx_vals = idx_dist & idx_fp
            if s==s_ref:  # for self-comparing
              self.model['counts_same'][i,j] += np.count_nonzero(idx_vals)
            else:         # for cross-comparing
              self.model['counts'][i,j,0] += np.count_nonzero(idx_vals)
              self.model['counts'][i,j,1] += np.count_nonzero(idx_vals & NN_idx)
              self.model['counts'][i,j,2] += np.count_nonzero(idx_vals & ~NN_idx)



    def position_kde(self,s,plot_bool=False):

      '''
        function to calculate kernel density estimate of neuron density in session s
        this is optional, but can be used to exclude highly dense and highly sparse regions from statistics in order to not skew statistics

      '''
      self.log.info('calculating kernel density estimates for session %d'%s)
      
      ## calculating kde from center of masses
      x_grid, y_grid = np.meshgrid(np.linspace(0,self.para['dims'][0]*self.para['pxtomu'],self.para['dims'][0]), np.linspace(0,self.para['dims'][1]*self.para['pxtomu'],self.para['dims'][1]))
      positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
      kde = sp.stats.gaussian_kde(self.data[s]['cm'][self.data[s]['idx_eval'],:].T)
      self.model['kernel']['kde'][s] = np.reshape(kde(positions),x_grid.shape)


      cm_px = (self.data[s]['cm'][self.data[s]['idx_eval'],:]/self.para['pxtomu']).astype('int')
      kde_at_com = np.zeros(self.data[s]['nA'][0])*np.NaN
      kde_at_com[self.data[s]['idx_eval']] = self.model['kernel']['kde'][s][cm_px[:,1],cm_px[:,0]]
      self.model['kernel']['idxes'][s] = (kde_at_com > np.quantile(self.model['kernel']['kde'][s],self.para['qtl'][0])) & (kde_at_com < np.quantile(self.model['kernel']['kde'][s],self.para['qtl'][1]))

      if plot_bool:
        plt.figure()
        h_kde = plt.imshow(self.model['kernel']['kde'][s],cmap=plt.cm.gist_earth_r,origin='lower',extent=[0,self.para['dims'][0]*self.para['pxtomu'],0,self.para['dims'][1]*self.para['pxtomu']])
        #if s>0:
          #col = self.data_cross['D_ROIs'].min(1)
        #else:
          #col = 'w'
        plt.scatter(self.data[s]['cm'][:,0],self.data[s]['cm'][:,1],c='w',s=5+10*self.model['kernel']['idxes'][s],clim=[0,10],cmap='YlOrRd')
        plt.xlim([0,self.para['dims'][0]*self.para['pxtomu']])
        plt.ylim([0,self.para['dims'][1]*self.para['pxtomu']])

        # cm_px = (self.data['cm'][s]/self.para['pxtomu']).astype('int')
        # kde_at_cm = self.model['kernel']['kde'][s][cm_px[:,1],cm_px[:,0]]
        plt.colorbar(h_kde)
        plt.show(block=False)


    def fit_model(self,model='shifted'):
      '''

      '''
      
      count_thr=0

      self.para['model'] = model
      if (not (self.para['model'] == 'unshifted')) & (not (self.para['model']=='shifted')):
          raise Exception('Please specify model to be either "shifted" or "unshifted"')

      key_counts = 'counts' if self.para['model']=='shifted' else 'counts_unshifted'
      counts = self.model[key_counts]

      nbins = self.para['nbins']
      step_dist = self.para['arrays']['distance_step']
      step_corr = self.para['arrays']['correlation_step']
      
      

      ## build single models
      ### distance
      fit_fun, fit_bounds = self.set_functions('distance','single')

      for p,pop in zip((1,2,0),['NN','nNN','all']):
        p0 = (counts[...,1].sum()/counts[...,0].sum(),) + \
            tuple(self.model['fit_parameter']['single']['distance']['NN']) + \
            tuple(self.model['fit_parameter']['single']['distance']['nNN']) if pop=='all' else None
        
        self.model['fit_parameter']['single']['distance'][pop] = curve_fit(
          fit_fun[pop],
          self.para['arrays']['distance'],
          counts[...,p].sum(1)/counts[...,p].sum()/step_dist,
          bounds=fit_bounds[pop],
          p0=p0
          )[0]

      ## build function
      d_NN = fun_wrapper(
          fit_fun['NN'],
          self.para['arrays']['distance'],
          self.model['fit_parameter']['single']['distance']['all'][1:3]
        ) * self.model['fit_parameter']['single']['distance']['all'][0]
      
      d_total = fun_wrapper(
          fit_fun['all'],
          self.para['arrays']['distance'],
          self.model['fit_parameter']['single']['distance']['all']
        )
      
      self.model['p_same']['single']['distance'] = d_NN/d_total
      self.log.warning('does something need to be stored from the parameters / bounds / functions?')


      ### to fp-correlation: NN - reverse lognormal, nNN - reverse lognormal
      fit_fun, fit_bounds = self.set_functions('correlation','single')

      for p,pop in zip((1,2,0),['NN','nNN','all']):
        p0 = (counts[...,1].sum()/counts[...,0].sum(),) + \
            tuple(self.model['fit_parameter']['single']['correlation']['NN']) + \
            tuple(self.model['fit_parameter']['single']['correlation']['nNN']) if pop=='all' else None
        
        self.model['fit_parameter']['single']['correlation'][pop] = curve_fit(
          fit_fun[pop],
          self.para['arrays']['correlation'],
          counts[...,p].sum(0)/counts[...,p].sum()/step_corr,
          bounds=fit_bounds[pop],
          p0=p0
          )[0]

      ## build function
      corr_NN = fun_wrapper(
          fit_fun['NN'],
          self.para['arrays']['correlation'],
          self.model['fit_parameter']['single']['correlation']['all'][1:3]
        ) * self.model['fit_parameter']['single']['correlation']['all'][0]
      
      corr_total = fun_wrapper(
          fit_fun['all'],
          self.para['arrays']['correlation'],
          self.model['fit_parameter']['single']['correlation']['all']
        )
      
      self.model['p_same']['single']['correlation'] = corr_NN/corr_total




      ## build joint model
      # preallocate

      ## define normalized histograms
      normalized_histogram = counts/counts.sum(0) * nbins/self.para['neighbor_distance']
      normalized_histogram[np.isnan(normalized_histogram)] = 0

      counts_thr = 20
      self.model['fit_parameter']['joint'] = {}

      fit_fun, fit_bounds = self.set_functions('distance','joint')
      self.model['fit_parameter']['joint']['distance'] = {
          'NN':np.zeros((self.para['nbins'],len(self.model['fit_parameter']['single']['distance']['NN'])))*np.NaN,
          'nNN':np.zeros((self.para['nbins'],fit_bounds['nNN'].shape[1]))*np.NaN
          #'all':np.zeros((self.para['nbins'],len(self.model['fit_parameter']['single']['distance']['all'])))*np.NaN},
        }
      
      for i in tqdm(range(nbins)):
        for p,pop in zip((1,2),['NN','nNN']):
        ### to distance distribution: NN - lognormal, nNN - large lognormal?!
          if counts[:,i,p].sum() > counts_thr:
            self.log.debug('data for %s distance-distribution: '%pop, normalized_histogram[:,i,1])
            self.model['fit_parameter']['joint']['distance'][pop][i,:] = curve_fit(
                fit_fun[pop],
                self.para['arrays']['distance'],
                normalized_histogram[:,i,p],
                bounds=fit_bounds[pop]
              )[0]

      fit_fun, fit_bounds = self.set_functions('correlation','joint')
      normalized_histogram = counts/counts.sum(1)[:,np.newaxis,:] * nbins
      normalized_histogram[np.isnan(normalized_histogram)] = 0
      self.model['fit_parameter']['joint']['correlation'] = {
          'NN':np.zeros((self.para['nbins'],len(self.model['fit_parameter']['single']['correlation']['NN'])))*np.NaN,
          'nNN':np.zeros((self.para['nbins'],fit_bounds['nNN'].shape[1]))*np.NaN
          #'all':np.zeros((self.para['nbins'],len(self.model['fit_parameter']['single']['correlation']['all'])))*np.NaN}
        }

      for i in tqdm(range(nbins)):
        for p,pop in zip((1,2),['NN','nNN']):
          ### to fp-correlation: NN - reverse lognormal, nNN - reverse lognormal
          if counts[i,:,p].sum() > counts_thr:
            self.log.debug('data for %s correlation-distribution: '%pop, normalized_histogram[i,:,1])
            self.model['fit_parameter']['joint']['correlation'][pop][i,:] = curve_fit(
                fit_fun[pop],
                self.para['arrays']['correlation'],
                normalized_histogram[i,:,p],
                bounds=fit_bounds[pop]
              )[0]
        #else:
          #self.model['fit_parameter']['joint']['correlation']['NN'][i,:] = 0


      ## smooth parameter functions
      for key in ['distance','correlation']:
        for pop in ['NN','nNN']:#,'all']
          #for ax in range(self.model['fit_parameter']['joint'][key][pop].shape(1)):
          self.model['fit_parameter']['joint'][key][pop] = sp.ndimage.median_filter(self.model['fit_parameter']['joint'][key][pop],[5,1])
          self.model['fit_parameter']['joint'][key][pop] = sp.ndimage.gaussian_filter(self.model['fit_parameter']['joint'][key][pop],[1,0])

          for ax in range(self.model['fit_parameter']['joint'][key][pop].shape[1]):
            ## find first/last index, at which parameter has a non-nan value
            nan_idx = np.isnan(self.model['fit_parameter']['joint'][key][pop][:,ax])
            
            if nan_idx[0] and (~nan_idx).sum()>1:  ## interpolate beginning
              idx = np.where(~nan_idx)[0][:20]
              y_arr = self.model['fit_parameter']['joint'][key][pop][idx,ax]
              f_interp = np.polyfit(self.para['arrays'][key][idx],y_arr,1)
              poly_fun = np.poly1d(f_interp)

              self.model['fit_parameter']['joint'][key][pop][:idx[0],ax] = poly_fun(self.para['arrays'][key][:idx[0]])
            
            if nan_idx[-1] and (~nan_idx).sum()>1:  ## interpolate end
              idx = np.where(~nan_idx)[0][-20:]
              y_arr = self.model['fit_parameter']['joint'][key][pop][idx,ax]
              f_interp = np.polyfit(self.para['arrays'][key][idx],y_arr,1)
              poly_fun = np.poly1d(f_interp)

              self.model['fit_parameter']['joint'][key][pop][idx[-1]+1:,ax] = poly_fun(self.para['arrays'][key][idx[-1]+1:])


      ## define probability density functions
      # joint_model = 'correlation'
      joint_model = 'distance'
      weight_model = 'distance' if joint_model=='correlation' else 'correlation'

      fit_fun, fit_bounds = self.set_functions(joint_model,'joint') # could also be distance
      self.model['pdf']['joint'] = np.zeros((2,nbins,nbins))
      
      for n in range(nbins):
        for p,pop in enumerate(['NN','nNN']):
          if not np.any(np.isnan(self.model['fit_parameter']['joint'][joint_model][pop][n,:])):
            f_pop = fun_wrapper(
                fit_fun[pop],
                self.para['arrays'][joint_model],
                self.model['fit_parameter']['joint'][joint_model][pop][n,:]
              )
            
            weight = self.model['p_same']['single'][weight_model][n]
            if pop=='nNN':
               weight = 1 - weight
            if joint_model=='correlation':
              self.model['pdf']['joint'][p,n,:] = f_pop*weight
            else:
              self.model['pdf']['joint'][p,:,n] = f_pop*weight

      
      ## obtain probability of being same neuron
      self.model['p_same']['joint'] = 1-self.model['pdf']['joint'][1,...]/np.nansum(self.model['pdf']['joint'],0)
      # if count_thr > 0:
        # self.model['p_same']['joint'] *= np.minimum(self.model[key_counts][...,0],count_thr)/count_thr
      # sp.ndimage.filters.gaussian_filter(self.model['p_same']['joint'],2,output=self.model['p_same']['joint'])
      self.status['model_calculated'] = True



    def set_functions(self,dimension,model='joint'):

        '''
          set up functions for model fitting to count histogram for both, single and joint model.
          This function defines both, the functional shape (from a set of predefined functions), and the boundaries for function parameters

          REMARK:
            * A more flexible approach with functions to be defined on class creation could be feasible, but makes everything a lot more complex for only minor pay-off. Could be implemented in future changes.
        '''
        
        fit_fun = {}
        fit_bounds = {}
        bounds_p = np.array([(0,1)]).T    # weight 'p' between NN and nNN function
        
        if dimension=='distance':
          
          ## set functions for model
          fit_fun['NN'] = functions['lognorm']
          fit_bounds['NN'] = np.array([(0,np.inf),(-np.inf,np.inf)]).T

          if model=='single':

            fit_fun['nNN'] = functions['linear_sigmoid']
            fit_bounds['nNN'] = np.array([(0,np.inf),(0,np.inf),(0,self.para['neighbor_distance']/2)]).T

            fit_fun['all'] = lambda x,p,sigma,mu,m,sig_slope,sig_center : p*functions['lognorm'](x,sigma,mu) + (1-p)*functions['linear_sigmoid'](x,m,sig_slope,sig_center)

          elif model=='joint':           
            fit_fun['nNN'] = functions['gauss']
            fit_bounds['nNN'] = np.array([(0,np.inf),(-np.inf,np.inf)]).T
            # fit_fun['nNN'] = functions['linear_sigmoid']
            # fit_bounds['nNN'] = np.array([(0,np.inf),(0,np.inf),(0,self.para['neighbor_distance']/2)]).T
          
          ## set bounds for fit-parameters
          fit_bounds['all'] = np.hstack([bounds_p,fit_bounds['NN'],fit_bounds['nNN']])
          
        elif dimension=='correlation':
          
          ## set functions for model
          fit_fun['NN'] = functions['lognorm_reverse']
          fit_bounds['NN'] = np.array([(0,np.inf),(-np.inf,np.inf)]).T

          if model == 'single':

            if self.para['model'] == 'shifted':
              fit_fun['nNN'] = functions['gauss']
              fit_bounds['nNN'] = np.array([(0,np.inf),(-np.inf,np.inf)]).T
              
              fit_fun['all'] = lambda x,p,sigma1,mu1,sigma2,mu2 : p*functions['lognorm_reverse'](x,sigma1,mu1) + (1-p)*functions['gauss'](x,sigma2,mu2)

            else:
              fit_fun['nNN'] = functions['beta']
              fit_bounds['nNN'] = np.array([(-np.inf,np.inf),(-np.inf,np.inf)]).T

              fit_fun['all'] = lambda x,p,sigma1,mu1,a,b : p*functions['lognorm_reverse'](x,sigma1,mu1) + (1-p)*functions['beta'](x,a,b)
            
          elif model=='joint':
            fit_fun['nNN'] = functions['gauss']
            fit_bounds['nNN'] = np.array([(0,np.inf),(-np.inf,np.inf)]).T

          ## set bounds for fit-parameters
          fit_bounds['all'] = np.hstack([bounds_p,fit_bounds['NN'],fit_bounds['nNN']])

        for pop in ['NN','nNN','all']:
          if pop in fit_fun:
            self.model['fit_function'][model][dimension][pop] = fit_fun[pop]
        
        return fit_fun, fit_bounds
    






    ### -------------------- SAVING & LOADING ---------------------- ###

    def save_model(self,suffix=''):
      
        pathMatching = os.path.join(self.para['pathData'],'matching')
        if ~os.path.exists(pathMatching):
          os.makedirs(pathMatching,exist_ok=True)

        pathSv = os.path.join(pathMatching,'match_model_%s.pkl'%suffix)

        results = {}
        for key in ['p_same','fit_parameter','pdf','counts','counts_unshifted','counts_same','counts_same_unshifted']:
          results[key] = self.model[key]
        pickleData(results,pathSv,'save')


    def load_model(self,suffix=''):
        
        pathLd = os.path.join(self.para['pathData'],'matching/match_model_%s.pkl'%suffix)
        results = pickleData([],pathLd,'load')
        for key in results.keys():
          self.model[key] = results[key]
        
        self.update_bins(self.model['p_same']['joint'].shape[0])
        self.model['model_calculated'] = True
    

    def save_registration(self,suffix=''):

        svDir = os.path.join(self.para['pathData'],'matching')
        if ~os.path.exists(svDir):
            os.makedirs(svDir,exist_ok=True)
        
        pathSv = os.path.join(self.para['pathData'],'matching','neuron_registration_%s.pkl'%suffix)
        # pickleData(self.results,pathSv,'save')
        pickleData({'results':self.results,'data':self.data},pathSv,'save')


    def load_registration(self,suffix=''):

        pathLd = os.path.join(self.para['pathData'],'matching','neuron_registration_%s.pkl'%suffix)
        dataLd = pickleData([],pathLd,'load')
        self.results = dataLd['results']
        self.data = dataLd['data']
        #try:
          #self.results['assignments'] = self.results['assignment']
        #except:
          #1






    ### ------------------- PLOTTING FUNCTIONS --------------------- ###

    def plot_fit_results(self,n=10,dim='correlation',model='joint',pop='nNN'):
      
      '''
        TODO:
          * build plot function to include widgets, with which parameter space can be visualized, showing fit results and histogram slice (bars), each (2 subplots: correlation & distance)
      '''

      # fit_fun, _ = self.set_functions(dim,model)
      
      key_counts = 'counts' if self.para['model']=='shifted' else 'counts_unshifted'
      counts = self.model[key_counts]

      nbins = self.para['nbins']
      joint_hist_norm_dist = counts/counts.sum(0) * nbins/self.para['neighbor_distance']
      joint_hist_norm_corr = counts/counts.sum(1)[:,np.newaxis,:] * nbins
      

      fig,ax = plt.subplots(2,2)

      ax[0][0].axis('off')

      h = {
         'distance': {},
         'distance_hist': {},
         'correlation': {},
         'correlation_hist': {},
      }

      for i,(pop,col) in enumerate(zip(['NN','nNN'],['g','r'])):
        h['distance'][pop], = ax[0][1].plot(
            self.para['arrays']['distance'],
            np.zeros(self.para['nbins']),
            c=col
          )
        
        h['distance_hist'][pop] = ax[0][1].bar(
           self.para['arrays']['distance'],
           joint_hist_norm_dist[:,0,i+1],
           width=self.para['arrays']['distance_step'],
           facecolor=col,
           alpha=0.5,
        )
      plt.setp(ax[0][1],ylim=[0,2])
      

      for pop,col in zip(['NN','nNN'],['g','r']):
        h['correlation'][pop], = ax[1][0].plot(
          self.para['arrays']['correlation'],
          np.zeros(self.para['nbins']),
          color=col
        )

        h['correlation_hist'][pop] = ax[1][0].bar(
           self.para['arrays']['correlation'],
           joint_hist_norm_corr[:,0,i+1],
           width=self.para['arrays']['correlation_step'],
           facecolor=col,
           alpha=0.5,
        )
      plt.setp(ax[1][0],ylim=[0,2])
      
      ax[1][1].imshow(
         self.model['counts'][...,0],
         origin='lower',aspect='auto',
         extent=tuple(self.para['arrays']['correlation'][[0,-1]])+tuple(self.para['arrays']['distance'][[0,-1]])
        )

      h['distance_marker'] = ax[1][1].axhline(0,c='r')
      h['correlation_marker'] = ax[1][1].axvline(0,c='r')
      plt.setp(ax[1][1],xlim=self.para['arrays']['correlation'][[0,-1]],ylim=self.para['arrays']['distance'][[0,-1]])

      slider_h = 0.02
      slider_w = 0.35
      axamp_dist = plt.axes([0.05, .8, slider_w, slider_h])
      axamp_corr = plt.axes([0.05, .65, slider_w, slider_h])

      
      # widget = {}
      # widget['distance'] = widgets.IntSlider(10,min=0,max=self.para['nbins']-1,orientation='horizontal',description=r'$\displaystyle d_{com}$')
      # widget['correlation'] = widgets.IntSlider(90,min=0,max=self.para['nbins']-1,orientation='horizontal',description=r'$\displaystyle c_{com}$')


      def update_plot_distance(n):
          
          n = int(n)
          # print(h)

          model = 'joint'
          y = {'NN':None,'nNN':None}
          for i,pop in enumerate(['NN','nNN']):
            weight = self.model['p_same']['single']['correlation'][n]
            if pop=='nNN':
               weight = 1-weight
            y[pop] = fun_wrapper(
                self.model['fit_function'][model]['distance'][pop],
                self.para['arrays']['distance'],
                self.model['fit_parameter'][model]['distance'][pop][n,:]
              ) * weight
            h['distance'][pop].set_ydata(y[pop])

            for rect,height in zip(h['distance_hist'][pop],joint_hist_norm_dist[:,n,i+1]):
                rect.set_height(height)
            
            h['correlation_marker'].set_xdata(self.para['arrays']['correlation'][n])
          
          fig.canvas.draw_idle()
        
      def update_plot_correlation(n):

          n = int(n)
          # print('\t\tn: ',n)
          # print('distance:', n,self.para['arrays']['distance'][n])
          model = 'joint'
          y = {'NN':None,'nNN':None}

          max_val = 0
          for i,pop in enumerate(['NN','nNN']):

            weight = self.model['p_same']['single']['distance'][n]
            if pop == 'nNN':
               weight = 1-weight
            y[pop] = fun_wrapper(
                self.model['fit_function'][model]['correlation'][pop],
                self.para['arrays']['correlation'],
                self.model['fit_parameter'][model]['correlation'][pop][n,:]
              ) * weight
            h['correlation'][pop].set_ydata(y[pop])
            max_val = max(max_val,y[pop].max())

            for rect,height in zip(h['correlation_hist'][pop],joint_hist_norm_corr[n,:,i+1]):
                rect.set_height(height)
                max_val = max(max_val,height)
            h['distance_marker'].set_ydata(self.para['arrays']['distance'][n])

          plt.setp(ax[1][0],ylim=[0,max_val*1.1])
          fig.canvas.draw_idle()

      distance_init = 110
      correlation_init = 10

      self.slider = {}
      self.slider['distance'] = Slider(axamp_dist, r'd_{com}', 0, self.para['nbins']-1, valinit=distance_init,orientation='horizontal',valstep=range(nbins))
      self.slider['correlation'] = Slider(axamp_corr, r'c_{com}', 0, self.para['nbins']-1, valinit=correlation_init,orientation='horizontal',valstep=range(nbins))
      
      self.slider['distance'].on_changed(update_plot_distance)
      self.slider['correlation'].on_changed(update_plot_correlation)

      update_plot_distance(distance_init)
      update_plot_correlation(correlation_init)

      plt.show(block=False)



    def plot_model(self,animate=False,sv=False,suffix=''):

      rc('font',size=10)
      rc('axes',labelsize=12)
      rc('xtick',labelsize=8)
      rc('ytick',labelsize=8)

      key_counts = 'counts' if self.para['model']=='shifted' else 'counts_unshifted'
      nbins = self.model[key_counts].shape[0]

      X, Y = np.meshgrid(self.para['arrays']['correlation'], self.para['arrays']['distance'])

      mean_corr_NN, var_corr_NN = mean_of_trunc_lognorm(self.model['fit_parameter']['joint']['correlation']['NN'][:,1],self.model['fit_parameter']['joint']['correlation']['NN'][:,0],[0,1])
      mean_dist_NN, var_dist_NN = mean_of_trunc_lognorm(self.model['fit_parameter']['joint']['distance']['NN'][:,1],self.model['fit_parameter']['joint']['distance']['NN'][:,0],[0,1])

      if self.para['model'] == 'unshifted':
        a = self.model['fit_parameter']['joint']['correlation']['nNN'][:,0]
        b = self.model['fit_parameter']['joint']['correlation']['nNN'][:,1]
        #mean_corr_nNN = a/(a+b)
        #var_corr_nNN = a*b/((a+b)**2*(a+b+1))
        mean_corr_nNN = b
        var_corr_nNN = a
      else:
        #mean_corr_nNN, var_corr_nNN = mean_of_trunc_lognorm(self.model['fit_parameter']['joint']['correlation']['nNN'][:,1],self.model['fit_parameter']['joint']['correlation']['nNN'][:,0],[0,1])
        mean_corr_nNN = self.model['fit_parameter']['joint']['correlation']['nNN'][:,1]
        var_corr_nNN = self.model['fit_parameter']['joint']['correlation']['nNN'][:,0]
      mean_corr_NN = 1-mean_corr_NN
      #mean_corr_nNN = 1-mean_corr_nNN

      fig = plt.figure(figsize=(7,4),dpi=150)
      ax_phase = plt.axes([0.3,0.13,0.2,0.4])
      add_number(fig,ax_phase,order=1,offset=[-250,200])
      #ax_phase.imshow(self.model[key_counts][:,:,0],extent=[0,1,0,self.para['neighbor_distance']],aspect='auto',clim=[0,0.25*self.model[key_counts][:,:,0].max()],origin='lower')
      NN_ratio = self.model[key_counts][:,:,1]/self.model[key_counts][:,:,0]
      cmap = plt.cm.RdYlGn
      NN_ratio = cmap(NN_ratio)
      NN_ratio[...,-1] = np.minimum(self.model[key_counts][...,0]/100,1)

      im_ratio = ax_phase.imshow(NN_ratio,extent=[0,1,0,self.para['neighbor_distance']],aspect='auto',clim=[0,0.5],origin='lower')
      nlev = 3
      col = (np.ones((nlev,3)).T*np.linspace(0,1,nlev)).T
      p_levels = ax_phase.contour(X,Y,self.model['p_same']['joint'],levels=[0.05,0.5,0.95],colors='b',linestyles=[':','--','-'])
      ax_phase.set_xlim([0,1])
      ax_phase.set_ylim([0,self.para['neighbor_distance']])
      ax_phase.tick_params(axis='x',which='both',bottom=True,top=True,labelbottom=False,labeltop=True)
      ax_phase.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=True)
      ax_phase.yaxis.set_label_position("right")
      #ax_phase.xaxis.tick_top()
      ax_phase.set_xlabel('correlation')
      ax_phase.xaxis.set_label_coords(0.5,-0.15)
      ax_phase.set_ylabel('distance')
      ax_phase.yaxis.set_label_coords(1.15,0.5)

      im_ratio.cmap = cmap
      if self.para['model'] == 'unshifted':
        cbaxes = plt.axes([0.41, 0.47, 0.07, 0.03])
        #cbar.ax.set_xlim([0,0.5])
      else:
        cbaxes = plt.axes([0.32, 0.2, 0.07, 0.03])
      cbar = plt.colorbar(im_ratio,cax=cbaxes,orientation='horizontal')
      #cbar.ax.set_xlabel('NN ratio')
      cbar.ax.set_xticks([0,0.5])
      cbar.ax.set_xticklabels(['nNN','NN'])

      #cbar.ax.set_xticks(np.linspace(0,1,2))
      #cbar.ax.set_xticklabels(np.linspace(0,1,2))


      ax_dist = plt.axes([0.05,0.13,0.2,0.4])

      ax_dist.barh(self.para['arrays']['distance'],self.model[key_counts][...,0].sum(1).flat,self.para['neighbor_distance']/nbins,facecolor='k',alpha=0.5,orientation='horizontal')
      ax_dist.barh(self.para['arrays']['distance'],self.model[key_counts][...,2].sum(1),self.para['arrays']['distance_step'],facecolor='salmon',alpha=0.5)
      ax_dist.barh(self.para['arrays']['distance'],self.model[key_counts][...,1].sum(1),self.para['arrays']['distance_step'],facecolor='lightgreen',alpha=0.5)
      ax_dist.invert_xaxis()
      #h_d_move = ax_dist.bar(self.para['arrays']['distance'],np.zeros(nbins),self.para['arrays']['distance_step'],facecolor='k')

      model_distance_all = (fun_wrapper(
          self.model['fit_function']['single']['distance']['NN'],
          self.para['arrays']['distance'],
          self.model['fit_parameter']['single']['distance']['NN']
        )*self.model[key_counts][...,1].sum() + fun_wrapper(self.model['fit_function']['single']['distance']['nNN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['nNN'])*self.model[key_counts][...,2].sum())*self.para['arrays']['distance_step']

      ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['all'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['all'])*self.model[key_counts][...,0].sum()*self.para['arrays']['distance_step'],self.para['arrays']['distance'],'k:')
      ax_dist.plot(model_distance_all,self.para['arrays']['distance'],'k')

      ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['NN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['all'][1:3])*self.model['fit_parameter']['single']['distance']['all'][0]*self.model[key_counts][...,0].sum()*self.para['arrays']['distance_step'],self.para['arrays']['distance'],'g')
      ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['NN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['NN'])*self.model[key_counts][...,1].sum()*self.para['arrays']['distance_step'],self.para['arrays']['distance'],'g:')

      ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['nNN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['all'][3:])*(1-self.model['fit_parameter']['single']['distance']['all'][0])*self.model[key_counts][...,0].sum()*self.para['arrays']['distance_step'],self.para['arrays']['distance'],'r')
      ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['nNN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['nNN'])*self.model[key_counts][...,2].sum()*self.para['arrays']['distance_step'],self.para['arrays']['distance'],'r',linestyle=':')
      ax_dist.set_ylim([0,self.para['neighbor_distance']])
      ax_dist.set_xlabel('counts')
      ax_dist.spines['left'].set_visible(False)
      ax_dist.spines['top'].set_visible(False)
      ax_dist.tick_params(axis='y',which='both',left=False,right=True,labelright=False,labelleft=False)

      ax_corr = plt.axes([0.3,0.63,0.2,0.325])
      ax_corr.bar(self.para['arrays']['correlation'],self.model[key_counts][...,0].sum(0).flat,1/nbins,facecolor='k',alpha=0.5)
      ax_corr.bar(self.para['arrays']['correlation'],self.model[key_counts][...,2].sum(0),1/nbins,facecolor='salmon',alpha=0.5)
      ax_corr.bar(self.para['arrays']['correlation'],self.model[key_counts][...,1].sum(0),1/nbins,facecolor='lightgreen',alpha=0.5)

      f_NN = fun_wrapper(self.model['fit_function']['single']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['NN'])
      f_nNN = fun_wrapper(self.model['fit_function']['single']['correlation']['nNN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['nNN'])
      model_fp_correlation_all = (f_NN*self.model[key_counts][...,1].sum() + f_nNN*self.model[key_counts][...,2].sum())*self.para['arrays']['correlation_step']
      #ax_corr.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['all'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['all'])*self.model[key_counts][...,0].sum()*self.para['arrays']['correlation_step'],'k')
      ax_corr.plot(self.para['arrays']['correlation'],model_fp_correlation_all,'k')

      #ax_corr.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['all'][1:3])*self.model['fit_parameter']['single']['correlation']['all'][0]*self.model[key_counts][...,0].sum()*self.para['arrays']['correlation_step'],'g')
      ax_corr.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['single']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['NN'])*self.model[key_counts][...,1].sum()*self.para['arrays']['correlation_step'],'g')

      #ax_corr.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['nNN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['all'][3:])*(1-self.model['fit_parameter']['single']['correlation']['all'][0])*self.model[key_counts][...,0].sum()*self.para['arrays']['correlation_step'],'r')
      ax_corr.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['single']['correlation']['nNN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['nNN'])*self.model[key_counts][...,2].sum()*self.para['arrays']['correlation_step'],'r')

      ax_corr.set_ylabel('counts')
      ax_corr.set_xlim([0,1])
      ax_corr.spines['right'].set_visible(False)
      ax_corr.spines['top'].set_visible(False)
      ax_corr.tick_params(axis='x',which='both',bottom=True,top=False,labelbottom=False,labeltop=False)

      #ax_parameter =
      p_steps, rates = self.RoC(100)

      ax_cum = plt.axes([0.675,0.7,0.3,0.225])
      add_number(fig,ax_cum,order=2)

      uncertain = {}
      idx_low = np.where(p_steps>0.05)[0][0]
      idx_high = np.where(p_steps<0.95)[0][-1]
      for key in rates['cumfrac'].keys():

        if (rates['cumfrac'][key][idx_low]>0.01) & (rates['cumfrac'][key][idx_high]<0.99):
          ax_cum.fill_between([rates['cumfrac'][key][idx_low],rates['cumfrac'][key][idx_high]],[0,0],[1,1],facecolor='y',alpha=0.5)
        uncertain[key] = (rates['cumfrac'][key][idx_high] - rates['cumfrac'][key][idx_low])#/(1-rates['cumfrac'][key][idx_high+1])

      ax_cum.plot([0,1],[0.05,0.05],'b',linestyle=':')
      ax_cum.plot([0.5,1],[0.95,0.95],'b',linestyle='-')

      ax_cum.plot(rates['cumfrac']['joint'],p_steps[:-1],'grey',label='Joint')
      # ax_cum.plot(rates['cumfrac']['distance'],p_steps[:-1],'k',label='Distance')
      if self.para['model']=='unshifted':
        ax_cum.plot(rates['cumfrac']['correlation'],p_steps[:-1],'lightgrey',label='Correlation')
      ax_cum.set_ylabel('$p_{same}$')
      ax_cum.set_xlabel('cumulative fraction')
      #ax_cum.legend(fontsize=10,frameon=False)
      ax_cum.spines['right'].set_visible(False)
      ax_cum.spines['top'].set_visible(False)

      ax_uncertain = plt.axes([0.75,0.825,0.05,0.1])
      # ax_uncertain.bar(2,uncertain['distance'],facecolor='k')
      ax_uncertain.bar(3,uncertain['joint'],facecolor='k')
      if self.para['model']=='unshifted':
          ax_uncertain.bar(1,uncertain['correlation'],facecolor='lightgrey')
          ax_uncertain.set_xlim([1.5,3.5])
          ax_uncertain.set_xticks(range(1,4))
          ax_uncertain.set_xticklabels(['Corr.','Dist.','Joint'],rotation=60,fontsize=10)
      else:
          ax_uncertain.set_xticks([])
          ax_uncertain.set_xlim([2.5,3.5])
          # ax_uncertain.set_xticklabels(['Dist.','Joint'],rotation=60,fontsize=10)
          ax_uncertain.set_xticklabels([])
      ax_uncertain.set_ylim([0,0.2])
      ax_uncertain.spines['right'].set_visible(False)
      ax_uncertain.spines['top'].set_visible(False)
      ax_uncertain.set_title('uncertain fraction',fontsize=10)

      #ax_rates = plt.axes([0.83,0.6,0.15,0.3])
      #ax_rates.plot(rates['fp']['joint'],p_steps[:-1],'r',label='false positive rate')
      #ax_rates.plot(rates['tp']['joint'],p_steps[:-1],'g',label='true positive rate')

      #ax_rates.plot(rates['fp']['distance'],p_steps[:-1],'r--')
      #ax_rates.plot(rates['tp']['distance'],p_steps[:-1],'g--')

      #ax_rates.plot(rates['fp']['correlation'],p_steps[:-1],'r:')
      #ax_rates.plot(rates['tp']['correlation'],p_steps[:-1],'g:')
      #ax_rates.legend()
      #ax_rates.set_xlabel('rate')
      #ax_rates.set_ylabel('$p_{same}$')

      idx = np.where(p_steps == 0.3)[0]

      ax_RoC = plt.axes([0.675,0.13,0.125,0.3])
      add_number(fig,ax_RoC,order=3)
      ax_RoC.plot(rates['fp']['joint'],rates['tp']['joint'],'k',label='Joint')
      # ax_RoC.plot(rates['fp']['distance'],rates['tp']['distance'],'k',label='Distance')
      if self.para['model']=='unshifted':
        ax_RoC.plot(rates['fp']['correlation'],rates['tp']['correlation'],'lightgrey',label='Correlation')
      ax_RoC.plot(rates['fp']['joint'][idx],rates['tp']['joint'][idx],'kx')
      # ax_RoC.plot(rates['fp']['distance'][idx],rates['tp']['distance'][idx],'kx')
      if self.para['model']=='unshifted':
        ax_RoC.plot(rates['fp']['correlation'][idx],rates['tp']['correlation'][idx],'kx')
      ax_RoC.set_ylabel('true positive')
      ax_RoC.set_xlabel('false positive')
      ax_RoC.spines['right'].set_visible(False)
      ax_RoC.spines['top'].set_visible(False)
      ax_RoC.set_xlim([0,0.1])
      ax_RoC.set_ylim([0.6,1])


      ax_fp = plt.axes([0.925,0.13,0.05,0.1])
      # ax_fp.bar(2,rates['fp']['distance'][idx],facecolor='k')
      ax_fp.bar(3,rates['fp']['joint'][idx],facecolor='k')
      ax_fp.set_xticks([])

      if self.para['model']=='unshifted':
          ax_fp.bar(1,rates['fp']['correlation'][idx],facecolor='lightgrey')
          ax_fp.set_xlim([1.5,3.5])
          ax_fp.set_xticks(range(1,4))
          ax_fp.set_xticklabels(['Corr.','Dist.','Joint'],rotation=60,fontsize=10)
      else:
          ax_fp.set_xticks([])
          ax_fp.set_xlim([2.5,3.5])
          # ax_fp.set_xticklabels(['Dist.','Joint'],rotation=60,fontsize=10)
          ax_fp.set_xticklabels([])

      ax_fp.set_ylim([0,0.05])
      ax_fp.spines['right'].set_visible(False)
      ax_fp.spines['top'].set_visible(False)
      ax_fp.set_ylabel('false pos.',fontsize=10)

      ax_tp = plt.axes([0.925,0.33,0.05,0.1])
      add_number(fig,ax_tp,order=4,offset=[-100,25])
      # ax_tp.bar(2,rates['tp']['distance'][idx],facecolor='k')
      ax_tp.bar(3,rates['tp']['joint'][idx],facecolor='k')
      ax_tp.set_xticks([])
      if self.para['model']=='unshifted':
        ax_tp.bar(1,rates['tp']['correlation'][idx],facecolor='lightgrey')
        ax_tp.set_xlim([1.5,3.5])
      else:
        ax_tp.set_xlim([2.5,3.5])
        ax_fp.set_xticklabels([])
      ax_tp.set_ylim([0.7,1])
      ax_tp.spines['right'].set_visible(False)
      ax_tp.spines['top'].set_visible(False)
      #ax_tp.set_ylabel('fraction',fontsize=10)
      ax_tp.set_ylabel('true pos.',fontsize=10)

      #plt.tight_layout()
      plt.show(block=False)
      if sv:
        ext = 'png'
        path = os.path.join(self.para['pathMouse'],'Sheintuch_matching_%s%s.%s'%(self.para['model'],suffix,ext))
        plt.savefig(path,format=ext,dpi=150)
      return
      #ax_cvc = plt.axes([0.65,0.1,0.2,0.4])
      #idx = self.data_cross['fp_corr_max']>0
      #ax_cvc.scatter(self.data_cross['fp_corr_max'][idx].toarray().flat,self.data_cross['fp_corr'][idx].toarray().flat,c='k',marker='.')
      #ax_cvc.plot([0,1],[0,1],'r--')
      #ax_cvc.set_xlim([0,1])
      #ax_cvc.set_ylim([0,1])
      #ax_cvc.set_xlabel('shifted correlation')
      #ax_cvc.set_ylabel('unshifted correlation')

      #plt.show(block=False)
      #return

      ## plot fit parameters
      plt.figure()
      plt.subplot(221)
      plt.plot(self.para['arrays']['correlation'],mean_dist_NN,'g',label='lognorm $\mu$')
      plt.plot(self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['nNN'][:,1],'r',label='gauss $\mu$')
      plt.title('distance models')
      #plt.plot(self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['all'][:,2],'g--')
      #plt.plot(self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['all'][:,5],'r--')
      plt.legend()

      plt.subplot(223)
      plt.plot(self.para['arrays']['correlation'],var_dist_NN,'g',label='lognorm $\sigma$')
      plt.plot(self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['nNN'][:,0],'r',label='gauss $\sigma$')
      #plt.plot(self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['nNN'][:,1],'r--',label='dist $\gamma$')
      plt.legend()

      plt.subplot(222)
      plt.plot(self.para['arrays']['distance'],mean_corr_NN,'g',label='lognorm $\mu$')#self.model['fit_parameter']['joint']['correlation']['NN'][:,1],'g')#
      plt.plot(self.para['arrays']['distance'],self.model['fit_parameter']['joint']['correlation']['nNN'][:,1],'r',label='gauss $\mu$')
      plt.title('correlation models')
      plt.legend()

      plt.subplot(224)
      plt.plot(self.para['arrays']['distance'],var_corr_NN,'g',label='lognorm $\sigma$')
      plt.plot(self.para['arrays']['distance'],self.model['fit_parameter']['joint']['correlation']['nNN'][:,0],'r',label='gauss $\sigma$')
      plt.legend()
      plt.tight_layout()
      plt.show(block=False)

      #return
      fig = plt.figure()
      plt.subplot(322)
      plt.plot(self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['NN'][:,1],'g')
      plt.plot(self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['nNN'][:,1],'r')
      plt.xlim([0,1])
      plt.ylim([0,self.para['neighbor_distance']])

      plt.subplot(321)
      plt.plot(self.para['arrays']['distance'],mean_corr_NN,'g')#self.model['fit_parameter']['joint']['correlation']['NN'][:,1],'g')#
      plt.plot(self.para['arrays']['distance'],mean_corr_nNN,'r')#self.model['fit_parameter']['joint']['correlation']['nNN'][:,1],'r')#
      plt.xlim([0,self.para['neighbor_distance']])
      plt.ylim([0.5,1])


      plt.subplot(323)
      plt.bar(self.para['arrays']['distance'],self.model[key_counts][...,0].sum(1).flat,self.para['neighbor_distance']/nbins,facecolor='k',alpha=0.5)
      plt.bar(self.para['arrays']['distance'],self.model[key_counts][...,2].sum(1),self.para['arrays']['distance_step'],facecolor='r',alpha=0.5)
      plt.bar(self.para['arrays']['distance'],self.model[key_counts][...,1].sum(1),self.para['arrays']['distance_step'],facecolor='g',alpha=0.5)
      h_d_move = plt.bar(self.para['arrays']['distance'],np.zeros(nbins),self.para['arrays']['distance_step'],facecolor='k')

      model_distance_all = (fun_wrapper(self.model['fit_function']['distance']['NN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['NN'])*self.model[key_counts][...,1].sum() + fun_wrapper(self.model['fit_function']['distance']['nNN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['nNN'])*self.model[key_counts][...,2].sum())*self.para['arrays']['distance_step']

      plt.plot(self.para['arrays']['distance'],fun_wrapper(self.model['fit_function']['distance']['all'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['all'])*self.model[key_counts][...,0].sum()*self.para['arrays']['distance_step'],'k')
      plt.plot(self.para['arrays']['distance'],model_distance_all,'k--')

      plt.plot(self.para['arrays']['distance'],fun_wrapper(self.model['fit_function']['distance']['NN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['all'][1:3])*self.model['fit_parameter']['single']['distance']['all'][0]*self.model[key_counts][...,0].sum()*self.para['arrays']['distance_step'],'g')
      plt.plot(self.para['arrays']['distance'],fun_wrapper(self.model['fit_function']['distance']['NN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['NN'])*self.model[key_counts][...,1].sum()*self.para['arrays']['distance_step'],'g--')

      plt.plot(self.para['arrays']['distance'],fun_wrapper(self.model['fit_function']['distance']['nNN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['all'][3:])*(1-self.model['fit_parameter']['single']['distance']['all'][0])*self.model[key_counts][...,0].sum()*self.para['arrays']['distance_step'],'r')
      plt.plot(self.para['arrays']['distance'],fun_wrapper(self.model['fit_function']['distance']['nNN'],self.para['arrays']['distance'],self.model['fit_parameter']['single']['distance']['nNN'])*self.model[key_counts][...,2].sum()*self.para['arrays']['distance_step'],'r--')
      plt.xlim([0,self.para['neighbor_distance']])
      plt.xlabel('distance')

      plt.subplot(324)
      plt.bar(self.para['arrays']['correlation'],self.model[key_counts][...,0].sum(0).flat,1/nbins,facecolor='k',alpha=0.5)
      plt.bar(self.para['arrays']['correlation'],self.model[key_counts][...,2].sum(0),1/nbins,facecolor='r',alpha=0.5)
      plt.bar(self.para['arrays']['correlation'],self.model[key_counts][...,1].sum(0),1/nbins,facecolor='g',alpha=0.5)
      h_fp_move = plt.bar(self.para['arrays']['correlation'],np.zeros(nbins),1/nbins,facecolor='k')

      model_fp_correlation_all = (fun_wrapper(self.model['fit_function']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['NN'])*self.model[key_counts][...,1].sum() + fun_wrapper(self.model['fit_function']['correlation']['nNN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['nNN'])*self.model[key_counts][...,2].sum())*self.para['arrays']['correlation_step']
      plt.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['all'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['all'])*self.model[key_counts][...,0].sum()*self.para['arrays']['correlation_step'],'k')
      plt.plot(self.para['arrays']['correlation'],model_fp_correlation_all,'k--')

      plt.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['all'][1:3])*self.model['fit_parameter']['single']['correlation']['all'][0]*self.model[key_counts][...,0].sum()*self.para['arrays']['correlation_step'],'g')
      plt.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['NN'])*self.model[key_counts][...,1].sum()*self.para['arrays']['correlation_step'],'g--')

      plt.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['nNN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['all'][3:])*(1-self.model['fit_parameter']['single']['correlation']['all'][0])*self.model[key_counts][...,0].sum()*self.para['arrays']['correlation_step'],'r')
      plt.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['nNN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['nNN'])*self.model[key_counts][...,2].sum()*self.para['arrays']['correlation_step'],'r--')

      plt.xlabel('correlation')
      plt.xlim([0,1])

      ax_d = plt.subplot(326)
      d_bar1 = ax_d.bar(self.para['arrays']['distance'],self.model[key_counts][:,0,0],self.para['neighbor_distance']/nbins,facecolor='k',alpha=0.5)
      d_bar2 = ax_d.bar(self.para['arrays']['distance'],self.model[key_counts][:,0,2],self.para['neighbor_distance']/nbins,facecolor='r',alpha=0.5)
      d_bar3 = ax_d.bar(self.para['arrays']['distance'],self.model[key_counts][:,0,1],self.para['neighbor_distance']/nbins,facecolor='g',alpha=0.5)

      ### now, plot model stuff
      d_model_nNN, = ax_d.plot(self.para['arrays']['distance'],fun_wrapper(self.model['fit_function']['distance']['nNN_joint'],self.para['arrays']['distance'],self.model['fit_parameter']['joint']['distance']['nNN'][0,:]),'r')
      d_model_NN, = ax_d.plot(self.para['arrays']['distance'],fun_wrapper(self.model['fit_function']['distance']['NN'],self.para['arrays']['distance'],self.model['fit_parameter']['joint']['distance']['NN'][0,:]),'g')

      h_d = [d_bar1,d_bar3,d_bar2,h_d_move,d_model_NN,d_model_nNN]
      #h_d = d_bar1
      ax_d.set_xlabel('distance')
      ax_d.set_xlim([0,self.para['neighbor_distance']])
      ax_d.set_ylim([0,self.model[key_counts][...,0].max()*1.1])


      ax_fp = plt.subplot(325)
      fp_bar1 = ax_fp.bar(self.para['arrays']['correlation'],self.model[key_counts][0,:,0],1/nbins,facecolor='k',alpha=0.5)
      fp_bar2 = ax_fp.bar(self.para['arrays']['correlation'],self.model[key_counts][0,:,2],1/nbins,facecolor='r',alpha=0.5)
      fp_bar3 = ax_fp.bar(self.para['arrays']['correlation'],self.model[key_counts][0,:,1],1/nbins,facecolor='g',alpha=0.5)

      ### now, plot model stuff
      fp_model_nNN, = ax_fp.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['nNN_joint'],self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['correlation']['nNN'][0,:]),'r')
      fp_model_NN, = ax_fp.plot(self.para['arrays']['correlation'],fun_wrapper(self.model['fit_function']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['correlation']['NN'][0,:]),'g')


      h_fp = [fp_bar1,fp_bar3,fp_bar2,h_fp_move,fp_model_NN,fp_model_nNN]
      ax_fp.set_xlabel('corr')
      ax_fp.set_xlim([0,1])
      ax_fp.set_ylim([0,self.model[key_counts][...,0].max()*1.1])


      def update_distr(i,h_d,h_fp):

        n = i%self.model[key_counts].shape[0]
        for k in range(3):
          [h.set_height(dat) for h,dat in zip(h_d[k],self.model[key_counts][:,n,k])]
          [h.set_height(dat) for h,dat in zip(h_fp[k],self.model[key_counts][n,:,k])]

        d_move = np.zeros(self.model[key_counts].shape[0])
        fp_move = np.zeros(self.model[key_counts].shape[1])

        d_move[n] = self.model[key_counts][n,:,0].sum()
        fp_move[n] = self.model[key_counts][:,n,0].sum()

        [h.set_height(dat) for h,dat in zip(h_d[3],d_move)]
        [h.set_height(dat) for h,dat in zip(h_fp[3],fp_move)]

        self.model['p_same']['single']['distance'][n]*self.model[key_counts][n,:,0].sum()
        (1-self.model['p_same']['single']['distance'][n])*self.model[key_counts][n,:,0].sum()


        h_fp[4].set_ydata(fun_wrapper(self.model['fit_function']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['correlation']['NN'][n,:])*self.model['p_same']['single']['distance'][n]*self.model[key_counts][n,:,0].sum()*self.para['arrays']['correlation_step'])
        h_fp[5].set_ydata(fun_wrapper(self.model['fit_function']['correlation']['nNN_joint'],self.para['arrays']['correlation'],self.model['fit_parameter']['joint']['correlation']['nNN'][n,:])*(1-self.model['p_same']['single']['distance'][n])*self.model[key_counts][n,:,0].sum()*self.para['arrays']['correlation_step'])

        self.model['p_same']['single']['correlation'][n]*self.model[key_counts][:,n,0].sum()
        (1-self.model['p_same']['single']['correlation'][n])*self.model[key_counts][:,n,0].sum()

        h_d[4].set_ydata(fun_wrapper(self.model['fit_function']['distance']['NN'],self.para['arrays']['distance'],self.model['fit_parameter']['joint']['distance']['NN'][n,:])*self.model['p_same']['single']['correlation'][n]*self.model[key_counts][:,n,0].sum()*self.para['arrays']['distance_step'])

        h_d[5].set_ydata(fun_wrapper(self.model['fit_function']['distance']['nNN_joint'],self.para['arrays']['distance'],self.model['fit_parameter']['joint']['distance']['nNN'][n,:])*(1-self.model['p_same']['single']['correlation'][n])*self.model[key_counts][:,n,0].sum()*self.para['arrays']['distance_step'])
        #print(tuple(h_d[0]))
        return tuple(h_d[0]) + tuple(h_d[1]) + tuple(h_d[2]) + tuple(h_d[3]) + (h_d[4],) + (h_d[5],) + tuple(h_fp[0]) + tuple(h_fp[1]) + tuple(h_fp[2]) + tuple(h_fp[3]) + (h_fp[4],) + (h_fp[5],)

      plt.tight_layout()
      if animate or False:

        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=900)

        anim = animation.FuncAnimation(fig, update_distr, fargs=(h_d,h_fp),frames=nbins,interval=100, blit=True)
        svPath = os.path.join(self.para['pathMouse'],'animation_single_models_%s.gif'%model)
        anim.save(svPath, writer='imagemagick',fps=15)#writer)
        print('animation saved at %s'%svPath)
        #anim
        plt.show()
      else:
        update_distr(20,h_d,h_fp)
        plt.show(block=False)
      #return


      counts1 = sp.ndimage.gaussian_filter(self.model['counts'][...,0],[1,1])
      counts2 = sp.ndimage.gaussian_filter(self.model['counts_same'],[1,1])

      counts1 /= counts1.sum(1)[-10:].sum()
      counts2 /= counts2.sum(1)[-10:].sum()

      counts_dif = counts1-counts2
      counts_dif[self.model['counts'][...,0]<2] = 0

      p_same = counts_dif/counts1
      p_same[counts1==0] = 0
      p_same = sp.ndimage.median_filter(p_same,[3,3],mode='nearest')
      self.f_same = sp.interpolate.RectBivariateSpline(self.para['arrays']['distance'],self.para['arrays']['correlation'],p_same,kx=1,ky=1)
      p_same = np.maximum(0,sp.ndimage.gaussian_filter(self.f_same(self.para['arrays']['distance'],self.para['arrays']['correlation']),[1,1]))

      #self.model['p_same']['joint'] = p_same
      plt.figure()
      X, Y = np.meshgrid(self.para['arrays']['correlation'], self.para['arrays']['distance'])

      ax = plt.subplot(221,projection='3d')
      ax.plot_surface(X,Y,counts1,alpha=0.5)
      ax.plot_surface(X,Y,counts2,alpha=0.5)

      ax = plt.subplot(222,projection='3d')
      ax.plot_surface(X,Y,counts_dif,cmap='jet')

      ax = plt.subplot(223,projection='3d')
      ax.plot_surface(X,Y,p_same,cmap='jet')

      ax = plt.subplot(224,projection='3d')
      ax.plot_surface(X,Y,self.model['p_same']['joint'],cmap='jet')

      plt.show(block=False)


      plt.figure()
      ax = plt.subplot(111,projection='3d')
      X, Y = np.meshgrid(self.para['arrays']['correlation'], self.para['arrays']['distance'])
      NN_ratio = self.model[key_counts][:,:,1]/self.model[key_counts][:,:,0]
      cmap = plt.cm.RdYlGn
      NN_ratio = cmap(NN_ratio)
      ax.plot_surface(X,Y,self.model[key_counts][:,:,0],facecolors=NN_ratio)
      ax.view_init(30,-120)
      ax.set_xlabel('footprint correlation',fontsize=14)
      ax.set_ylabel('distance',fontsize=14)
      ax.set_zlabel('# pairs',fontsize=14)
      plt.tight_layout()
      plt.show(block=False)

      if sv:
        ext = 'png'
        path = os.path.join(self.para['pathMouse'],'Sheintuch_matching_phase_%s%s.%s'%(self.para['model'],suffix,ext))
        plt.savefig(path,format=ext,dpi=300)

      return


      plt.figure()
      plt.subplot(221)
      plt.imshow(self.model[key_counts][:,:,0],extent=[0,1,0,self.para['neighbor_distance']],aspect='auto',clim=[0,self.model[key_counts][:,:,0].max()],origin='lower')
      nlev = 3
      col = (np.ones((nlev,3)).T*np.linspace(0,1,nlev)).T
      p_levels = plt.contour(X,Y,self.model['p_same']['joint'],levels=[0.05,0.5,0.95],colors=col)
      #plt.colorbar(p_levels)
      #plt.imshow(self.model[key_counts][...,0],extent=[0,1,0,self.para['neighbor_distance']],aspect='auto',origin='lower')
      plt.subplot(222)
      plt.imshow(self.model[key_counts][...,0],extent=[0,1,0,self.para['neighbor_distance']],aspect='auto',origin='lower')
      plt.subplot(223)
      plt.imshow(self.model[key_counts][...,1],extent=[0,1,0,self.para['neighbor_distance']],aspect='auto',origin='lower')
      plt.subplot(224)
      plt.imshow(self.model[key_counts][...,2],extent=[0,1,0,self.para['neighbor_distance']],aspect='auto',origin='lower')
      plt.show(block=False)




      plt.figure()
      W_NN = self.model[key_counts][...,1].sum() / self.model[key_counts][...,0].sum()
      #W_NN = 0.5
      W_nNN = 1-W_NN
      plt.subplot(211)
      pdf_NN = fun_wrapper(self.model['fit_function']['correlation']['NN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['NN'])*self.para['arrays']['correlation_step']
      pdf_nNN = fun_wrapper(self.model['fit_function']['correlation']['nNN'],self.para['arrays']['correlation'],self.model['fit_parameter']['single']['correlation']['nNN'])*self.para['arrays']['correlation_step']
      pdf_all = pdf_NN*W_NN+pdf_nNN*W_nNN

      plt.plot(self.para['arrays']['correlation'],pdf_NN*W_NN,'g')
      plt.plot(self.para['arrays']['correlation'],pdf_nNN*W_nNN,'r')
      plt.plot(self.para['arrays']['correlation'],pdf_all)

      plt.subplot(212)
      plt.plot(self.para['arrays']['correlation'],pdf_NN*W_NN/pdf_all,'k')
      plt.ylim([0,1])

      plt.show(block=False)



      X, Y = np.meshgrid(self.para['arrays']['correlation'], self.para['arrays']['distance'])

      fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,8),subplot_kw={'projection':'3d'})
      #ax = plt.subplot(221,projection='3d')
      prob = ax1.plot_surface(X,Y,self.model['pdf']['joint'][0,...],cmap='jet')
      prob.set_clim(0,6)
      ax1.set_xlabel('corr')
      ax1.set_ylabel('d')
      ax1.set_zlabel('model')
      #ax1 = plt.subplot(222,projection='3d')
      prob = ax2.plot_surface(X,Y,self.model['pdf']['joint'][1,...],cmap='jet')
      prob.set_clim(0,6)
      ax2.set_xlabel('corr')
      ax2.set_ylabel('d')
      ax2.set_zlabel('model')

      prob = ax3.plot_surface(X,Y,self.model['p_same']['joint'],cmap='jet')
      prob.set_clim(0,1)
      ax3.set_zlim([0,1])
      ax3.set_xlabel('corr')
      ax3.set_ylabel('d')
      ax3.set_zlabel('model')

      #ax = plt.subplot(224,projection='3d')
      prob = ax4.plot_surface(X,Y,self.model[key_counts][...,0],cmap='jet')
      #prob = ax.bar3d(X.flatten(),Y.flatten(),np.zeros((nbins,nbins)).flatten(),np.ones((nbins,nbins)).flatten()*self.para['arrays']['correlation_step'],np.ones((nbins,nbins)).flatten()*self.para['arrays']['distance_step'],self.model[key_counts][...,0].flatten(),cmap='jet')
      #prob.set_clim(0,1)
      ax4.set_xlabel('corr')
      ax4.set_ylabel('d')
      ax4.set_zlabel('occurence')
      plt.tight_layout()
      axes = [ax1,ax2,ax3,ax4]

      def rotate_view(i,axes,fixed_angle=30):
        for ax in axes:
          ax.view_init(fixed_angle,(i*2)%360)
        #return ax

      if animate:
        anim = animation.FuncAnimation(fig, rotate_view, fargs=(axes,30), frames=180, interval=100, blit=False)
        svPath = os.path.join(self.para['pathMouse'],'animation_p_same.gif')
        anim.save(svPath, writer='imagemagick',fps=15)#writer)
        print('animation saved at %s'%svPath)
        #anim
        #plt.show()
      else:
        rotate_view(100,axes,fixed_angle=30)
        plt.show(block=False)
      #anim.save('animation.mp4', writer=writer)
      #print('animation saved')

      print('proper weighting of bin counts')
      print('smoothing by gaussian')
    

    def plot_matches(self, s, p_thr=0.5,plot_results=False):
    
      '''

        TODO:
          * rewrite function, such that it calculates and plots footprint matching for 2 arbitrary sessions (s,s_ref)
          * write function description and document code properly
          * optimize plotting, so it doesn't take forever
      '''
      

      matches = linear_sum_assignment(1 - self.data[s]['p_same'].toarray())
      p_matched = self.data[s]['p_same'].toarray()[matches]

      if plot_results:
          idx_TP = np.where(np.array(p_matched) > p_thr)[0] ## thresholding results
          nA_ref = self.data[s]['p_same'].shape[0]
          if len(idx_TP) > 0:
              matched_ROIs1 = matches[0][idx_TP]    # ground truth
              matched_ROIs2 = matches[1][idx_TP]   # algorithm - comp
              non_matched1 = np.setdiff1d(list(range(nA_ref)), matches[0][idx_TP])
              non_matched2 = np.setdiff1d(list(range(self.data[s]['nA'][0])), matches[1][idx_TP])
              TP = np.sum(np.array(p_matched) > p_thr).astype('float32')
          else:
              TP = 0.
              plot_results = False
              matched_ROIs1 = []
              matched_ROIs2 = []
              non_matched1 = list(range(nA_ref))
              non_matched2 = list(range(self.data[s]['nA'][0]))

              FN = nA_ref - TP
              FP = self.data[s]['nA'][0] - TP
              TN = 0

              performance = dict()
              performance['recall'] = TP / (TP + FN)
              performance['precision'] = TP / (TP + FP)
              performance['accuracy'] = (TP + TN) / (TP + FP + FN + TN)
              performance['f1_score'] = 2 * TP / (2 * TP + FP + FN)

          print('plotting...')
          t_start = time.time()
          cmap = 'viridis'

          Cn = self.A_ref.sum(1).reshape(512,512)

          level = 0.1
          plt.figure(figsize=(15,12))
          plt.rcParams['pdf.fonttype'] = 42
          font = {'family': 'Myriad Pro',
                'weight': 'regular',
                'size': 10}
          plt.rc('font', **font)
          lp, hp = np.nanpercentile(np.array(Cn), [5, 95])

          ax_matches = plt.subplot(121)
          ax_nonmatches = plt.subplot(122)

          ax_matches.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
          ax_nonmatches.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)

          A = np.reshape(self.A_ref.astype('float32').toarray(), self.para['dims'] + (-1,), order='F').transpose(2, 0, 1)
          [ax_matches.contour(a, levels=[level], colors='w', linewidths=1) for a in A[matched_ROIs1,...]]
          [ax_nonmatches.contour(a, levels=[level], colors='w', linewidths=1) for a in A[non_matched1,...]]

          print('first half done - %5.3f'%(time.time()-t_start))
          A = None
          A = np.reshape(self.A.astype('float32').toarray(), self.para['dims'] + (-1,), order='F').transpose(2, 0, 1)

          [ax_matches.contour(a, levels=[level], colors='r', linewidths=1) for a in A[matched_ROIs2,...]]
          [ax_nonmatches.contour(a, levels=[level], colors='r', linewidths=1) for a in A[non_matched2,...]]
          A = None

          plt.draw()
          print('done. time taken: %5.3f'%(time.time()-t_start))
          plt.show(block=False)

      return matches, p_matched

    def RoC(self,steps):
      key_counts = 'counts' if self.para['model']=='shifted' else 'counts_unshifted'
      p_steps = np.linspace(0,1,steps+1)

      rates = {'tp':      {},
              'tn':      {},
              'fp':      {},
              'fn':      {},
              'cumfrac': {}}

      for key in rates.keys():
        rates[key] = {'joint':np.zeros(steps),
                      'distance':np.zeros(steps),
                      'correlation':np.zeros(steps)}

      nTotal = self.model[key_counts][...,0].sum()
      for i in range(steps):
        p = p_steps[i]

        for key in ['joint','distance','correlation']:

          if key == 'joint':
            idxes_negative = self.model['p_same']['joint'] < p
            idxes_positive = self.model['p_same']['joint'] >= p

            tp = self.model[key_counts][idxes_positive,1].sum()
            tn = self.model[key_counts][idxes_negative,2].sum()
            fp = self.model[key_counts][idxes_positive,2].sum()
            fn = self.model[key_counts][idxes_negative,1].sum()

            rates['cumfrac']['joint'][i] = self.model[key_counts][idxes_negative,0].sum()/nTotal
          elif key == 'distance':
            idxes_negative = self.model['p_same']['single']['distance'] < p
            idxes_positive = self.model['p_same']['single']['distance'] >= p

            tp = self.model[key_counts][idxes_positive,:,1].sum()
            tn = self.model[key_counts][idxes_negative,:,2].sum()
            fp = self.model[key_counts][idxes_positive,:,2].sum()
            fn = self.model[key_counts][idxes_negative,:,1].sum()

            rates['cumfrac']['distance'][i] = self.model[key_counts][idxes_negative,:,0].sum()/nTotal
          else:
            idxes_negative = self.model['p_same']['single']['correlation'] < p
            idxes_positive = self.model['p_same']['single']['correlation'] >= p

            tp = self.model[key_counts][:,idxes_positive,1].sum()
            tn = self.model[key_counts][:,idxes_negative,2].sum()
            fp = self.model[key_counts][:,idxes_positive,2].sum()
            fn = self.model[key_counts][:,idxes_negative,1].sum()

            rates['cumfrac']['correlation'][i] = self.model[key_counts][:,idxes_negative,0].sum()/nTotal

          rates['tp'][key][i] = tp/(fn+tp)
          rates['tn'][key][i] = tn/(fp+tn)
          rates['fp'][key][i] = fp/(fp+tn)
          rates['fn'][key][i] = fn/(fn+tp)

      return p_steps, rates


    def plot_neuron_numbers(self):

        ### plot occurence of neurons
        colors = [(1,0,0,0),(1,0,0,1)]
        RedAlpha = mcolors.LinearSegmentedColormap.from_list('RedAlpha',colors,N=2)
        colors = [(0,0,0,0),(0,0,0,1)]
        BlackAlpha = mcolors.LinearSegmentedColormap.from_list('BlackAlpha',colors,N=2)

        idxes = np.ones(self.results['assignments'].shape,'bool')

        ### plot occurence of neurons
        ax_oc = plt.subplot(111)
        #ax_oc2 = ax_oc.twinx()
        ax_oc.imshow((~np.isnan(self.results['assignments']))&idxes,cmap=BlackAlpha,aspect='auto',interpolation='none')
        # ax_oc2.imshow((~np.isnan(self.results['assignments']))&(~idxes),cmap=RedAlpha,aspect='auto')
        #ax_oc.imshow(self.results['p_matched'],cmap='binary',aspect='auto')
        ax_oc.set_xlabel('session')
        ax_oc.set_ylabel('neuron ID')
        plt.tight_layout()
        plt.show(block=False)

        # ext = 'png'
        # path = pathcat([self.para['pathMouse'],'Figures/Sheintuch_registration_score_stats_raw_%s_%s.%s'%(self.para['model'],suffix,ext)])
        # plt.savefig(path,format=ext,dpi=300)



    def plot_registration(self,suffix='',sv=False):

      rc('font',size=10)
      rc('axes',labelsize=12)
      rc('xtick',labelsize=8)
      rc('ytick',labelsize=8)

      # fileName = 'clusterStats_%s.pkl'%dataSet
      idxes = np.ones(self.results['assignments'].shape,'bool')

      fileName = f'cluster_stats_{suffix}.pkl'
      pathLoad = os.path.join(self.para['pathMouse'],fileName)
      if os.path.exists(pathLoad):
        ld = pickleData([],pathLoad,'load')
        if ~np.all(np.isnan(ld['SNR_comp'])):
          idxes = (ld['SNR_comp']>2) & (ld['r_values']>0) & (ld['cnn_preds']>0.3) & (ld['firingrate']>0)

      
      
      
      # plt.figure(figsize=(3,6))

      # return
      

      # plt.figure(figsize=(7,3.5))

      # ax_oc = plt.axes([0.1,0.15,0.25,0.6])
      # ax_oc2 = ax_oc.twinx()
      # ax_oc.imshow((~np.isnan(self.results['assignments']))&idxes,cmap=BlackAlpha,aspect='auto')
      # ax_oc2.imshow((~np.isnan(self.results['assignments']))&(~idxes),cmap=RedAlpha,aspect='auto')
      # #ax_oc.imshow(self.results['p_matched'],cmap='binary',aspect='auto')
      # ax_oc.set_xlabel('session')
      # ax_oc.set_ylabel('neuron ID')
      #
      self.results['p_matched'][self.results['p_matched']==0] = np.NaN
      nC,nS = self.results['assignments'].shape
      ### plot point statistics
      if not ('match_2nd' in self.results.keys()):
        self.results['match_2nd'] = np.zeros((nC,nS))
        for s in tqdm(range(1,nS)):
          if s in self.results['data'].keys():
            idx_c = np.where(~np.isnan(self.results['assignments'][:,s]))[0]
            idx_c = idx_c[idx_c<self.results['data'][s]['p_same'].shape[0]]
            scores_now = self.results['data'][s]['p_same'].toarray()

            self.results['match_2nd'][idx_c,s] = [max(scores_now[c,np.where(scores_now[c,:]!=self.results['p_matched'][c,s])[0]]) for c in idx_c]
      #
      # ax = plt.axes([0.1,0.75,0.25,0.2])
      # ax.plot(np.linspace(0,nS,nS),(~np.isnan(self.results['assignments'])).sum(0),'ro',markersize=1)
      # ax.plot(np.linspace(0,nS,nS),((~np.isnan(self.results['assignments'])) & idxes).sum(0),'ko',markersize=1)
      # ax.set_xlim([0,nS])
      # ax.set_ylim([0,3500])
      # ax.set_xticks([])
      # ax.set_ylabel('# neurons')
      #
      # ax = plt.axes([0.35,0.15,0.1,0.6])
      # ax.plot(((~np.isnan(self.results['assignments'])) & idxes).sum(1),np.linspace(0,nC,nC),'ko',markersize=0.5)
      # ax.invert_yaxis()
      # ax.set_ylim([nC,0])
      # ax.set_yticks([])
      # ax.set_xlabel('occurence')
      #
      # ax = plt.axes([0.35,0.75,0.1,0.2])
      # ax.hist((~np.isnan(self.results['assignments'])).sum(1),np.linspace(0,nS,nS),color='r',cumulative=True,density=True,histtype='step')
      # ax.hist(((~np.isnan(self.results['assignments'])) & idxes).sum(1),np.linspace(0,nS,nS),color='k',alpha=0.5,cumulative=True,density=True,histtype='step')
      # ax.set_xticks([])
      # #ax.set_yticks([])
      # ax.yaxis.tick_right()
      # ax.yaxis.set_label_position("right")
      # ax.set_ylim([0,1])
      # #ax.set_ylabel('# neurons')
      # ax.spines['top'].set_visible(False)
      # #ax.spines['right'].set_visible(False)

      pm_thr = 0.5
      idx_pm = ((self.results['p_matched']-self.results['match_2nd'])>pm_thr) | (self.results['p_matched']>0.95)

      plt.figure(figsize=(7,1.5))
      ax_sc1 = plt.axes([0.1,0.3,0.35,0.65])

      ax = ax_sc1.twinx()
      ax.hist(self.results['match_2nd'][idxes].flat,np.linspace(0,1,51),facecolor='tab:red',alpha=0.3)
      #ax.invert_yaxis()
      ax.set_yticks([])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax = ax_sc1.twiny()
      ax.hist(self.results['p_matched'][idxes].flat,np.linspace(0,1,51),facecolor='tab:blue',orientation='horizontal',alpha=0.3)
      ax.set_xticks([])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax_sc1.plot(self.results['match_2nd'][idxes].flat,self.results['p_matched'][idxes].flat,'.',markeredgewidth=0,color='k',markersize=1)
      ax_sc1.plot([0,1],[0,1],'--',color='tab:red',lw=0.5)
      ax_sc1.plot([0,0.45],[0.5,0.95],'--',color='tab:orange',lw=1)
      ax_sc1.plot([0.45,1],[0.95,0.95],'--',color='tab:orange',lw=1)
      ax_sc1.set_ylabel('$p^{\\asterisk}$')
      ax_sc1.set_xlabel('max($p\\backslash p^{\\asterisk}$)')
      ax_sc1.spines['top'].set_visible(False)
      ax_sc1.spines['right'].set_visible(False)

      # match vs max
      # idxes &= idx_pm

      p_matched = np.copy(self.results['p_matched'])
      # p_matched[~idx_pm] = np.NaN
      # avg matchscore per cluster, min match score per cluster, ...
      ax_sc2 = plt.axes([0.6,0.3,0.35,0.65])
      #plt.hist(np.nanmean(self.results['p_matched'],1),np.linspace(0,1,51))
      ax = ax_sc2.twinx()
      ax.hist(np.nanmin(p_matched,1),np.linspace(0,1,51),facecolor='tab:red',alpha=0.3)
      ax.set_yticks([])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax = ax_sc2.twiny()
      ax.hist(np.nanmean(p_matched,axis=1),np.linspace(0,1,51),facecolor='tab:blue',orientation='horizontal',alpha=0.3)
      ax.set_xticks([])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax_sc2.plot(np.nanmin(p_matched,1),np.nanmean(p_matched,axis=1),'.',markeredgewidth=0,color='k',markersize=1)
      ax_sc2.set_xlabel('min($p^{\\asterisk}$)')
      ax_sc2.set_ylabel('$\left\langle p^{\\asterisk} \\right\\rangle$')
      ax_sc2.spines['top'].set_visible(False)
      ax_sc2.spines['right'].set_visible(False)

      ### plot positions of neurons
      plt.tight_layout()
      plt.show(block=False)

      if sv:
          ext = 'png'
          path = os.path.join(self.para['pathMouse'],'Figures/Sheintuch_registration_score_stats_%s_%s.%s'%(self.para['model'],suffix,ext))
          plt.savefig(path,format=ext,dpi=300)



def mean_of_trunc_lognorm(mu,sigma,trunc_loc):

  alpha = (trunc_loc[0]-mu)/sigma
  beta = (trunc_loc[1]-mu)/sigma

  phi = lambda x : 1/np.sqrt(2*np.pi)*np.exp(-1/2*x**2)
  psi = lambda x : 1/2*(1 + sp.special.erf(x/np.sqrt(2)))

  trunc_mean = mu + sigma * (phi(alpha) - phi(beta))/(psi(beta) - psi(alpha))
  trunc_var = np.sqrt(sigma**2 * (1 + (alpha*phi(alpha) - beta*phi(beta))/(psi(beta) - psi(alpha)) - ((phi(alpha) - phi(beta))/(psi(beta) - psi(alpha)))**2))

  return trunc_mean,trunc_var

def norm_nrg(a_):

  a = a_.copy()
  dims = a.shape
  a = a.reshape(-1, order='F')
  indx = np.argsort(a, axis=None)[::-1]
  cumEn = np.cumsum(a.flatten()[indx]**2)
  cumEn /= cumEn[-1]
  a = np.zeros(np.prod(dims))
  a[indx] = cumEn
  return a.reshape(dims, order='F')


def add_number(fig,ax,order=1,offset=None):

    # offset = [-175,50] if offset is None else offset
    offset = [-75,25] if offset is None else offset
    pos = fig.transFigure.transform(plt.get(ax,'position'))
    x = pos[0,0]+offset[0]
    y = pos[1,1]+offset[1]
    ax.text(x=x,y=y,s='%s)'%chr(96+order),ha='center',va='center',transform=None,weight='bold',fontsize=14)
