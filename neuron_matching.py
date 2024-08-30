'''
  function written by Alexander Schmidt, based on the paper "Sheintuch et al., ...", allowing for complete registration of neuron footprints across several sessions

  TODO:
    * write plotting procedure for cluster footprints (3D), to allow manual corrections
    * save data-attribute / structure after model-building, not only after registration
    * change save structure, such that all that is needed for further analysis is readily accessible:
        - filePath of results file
        - no redundancy in SNR, r_values, cnn saving
        - 'remap' into 'alignment' structure in results
        - cm only needed once
  
  last updated on January 28th, 2024
'''


import os, cv2, copy, time, logging, pickle, tqdm, h5py
import numpy as np
from inspect import signature

import scipy as sp
from scipy import signal
from scipy.optimize import curve_fit, linear_sum_assignment
import scipy.stats as sstats

from matplotlib import pyplot as plt, rc, colors as mcolors, patches as mppatches, lines as mplines, cm

from matplotlib.widgets import Slider

from .utils import *

logging.basicConfig(level=logging.INFO)

class matching:

    def __init__(self,mousePath,paths=None,suffix='',matlab=False,logLevel=logging.ERROR):
        '''
            TODO:
                * put match results into more handy shape:
                    - list of filePaths
                    - SNR, etc not twice (only in results)
                    - remap in results, not data (shift=0,corr=1)
                    - p_same containing best and best_without_match value and being in results
        '''

        self.matlab = matlab
        ## make sure suffix starts with '_'
        if suffix and suffix[0] != '_':
            suffix = '_' + suffix
        
        assert not (paths is None), 'You have to provide a list of paths to the CaImAn results files to be processed!'
        # if not mousePath:
        #     mousePath = 'data/555wt'
        # if not paths:
        #     ## create paths that should be processed
        #     if fileName_results[-1] == '*':
        #         paths = [
        #             os.path.join(mousePath,sessionPath,os.path.splitext(fileName)[0] + suffix + ('.mat' if self.matlab else '.hdf5') ) 
        #             for sessionPath in sorted(os.listdir(mousePath)) if sessionPath.startswith('Session') 
        #             for fileName in os.listdir(os.path.join(mousePath,sessionPath)) if fileName.startswith(fileName_results[:-1])]
        #     else:
        #         paths = [
        #             os.path.join(mousePath,sessionPath,os.path.splitext(fileName_results)[0] + suffix + ('.mat' if self.matlab else '.hdf5') ) 
        #             for sessionPath in sorted(os.listdir(mousePath)) if 'Session' in sessionPath]
        #     # paths.sort()
        #     print(f'{paths=}')

        self.log = logging.getLogger("matchinglogger")
        self.log.setLevel(logLevel)

        
        self.paths = {
            'sessions':     paths,      # list of paths to CaImAn result files to be processed in order
            'data':         mousePath,   # path to which results are stored and loaded from
            'suffix':       suffix,
        }

        # mP = matching_params
        self.params = matching_params
        self.params['model'] = 'correlation'
        self.params['correlation_model'] = 'shifted'
        self.update_bins(self.params['nbins'])

        ## initialize the main data dictionaries
        self.data_blueprint = {
            'nA':       np.zeros(2,'int'),
            'cm':       None,
            'p_same':   None,
            'idx_eval': None,
            'SNR_comp': None,
            'r_values': None,
            'cnn_preds':None,
            'filePath': None,
        }

        self.data = {}
        self.data_tmp = {}

        self.data_cross = {
            'D_ROIs':   [],
            'fp_corr':  [],
        }

        
        self.status = {
            'counts_calculated': False,
            'model_calculated': False,

            'neurons_matched': False,
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
        
        self.params['nbins'] = nbins
        self.params['arrays'] = self.build_arrays(nbins)
        
        

    def build_arrays(self,nbins):
       
        ## create value arrays for distance and footprint correlation
        arrays = {}
        arrays['distance_bounds'] = np.linspace(0,self.params['neighbor_distance'],nbins+1)#[:-1]
        arrays['correlation_bounds'] = np.linspace(0,1,nbins+1)#[:-1]

        arrays['distance_step'] = self.params['neighbor_distance']/nbins
        arrays['correlation_step'] = 1./nbins


        arrays['distance'] = arrays['distance_bounds'][:-1] + arrays['distance_step']/2
        arrays['correlation'] = arrays['correlation_bounds'][:-1] + arrays['correlation_step']/2


        ## update histogram counts (if nbins_new < nbins), merge counts, otherwise split evenly
        # self.log.warning('implement method to update count histograms!')
        return arrays
       
       

    
    def run_matching(self,p_thr=[0.3,0.05],save_results=True):
        print('Now running matching procedure in %s'%self.paths['data'])

        print('Building model for matching ...')
        self.build_model(save_results=save_results)
        
        print('Matching neurons ...')
        self.register_neurons(save_results=save_results,p_thr=p_thr)
        
        print('Done!')


    def build_model(self,save_results=False):

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

        self.update_bins(128)

        self.model = {
            'counts':                 np.zeros((self.params['nbins'],self.params['nbins'],3),'int'),
            'counts_unshifted':       np.zeros((self.params['nbins'],self.params['nbins'],3),'int'),
            'counts_same':            np.zeros((self.params['nbins'],self.params['nbins']),'int'),
            'counts_same_unshifted':  np.zeros((self.params['nbins'],self.params['nbins']),'int'),

            'kernel': {
                'idxes':   {},
                'kde':     {}
              },
        }

        self.nS = len(self.paths['sessions'])
        self.progress = tqdm.tqdm(enumerate(self.paths['sessions']),total=self.nS)
        # print(self.paths['sessions'])

        has_reference = False

        # s_ref = 0
        for (s,self.currentPath) in self.progress:
            self.data[s] = copy.deepcopy(self.data_blueprint)
            self.data[s]['filePath'] = self.currentPath
            self.data[s]['skipped'] = False

            self.A = self.load_footprints(self.currentPath,s)
            
            if isinstance(self.A,bool): 
                self.data[s]['skipped'] = True
                continue
            
            self.progress.set_description(f'Aligning data from {self.currentPath}')
            
            ## prepare (and align) footprints
            prepared,self.data[s]['remap'] = self.prepare_footprints(align_to_reference=has_reference)
            self.calculate_footprint_data(s)

            if not prepared: 
                self.data[s]['skipped'] = True
                continue

            if self.params['use_kde']:
                # self.progress.set_description('Calculate kernel density for Session %d'%s0)
                self.position_kde(s)         # build kernel

            self.update_joint_model(s,s)

            if has_reference:
                self.progress.set_description('Calculate cross-statistics for %s'%self.currentPath)
                self.data_cross['D_ROIs'],self.data_cross['fp_corr'],_ = calculate_statistics(
                    self.A,A_ref=self.A_ref,idx_eval=self.data[s]['idx_eval'],idx_eval_ref=self.data[s_ref]['idx_eval'],
                    binary=self.params['binary'],neighbor_distance=self.params['neighbor_distance'],convert=self.params['pxtomu'],model=self.params['correlation_model'],
                    dims=self.params['dims']
                )        # calculating distances and footprint correlations
                self.progress.set_description('Update model with data from %s'%self.currentPath)
                self.update_joint_model(s,s_ref)

            self.A_ref = self.A.copy()
            self.Cn_ref = self.data_tmp['Cn']
            s_ref = s
            has_reference = True

        self.dynamic_fit()
        
        if save_results:
            self.save_model(suffix=self.paths['suffix'])

    def dynamic_fit(self):
        times = 0
        while times<3:
            try:
                self.fit_model(times=times)
                success = True
            except:
                success = False
            
            if success and self.model['p_same']['which']=='joint':
                print(f'found proper fit @times={times}')
                break
            times += 1
        

    def register_neurons(self, p_thr=[0.3,0.05], save_results=False, model='shifted'):
        
        '''
          This function iterates through sessions chronologically to create a set of all detected neurons, matched to one another based on the created model

            p_thr - (list of floats)
                2 entry-list, specifying probability above which matches are accepted [0] 
                and above which losing contenders for a match are removed from the data [1]
        '''

        self.params['correlation_model'] = model
        
        self.nS = len(self.paths['sessions'])

        self.data = {}


        # if not self.model['f_same']:
        assert self.status['model_calculated'], 'Model not yet created - please run build_model first'
        
        ## load and prepare first set of footprints
        s=0
        while True:
            self.data[s] = copy.deepcopy(self.data_blueprint)
            self.data[s]['filePath'] = self.paths['sessions'][s]
            self.data[s]['skipped'] = False

            self.A = self.load_footprints(self.paths['sessions'][s],s)
            if isinstance(self.A,bool):
                self.data[s]['skipped'] = True
                s += 1
                continue
            else:
                break
        self.progress = tqdm.tqdm(enumerate(self.paths['sessions'][s+1:],start=s+1),total=self.nS-(s+1),leave=True)
        
        self.prepare_footprints(align_to_reference=False)
        self.calculate_footprint_data(s)
        
        self.Cn_ref = self.data_tmp['Cn']
        self.A_ref = self.A[:,self.data[s]['idx_eval']]
        self.A0 = self.A.copy()   ## store initial footprints for common reference
        
        ## initialize reference session, containing the union of all neurons
        self.data['joint'] = copy.deepcopy(self.data_blueprint)
        self.data['joint']['nA'][0] = self.A_ref.shape[1]
        self.data['joint']['idx_eval'] = np.ones(self.data['joint']['nA'][0],'bool')
        self.data['joint']['cm'] = center_of_mass(self.A_ref,self.params['dims'][0],self.params['dims'][1],convert=self.params['pxtomu'])

        ## prepare and initialize assignment- and p_matched-arrays for storing results
        self.results = {
          'assignments': np.full((self.data[s]['nA'][1],self.nS),np.NaN),
          'p_matched': np.full((self.data[s]['nA'][1],self.nS,2),np.NaN),
          'Cn_corr': np.full((self.nS,self.nS),np.NaN),
        }
        self.results['assignments'][:,s] = np.where(self.data[s]['idx_eval'])[0]
        self.results['p_matched'][:,0,0] = 1

        for (s,self.currentPath) in self.progress:
            if not self.currentPath:
                continue

            session_name = os.path.dirname(self.paths['sessions'][s]).split('/')[-1]
            self.data[s] = copy.deepcopy(self.data_blueprint)
            self.data[s]['filePath'] = self.currentPath
            self.data[s]['skipped'] = False

            self.A = self.load_footprints(self.currentPath,s)

            if isinstance(self.A,bool):
                self.data[s]['skipped'] = True
                continue
                
            for s0 in range(s):
                # self.progress.set_description('Calculate image correlations for Session %d'%s)
                if (self.paths['sessions'][s0]) and ('Cn' in self.data[s0].keys()) and ('Cn' in self.data[s].keys()):
                    c_max,_,_ = calculate_img_correlation(self.data[s0]['Cn'],self.data[s]['Cn'],plot_bool=False)
                    self.results['Cn_corr'][s0,s] = self.results['Cn_corr'][s,s0] = c_max      
            
            self.progress.set_description(f"A union size: {self.data['joint']['nA'][0]}, Preparing footprints from {session_name}")

            ## preparing data of current session
            prepared,self.data[s]['remap'] = self.prepare_footprints(A_ref=self.A0)
            self.calculate_footprint_data(s)

            if not prepared:
                self.data[s]['skipped'] = True
                continue


            ## calculate matching probability between each pair of neurons 
            self.progress.set_description(f"A union size: {self.data['joint']['nA'][0]}, Calculate statistics for {session_name}")
            self.data_cross['D_ROIs'],self.data_cross['fp_corr'],_ = calculate_statistics(
                self.A,A_ref=self.A_ref,idx_eval=self.data[s]['idx_eval'],idx_eval_ref=self.data['joint']['idx_eval'],
                binary=self.params['binary'],neighbor_distance=self.params['neighbor_distance'],convert=self.params['pxtomu'],model=self.params['correlation_model'],
                dims=self.params['dims']
            )

            idx_fp = 1 if self.params['correlation_model'] == 'shifted' else 0
            self.data[s]['p_same'] = calculate_p(
                self.data_cross['D_ROIs'],self.data_cross['fp_corr'][idx_fp,...],
                self.model['f_same'],self.params['neighbor_distance']
            )


            ## run hungarian algorithm (HA) with (1-p_same) as score
            self.progress.set_description(f"A union size: {self.data['joint']['nA'][0]}, Perform Hungarian matching on {session_name}")
            matches = linear_sum_assignment(1 - self.data[s]['p_same'].toarray())
            p_matched = self.data[s]['p_same'].toarray()[matches]
            
            idx_TP = np.where(p_matched > p_thr[0])[0] ## thresholding results (HA matches all pairs, but we only want matches above p_thr)
            if len(idx_TP) > 0:
                matched_ref = matches[0][idx_TP]    # matched neurons in s_ref
                matched = matches[1][idx_TP]        # matched neurons in s
                # print(idx_TP,matched_ref)
            else:
                matched_ref = np.array([],'int')
                matched = np.array([],'int')
            

            ## find neurons which were not matched in current and reference session
            non_matched_ref = np.setdiff1d(list(range(self.data['joint']['nA'][0])), matched_ref)
            non_matched = np.setdiff1d(list(np.where(self.data[s]['idx_eval'])[0]), matches[1][idx_TP])
            non_matched = non_matched[self.data[s]['idx_eval'][non_matched]]

            # print(matched,non_matched)
            ## calculate number of matches found
            TP = np.sum(p_matched > p_thr[0]).astype('float32')
            self.A_ref = self.A_ref.tolil()
            
            ## update footprint shapes of matched neurons with A_ref = (1-p/2)*A_ref + p/2*A to maintain part or all of original shape, depending on p_matched
            self.A_ref[:,matched_ref] = self.A_ref[:,matched_ref].multiply(1-p_matched[idx_TP]/2) + self.A[:,matched].multiply(p_matched[idx_TP]/2)

            ## removing footprints from the data which were competing with another one 
            ## to be matched and lost, but have significant probability to be the same
            ## this step ensures, that downstream session don't confuse this one and the
            ## 'winner', leading to arbitrary assignments between two clusters
            for nm in non_matched:
                p_all = self.data[s]['p_same'][:,nm].todense()
                if np.any(p_all>p_thr[1]):
                #    print(f'!! neuron {nm} is removed, as it is nonmatched and has high match probability:',p_all)[p_all>0])
                #    print(np.where(p_all>0))
                    non_matched = non_matched[non_matched!=nm]

            
            ## append new neuron footprints to union
            self.A_ref = sp.sparse.hstack([self.A_ref, self.A[:,non_matched]]).asformat('csc')

            ## update union data
            self.data['joint']['nA'][0] = self.A_ref.shape[1]
            self.data['joint']['idx_eval'] = np.ones(self.data['joint']['nA'][0],'bool')
            self.data['joint']['cm'] = center_of_mass(self.A_ref,self.params['dims'][0],self.params['dims'][1],convert=self.params['pxtomu'])

            
            ## write neuron indices of neurons from this session
            self.results['assignments'][matched_ref,s] = matched   # ... matched neurons are added

            ## ... and non-matched (new) neurons are appended 
            N_add = len(non_matched)
            match_add = np.zeros((N_add,self.nS))*np.NaN
            match_add[:,s] = non_matched
            self.results['assignments'] = np.concatenate([self.results['assignments'],match_add],axis=0)

            ## write match probabilities to matched neurons and reshape array to new neuron number
            self.results['p_matched'][matched_ref,s,0] = p_matched[idx_TP]

            ## write best non-matching probability
            p_all = self.data[s]['p_same'].toarray()
            self.results['p_matched'][matched_ref,s,1] = [max(p_all[c,np.where(p_all[c,:]!=self.results['p_matched'][c,s,0])[0]]) for c in matched_ref]
            # self.results['p_matched'][non_matched,s,1] = [max(p_all[c,np.where(p_all[c,:]!=self.results['p_matched'][c,s,0])[0]]) for c in matched_ref]

            
            p_same_add = np.full((N_add,self.nS,2),np.NaN)
            p_same_add[:,s,0] = 1
            
            self.results['p_matched'] = np.concatenate([self.results['p_matched'],p_same_add],axis=0)

            # if np.any(np.all(self.results['p_matched']>0.9,axis=2)):
            #     print('double match!')
            #     return


        ## some post-processing to create cluster-structures values / statistics
        self.store_results()

        # finally, save results
        if save_results:
            self.save_registration(suffix=self.paths['suffix'])
            self.save_data(suffix=self.paths['suffix'])


    def store_results(self):
        self.results['cm'] = np.zeros(self.results['assignments'].shape + (2,)) * np.NaN
        for key in ['SNR_comp','r_values','cnn_preds']:
            self.results[key] = np.zeros_like(self.results['assignments'])
        
        # for s in range(self.nS):
        self.results['remap'] = {
            'shift': np.zeros((self.nS,2)),
            'transposed': np.zeros(self.nS,'bool'),
            'corr': np.zeros(self.nS),
            'corr_zscored': np.zeros(self.nS),
        }
        self.results['filePath'] = ['']*self.nS

        has_reference = False
        for s in self.data:
            if s=='joint': continue

            if 'remap' in self.data[s].keys():
                self.results['remap']['transposed'][s] = self.data[s]['remap']['transposed'] if has_reference else False
                self.results['remap']['shift'][s,:] = self.data[s]['remap']['shift'] if has_reference else [0,0]
                self.results['remap']['corr'][s] = self.data[s]['remap']['c_max'] if has_reference else 1
                self.results['remap']['corr_zscored'][s] = self.data[s]['remap']['c_zscored'] if has_reference else 1

                
            if self.data[s]['skipped']: continue
            
            self.results['filePath'][s] = self.data[s]['filePath']
            
            idx_c = np.where(~np.isnan(self.results['assignments'][:,s]))[0]
            idx_n = self.results['assignments'][idx_c,s].astype('int')

            for key in ['cm','SNR_comp','r_values','cnn_preds']:
                try:
                    self.results[key][idx_c,s,...] = self.data[s][key][idx_n,...]
                except:
                    pass
            
            has_reference = True


    def load_footprints(self,loadPath,s=None):
        '''
          function to load results from neuron detection (CaImAn, OnACID) and store in according dictionaries

          TODO:
            * implement min/max thresholding (maybe shift thresholding to other part of code?)
        '''
        
        if loadPath and os.path.exists(loadPath):
            ld = load_data(loadPath)
            
            if 'Cn' in ld.keys():
                Cn = ld['Cn'].T
            else:
                self.log.warning('Cn not in result files. constructing own Cn from footprints!')
                Cn = np.array(ld['A'].sum(axis=1).reshape(*self.params['dims']))
            
            self.data_tmp['Cn'] = Cn
            self.data_tmp['C'] = ld['C']

            if not s is None:
                self.data[s]['Cn'] = Cn
            
                ## load some data necessary for further processing
                if np.all([key in ld.keys() for key in ['SNR_comp','r_values','cnn_preds']]):
                    ## if evaluation parameters are present, use them to define used neurons
                    for key in ['SNR_comp','r_values','cnn_preds']:
                        self.data[s][key] = ld[key]

                    ## threshold neurons according to evaluation parameters
                    self.data[s]['idx_eval'] = ((ld['SNR_comp']>self.params['SNR_lowest']) & (ld['r_values']>self.params['rval_lowest']) & (ld['cnn_preds']>self.params['cnn_lowest'])) & ((ld['SNR_comp']>self.params['min_SNR']) | (ld['r_values']>self.params['rval_thr']) | (ld['cnn_preds']>self.params['min_cnn_thr']))
                else:
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
        
        remap = {
           'shift':     np.full((2,),np.NaN),
           'flow':      np.zeros((2,)+self.params['dims']),
           'c_max':     np.NaN,
           'c_zscored': np.NaN, ## z-score of correlation
           'transposed':False,
        }
        
        if align_to_reference:  
            ## align footprints A to reference set A_ref
            
            # if no reference set of footprints is specified, use current reference set
            if A_ref is None: 
                A_ref = self.A_ref
            
            ## cast this one to sparse as well, if needed
            if 'csc_matrix' not in str(type(A_ref)):
                A_ref = sp.sparse.csc_matrix(A_ref)
            
            ## test whether images might be transposed (sometimes happens...) and flip accordingly
            # Cn = np.array(self.A.sum(1).reshape(512,512))
            # Cn_ref = np.array(A_ref.sum(1).reshape(512,512))
            Cn = self.data_tmp['Cn']
            Cn_ref = self.Cn_ref
            c_max,c_zscored,_ = calculate_img_correlation(Cn_ref,Cn,plot_bool=False)
            c_max_T,c_zscored_T,_ = calculate_img_correlation(Cn_ref,Cn.T,plot_bool=False)
            # print('z scores:',c_zscored,c_zscored_T)
            remap['c_max'] = c_max

            ##  if no good alignment is found, don't include this session in the matching procedure (e.g. if imaging window is shifted too much)
            if (c_zscored < self.params['min_session_correlation_zscore']) & \
                (c_zscored_T < self.params['min_session_correlation_zscore']):
            # if (c_max < self.params['min_session_correlation']) & \
            #   (c_max_T < self.params['min_session_correlation']):
                print(f'Identified correlation {c_max} too low in session {self.currentPath}, skipping alignment, as sessions appear to have no common imaging window!')
                return False, remap
            
            if (c_max>0.95) | (c_max_T>0.95):
                print(f'High correlation {c_max} found in session {self.currentPath}, skipping alignment as it is likely to be the same imaging data')
                return False, remap
            

            if (c_max_T > c_max) & (c_max_T > self.params['min_session_correlation']):
                print('Transposed image')
                self.A = sp.sparse.hstack([sp.sparse.csc_matrix(img.reshape(self.params['dims']).transpose().reshape(-1,1)) for img in self.A.transpose()])
                remap['transposed'] = True
            
            ## calculate rigid shift and optical flow from reduced (cumulative) footprint arrays
            remap['shift'],flow,remap['c_max'],remap['c_zscored'] = get_shift_and_flow(A_ref,self.A,self.params['dims'],projection=1,plot_bool=False)
            # remap['shift'],flow,remap['c_max'],remap['c_zscored'] = get_shift_and_flow(Cn_ref,Cn,self.params['dims'],projection=None,plot_bool=False)
            remap['flow'][0,...] = flow[...,0]
            remap['flow'][1,...] = flow[...,1]

            total_shift = np.sum(np.sqrt(np.array([s**2 for s in remap['shift']]).sum()))
            if total_shift > self.params['max_session_shift']:
                print(f'Large shift {total_shift} identified in session {self.currentPath}, skipping alignment as sessions appear to have no significantly common imaging window!')
                return False, remap

            ## use shift and flow to align footprints - define reverse mapping
            x_remap,y_remap = build_remap_from_shift_and_flow(self.params['dims'],remap['shift'],remap['flow'] if use_opt_flow else None)
            
            ## use shift and flow to align footprints - apply reverse mapping
            self.A = sp.sparse.hstack([
               sp.sparse.csc_matrix(                    # cast results to sparse type
                  cv2.remap(
                      img.reshape(self.params['dims']),   # reshape image to original dimensions
                      x_remap, y_remap,                 # apply reverse identified shift and flow
                      cv2.INTER_CUBIC                 
                  ).reshape(-1,1)                       # reshape back to allow sparse storage
                ) for img in self.A.toarray().T        # loop through all footprints
            ])

        ## finally, apply normalization (watch out! changed normalization to /sum instead of /max)
        # self.A_ref = normalize_sparse_array(self.A_ref)
        self.A = normalize_sparse_array(self.A)
        
        return True, remap
    

    def calculate_footprint_data(self,s):

        # print(f'Session: {s}, A: {self.A.shape}')

        ## calculating various statistics
        self.data[s]['idx_eval'] &= (np.diff(self.A.indptr) != 0) ## finding non-empty rows in sparse array (https://mike.place/2015/sparse/)

        # print('\n idx evals:',self.data[s]['idx_eval'].sum())
        self.data[s]['nA'][0] = self.A.shape[1]    # number of neurons
        self.data[s]['cm'] = center_of_mass(self.A,*self.params['dims'],convert=self.params['pxtomu'])

        self.progress.set_description(f"Calculate self-referencing statistics for {self.paths['sessions'][s]}")

        ## find and mark potential duplicates of neuron footprints in session
        self.data_cross['D_ROIs'],self.data_cross['fp_corr'],idx_remove = calculate_statistics(
            self.A,idx_eval=self.data[s]['idx_eval'],
            SNR_comp=self.data[s]['SNR_comp'],C=self.data_tmp['C'],
            binary=self.params['binary'],neighbor_distance=self.params['neighbor_distance'],convert=self.params['pxtomu'],model=self.params['correlation_model'],
            dims=self.params['dims']
        )
        # print('\n idx remove:',idx_remove)
        self.data[s]['idx_eval'][idx_remove] = False
        self.data[s]['nA'][1] = self.data[s]['idx_eval'].sum()


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
        idxes = self.model['kernel']['idxes'][s_ref] if self.params['use_kde'] else np.ones(self.data[s_ref]['nA'][0],'bool')

        ## find all neuron pairs below a distance threshold
        neighbors = self.data_cross['D_ROIs'][idxes,:] < self.params['neighbor_distance']

        NN_idx = np.zeros((self.data[s_ref]['nA'][0],self.data[s]['nA'][0]),'bool')
        if s!=s_ref:
            ## identifying next-neighbours
            idx_NN = np.nanargmin(self.data_cross['D_ROIs'][self.data[s_ref]['idx_eval'],:],axis=1)

            NN_idx[self.data[s_ref]['idx_eval'],idx_NN] = True
            NN_idx = NN_idx[idxes,:][neighbors]
        else:
            np.fill_diagonal(NN_idx,True)
            # NN_idx[~idxes,:] = False
            NN_idx = NN_idx[idxes,:][neighbors]


        ## obtain distance and correlation values of close neighbors, only
        D_ROIs = self.data_cross['D_ROIs'][idxes,:][neighbors]
        fp_corr = self.data_cross['fp_corr'][0,idxes,:][neighbors]
        fp_corr_shifted = self.data_cross['fp_corr'][1,idxes,:][neighbors]

        ## update count histogram with data from current session pair
        for i in tqdm.tqdm(range(self.params['nbins']),desc='updating joint model',leave=False):

            ## find distance indices of values falling into the current bin
            idx_dist = (D_ROIs >= self.params['arrays']['distance_bounds'][i]) & (D_ROIs < self.params['arrays']['distance_bounds'][i+1])

            for j in range(self.params['nbins']):

                ## differentiate between the two models for calculating footprint-correlation
                if (self.params['correlation_model']=='unshifted') | (self.params['correlation_model']=='both'):
                    ## find correlation indices of values falling into the current bin
                    idx_fp = (fp_corr > self.params['arrays']['correlation_bounds'][j]) & (self.params['arrays']['correlation_bounds'][j+1] > fp_corr)
                    idx_vals = idx_dist & idx_fp

                    if s==s_ref:  # for self-comparing
                        self.model['counts_same_unshifted'][i,j] += np.count_nonzero(idx_vals & ~NN_idx)

                    else:         # for cross-comparing
                        self.model['counts_unshifted'][i,j,0] += np.count_nonzero(idx_vals)
                        self.model['counts_unshifted'][i,j,1] += np.count_nonzero(idx_vals & NN_idx)
                        self.model['counts_unshifted'][i,j,2] += np.count_nonzero(idx_vals & ~NN_idx)

                if (self.params['correlation_model']=='shifted') | (self.params['correlation_model']=='both'):
                    idx_fp = (fp_corr_shifted > self.params['arrays']['correlation_bounds'][j]) & (self.params['arrays']['correlation_bounds'][j+1] > fp_corr_shifted)
                    idx_vals = idx_dist & idx_fp
                    if s==s_ref:  # for self-comparing
                        # self.model['counts_same'][i,j] += np.count_nonzero(idx_vals)
                        self.model['counts_same'][i,j] += np.count_nonzero(idx_vals & ~NN_idx)
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
        x_grid, y_grid = np.meshgrid(np.linspace(0,self.params['dims'][0]*self.params['pxtomu'],self.params['dims'][0]), np.linspace(0,self.params['dims'][1]*self.params['pxtomu'],self.params['dims'][1]))
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        kde = sp.stats.gaussian_kde(self.data[s]['cm'][self.data[s]['idx_eval'],:].T)
        self.model['kernel']['kde'][s] = np.reshape(kde(positions),x_grid.shape)


        cm_px = (self.data[s]['cm'][self.data[s]['idx_eval'],:]/self.params['pxtomu']).astype('int')
        kde_at_com = np.zeros(self.data[s]['nA'][0])*np.NaN
        kde_at_com[self.data[s]['idx_eval']] = self.model['kernel']['kde'][s][cm_px[:,1],cm_px[:,0]]
        self.model['kernel']['idxes'][s] = (kde_at_com > np.quantile(self.model['kernel']['kde'][s],self.params['qtl'][0])) & (kde_at_com < np.quantile(self.model['kernel']['kde'][s],self.params['qtl'][1]))

        if plot_bool:
            plt.figure()
            h_kde = plt.imshow(self.model['kernel']['kde'][s],cmap=plt.cm.gist_earth_r,origin='lower',extent=[0,self.params['dims'][0]*self.params['pxtomu'],0,self.params['dims'][1]*self.params['pxtomu']])
            #if s>0:
            #col = self.data_cross['D_ROIs'].min(1)
            #else:
            #col = 'w'
            plt.scatter(self.data[s]['cm'][:,0],self.data[s]['cm'][:,1],c='w',s=5+10*self.model['kernel']['idxes'][s],clim=[0,10],cmap='YlOrRd')
            plt.xlim([0,self.params['dims'][0]*self.params['pxtomu']])
            plt.ylim([0,self.params['dims'][1]*self.params['pxtomu']])

            # cm_px = (self.data['cm'][s]/self.params['pxtomu']).astype('int')
            # kde_at_cm = self.model['kernel']['kde'][s][cm_px[:,1],cm_px[:,0]]
            plt.colorbar(h_kde)
            plt.show(block=False)


    def fit_model(self,correlation_model='shifted',times=0):
        '''

        '''

        self.params['correlation_model'] = correlation_model

        neighbor_dictionary = {
            'NN':    [],
            'nNN':   [],
            'all':   []
          }
        
        self.model = self.model | {            
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
                  'correlation':    copy.deepcopy(neighbor_dictionary)
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
                'joint': np.zeros(0)
              },
            
            'f_same':         False,
          }
        
        # print(self.params['correlation_model'])
        if (not (self.params['correlation_model'] == 'unshifted')) & (not (self.params['correlation_model']=='shifted')):
            raise Exception('Please specify model to be either "shifted" or "unshifted"')

        key_counts = 'counts' if self.params['correlation_model']=='shifted' else 'counts_unshifted'
        counts = scale_down_counts(self.model[key_counts].astype('float32'),times=times)

        nbins = counts.shape[0]
        self.update_bins(nbins)
        
        
        def build_single_model(model,times=0):
            
            m = 0 if model=='correlation' else 1
            max_val = 1. if model=='correlation' else self.params['neighbor_distance']

            fit_fun, fit_bounds = self.set_functions(model,'single')

            ## obtain fit parameters from fit to count histograms
            for p,pop in zip((1,2,0),['NN','nNN','all']):

                if pop=='all':
                    p0 = (self.model[key_counts][...,1].sum()/self.model[key_counts][...,0].sum(),) + \
                        tuple(self.model['fit_parameter']['single'][model]['NN']) + \
                        tuple(self.model['fit_parameter']['single'][model]['nNN'])
                else:
                    p0 = None

                failed=True
                nbins = self.model[key_counts].shape[0]
                # while failed==True and nbins>10:
                #     try:
                self.log.debug(model,pop)

                counts = scale_down_counts(self.model[key_counts].astype('float32'),times=times)
                nbins = counts.shape[0]
                self.update_bins(nbins)
                
                normed_counts = counts[...,p].sum(m)/counts[...,p].sum()/self.params['arrays'][f'{model}_step']
                plot=False
                if plot:
                    plt.figure()
                    plt.plot(self.params['arrays'][model],normed_counts)

                    if pop=='all':
                        plt.plot(self.params['arrays'][model],fit_fun[pop](self.params['arrays'][model],*p0),'k--')
                
                self.model['fit_parameter']['single'][model][pop] = curve_fit(
                    fit_fun[pop],
                    self.params['arrays'][model],
                    normed_counts,
                    bounds=fit_bounds[pop],
                    p0=p0
                )[0]

                if plot:
                    plt.plot(self.params['arrays'][model],fit_fun[pop](self.params['arrays'][model],*self.model['fit_parameter']['single'][model][pop]))
                    plt.show(block=False)

                failed=False
                    # except:
                    #     times += 1
                    #     print(f'Finding solution failed. Scaling down counts by factor {times} ({nbins} bins)')


            ## build functions for NN population and overall
            ## using "all model", as it doesn't overestimate wrongly assigned tails of NN distribution

            fun_NN = fun_wrapper(
                fit_fun['NN'],
                self.params['arrays'][model],
                self.model['fit_parameter']['single'][model]['all'][1:self.model['fit_function']['single'][model]['NN_Nparams']]
            ) * self.model['fit_parameter']['single'][model]['all'][0]

            fun_nNN = fun_wrapper(
                fit_fun['nNN'],
                self.params['arrays'][model],
                self.model['fit_parameter']['single'][model]['all'][self.model['fit_function']['single'][model]['NN_Nparams']:]
            ) * (1-self.model['fit_parameter']['single'][model]['all'][0])
            

            if model=='correlation':
                cdf_NN = np.cumsum(fun_NN)/np.sum(fun_NN)
                cdf_nNN = np.cumsum(fun_nNN[::-1])[::-1]/np.sum(fun_nNN)
            else:
                cdf_NN = np.cumsum(fun_NN[::-1])[::-1]/np.sum(fun_NN)
                cdf_nNN = np.cumsum(fun_nNN)/np.sum(fun_nNN)
            
            self.model['p_same']['single'][model] = cdf_NN/(cdf_NN+cdf_nNN)
        
            
        

        def build_joint_model(model,times=0,counts_thr = 20):
            
            # 'model' defines the single model functions which are used to calculate the joint model, 
            # while the 'weight model' merely specifies the weighting between NN and nNN functions

            counts = scale_down_counts(self.model[key_counts].astype('float32'),times=times)
            nbins = counts.shape[0]
            self.update_bins(nbins)

            m = 1 if model=='correlation' else 0
            
            fit_fun, fit_bounds = self.set_functions(model,'joint')

            # normalized_histogram = counts/counts.sum(m,keepdims=True)/self.params['arrays'][f'{model}_step']# * nbins/max_val
            # normalized_histogram[np.isnan(normalized_histogram)] = 0
            # print(normalized_histogram.shape)

            self.model['fit_parameter']['joint'][model] = {
                'NN':np.full((nbins,self.model['fit_function']['single'][model]['NN_Nparams']-1),np.NaN),
                'nNN':np.full((nbins,self.model['fit_function']['single'][model]['nNN_Nparams']-1),np.NaN)
                #'all':np.zeros((nbins,len(self.model['fit_parameter']['single']['correlation']['all'])))*np.NaN}
                }

            for p,pop in enumerate(['NN','nNN'],start=1):
                fitted = False
                for n in range(nbins):
                    ### to fp-correlation: NN - reverse lognormal, nNN - reverse lognormal
                    fit_data = np.take(counts[...,p],n,axis=abs(m-1))
                    # print(f'n data (@bin {n}):',fit_data.sum())
                    if fit_data.sum() > counts_thr:
                        fit_data /= fit_data.sum()
                        # print(fit_data)
                        self.log.debug(f'data for {pop} {model}-distribution: ', fit_data)
                        
                        # scale_times = 0
                        # while nbins > 16:
                            # fit_histogram = np.copy(normalized_histogram)
                            # fit_histogram = scale_down_counts(normalized_histogram,times=scale_times)
                            # nbins = fit_histogram.shape[0]
                            # self.update_bins(nbins)
                            # print(scale_times,nbins)
                            # print(self.params['arrays'][model])
                            # print('vals:',np.take(fit_histogram[...,p],i,axis=abs(m-1)))
                            # print(fit_bounds[pop])
                            # try:
                            
                                
                                # if pop=='all':
                                #     plt.plot(self.params['arrays'][model],fit_fun[pop](self.params['arrays'][model],*p0),'k--')
                        plot=False
                        if plot:
                            plt.figure()
                            plt.plot(self.params['arrays'][model],fit_data)

                        try:
                            self.model['fit_parameter']['joint'][model][pop][n,:] = curve_fit(
                                fit_fun[pop],
                                self.params['arrays'][model],
                                fit_data,
                                bounds=fit_bounds[pop],
                                # p0=self.model['fit_parameter']['joint'][model][pop][i-1,:] if fitted else None
                            )[0]
                            self.log.debug(f'fit results @bin {n}:',self.model['fit_parameter']['joint'][model][pop][n,:])
                        except:
                            print(f'fit failed for {model} model and population {pop} @bin {n}/{nbins}')
                            if plot:
                                plt.plot(self.params['arrays'][model],fit_fun[pop](self.params['arrays'][model],*self.model['fit_parameter']['joint'][model][pop][n,:]))
                                plt.title(f'{model} model, {pop} population, bin {n} ({nbins} bins)')
                                plt.show(block=False)
                            continue

                            #     break
                            # except:
                            #     scale_times += 1
                            #     self.log.debug(f'Finding solution failed. Scaling down counts by factor {scale_times} ({nbins/2} bins)')
                            
                        if plot:
                            plt.plot(self.params['arrays'][model],fit_fun[pop](self.params['arrays'][model],*self.model['fit_parameter']['joint'][model][pop][n,:]))
                            plt.title(f'{model} model, {pop} population, bin {n} ({nbins} bins)')
                            plt.show(block=False)
                            
                        # nbins = normalized_histogram.shape[0]
                        # self.update_bins(nbins)
                        fitted = True
                    else:
                        fitted = False
            
            extrapolate = True
            postprocess = True
            ## running smoothing, intra- and extrapolation
            filter_footprint = (int(nbins/10),0)
            # print('blabla')
            if postprocess:
                # print('smoothing')
                # print(filter_footprint)
                for pop in ['NN','nNN']:#,'all']
                    # print(pop)
                    # print('params before smoothing:',self.model['fit_parameter']['joint'][model][pop])
                    #for ax in range(self.model['fit_parameter']['joint'][key][pop].shape(1)):
                    # self.model['fit_parameter']['joint'][model][pop] = nanmedian_filter(self.model['fit_parameter']['joint'][model][pop],np.ones(filter_footprint))
                    self.model['fit_parameter']['joint'][model][pop] = nanmedian_filter(self.model['fit_parameter']['joint'][model][pop],np.ones((3,1)))
                    # print('params after median:',self.model['fit_parameter']['joint'][model][pop])
                    self.model['fit_parameter']['joint'][model][pop] = nangauss_filter(self.model['fit_parameter']['joint'][model][pop],filter_footprint)
                    # print(self.model['fit_parameter']['joint'][key][pop])
                    # print('params after smoothing:',self.model['fit_parameter']['joint'][model][pop])

                    if extrapolate:
                        # print('extrapolating data...')

                        for ax in range(self.model['fit_parameter']['joint'][model][pop].shape[1]):
                            ## find first/last index, at which parameter has a non-nan value
                            nan_idx = np.isnan(self.model['fit_parameter']['joint'][model][pop][:,ax])
                            
                            if nan_idx[0] and (~nan_idx).sum()>1:  ## extrapolate beginning
                                idx = np.where(~nan_idx)[0][:int(nbins/5)]
                                y_arr = self.model['fit_parameter']['joint'][model][pop][idx,ax]
                                f_interp = np.polyfit(self.params['arrays'][model][idx],y_arr,1)
                                poly_fun = np.poly1d(f_interp)

                                self.model['fit_parameter']['joint'][model][pop][:idx[0],ax] = poly_fun(self.params['arrays'][model][:idx[0]])
                            
                            if nan_idx[-1] and (~nan_idx).sum()>1:  ## extrapolate end
                                idx = np.where(~nan_idx)[0][-int(nbins/5):]
                                y_arr = self.model['fit_parameter']['joint'][model][pop][idx,ax]
                                f_interp = np.polyfit(self.params['arrays'][model][idx],y_arr,1)
                                poly_fun = np.poly1d(f_interp)

                                self.model['fit_parameter']['joint'][model][pop][idx[-1]+1:,ax] = poly_fun(self.params['arrays'][model][idx[-1]+1:])


        build_single_model('distance',times)
        build_single_model('correlation',times)
        
        ## build both models first        
        build_joint_model('distance',times)
        build_joint_model('correlation',times)

        ## define probability density functions from "major" model
        weight_model = 'distance' if self.params['model']=='correlation' else 'correlation'

        fit_fun,_ = self.set_functions(self.params['model'],'joint') # could also be distance
        # self.model['pdf']['joint'] = np.zeros((2,nbins,nbins))
        
        self.model['p_same']['joint'] = np.zeros((nbins,nbins))
        for n in range(nbins):

            fun_NN = fun_wrapper(
                fit_fun['NN'],
                self.params['arrays'][self.params['model']],
                self.model['fit_parameter']['joint'][self.params['model']]['NN'][n,:]
            )

            fun_nNN = fun_wrapper(
                fit_fun['nNN'],
                self.params['arrays'][self.params['model']],
                self.model['fit_parameter']['joint'][self.params['model']]['nNN'][n,:]
            )
            

            # for p,pop in enumerate(['NN','nNN']):
            #     # if not np.any(np.isnan(self.model['fit_parameter']['joint'][self.params['model']][pop][n,:])):
            #         f_pop = fun_wrapper(
            #             fit_fun[pop],
            #             self.params['arrays'][self.params['model']],
            #             self.model['fit_parameter']['joint'][self.params['model']][pop][n,:]
            #         )
                    
            weight = self.model['p_same']['single'][weight_model][n]
            # if pop=='nNN':
            #     weight = 1 - weight
            # if self.params['model']=='correlation':
            #     self.model['pdf']['joint'][p,n,:] = f_pop*weight
            # else:
            #     self.model['pdf']['joint'][p,:,n] = f_pop*weight


            if self.params['model']=='correlation':
                cdf_NN = np.cumsum(fun_NN)/np.sum(fun_NN)*weight
                cdf_nNN = np.cumsum(fun_nNN[::-1])[::-1]/np.sum(fun_nNN)*(1-weight)
                self.model['p_same']['joint'][n,:] = cdf_NN/(cdf_NN+cdf_nNN)
            else:
                cdf_NN = np.cumsum(fun_NN[::-1])[::-1]/np.sum(fun_NN)*weight
                cdf_nNN = np.cumsum(fun_nNN)/np.sum(fun_nNN)*(1-weight)
                self.model['p_same']['joint'][:,n] = cdf_NN/(cdf_NN+cdf_nNN)
            

        ## obtain probability of being same neuron

        # self.model['p_same']['joint'] = 1-self.model['pdf']['joint'][1,...]/np.nansum(self.model['pdf']['joint'],0)
        
        # self.model['p_same']['joint'] = nangauss_filter(self.model['p_same']['joint'],2)
        if np.any(np.isnan(self.model['p_same']['joint'])):
            self.model['p_same']['which'] = 'single'
        else:
            self.model['p_same']['which'] = 'joint'

        # if count_thr > 0:
            # self.model['p_same']['joint'] *= np.minimum(self.model[key_counts][...,0],count_thr)/count_thr
        # sp.ndimage.filters.gaussian_filter(self.model['p_same']['joint'],2,output=self.model['p_same']['joint'])
        self.create_model_evaluation()
        self.status['model_calculated'] = True
        # return self.model['pdf']['joint']


    def create_model_evaluation(self):
        
        if self.model['p_same']['which'] == 'single':
            fun = sp.interpolate.interp1d(self.params['arrays']['distance'],self.model['p_same']['single']['distance'],fill_value="extrapolate")
            self.model['f_same'] = lambda distance,correlation=None : fun(distance)
        else:
            self.model['f_same'] = lambda distance,correlation : sp.interpolate.interpn((self.params['arrays']['distance'],self.params['arrays']['correlation']),self.model['p_same']['joint'],(distance,correlation),bounds_error=False,fill_value=None)


        


    def set_functions(self,model,model_type='joint'):

        '''
          set up functions for model fitting to count histogram for both, single and joint model.
          This function defines both, the functional shape (from a set of predefined functions), and the boundaries for function parameters

          REMARK:
            * A more flexible approach with functions to be defined on class creation could be feasible, but makes everything a lot more complex for only minor pay-off. Could be implemented in future changes.
        '''
        
        fit_fun = {}
        fit_bounds = {}
        bounds_p = np.array([(0,1)]).T    # weight 'p' between NN and nNN function
        
        if model=='distance':
          
            ## set functions for model
            fit_fun['NN'] = functions['lognorm']
            fit_bounds['NN'] = np.array([(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)]).T
            # fit_fun['NN'] = functions['gamma']
            # fit_bounds['NN'] = np.array([(0,np.inf),(0,np.inf),(0,np.inf)]).T

            fit_fun['nNN'] = functions['linear_sigmoid']
            fit_bounds['nNN'] = np.array([(0,np.inf),(0,np.inf),(0,self.params['neighbor_distance']*2/3)]).T

            

          
        elif model=='correlation':
          
            ## set functions for model
            fit_fun['NN'] = functions['lognorm_reverse']
            fit_bounds['NN'] = np.array([(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)]).T

            # fit_fun['NN'] = functions['gamma_reverse']
            # fit_bounds['NN'] = np.array([(0,np.inf),(0,np.inf),(0,np.inf)]).T

            if self.params['correlation_model'] == 'shifted':
                fit_fun['nNN'] = functions['gauss']
                # fit_fun['nNN'] = functions['skewed_gauss']
                fit_bounds['nNN'] = np.array([(0,np.inf),(0,np.inf),(0,1)]).T
                
            else:
                fit_fun['nNN'] = functions['beta']
                fit_bounds['nNN'] = np.array([(-np.inf,np.inf),(-np.inf,np.inf)]).T
        

        sig_NN = signature(fit_fun['NN'])
        NN_parameters = len(sig_NN.parameters)-1

        ## set bounds for fit-parameters
        fit_fun['all'] = lambda x,p,*args : p*fit_fun['NN'](x,*args[:NN_parameters]) + (1-p)*fit_fun['nNN'](x,*args[NN_parameters:])
        fit_bounds['all'] = np.hstack([bounds_p,fit_bounds['NN'],fit_bounds['nNN']])
        
        
        for pop in ['NN','nNN','all']:
            if pop in fit_fun:
                self.model['fit_function'][model_type][model][pop] = fit_fun[pop]

                sig = signature(fit_fun[pop])
                self.model['fit_function'][model_type][model][f'{pop}_Nparams'] = len(sig.parameters)
        
        return fit_fun, fit_bounds
    






    ### -------------------- SAVING & LOADING ---------------------- ###

    def save_model(self,suffix='',matlab=None):
        
        fix_suffix(suffix)
        pathMatching = os.path.join(self.paths['data'],'matching')
        if ~os.path.exists(pathMatching):
            os.makedirs(pathMatching,exist_ok=True)


        results = {}
        for key in ['p_same','fit_parameter','pdf','counts','counts_unshifted','counts_same','counts_same_unshifted']:#,'f_same']:
            results[key] = self.model[key]
        
        pathSv = os.path.join(pathMatching,f'match_model{suffix}')
        pathSv += '.mat' if (matlab is None and self.matlab) or matlab else '.pkl'
        save_data(results,pathSv)


    def load_model(self,suffix='',matlab=None):
        
        suffix = fix_suffix(suffix)

        pathLd = os.path.join(self.paths['data'],f'matching/match_model{suffix}')
        pathLd += '.mat' if (matlab is None and self.matlab) or matlab else '.pkl'
        results = load_data(pathLd)

        self.model = {}
        for key in results.keys():
            self.model[key] = results[key]
        
        self.update_bins(self.model['p_same']['joint'].shape[0])

        self.create_model_evaluation()
        self.model['model_calculated'] = True
    

    def save_registration(self,suffix='',matlab=None):
        
        suffix = fix_suffix(suffix)
        pathMatching = os.path.join(self.paths['data'],'matching')
        if ~os.path.exists(pathMatching):
            os.makedirs(pathMatching,exist_ok=True)
        
        pathSv = os.path.join(pathMatching,f'neuron_registration{suffix}')
        pathSv += '.mat' if (matlab is None and self.matlab) or matlab else '.pkl'
        save_data(self.results,pathSv)


    def save_data(self,suffix='',matlab=None):

        suffix = fix_suffix(suffix)
        pathMatching = os.path.join(self.paths['data'],'matching')
        if ~os.path.exists(pathMatching):
            os.makedirs(pathMatching,exist_ok=True)

        pathSv = os.path.join(pathMatching,f'matching_data{suffix}')
        pathSv += '.mat' if (matlab is None and self.matlab) or matlab else '.pkl'
        save_data(self.data,pathSv)


    def load_registration(self,suffix='',matlab=None):

        suffix = fix_suffix(suffix)

        pathLd = os.path.join(self.paths['data'],f'matching/neuron_registration{suffix}')
        pathLd += '.mat' if (matlab is None and self.matlab) or matlab else '.pkl'
        self.results = load_data(pathLd)

        self.paths['sessions'] = replace_relative_path(self.paths['sessions'],self.paths['data'])
        #try:
          #self.results['assignments'] = self.results['assignment']
        #except:
          #1

    def load_data(self,suffix='',matlab=None):

        suffix = fix_suffix(suffix)

        self.data = {}
        if (matlab is None and self.matlab) or matlab:
            pathLd = os.path.join(self.paths['data'],f'matching/matching_data{suffix}.mat')
            dataLd = load_data(pathLd)
            for key in dataLd.keys():
                try:
                    self.data[int(key)] = dataLd[key]
                except:
                    self.data[key] = dataLd[key]
        else:
            pathLd = os.path.join(self.paths['data'],f'matching/matching_data{suffix}.pkl')
            with open(pathLd,'rb') as f:
                self.data = pickle.load(f)
        # self.paths['sessions'] = replace_relative_path(self.paths['sessions'],self.paths['data'])



    def find_confusion_candidates(self,confusion_distance=5):
       
        # cm = self.results['com']

        cm_mean = np.nanmean(self.results['cm'],axis=1)
        cm_dists = sp.spatial.distance.squareform(sp.spatial.distance.pdist(cm_mean))
        
        confusion_candidates = np.where(
           np.logical_and(cm_dists > 0,cm_dists<confusion_distance)
        )

        # for i,j in zip(*confusion_candidates):
        #     if i<j:
        #         print('clusters:',i,j)
        #         assignments = self.results['assignments'][(i,j),:].T

        #         occ = np.isfinite(assignments)

        #         nOcc = occ.sum(axis=0)
        #         nJointOcc = np.prod(occ,axis=1).sum()
        #         print(nOcc,nJointOcc)
        #         # self.results['assignments'][i,:]
        # return
        print(confusion_candidates)
        print(len(confusion_candidates[0])/2,' candidates found')
        ct = 0
        for i,j in zip(*confusion_candidates):
            
            if i<j:
                assignments = self.results['assignments'][(i,j),:].T
                occ = np.isfinite(assignments)
                nOcc = occ.sum(axis=0)
                print('clusters:',i,j,'occurences:',nOcc)
                
                # print(assignments)

                ### confusion can occur if in one session two footprints are competing for a match
                confused_sessions = np.where(np.all(np.isfinite(assignments),axis=1))[0]
                if len(confused_sessions)>0:
                    confused_session = confused_sessions[0]
                    print(assignments[:confused_session+1])

                    # print(self.data[confused_session]['p_same'][i,:])
                    fig,ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
                    self.plot_footprints(i,fp_color='k',ax_in=ax,use_plotly=True)
                    self.plot_footprints(j,fp_color='r',ax_in=ax,use_plotly=True)
                    plt.show(block=False)

                    ct+=1
                
                ### or can occur when matching probability is quite low
                if ct > 3: break
        return


    def scale_counts(self,times=0):

        key_counts = 'counts' if self.params['correlation_model']=='shifted' else 'counts_unshifted'
        counts = scale_down_counts(self.model[key_counts],times)
        self.params['nbins'] = counts.shape[0]
        self.update_bins(self.params['nbins'])

        if self.model['p_same']['joint'].shape[0] != self.params['nbins']:
            self.fit_model(times=times)

        return counts





    ### ------------------- PLOTTING FUNCTIONS --------------------- ###

    '''
      this specifies the different plot functions:

        all plots have inputs sv, suffix to specify saving behavior

        1. plot_fit_results
            inputs:
              model
              times
            creates interactive plot of joint model results
        
        2. plot_model
            creates general overview of model results and matching performance compared to guess based on nearest neighbours

        3. plot_fit_parameters
          MERGE THIS INTO #1

        4. plot_count_histogram
            inputs:
              times
            plots the histogram for different populations in 2- and 3D

        5. plot_something
            plots 3D visualization of matching probability
        
        6. plot_matches
            inputs:
              s_ref
              s
            plots neuron footprints of 2 sessions, colorcoded by whether they are matched, or not
        
        7. plot_neuron_numbers
            shows sessions in which each neuron is active
            ADJUST ACCORDING TO PLOTTING SCRIPT TO SHOW PHD LIKE FIGURE

        8. plot_registration
            shows distribution of match probabilities and 2nd best probability per match (how much "confusion" could there be?)
    '''


    def plot_fit_results(self,times=1,sv=False,suffix=''):
      
        '''
            TODO:
            * build plot function to include widgets, with which parameter space can be visualized, showing fit results and histogram slice (bars), each (2 subplots: correlation & distance)
        '''

        counts = self.scale_counts(times)
        nbins = self.params['nbins']

        model = self.params['model']


        # print(joint_hist_norm_dist.sum(axis=1)*self.params['arrays']['distance_step'])

        fig,ax = plt.subplots(3,2)

        mean, var = self.get_population_mean_and_var()
        model = 'correlation'

        for axx,model in zip([ax[0][1],ax[1][0]],['correlation','distance']):

            weight_model = 'distance' if model=='correlation' else 'correlation'

            axx.plot(self.params['arrays'][weight_model],mean[model]['NN'],'g',label='lognorm $\mu$')
            axx.plot(self.params['arrays'][weight_model],mean[model]['nNN'],'r',label='$A$ sigm.')
            axx.set_title(f'{model} models')
            #plt.plot(self.params['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['all'][:,2],'g--')
            #plt.plot(self.params['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['all'][:,5],'r--')
            axx.legend()
            plt.setp(axx,ylim=[0,self.params['arrays'][model][-1]])

        # ax[1][0].plot(self.params['arrays'][weight_model],var[model]['NN'],'g',label='lognorm $\sigma$')
        # ax[1][0].plot(self.params['arrays'][weight_model],var[model]['nNN'],'r',label='slope sigm')
        # #plt.plot(self.params['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['nNN'][:,1],'r--',label='dist $\gamma$')
        # ax[1][0].legend()

        # plt.subplot(222)
        # plt.plot(self.params['arrays']['distance'],mean['correlation']['NN'],'g',label='lognorm $\mu$')#self.model['fit_parameter']['joint']['correlation']['NN'][:,1],'g')#
        # plt.plot(self.params['arrays']['distance'],self.model['fit_parameter']['joint']['correlation']['nNN'][:,1],'r',label='gauss $\mu$')
        # plt.title('correlation models')
        # plt.legend()

        # plt.subplot(224)
        # plt.plot(self.params['arrays']['distance'],var['correlation']['NN'],'g',label='lognorm $\sigma$')
        # plt.plot(self.params['arrays']['distance'],self.model['fit_parameter']['joint']['correlation']['nNN'][:,0],'r',label='gauss $\sigma$')
        # plt.legend()
        # plt.tight_layout()
        # plt.show(block=False)

        ax[0][0].axis('off')

        h = {
            'distance': {},
            'distance_hist': {},
            'correlation': {},
            'correlation_hist': {},
        }

        for axx,model in zip([ax[1][1],ax[2][0]],['correlation','distance']):
            for pop,col in zip(['NN','nNN'],['g','r']):
                h[model][pop], = axx.plot(
                    self.params['arrays'][model],
                    np.zeros(nbins),
                    c=col
                )
            
                h[f'{model}_hist'][pop] = axx.bar(
                    self.params['arrays'][model],
                    np.zeros(nbins),
                    width=self.params['arrays'][f'{model}_step'],
                    facecolor=col,
                    alpha=0.5,
                )
        plt.setp(ax[1][1],xlim=self.params['arrays']['correlation'][[0,-1]])
        plt.setp(ax[2][0],xlim=self.params['arrays']['distance'][[0,-1]])
        
        
        ax[2][1].imshow(
            counts[...,0],
            origin='lower',aspect='auto',
            extent=tuple(self.params['arrays']['correlation'][[0,-1]])+tuple(self.params['arrays']['distance'][[0,-1]])
        )

        h['distance_marker'] = ax[2][1].axhline(0,c='r')
        h['correlation_marker'] = ax[2][1].axvline(0,c='r')
        plt.setp(ax[2][1],xlim=self.params['arrays']['correlation'][[0,-1]],ylim=self.params['arrays']['distance'][[0,-1]])

        slider_h = 0.02
        slider_w = 0.3
        axamp_dist = plt.axes([0.15, .85, slider_w, slider_h])
        axamp_corr = plt.axes([0.15, .75, slider_w, slider_h])

        counts_norm_along_corr = counts/counts.sum(1,keepdims=True)# / self.params['arrays']['distance_step']# * nbins/self.params['neighbor_distance']
        counts_norm_along_dist = counts/counts.sum(0,keepdims=True)# / self.params['arrays']['correlation_step']
        
        def update_plot(ax,n,weight_model='correlation',model_type='joint'):

            # print('somehow, joint model functions are still not working nicely - fits look aweful! is it an issue of displaying, or of construction?')
            model = 'distance' if weight_model=='correlation' else 'correlation'
            m = 0 if model=='correlation' else 1
            joint_hist = counts_norm_along_corr if model=='correlation' else counts_norm_along_dist
            
            fit_fun, _ = self.set_functions(model,model_type)

            max_val = 0
            for p,pop in enumerate(['NN','nNN'],start=1):

                weight = self.model['p_same']['single'][weight_model][n]
                if pop == 'nNN':
                    weight = 1-weight
                
                # print(self.params['arrays'][model])

                fun_pop = fun_wrapper(
                    fit_fun[pop],
                    self.params['arrays'][model],
                    self.model['fit_parameter'][model_type][model][pop][n,:]
                ) * weight / self.params['arrays'][f'{model}_step']
                
                # print(fun_pop)
                # print('parameter:',self.model['fit_parameter'][model_type][model][pop][n,:])
                # print('mean&var:',mean[model][pop][n],var[model][pop][n])

                norm_counts = np.take(joint_hist[...,p],n,axis=m)/self.params['arrays'][f'{model}_step']
                # print(f'counts: ',norm_counts.sum())
                h[model][pop].set_ydata(fun_pop)
                max_val = max(max_val,fun_pop.max())

                for rect,height in zip(h[f'{model}_hist'][pop],norm_counts):
                    rect.set_height(height)
                    max_val = max(max_val,height)

                marker_value = self.params['arrays'][weight_model][n]
                if weight_model=='distance':
                    h[f'{weight_model}_marker'].set_ydata(marker_value)
                else:
                    h[f'{weight_model}_marker'].set_xdata(marker_value)
                    

            plt.setp(ax,ylim=[0,max_val*1.1])
            fig.canvas.draw_idle()
        

        distance_init = int(nbins*0.1)
        correlation_init = int(nbins*0.9)

        self.slider = {}
        self.slider['distance'] = Slider(axamp_dist, r'd_{com}', 0, nbins-1, valinit=distance_init,orientation='horizontal',valstep=range(nbins))
        self.slider['correlation'] = Slider(axamp_corr, r'c_{com}', 0, nbins-1, valinit=correlation_init,orientation='horizontal',valstep=range(nbins))

        update_plot_correlation = lambda n: update_plot(ax[2][0],n=n,weight_model='correlation')
        update_plot_distance = lambda n: update_plot(ax[1][1],n=n,weight_model='distance')
        
        self.slider['distance'].on_changed(update_plot_distance)
        self.slider['correlation'].on_changed(update_plot_correlation)

        update_plot_distance(distance_init)
        update_plot_correlation(correlation_init)

        plt.show(block=False)


    def plot_model(self,model='correlation',sv=False,suffix='',times=1):

        rc('font',size=10)
        rc('axes',labelsize=12)
        rc('xtick',labelsize=8)
        rc('ytick',labelsize=8)

        counts = self.scale_counts(times)
        nbins = self.params['nbins']
        

        arrays = self.params['arrays']
        X, Y = np.meshgrid(arrays['correlation'], arrays['distance'])

        fig = plt.figure(figsize=(7,4),dpi=150)
        ax_phase = plt.axes([0.3,0.13,0.2,0.4])
        add_number(fig,ax_phase,order=1,offset=[-250,200])
        #ax_phase.imshow(self.model[key_counts][:,:,0],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',clim=[0,0.25*self.model[key_counts][:,:,0].max()],origin='lower')
        NN_ratio = counts[:,:,1]/counts[:,:,0]
        cmap = plt.cm.RdYlGn
        NN_ratio = cmap(NN_ratio)
        NN_ratio[...,-1] = np.minimum(counts[...,0]/(np.max(counts)/3.),1)

        im_ratio = ax_phase.imshow(NN_ratio,extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',clim=[0,0.5],origin='lower')
        nlev = 3
        # col = (np.ones((nlev,3)).T*np.linspace(0,1,nlev)).T
        p_levels = ax_phase.contour(X,Y,self.model['p_same']['joint'],levels=[0.05,0.5,0.95],colors='k',linestyles=[':','--','-'],linewidths=1.)
        plt.setp(
            ax_phase,xlim=[0,1],ylim=[0,self.params['neighbor_distance']],
            xlabel='correlation',
            ylabel='distance'
        )
        ax_phase.tick_params(axis='x',which='both',bottom=True,top=True,labelbottom=False,labeltop=True)
        ax_phase.tick_params(axis='y',which='both',left=True,right=True,labelright=False,labelleft=True)
        ax_phase.yaxis.set_label_position("right")
        #ax_phase.xaxis.tick_top()
        ax_phase.xaxis.set_label_coords(0.5,-0.15)
        ax_phase.yaxis.set_label_coords(1.15,0.5)

        im_ratio.cmap = cmap
        if self.params['correlation_model'] == 'unshifted':
            cbaxes = plt.axes([0.41, 0.47, 0.07, 0.03])
            #cbar.ax.set_xlim([0,0.5])
        else:
            cbaxes = plt.axes([0.32, 0.47, 0.07, 0.03])
        cbar = plt.colorbar(im_ratio,cax=cbaxes,orientation='horizontal')
        #cbar.ax.set_xlabel('NN ratio')
        cbar.ax.set_xticks([0,0.5])
        cbar.ax.set_xticklabels(['nNN','NN'])

        #cbar.ax.set_xticks(np.linspace(0,1,2))
        #cbar.ax.set_xticklabels(np.linspace(0,1,2))


        ax_dist = plt.axes([0.05,0.13,0.2,0.4])

        ax_dist.barh(arrays['distance'],counts[...,0].sum(1).flat,self.params['neighbor_distance']/nbins,facecolor='k',alpha=0.5,orientation='horizontal')
        ax_dist.barh(arrays['distance'],counts[...,2].sum(1),arrays['distance_step'],facecolor='salmon',alpha=0.5)
        ax_dist.barh(arrays['distance'],counts[...,1].sum(1),arrays['distance_step'],facecolor='lightgreen',alpha=0.3)
        ax_dist.invert_xaxis()
        #h_d_move = ax_dist.bar(arrays['distance'],np.zeros(nbins),arrays['distance_step'],facecolor='k')

        model_distance_all = (
           fun_wrapper(
            self.model['fit_function']['single']['distance']['NN'],
            arrays['distance'],
            self.model['fit_parameter']['single']['distance']['NN']
            ) * counts[...,1].sum() + \
            fun_wrapper(self.model['fit_function']['single']['distance']['nNN'],arrays['distance'],self.model['fit_parameter']['single']['distance']['nNN'])*counts[...,2].sum()
            ) * arrays['distance_step']

        NN_params = self.model['fit_function']['single']['distance']['NN_Nparams']

        ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['all'],arrays['distance'],self.model['fit_parameter']['single']['distance']['all'])*counts[...,0].sum()*arrays['distance_step'],arrays['distance'],'k:')
        ax_dist.plot(model_distance_all,arrays['distance'],'k')

        ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['NN'],arrays['distance'],self.model['fit_parameter']['single']['distance']['all'][1:NN_params])*self.model['fit_parameter']['single']['distance']['all'][0]*counts[...,0].sum()*arrays['distance_step'],arrays['distance'],'g')
        ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['NN'],arrays['distance'],self.model['fit_parameter']['single']['distance']['NN'])*counts[...,1].sum()*arrays['distance_step'],arrays['distance'],'g:')

        ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['nNN'],arrays['distance'],self.model['fit_parameter']['single']['distance']['all'][NN_params:])*(1-self.model['fit_parameter']['single']['distance']['all'][0])*counts[...,0].sum()*arrays['distance_step'],arrays['distance'],'r')
        ax_dist.plot(fun_wrapper(self.model['fit_function']['single']['distance']['nNN'],arrays['distance'],self.model['fit_parameter']['single']['distance']['nNN'])*counts[...,2].sum()*arrays['distance_step'],arrays['distance'],'r',linestyle=':')
        ax_dist.set_ylim([0,self.params['neighbor_distance']])
        ax_dist.set_xlabel('counts')
        ax_dist.spines['left'].set_visible(False)
        ax_dist.spines['top'].set_visible(False)
        ax_dist.tick_params(axis='y',which='both',left=False,right=True,labelright=False,labelleft=False)

        ax_corr = plt.axes([0.3,0.63,0.2,0.325])
        ax_corr.bar(arrays['correlation'],counts[...,0].sum(0).flat,1/nbins,facecolor='k',alpha=0.5)
        ax_corr.bar(arrays['correlation'],counts[...,2].sum(0),1/nbins,facecolor='salmon',alpha=0.5)
        ax_corr.bar(arrays['correlation'],counts[...,1].sum(0),1/nbins,facecolor='lightgreen',alpha=0.3)

        f_NN = fun_wrapper(self.model['fit_function']['single']['correlation']['NN'],arrays['correlation'],self.model['fit_parameter']['single']['correlation']['NN'])
        f_nNN = fun_wrapper(self.model['fit_function']['single']['correlation']['nNN'],arrays['correlation'],self.model['fit_parameter']['single']['correlation']['nNN'])
        model_fp_correlation_all = (f_NN*counts[...,1].sum() + f_nNN*counts[...,2].sum())*arrays['correlation_step']
        #ax_corr.plot(arrays['correlation'],fun_wrapper(self.model['fit_function']['correlation']['all'],arrays['correlation'],self.model['fit_parameter']['single']['correlation']['all'])*counts[...,0].sum()*arrays['correlation_step'],'k')
        ax_corr.plot(arrays['correlation'],model_fp_correlation_all,'k')

        ax_corr.plot(arrays['correlation'],fun_wrapper(self.model['fit_function']['single']['correlation']['NN'],arrays['correlation'],self.model['fit_parameter']['single']['correlation']['NN'])*counts[...,1].sum()*arrays['correlation_step'],'g')

        #ax_corr.plot(arrays['correlation'],fun_wrapper(self.model['fit_function']['correlation']['nNN'],arrays['correlation'],self.model['fit_parameter']['single']['correlation']['all'][3:])*(1-self.model['fit_parameter']['single']['correlation']['all'][0])*counts[...,0].sum()*arrays['correlation_step'],'r')
        ax_corr.plot(arrays['correlation'],fun_wrapper(self.model['fit_function']['single']['correlation']['nNN'],arrays['correlation'],self.model['fit_parameter']['single']['correlation']['nNN'])*counts[...,2].sum()*arrays['correlation_step'],'r')

        ax_corr.set_ylabel('counts')
        ax_corr.set_xlim([0,1])
        ax_corr.spines['right'].set_visible(False)
        ax_corr.spines['top'].set_visible(False)
        ax_corr.tick_params(axis='x',which='both',bottom=True,top=False,labelbottom=False,labeltop=False)

        #ax_parameter =
        p_steps, rates = self.calculate_RoC(100)

        ax_cum = plt.axes([0.675,0.7,0.3,0.225])
        add_number(fig,ax_cum,order=2)

        uncertain = {}
        idx_low = np.where(p_steps>0.05)[0][0]
        idx_high = np.where(p_steps<0.95)[0][-1]
        for key in rates['cumfrac'].keys():

            if key=='joint' and (rates['cumfrac'][key][idx_low]>0.01) & (rates['cumfrac'][key][idx_high]<0.99):
                ax_cum.fill_between([rates['cumfrac'][key][idx_low],rates['cumfrac'][key][idx_high]],[0,0],[1,1],facecolor='y',alpha=0.5)
            uncertain[key] = (rates['cumfrac'][key][idx_high] - rates['cumfrac'][key][idx_low])#/(1-rates['cumfrac'][key][idx_high+1])

        ax_cum.plot([0,1],[0.05,0.05],'b',linestyle=':')
        ax_cum.plot([0.5,1],[0.95,0.95],'b',linestyle='-')

        ax_cum.plot(rates['cumfrac']['joint'],p_steps[:-1],'grey',label='Joint')
        # ax_cum.plot(rates['cumfrac']['distance'],p_steps[:-1],'k',label='Distance')
        if self.params['correlation_model']=='unshifted':
            ax_cum.plot(rates['cumfrac']['correlation'],p_steps[:-1],'lightgrey',label='Correlation')
        ax_cum.set_ylabel('$p_{same}$')
        ax_cum.set_xlabel('cumulative fraction')
        #ax_cum.legend(fontsize=10,frameon=False)
        ax_cum.spines['right'].set_visible(False)
        ax_cum.spines['top'].set_visible(False)

        ax_uncertain = plt.axes([0.75,0.825,0.05,0.1])
        # ax_uncertain.bar(2,uncertain['distance'],facecolor='k')
        ax_uncertain.bar(3,uncertain['joint'],facecolor='k')
        if self.params['correlation_model']=='unshifted':
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
        if self.params['correlation_model']=='unshifted':
            ax_RoC.plot(rates['fp']['correlation'],rates['tp']['correlation'],'lightgrey',label='Correlation')
        ax_RoC.plot(rates['fp']['joint'][idx],rates['tp']['joint'][idx],'kx')
        # ax_RoC.plot(rates['fp']['distance'][idx],rates['tp']['distance'][idx],'kx')
        if self.params['correlation_model']=='unshifted':
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

        if self.params['correlation_model']=='unshifted':
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
        if self.params['correlation_model']=='unshifted':
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
            path = os.path.join(self.params['pathMouse'],'Sheintuch_matching_%s%s.%s'%(self.params['correlation_model'],suffix,ext))
            plt.savefig(path,format=ext,dpi=150)
        # return
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


    def plot_count_histogram(self,times=0):

        counts = self.scale_counts(times)
        nbins = self.params['nbins']

        arrays = self.params['arrays']

        plt.figure(figsize=(12,9))
        # plt.subplot(221)
        ax = plt.subplot(221,projection='3d')
        X, Y = np.meshgrid(arrays['correlation'], arrays['distance'])
        NN_ratio = counts[:,:,1]/counts[:,:,0]
        cmap = plt.cm.RdYlGn
        NN_ratio = cmap(NN_ratio)
        ax.plot_surface(X,Y,counts[:,:,0],facecolors=NN_ratio)
        ax.view_init(30,-120)
        ax.set_xlabel('footprint correlation',fontsize=14)
        ax.set_ylabel('distance',fontsize=14)
        ax.set_zlabel('# pairs',fontsize=14)

        # plt.imshow(counts[...,0],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',clim=[0,counts[...,0].max()],origin='lower')
        # nlev = 3
        # col = (np.ones((nlev,3)).T*np.linspace(0,1,nlev)).T
        # p_levels = plt.contour(X,Y,self.model['p_same']['joint'],levels=[0.05,0.5,0.95],colors=col)
        #plt.colorbar(p_levels)
        #plt.imshow(self.model[key_counts][...,0],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',origin='lower')
        ax2 = plt.subplot(222)
        ax2.imshow(counts[...,0],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',origin='lower')
        ax2.set_title('all counts',y=1,pad=-14,color='white',fontweight='bold')
        
        ax3 = plt.subplot(223)
        ax3.imshow(counts[...,1],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',origin='lower')
        ax3.set_title('nearest neighbour counts',y=1,pad=-14,color='white',fontweight='bold')

        ax4 = plt.subplot(224)
        ax4.imshow(counts[...,2],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',origin='lower')
        ax4.set_title('non-nearest neighbour counts',y=1,pad=-14,color='white',fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)


        # plt.figure()
        # cdf = {'correlation':{},'distance':{}}

        # cdf['correlation']['NN'] = np.cumsum(counts[...,1],axis=1)
        # cdf['correlation']['nNN'] = np.cumsum(counts[:,::-1,2],axis=1)[:,::-1]

        # cdf['distance']['NN'] = np.cumsum(counts[::-1,:,1],axis=0)[::-1,:]
        # cdf['distance']['nNN'] = np.cumsum(counts[:,:,2],axis=0)

        # # ax=0 -> distance , ax=1 -> correlation
        # for m,model in enumerate(['distance','correlation']):

        #     for p,pop in enumerate(['NN','nNN']):
        #         print(m,p,2*m+p+1)
        #         cdf[model][pop] = cdf[model][pop]/counts[...,p+1].sum(axis=m,keepdims=True)

        #         ax = plt.subplot(3,2,2*m+p+1,projection='3d')
        #         ax.plot_surface(X,Y,cdf[model][pop])

        # ax_p_same = plt.subplot(325,projection='3d')
        # p_same = cdf['correlation']['NN']/(cdf['correlation']['NN']+cdf['correlation']['nNN'])
        # ax_p_same.plot_surface(X,Y,p_same)
        
        # ax_p_same = plt.subplot(326,projection='3d')
        # p_same = cdf['distance']['NN']/(cdf['distance']['NN']+cdf['distance']['nNN'])
        # ax_p_same.plot_surface(X,Y,p_same)
      
        # plt.show(block=False)



    def plot_p_same(self,times=0,animate=False):
      
        counts = self.scale_counts(times)
        nbins = counts.shape[0]
        self.update_bins(nbins)

        X, Y = np.meshgrid(self.params['arrays']['correlation'], self.params['arrays']['distance'])

        fig = plt.figure(figsize=(8,10))
        ax_corr = [fig.add_subplot(4,2,1), fig.add_subplot(4,2,3)]
        ax_dist= [fig.add_subplot(4,2,2), fig.add_subplot(4,2,4)]

        for m,(ax,model) in enumerate(zip([ax_corr,ax_dist],['correlation','distance'])):

            fit_fun, _ = self.set_functions(model,'single')
            fun_NN = fun_wrapper(
                fit_fun['NN'],
                self.params['arrays'][model],
                self.model['fit_parameter']['single'][model]['all'][1:self.model['fit_function']['single'][model]['NN_Nparams']]
            ) * self.model['fit_parameter']['single'][model]['all'][0]

            fun_nNN = fun_wrapper(
                fit_fun['nNN'],
                self.params['arrays'][model],
                self.model['fit_parameter']['single'][model]['all'][self.model['fit_function']['single'][model]['NN_Nparams']:]
            ) * (1-self.model['fit_parameter']['single'][model]['all'][0])
            
            fun_total = fun_NN + fun_nNN
            
            ax[0].plot(self.params['arrays'][model],counts[...,0].sum(m)/counts[...,0].sum()/self.params['arrays'][f'{model}_step'],'g',label='data')
            # for p,pop in enumerate(['NN','nNN'],start=1):
            ax[0].plot(self.params['arrays'][model],fun_NN,'k:')
            ax[0].plot(self.params['arrays'][model],fun_nNN,'k:')
            ax[0].plot(self.params['arrays'][model],fun_total,'k--',label='model')
            plt.setp(ax[0],xlim=[0,self.params['arrays'][model][-1]],yticks=[])
            ax[0].spines[['top','right']].set_visible(False)
            ax[0].legend()

            ax[0].set_title(f'single model: {model}')


            ax[1].plot(self.params['arrays'][model],self.model['p_same']['single'][model],'r')
            plt.setp(ax[1],xlim=[0,self.params['arrays'][model][-1]],xlabel=model,ylabel='$p_{same}$')
            ax[1].spines[['top','right']].set_visible(False)
            # plt.show()
        

        ax_p_same = fig.add_subplot(2,2,4,projection='3d')
        prob = ax_p_same.plot_surface(X,Y,self.model['p_same']['joint'],cmap='jet')
        prob.set_clim(0,1)
        ax_p_same.set_zlim([0,1])
        ax_p_same.set_xlabel('correlation')
        ax_p_same.set_ylabel('distance')
        ax_p_same.set_zlabel('$p_{same}$')


        #ax = plt.subplot(224,projection='3d')
        ax_counts = fig.add_subplot(2,2,3,projection='3d')
        prob = ax_counts.plot_surface(X,Y,counts[...,0],cmap='jet')
        #prob = ax.bar3d(X.flatten(),Y.flatten(),np.zeros((nbins,nbins)).flatten(),np.ones((nbins,nbins)).flatten()*self.params['arrays']['correlation_step'],np.ones((nbins,nbins)).flatten()*self.params['arrays']['distance_step'],self.model[key_counts][...,0].flatten(),cmap='jet')
        #prob.set_clim(0,1)
        ax_counts.set_xlabel('correlation')
        ax_counts.set_ylabel('distance')
        ax_counts.set_zlabel('counts')
        ax_counts.set_title(f'joint model')
        plt.tight_layout()

        def rotate_view(i,axes,fixed_angle=30):
            for axx in axes:
                axx.view_init(fixed_angle,(i*2)%360)
            #return ax

        rotate_view(100,[ax_p_same,ax_counts],fixed_angle=30)
        plt.show(block=False)

        print('proper weighting of bin counts')
        print('smoothing by gaussian')
    

    def plot_matches(self, s_ref, s,
        color_s_ref='coral',
        color_s='lightgreen',
        level=[0.05]):
    
        '''

            TODO:
            * rewrite function, such that it calculates and plots footprint matching for 2 arbitrary sessions (s,s_ref)
            * write function description and document code properly
            * optimize plotting, so it doesn't take forever
        '''
        
        matched1 = self.results['assignments']
        matched_c = np.all(np.isfinite(self.results['assignments'][:,(s_ref,s)]),axis=1)
        matched1 = self.results['assignments'][matched_c,s_ref].astype(int)
        matched2 = self.results['assignments'][matched_c,s].astype(int)
        # print('matched: ',matched_c.sum())
        nMatched = matched_c.sum()
        
        non_matched_c = np.isfinite(self.results['assignments'][:,s_ref]) & np.isnan(self.results['assignments'][:,s])
        non_matched1 = self.results['assignments'][non_matched_c,s_ref].astype(int)
        # print('non_matched 1: ',non_matched_c.sum())
        nNonMatched1 = non_matched_c.sum()
        
        non_matched_c = np.isnan(self.results['assignments'][:,s_ref]) & np.isfinite(self.results['assignments'][:,s])
        non_matched2 = self.results['assignments'][non_matched_c,s].astype(int)
        # print('non_matched 1: ',non_matched_c.sum())
        nNonMatched2 = non_matched_c.sum()

        
        print('plotting...')
        t_start = time.time()

        def load_and_align(s):
            
            ld = load_data(self.paths['sessions'][s])
            A = ld['A']
            Cn = ld['Cn'].T

            if 'remap' in self.data[s].keys():
          
              x_remap,y_remap = build_remap_from_shift_and_flow(self.params['dims'],self.data[s]['remap']['shift'],self.data[s]['remap']['flow'])
          
              Cn = cv2.remap(
                  Cn,   # reshape image to original dimensions
                  x_remap, y_remap,                 # apply reverse identified shift and flow
                  cv2.INTER_CUBIC                 
              )

              A = sp.sparse.hstack([
                  sp.sparse.csc_matrix(                    # cast results to sparse type
                      cv2.remap(
                          fp.reshape(self.params['dims']),   # reshape image to original dimensions
                          x_remap, y_remap,                 # apply reverse identified shift and flow
                          cv2.INTER_CUBIC                 
                      ).reshape(-1,1)                       # reshape back to allow sparse storage
                      ) for fp in A.toarray().T        # loop through all footprints
                  ])
               
            lp, hp = np.nanpercentile(Cn, [25, 99])
            Cn -= lp
            Cn /= (hp-lp)

            Cn = np.clip(Cn,a_min=0,a_max=1)

            return A,Cn

        Cn = np.zeros(self.params['dims']+(3,))
        
        A_ref, Cn[...,0] = load_and_align(s_ref)
        A, Cn[...,1] = load_and_align(s)
        
        plt.figure(figsize=(15,12))

        ax_matches = plt.subplot(111)
        ax_matches.imshow(np.transpose(Cn,(1,0,2)))#, origin='lower')

        [ax_matches.contour(np.reshape(a.todense(),self.params['dims']).T, levels=level, colors=color_s_ref, linewidths=2) for a in A_ref[:,matched1].T]
        [ax_matches.contour(np.reshape(a.todense(),self.params['dims']).T, levels=level, colors=color_s_ref, linewidths=2, linestyles='--') for a in A_ref[:,non_matched1].T]

        print('first half done: %5.3f'%(time.time()-t_start))
        [ax_matches.contour(np.reshape(a.todense(),self.params['dims']).T, levels=level, colors=color_s, linewidths=2) for a in A[:,matched2].T]
        [ax_matches.contour(np.reshape(a.todense(),self.params['dims']).T, levels=level, colors=color_s, linewidths=2, linestyles='--') for a in A[:,non_matched2].T]

        ax_matches.legend(handles=[
           mppatches.Patch(color=color_s_ref,label='reference session'),
           mppatches.Patch(color=color_s,label='session'),
           mplines.Line2D([0],[0],color='k',linestyle='-',label=f'matched ({nMatched})'),
           mplines.Line2D([0],[0],color='k',linestyle='--',label=f'non-matched ({nNonMatched1}/{nNonMatched2})'),
        ],loc='lower left', framealpha=0.9)
        plt.setp(ax_matches,xlabel='x [px]',ylabel='y [px]')
        print('done. time taken: %5.3f'%(time.time()-t_start))
        plt.show(block=False)



    def plot_neuron_numbers(self):

        ### plot occurence of neurons
        colors = [(1,0,0,0),(1,0,0,1)]
        RedAlpha = mcolors.LinearSegmentedColormap.from_list('RedAlpha',colors,N=2)
        colors = [(0,0,0,0),(0,0,0,1)]
        BlackAlpha = mcolors.LinearSegmentedColormap.from_list('BlackAlpha',colors,N=2)

        idxes = np.ones(self.results['assignments'].shape,'bool')

        fig = plt.figure(figsize=(5,8))
        ### plot occurence of neurons
        ax_oc = fig.add_subplot([0.1,0.4,0.6,0.5])
        #ax_oc2 = ax_oc.twinx()
        ax_oc.imshow((~np.isnan(self.results['assignments']))&idxes,cmap=BlackAlpha,aspect='auto',interpolation='none')
        # ax_oc2.imshow((~np.isnan(self.results['assignments']))&(~idxes),cmap=RedAlpha,aspect='auto')
        #ax_oc.imshow(self.results['p_matched'],cmap='binary',aspect='auto')
        ax_oc.set_xlabel('session')
        ax_oc.set_ylabel('neuron ID')

        ax_per_session = fig.add_subplot([0.1,0.1,0.6,0.3])
        ax_per_session.plot((~np.isnan(self.results['assignments'])).sum(0),'ko',markersize=3)
        

        ax_per_cluster = fig.add_subplot([0.7,0.4,0.25,0.5])
        nC,nS = self.results['assignments'].shape
        ax_per_cluster.plot((~np.isnan(self.results['assignments'])).sum(1),np.linspace(0,nC,nC),'ko',markersize=3)
        
        nS = ((~np.isnan(self.results['assignments'])).sum(0)>0).sum()
        plt.setp(ax_per_cluster,xlim=[0,nS],ylim=[nC,0],yticks=[],xlabel='# sessions')

        ax_per_cluster_histogram = fig.add_subplot([0.7,0.9,0.25,0.075])
        ax_per_cluster_histogram.hist((~np.isnan(self.results['assignments'])).sum(1),np.linspace(0,nS,nS),color='k',cumulative=True,density=True,histtype='step')
        plt.setp(ax_per_cluster_histogram,xlim=[0,nS])



        plt.tight_layout()
        plt.show(block=False)

        # ext = 'png'
        # path = pathcat([self.params['pathMouse'],'Figures/Sheintuch_registration_score_stats_raw_%s_%s.%s'%(self.params['correlation_model'],suffix,ext)])
        # plt.savefig(path,format=ext,dpi=300)



    def plot_registration(self,suffix='',sv=False):

      rc('font',size=10)
      rc('axes',labelsize=12)
      rc('xtick',labelsize=8)
      rc('ytick',labelsize=8)

      # fileName = 'clusterStats_%s.pkl'%dataSet
      idxes = np.ones(self.results['assignments'].shape,'bool')

      fileName = f'cluster_stats_{suffix}.pkl'
      pathLoad = os.path.join(self.paths['data'],fileName)
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

    #   pm_thr = 0.3
    #   idx_pm = ((self.results['p_matched'][...,0]-self.results['p_matched'][...,1])>pm_thr) | (self.results['p_matched'][...,0]>0.95)

      plt.figure(figsize=(7,1.5))
      ax_sc1 = plt.axes([0.1,0.3,0.35,0.65])

      ax = ax_sc1.twinx()
      ax.hist(self.results['p_matched'][idxes,1].flat,np.linspace(0,1,51),facecolor='tab:red',alpha=0.3)
      #ax.invert_yaxis()
      ax.set_yticks([])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax = ax_sc1.twiny()
      ax.hist(self.results['p_matched'][idxes,0].flat,np.linspace(0,1,51),facecolor='tab:blue',orientation='horizontal',alpha=0.3)
      ax.set_xticks([])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax_sc1.plot(self.results['p_matched'][idxes,1].flat,self.results['p_matched'][idxes,0].flat,'.',markeredgewidth=0,color='k',markersize=1)
      ax_sc1.plot([0,1],[0,1],'--',color='tab:red',lw=0.5)
      ax_sc1.plot([0,0.45],[0.5,0.95],'--',color='tab:orange',lw=1)
      ax_sc1.plot([0.45,1],[0.95,0.95],'--',color='tab:orange',lw=1)
      ax_sc1.set_ylabel('$p^{\\asterisk}$')
      ax_sc1.set_xlabel('max($p\\backslash p^{\\asterisk}$)')
      ax_sc1.spines['top'].set_visible(False)
      ax_sc1.spines['right'].set_visible(False)

      # match vs max
      # idxes &= idx_pm

      # avg matchscore per cluster, min match score per cluster, ...
      ax_sc2 = plt.axes([0.6,0.3,0.35,0.65])
      #plt.hist(np.nanmean(self.results['p_matched'],1),np.linspace(0,1,51))
      ax = ax_sc2.twinx()
      ax.hist(np.nanmin(self.results['p_matched'][...,0],1),np.linspace(0,1,51),facecolor='tab:red',alpha=0.3)
      ax.set_yticks([])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax = ax_sc2.twiny()
      ax.hist(np.nanmean(self.results['p_matched'][...,0],axis=1),np.linspace(0,1,51),facecolor='tab:blue',orientation='horizontal',alpha=0.3)
      ax.set_xticks([])
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

      ax_sc2.plot(np.nanmin(self.results['p_matched'][...,0],1),np.nanmean(self.results['p_matched'][...,0],axis=1),'.',markeredgewidth=0,color='k',markersize=1)
      ax_sc2.set_xlabel('min($p^{\\asterisk}$)')
      ax_sc2.set_ylabel('$\left\langle p^{\\asterisk} \\right\\rangle$')
      ax_sc2.spines['top'].set_visible(False)
      ax_sc2.spines['right'].set_visible(False)

      ### plot positions of neurons
      plt.tight_layout()
      plt.show(block=False)

      if sv:
          ext = 'png'
          path = os.path.join(self.params['pathMouse'],'Figures/Sheintuch_registration_score_stats_%s_%s.%s'%(self.params['correlation_model'],suffix,ext))
          plt.savefig(path,format=ext,dpi=300)


    def plot_cluster_stats(self):

        print('### Plotting ROI and cluster statistics of matching ###')

        # idx_unsure = cluster.stats['match_score'][...,0]<(cluster.stats['match_score'][...,1]+0.5)

        nC,nSes = self.results['assignments'].shape
        active = ~np.isnan(self.results['assignments'])

        idx_unsure = self.results['p_matched'][...,0]<0.95

        fig = plt.figure(figsize=(7,4),dpi=300)

        nDisp = 20
        ax_3D = plt.subplot(221,projection='3d')
        # ax_3D.set_position([0.2,0.5,0.2,0.3])
        ##fig.gca(projection='3d')
        #a = np.arange(30)
        #for c in range(30):
        
        n_arr = np.random.choice(np.where(active.sum(1)>10)[0],nDisp)
        # n_arr = np.random.randint(0,cluster.meta['nC'],nDisp)
        cmap = cm.get_cmap('tab20')
        ax_3D.set_prop_cycle(color=cmap.colors)
        # print(self.results['cm'][n_arr,:,0],self.results['cm'][n_arr,:,0].shape)
        for n in n_arr:
          ax_3D.scatter(self.results['cm'][n,:,0],self.results['cm'][n,:,1],np.arange(nSes),s=0.5)#linewidth=2)
        ax_3D.set_xlim([0,512*self.params['pxtomu']])
        ax_3D.set_ylim([0,512*self.params['pxtomu']])

        ax_3D.set_xlabel('x [$\mu$m]')
        ax_3D.set_ylabel('y [$\mu$m]')
        ax_3D.invert_zaxis()
        # ax_3D.zaxis._axinfo['label']['space_factor'] = 2.8
        ax_3D.set_zlabel('session')

        ax_proxy = plt.axes([0.1,0.925,0.01,0.01])
        add_number(fig,ax_proxy,order=1,offset=[-50,25])
        ax_proxy.spines[['top','right','bottom','left']].set_visible(False)
        # pl_dat.remove_frame(ax_proxy)
        ax_proxy.set_xticks([])
        ax_proxy.set_yticks([])

        #ax = plt.subplot(243)
        ax = plt.axes([0.65,0.65,0.125,0.275])
        add_number(fig,ax,order=2,offset=[-50,25])
        dx = np.diff(self.results['cm'][...,0],axis=1)*self.params['pxtomu']
        ax.hist(dx.flatten(),np.linspace(-10,10,101),facecolor='tab:blue',alpha=0.5)
        ax.hist(dx[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='tab:red',alpha=0.5)
        ax.set_xlabel('$\Delta$x [$\mu$m]')
        ax.set_ylabel('density')
        ax.spines[['top','left','right']].set_visible(False)
        ax.set_ylim([0,10000])
        ax.set_yticks([])

        #ax = plt.subplot(244)
        ax = plt.axes([0.8,0.65,0.125,0.275])
        dy = np.diff(self.results['cm'][...,1],axis=1)*self.params['pxtomu']
        ax.hist(dy.flatten(),np.linspace(-10,10,101),facecolor='tab:blue',alpha=0.5)
        ax.hist(dy[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='tab:red',alpha=0.5)
        ax.set_xlabel('$\Delta$y [$\mu$m]')
        ax.spines[['top','left','right']].set_visible(False)
        ax.set_ylim([0,10000])
        ax.set_yticks([])

        ax = plt.axes([0.73,0.85,0.075,0.05])
        ax.hist(dx.flatten(),np.linspace(-10,10,101),facecolor='tab:blue',alpha=0.5)
        ax.hist(dx[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='tab:red',alpha=0.5)
        # ax.set_xlabel('$\Delta$x [$\mu$m]',fontsize=10)
        ax.set_yticks([])
        ax.spines[['top','left','right']].set_visible(False)
        ax.set_ylim([0,500])

        ax = plt.axes([0.88,0.85,0.075,0.05])
        ax.hist(dy.flatten(),np.linspace(-10,10,101),facecolor='tab:blue',alpha=0.5)
        ax.hist(dy[idx_unsure[:,1:]].flatten(),np.linspace(-10,10,101),facecolor='tab:red',alpha=0.5)
        # ax.set_xlabel('$\Delta$y [$\mu$m]',fontsize=10)
        ax.spines[['top','left','right']].set_visible(False)
        ax.set_ylim([0,500])
        ax.set_yticks([])

        ROI_diff = np.full((nC,nSes,2),np.NaN)
        com_ref = np.full((nC,2),np.NaN)
        for n in range(nC):
            s_ref = np.where(active[n,:])[0]
            if len(s_ref)>0:
                com_ref[n,:] = self.results['cm'][n,s_ref[0],:]
                ROI_diff[n,:nSes-s_ref[0],:] = self.results['cm'][n,s_ref[0]:,:]-com_ref[n,:]
                # print('neuron %d, first session: %d, \tposition: (%.2f,%.2f)'%(n,s_ref[0],com_ref[n,0],com_ref[n,1]))

        ax_mv = plt.axes([0.1,0.11,0.35,0.3])
        add_number(fig,ax_mv,order=3,offset=[-75,50])
        # ROI_diff = (self.results['cm'].transpose(1,0,2)-self.results['cm'][:,0,:]).transpose(1,0,2)#*cluster.para['pxtomu']
        # for n in range(nC):
            # ROI_diff[n,:]
        # ROI_diff = (self.results['cm'].transpose(1,0,2)-com_ref).transpose(1,0,2)#*cluster.para['pxtomu']
        ROI_diff_abs = np.array([np.sqrt(x[:,0]**2+x[:,1]**2) for x in ROI_diff])
        # ROI_diff_abs[~cluster.status[...,1]] = np.NaN


        for n in n_arr:
            ax_mv.plot(range(nSes),ROI_diff_abs[n,:],linewidth=0.5,color=[0.6,0.6,0.6])
        ax_mv.plot(range(nSes),ROI_diff_abs[n,:]*np.NaN,linewidth=0.5,color=[0.6,0.6,0.6],label='displacement')

        plot_with_confidence(ax_mv,range(nSes),np.nanmean(ROI_diff_abs,0),np.nanstd(ROI_diff_abs,0),col='tab:red',ls='-',label='average')
        ax_mv.set_xlabel('session')
        ax_mv.set_ylabel('$\Delta$d [$\mu$m]')
        ax_mv.set_ylim([0,11])
        ax_mv.legend(fontsize=10)
        ax_mv.spines[['top','right']].set_visible(False)

        idx_c_unsure = idx_unsure.any(1)

        ax_mv_max = plt.axes([0.6,0.11,0.35,0.325])
        add_number(fig,ax_mv_max,order=4,offset=[-75,50])
        ROI_max_mv = np.nanmax(ROI_diff_abs,1)
        ax_mv_max.hist(ROI_max_mv,np.linspace(0,20,41),facecolor='tab:blue',alpha=0.5,label='certain')
        ax_mv_max.hist(ROI_max_mv[idx_c_unsure],np.linspace(0,20,41),facecolor='tab:red',alpha=0.5,label='uncertain')
        ax_mv_max.set_xlabel('max($\Delta$d) [$\mu$m]')
        ax_mv_max.set_ylabel('# cluster')
        ax_mv_max.legend(fontsize=10)

        ax_mv_max.spines[['top','right']].set_visible(False)

        plt.tight_layout()
        plt.show(block=False)

        # if sv:
            # pl_dat.save_fig('ROI_positions')


    def plot_match_statistics(self):

        print('### Plotting matching score statistics ###')

        print('now add example how to calculate footprint correlation(?), sketch how to fill cost-matrix')
        
        s = 15
        margins = 20

        nC,nSes = self.results['assignments'].shape
        sessions_bool = np.ones(nSes,'bool')
        active = ~np.isnan(self.results['assignments'])

        D_ROIs = sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.results['cm'][:,s,:]))
        np.fill_diagonal(D_ROIs,np.NaN)

        idx_dense = np.where((np.sum(D_ROIs<margins,1)<=8) & active[:,s] & active[:,s+1])[0]
        c = np.random.choice(idx_dense)
        # c = idx_dense[0]
        # c = 375
        # print(c)
        # print(cluster.IDs['neuronID'][c,s,1])
        n = int(self.results['assignments'][c,s])
        #n = 328
        print(c,n)
        fig = plt.figure(figsize=(7,4),dpi=150)
        props = dict(boxstyle='round', facecolor='w', alpha=0.8)

        ## plot ROIs from a single session

        # c = np.where(cluster.IDs['neuronID'][:,s,1] == n)[0][0]
        # idx_close = np.where(D_ROIs[c,:]<margins*2)[0]

        n_close = self.results['assignments'][D_ROIs[c,:]<margins*1.5,s].astype('int')

        print('load from %s'%self.paths['CaImAn_results'][s])
        A = self.load_footprints(self.paths['CaImAn_results'][s],None)
        Cn = self.data_tmp['Cn']
        
        x,y = self.results['cm'][c,s,:].astype('int')
        print(x,y)

        # x = int(self.results['cm'][c,s,0])#+cluster.sessions['shift'][s,0])
        # y = int(self.results['cm'][c,s,1])#+cluster.sessions['shift'][s,1])
        # x = int(cm[0])#-cluster.sessions['shift'][s,0])
        # y = int(cm[1])#-cluster.sessions['shift'][s,1])

        ax_ROIs1 = plt.axes([0.05,0.55,0.25,0.4])
        add_number(fig,ax_ROIs1,order=1,offset=[-25,25])

        #margins = 10
        Cn_tmp = Cn[y-margins:y+margins,x-margins:x+margins]
        Cn -= Cn_tmp.min()
        Cn_tmp -= Cn_tmp.min()
        Cn /= Cn_tmp.max()

        ax_ROIs1.imshow(Cn,origin='lower',clim=[0,1])
        An = A[...,n].reshape(self.params['dims']).toarray()
        for nn in n_close:
            cc = np.where(self.results['assignments'][:,s]==nn)[0]
            print(cc,nn)
            # print('SNR: %.2g'%cluster.stats['SNR'][cc,s])
            ax_ROIs1.contour(A[...,nn].reshape(self.params['dims']).toarray(),[0.2*A[...,nn].max()],colors='w',linestyles='--',linewidths=1)
        ax_ROIs1.contour(An,[0.2*An.max()],colors='w',linewidths=3)
        # ax_ROIs1.plot(cluster.sessions['com'][c,s,0],cluster.sessions['com'][c,s,1],'kx')

        # sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
        # ax_ROIs1.add_artist(sbar)
        ax_ROIs1.set_xlim([x-margins,x+margins])
        ax_ROIs1.set_ylim([y-margins,y+margins])
        ax_ROIs1.text(x-margins+3,y+margins-5,'Session s',bbox=props,fontsize=10)
        ax_ROIs1.set_xticklabels([])
        ax_ROIs1.set_yticklabels([])

        # plt.show(block=False)
        # return
        D_ROIs_cross = sp.spatial.distance.cdist(self.results['cm'][:,s,:],self.results['cm'][:,s+1,:])
        n_close = self.results['assignments'][D_ROIs_cross[c,:]<margins*2,s+1].astype('int')

        A = self.load_footprints(self.paths['CaImAn_results'][s+1],None)
        
        ## plot ROIs of session 2 compared to one of session 1

        #Cn = cv2.remap(Cn,x_remap,y_remap, interpolation=cv2.INTER_CUBIC)

        shift = np.array(self.data[s+1]['remap']['shift']) - np.array(self.data[s]['remap']['shift'])
        # y_shift = self.data[s+1]['remap']['shift'][0] - self.data[s]['remap']['shift'][0]
        print('shift',shift)

        x_remap,y_remap = build_remap_from_shift_and_flow(self.params['dims'],shift)

        # x_grid, y_grid = np.meshgrid(np.arange(0., cluster.meta['dims'][0]).astype(np.float32), np.arange(0., cluster.meta['dims'][1]).astype(np.float32))
        # x_remap = (x_grid - \
        #               cluster.sessions['shift'][s+1,0] + cluster.sessions['shift'][s,0] + \
        #               cluster.sessions['flow_field'][s+1,:,:,0] - cluster.sessions['flow_field'][s,:,:,0]).astype('float32')
        # y_remap = (y_grid - \
        #               cluster.sessions['shift'][s+1,1] + cluster.sessions['shift'][s,1] + \
        #               cluster.sessions['flow_field'][s+1,:,:,1] - cluster.sessions['flow_field'][s,:,:,1]).astype('float32')
        # # Cn = cv2.remap(Cn,x_remap,y_remap, interpolation=cv2.INTER_CUBIC)

        ax_ROIs2 = plt.axes([0.35,0.55,0.25,0.4])
        add_number(fig,ax_ROIs2,order=2,offset=[-25,25])
        ax_ROIs2.imshow(Cn,origin='lower',clim=[0,1])
        n_match = int(self.results['assignments'][c,s+1])
        for nn in n_close:
            cc = np.where(self.results['assignments'][:,s+1]==nn)
            # print('SNR: %.2g'%cluster.stats['SNR'][cc,s+1])
            if (not (nn==n_match)):# & (cluster.stats['SNR'][cc,s+1]>3):
                A_tmp = cv2.remap(A[...,nn].reshape(self.params['dims']).toarray(),x_remap,y_remap, interpolation=cv2.INTER_CUBIC)
            # A_tmp = A[...,nn].reshape(self.params['dims']).toarray()
                ax_ROIs2.contour(A_tmp,[0.2*A_tmp.max()],colors='r',linestyles='--',linewidths=1)
        ax_ROIs2.contour(An,[0.2*An.max()],colors='w',linewidths=3)
        A_tmp = cv2.remap(A[...,n_match].reshape(self.params['dims']).toarray(),x_remap,y_remap, interpolation=cv2.INTER_CUBIC)
        # A_tmp = A[...,n_match].reshape(self.params['dims']).toarray()
        ax_ROIs2.contour(A_tmp,[0.2*A_tmp.max()],colors='g',linewidths=3)

        ax_ROIs2.set_xlim([x-margins,x+margins])
        ax_ROIs2.set_ylim([y-margins,y+margins])
        ax_ROIs2.text(x-margins+3,y+margins-5,'Session s+1',bbox=props,fontsize=10)
        ax_ROIs2.set_xticklabels([])
        ax_ROIs2.set_yticklabels([])

        ax_zoom1 = plt.axes([0.075,0.125,0.225,0.275])
        add_number(fig,ax_zoom1,order=3,offset=[-50,25])
        ax_zoom1.hist(D_ROIs.flatten(),np.linspace(0,15,31),facecolor='k',density=True)
        ax_zoom1.set_xlabel('distance [$\mu$m]')
        ax_zoom1.spines[['top','left','right']].set_visible(False)
        ax_zoom1.set_yticks([])
        ax_zoom1.set_ylabel('counts')

        ax = plt.axes([0.1,0.345,0.075,0.125])
        plt.hist(D_ROIs.flatten(),np.linspace(0,np.sqrt(2*512**2),101),facecolor='k',density=True)
        ax.set_xlabel('d [$\mu$m]',fontsize=10)
        ax.spines[['top','left','right']].set_visible(False)        
        ax.set_yticks([])

        D_matches = np.copy(D_ROIs_cross.diagonal())
        np.fill_diagonal(D_ROIs_cross,np.NaN)

        ax_zoom2 = plt.axes([0.35,0.125,0.225,0.275])
        add_number(fig,ax_zoom2,order=4,offset=[-50,25])
        ax_zoom2.hist(D_ROIs_cross.flatten(),np.linspace(0,15,31),facecolor='tab:red',alpha=0.5)
        ax_zoom2.hist(D_ROIs.flatten(),np.linspace(0,15,31),facecolor='k',edgecolor='k',histtype='step')
        ax_zoom2.hist(D_matches,np.linspace(0,15,31),facecolor='tab:green',alpha=0.5)
        ax_zoom2.set_xlabel('distance [$\mu$m]')
        ax_zoom2.spines[['top','left','right']].set_visible(False)
        ax_zoom2.set_yticks([])

        ax = plt.axes([0.38,0.345,0.075,0.125])
        ax.hist(D_ROIs_cross.flatten(),np.linspace(0,np.sqrt(2*512**2),101),facecolor='tab:red',alpha=0.5)
        ax.hist(D_matches,np.linspace(0,np.sqrt(2*512**2),101),facecolor='tab:green',alpha=0.5)
        ax.set_xlabel('d [$\mu$m]',fontsize=10)
        ax.spines[['top','left','right']].set_visible(False)
        ax.set_yticks([])

        plt.show(block=False)
        return

        ax = plt.axes([0.7,0.775,0.25,0.125])#ax_sc1.twinx()
        add_number(fig,ax,order=5,offset=[-75,50])
        ax.hist(cluster.stats['match_score'][:,:,0].flat,np.linspace(0,1,51),facecolor='tab:blue',alpha=1,label='$p^*$')
        ax.hist(cluster.stats['match_score'][:,:,1].flat,np.linspace(0,1,51),facecolor='tab:orange',alpha=1,label='max($p\\backslash p^*$)')
        #ax.invert_yaxis()
        ax.set_xlim([0,1])
        ax.set_yticks([])
        ax.set_xlabel('p')
        ax.legend(fontsize=8,bbox_to_anchor=[0.3,0.2],loc='lower left',handlelength=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)


        ax_sc1 = plt.axes([0.7,0.45,0.25,0.125])
        add_number(fig,ax_sc1,order=6,offset=[-75,50])
        # ax = plt.axes([0.925,0.85,0.225,0.05])#ax_sc1.twiny()
        # ax.set_xticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        ax_sc1.plot(cluster.stats['match_score'][:,:,1].flat,cluster.stats['match_score'][:,:,0].flat,'.',markeredgewidth=0,color='k',markersize=1)
        ax_sc1.plot([0,1],[0,1],'--',color='tab:red',lw=1)
        # ax_sc1.plot([0,0.45],[0.5,0.95],'--',color='tab:blue',lw=2)
        # ax_sc1.plot([0.45,1],[0.95,0.95],'--',color='tab:blue',lw=2)
        ax_sc1.set_ylabel('$p^{\\asterisk}$')
        ax_sc1.set_xlabel('max($p\\backslash p^*$)')
        ax_sc1.set_xlim([0,1])
        ax_sc1.set_ylim([0.5,1])
        ax_sc1.spines['top'].set_visible(False)
        ax_sc1.spines['right'].set_visible(False)


        ax_sc2 = plt.axes([0.7,0.125,0.25,0.125])
        add_number(fig,ax_sc2,order=7,offset=[-75,50])
        #plt.hist(np.nanmean(self.results['p_matched'],1),np.linspace(0,1,51))
        ax = ax_sc2.twinx()
        ax.hist(np.nanmin(cluster.stats['match_score'][:,:,0],1),np.linspace(0,1,51),facecolor='tab:red',alpha=0.3)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = ax_sc2.twiny()
        ax.hist(np.nanmean(cluster.stats['match_score'][:,:,0],axis=1),np.linspace(0,1,51),facecolor='tab:blue',orientation='horizontal',alpha=0.3)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax_sc2.plot(np.nanmin(cluster.stats['match_score'][:,:,0],1),np.nanmean(cluster.stats['match_score'][:,:,0],axis=1),'.',markeredgewidth=0,color='k',markersize=1)
        ax_sc2.set_xlabel('min($p^{\\asterisk}$)')
        ax_sc2.set_ylabel('$\left\langle p^{\\asterisk} \\right\\rangle$')
        ax_sc2.set_xlim([0.5,1])
        ax_sc2.set_ylim([0.5,1])
        ax_sc2.spines['top'].set_visible(False)
        ax_sc2.spines['right'].set_visible(False)


        # ax = plt.subplot(248)
        # ax.plot([0,1],[0,1],'--',color='r')
        # ax.scatter(cluster.stats['match_score'][:,:,0],cluster.stats['match_score'][:,:,1],s=1,color='k')
        # ax.set_xlim([0.3,1])
        # ax.set_ylim([-0.05,1])
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
        # ax.set_xlabel('matched score',fontsize=14)
        # ax.set_ylabel('2nd best score',fontsize=14)
        # pl_dat.remove_frame(ax,['top'])
        #
        # ax = plt.subplot(244)
        # #ax.hist(cluster.sessions['match_score'][...,1].flatten(),np.linspace(0,1,101),facecolor='r',alpha=0.5)
        # ax.hist(cluster.stats['match_score'][...,0].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5,density=True,label='match score')
        # pl_dat.remove_frame(ax,['left','right','top'])
        # ax.yaxis.set_label_position("right")
        # #ax.yaxis.tick_right()
        # ax.set_xlim([0.3,1])
        # ax.set_xticks([])
        # ax.set_ylabel('density',fontsize=14)
        # ax.legend(loc='upper left',fontsize=10)
        #
        # ax = plt.subplot(247)
        # ax.hist(cluster.stats['match_score'][...,1].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5,density=True,orientation='horizontal',label='2nd best score')
        # #ax.hist(cluster.sessions['match_score'][...,0].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5)
        # pl_dat.remove_frame(ax,['left','bottom','top'])
        # ax.set_ylim([-0.05,1])
        # ax.set_xlim([1.2,0])
        # ax.set_yticks([])
        # ax.legend(loc='upper right',fontsize=10)
        # ax.set_xlabel('density',fontsize=14)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig('match_stats')

    
    def plot_alignment_statistics(self,s_compare):

        print('### Plotting session alignment procedure and statistics ###')


        nC,nSes = self.results['assignments'].shape
        sessions_bool = np.ones(nSes,'bool')
        active = ~np.isnan(self.results['assignments'])

        # s = s-1

        dims = self.params['dims']
        com_mean = np.nanmean(self.results['cm'],1)

        W = sstats.norm.pdf(range(dims[0]),dims[0]/2,dims[0]/(0.5*1.96))
        W /= W.sum()
        W = np.sqrt(np.diag(W))
        # x_w = np.dot(W,x)

        y = np.hstack([np.ones((512,1)),np.arange(512).reshape(512,1)])
        y_w = np.dot(W,y)
        x = np.hstack([np.ones((512,1)),np.arange(512).reshape(512,1)])
        x_w = np.dot(W,x)

        # pathSession1 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%1])
        # ROIs1_ld = loadmat(pathSession1)
        s_ = 0
        print(self.paths['CaImAn_results'][s_])
        ROIs1_ld = load_data(self.paths['CaImAn_results'][s_])
        print(ROIs1_ld.keys())
        Cn = np.array(ROIs1_ld['A'].sum(1).reshape(dims))
        # Cn = ROIs1_ld['Cn'].T
        Cn -= Cn.min()
        Cn /= Cn.max()
        # if self.data[_s]['remap']['transposed']:
          # Cn2 = Cn2.T
        # dims = Cn.shape

        # p_vals = np.zeros((cluster.meta['nSes'],4))*np.NaN
        p_vals = np.zeros((nSes,2))*np.NaN
        # fig1 = plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        fig = plt.figure(figsize=(7,5),dpi=150)
        for s in tqdm.tqdm(np.where(sessions_bool)[0][1:]):#cluster.meta['nSes'])):

            # try:
                # pathSession2 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
                # ROIs2_ld = load_dict_from_hdf5(self.paths['sessions'][s])

                # Cn2 = np.array(ROIs2_ld['A'].sum(1).reshape(dims))
                # Cn2 = ROIs2_ld['Cn']
                # Cn2 -= Cn2.min()
                # Cn2 /= Cn2.max()
                # if self.data[s]['remap']['transpose']:
                #     Cn2 = Cn2.T
                # print('adjust session position')

                # t_start = time.time()
                # (x_shift,y_shift), flow, corr = get_shift_and_flow(Cn,Cn2,dims,projection=None,plot_bool=False)
                # (x_shift,y_shift) = cluster.sessions['shift'][s,:]
                # flow = cluster.sessions['flow_field'][s,...]

                # x_remap = (x_grid - x_shift + flow[...,0])
                # y_remap = (y_grid - y_shift + flow[...,1])

                # flow = self.data[s]['remap']['flow']
                try:
                  x_remap, y_remap = build_remap_from_shift_and_flow(self.params['dims'],self.data[s]['remap']['shift'],self.data[s]['remap']['flow'])

                  flow_w_y = np.dot(self.data[s]['remap']['flow'][0,...],W)
                  y0,res,rank,tmp = np.linalg.lstsq(y_w,flow_w_y)
                  dy = -y0[0,:]/y0[1,:]
                  idx_out = (dy>512) | (dy<0)
                  r_y = sstats.linregress(np.where(~idx_out),dy[~idx_out])
                  tilt_ax_y = r_y.intercept+r_y.slope*range(512)

                  # print((res**2).sum())
                  res_y = np.sqrt(((tilt_ax_y-dy)**2).sum())/dims[0]
                  # print('y: %.3f'%(np.sqrt(((tilt_ax_y-dy)**2).sum())/dims[0]))

                  flow_w_x = np.dot(self.data[s]['remap']['flow'][1,...],W)
                  x0,res,rank,tmp = np.linalg.lstsq(x_w,flow_w_x)
                  dx = -x0[0,:]/x0[1,:]
                  idx_out = (dx>512) | (dx<0)
                  r_x = sstats.linregress(np.where(~idx_out),dx[~idx_out])
                  tilt_ax_x = r_x.intercept+r_x.slope*range(512)
                  # print(r_x)
                  # print('x:')
                  # print((res**2).sum())
                  # print('x: %.3f'%(np.sqrt(((tilt_ax_x-dx)**2).sum())/dims[0]))
                  res_x = np.sqrt(((tilt_ax_x-dx)**2).sum())/dims[0]
                  r = r_y if (res_y < res_x) else r_x
                  d = dy if (res_y < res_x) else dx
                  tilt_ax = r.intercept+r.slope*range(512)

                  com_silent = com_mean[~active[:,s],:]
                  com_active = com_mean[active[:,s],:]
                  # com_PCs = com_mean[cluster.status[cluster.stats['cluster_bool'],s,2],:]

                  dist_mean = np.abs((r.slope*com_mean[:,0]-com_mean[:,1]+r.intercept)/np.sqrt(r.slope**2+1**2))
                  dist_silent = np.abs((r.slope*com_silent[:,0]-com_silent[:,1]+r.intercept)/np.sqrt(r.slope**2+1**2))
                  dist_active = np.abs((r.slope*com_active[:,0]-com_active[:,1]+r.intercept)/np.sqrt(r.slope**2+1**2))
                  # dist_PCs = np.abs((r.slope*com_PCs[:,0]-com_PCs[:,1]+r.intercept)/np.sqrt(r.slope**2+1**2))

                  # dist = np.abs((r.slope*x_grid-y_grid+r.intercept)/np.sqrt(r.slope**2+1**2))

                  # plt.figure()
                  # ax_dist = plt.subplot(111)
                  # im = ax_dist.imshow(dist,cmap='jet',origin='lower')
                  # cb = plt.colorbar(im)
                  # cb.set_label('distance [$\mu$m]',fontsize=10)
                  # ax_dist.set_xlim([0,dims[0]])
                  # ax_dist.set_ylim([0,dims[0]])
                  # ax_dist.set_xlabel('x [$\mu$m]')
                  # ax_dist.yaxis.tick_right()
                  # ax_dist.yaxis.set_label_position("right")
                  # ax_dist.set_ylabel('y [$\mu$m]')
                  # plt.show(block=False)

                  # plt.figure(fig1.number)
                  # t_start = time.time()
                  r_silent = sstats.ks_2samp(dist_silent,dist_mean)
                  r_active = sstats.ks_2samp(dist_active,dist_mean)
                  # r_cross = sstats.ks_2samp(dist_active,dist_silent)
                  # r_PCs = sstats.ks_2samp(dist_PCs,dist_mean)
                  # p_vals[s,:] = [r_silent.pvalue,r_active.pvalue,r_cross.pvalue,r_PCs.pvalue]
                  # p_vals[s] = r_cross.pvalue
                  p_vals[s,:] = [r_silent.statistic,r_active.statistic]
                except:
                   pass
                # print('time (KS): %.3f'%(time.time()-t_start))
                if s == s_compare:

                    ROIs2_ld = load_data(self.paths['CaImAn_results'][s])

                    Cn2 = np.array(ROIs2_ld['A'].sum(1).reshape(dims))
                    # Cn2 = ROIs2_ld['Cn'].T
                    Cn2 -= Cn2.min()
                    Cn2 /= Cn2.max()
                    # if self.data[s]['remap']['transposed']:
                    #     Cn2 = Cn2.T
                    

                    props = dict(boxstyle='round', facecolor='w', alpha=0.8)

                    ax_im1 = plt.axes([0.1,0.625,0.175,0.35])
                    add_number(fig,ax_im1,order=1,offset=[-50,-5])
                    im_col = np.zeros((512,512,3))
                    im_col[:,:,0] = Cn2
                    ax_im1.imshow(im_col,origin='lower')
                    ax_im1.text(50,430,'Session %d'%(s+1),bbox=props,fontsize=8)
                    ax_im1.set_xticks([])
                    ax_im1.set_yticks([])

                    im_col = np.zeros((512,512,3))
                    im_col[:,:,1] = Cn

                    ax_im2 = plt.axes([0.05,0.575,0.175,0.35])
                    ax_im2.imshow(im_col,origin='lower')
                    ax_im2.text(50,430,'Session %d'%1,bbox=props,fontsize=8)
                    ax_im2.set_xticks([])
                    ax_im2.set_yticks([])
                    # ax_im2.set_xlabel('x [px]',fontsize=14)
                    # ax_im2.set_ylabel('y [px]',fontsize=14)
                    # sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
                    # ax_im2.add_artist(sbar)

                    ax_sShift = plt.axes([0.4,0.575,0.175,0.35])
                    add_number(fig,ax_sShift,order=2)
                    cbaxes = plt.axes([0.4, 0.88, 0.05, 0.02])

                    C = signal.convolve(Cn-Cn.mean(),Cn2[::-1,::-1]-Cn2.mean(),mode='same')/(np.prod(dims)*Cn.std()*Cn2.std())
                    C -= np.percentile(C,95)
                    C /= C.max()
                    im = ax_sShift.imshow(C,origin='lower',extent=[-dims[0]/2,dims[0]/2,-dims[1]/2,dims[1]/2],cmap='jet',clim=[0,1])
                    
                    cb = fig.colorbar(im,cax = cbaxes,orientation='horizontal')
                    cbaxes.xaxis.set_label_position('top')
                    cbaxes.xaxis.tick_top()
                    cb.set_ticks([0,1])
                    cb.set_ticklabels(['low','high'])
                    cb.set_label('corr.',fontsize=10)
                    ax_sShift.arrow(0,0,float(self.data[s]['remap']['shift'][0]),float(self.data[s]['remap']['shift'][1]),head_width=1.5,head_length=2,color='k',width=0.1,length_includes_head=True)
                    ax_sShift.text(-13, -13, 'shift: (%d,%d)'%(self.data[s]['remap']['shift'][0],self.data[s]['remap']['shift'][1]), size=10, ha='left', va='bottom',color='k',bbox=props)

                    #ax_sShift.colorbar()
                    ax_sShift.set_xlim([-15,15])
                    ax_sShift.set_ylim([-15,15])
                    ax_sShift.set_xlabel('$\Delta x [\mu m]$')
                    ax_sShift.set_ylabel('$\Delta y [\mu m]$')

                    ax_sShift_all = plt.axes([0.54,0.79,0.1,0.15])
                    for ss in range(1,nSes):
                        if sessions_bool[ss]:
                            col = [0.6,0.6,0.6]
                        else:
                            col = 'tab:red'
                        try:
                            ax_sShift_all.arrow(0,0,self.data[ss]['remap']['shift'][0],self.data[ss]['remap']['shift'][1],color=col,linewidth=0.5)
                        except:
                            pass
                    ax_sShift_all.arrow(0,0,self.data[s]['remap']['shift'][0],self.data[s]['remap']['shift'][1],color='k',linewidth=0.5)
                    ax_sShift_all.yaxis.set_label_position("right")
                    ax_sShift_all.yaxis.tick_right()
                    ax_sShift_all.xaxis.set_label_position("top")
                    ax_sShift_all.xaxis.tick_top()
                    ax_sShift_all.set_xlim([-25,50])


    

    
                    ax_sShift_all.set_ylim([-25,50])
                    # ax_sShift_all.set_xlabel('x [px]',fontsize=10)
                    # ax_sShift_all.set_ylabel('y [px]',fontsize=10)

                    idxes = 50
                    # tx = dims[0]/2 - 1
                    # ty = tilt_ax_y[int(tx)]
                    ax_OptFlow = plt.axes([0.8,0.625,0.175,0.25])
                    add_number(fig,ax_OptFlow,order=3)

                    x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32), np.arange(0., dims[1]).astype(np.float32))
    
                    ax_OptFlow.quiver(x_grid[::idxes,::idxes], y_grid[::idxes,::idxes], self.data[s]['remap']['flow'][0,::idxes,::idxes], self.data[s]['remap']['flow'][1,::idxes,::idxes], angles='xy', scale_units='xy', scale=0.1, headwidth=4,headlength=4, width=0.002, units='width')#,label='x-y-shifts')
                    ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),d,':',color='tab:green')
                    # ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),dx,'g:')
                    # ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax,'g-')
                    ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax,'-',color='tab:green')

                    ax_OptFlow.set_xlim([0,dims[0]])
                    ax_OptFlow.set_ylim([0,dims[1]])
                    ax_OptFlow.set_xlabel('$x [\mu m]$')
                    ax_OptFlow.set_ylabel('$y [\mu m]$')

                    # ax_OptFlow_stats = plt.axes([0.65,0.6,0.075,0.125])
                    # ax_OptFlow_stats.scatter(flow[:,:,0].reshape(-1,1),flow[:,:,1].reshape(-1,1),s=0.2,marker='.',color='k')#,label='xy-shifts')
                    # ax_OptFlow_stats.plot(np.mean(flow[:,:,0]),np.mean(flow[:,:,1]),marker='.',color='r')
                    # ax_OptFlow_stats.set_xlim(-10,10)
                    # ax_OptFlow_stats.set_ylim(-10,10)
                    # ax_OptFlow_stats.set_xlabel('$\Delta$x [px]',fontsize=10)
                    # ax_OptFlow_stats.set_ylabel('$\Delta$y [px]',fontsize=10)
                    # # ax_OptFlow_stats.yaxis.set_label_position("right")
                    # # ax_OptFlow_stats.yaxis.tick_right()
                    # #ax_OptFlow_stats.legend()


                    # dist_mat = np.abs((r.slope*x_grid-y_grid+r.intercept)/np.sqrt(r.slope**2+1**2))
                    # slope_normal = np.array([-r.slope,1])
                    # slope_normal /= np.linalg.norm(slope_normal)
                    # f_perp = np.dot(flow[:,:,:2],slope_normal)
                    # # print(f_perp)
                    # # print(flow[:,:,0]*slope_normal[0] + flow[:,:,1]*slope_normal[1])
                    # h_dat = np.sign(f_perp)*np.sin(np.arccos((dist_mat - np.abs(f_perp))/dist_mat))*dist_mat

                    # ax = plt.axes([0.575,0.125,0.175,0.35])
                    # ax.yaxis.set_label_position("right")
                    # ax.yaxis.tick_right()
                    # im = ax.imshow(h_dat,origin='lower',cmap='jet',clim=[-30,30])
                    # im = ax.imshow(f_perp,origin='lower',cmap='jet',clim=[-3,3])

                    # cbaxes = plt.axes([0.548, 0.3, 0.01, 0.175])
                    # cb = plt.colorbar(im,cax = cbaxes)
                    # cbaxes.yaxis.set_label_position('left')
                    # cbaxes.yaxis.set_ticks_position('left')
                    # cb.set_label('z [$\mu$m]',fontsize=10)
                    print('shift:',self.data[s]['remap']['shift'])
                    x_remap,y_remap = build_remap_from_shift_and_flow(self.params['dims'],self.data[s]['remap']['shift'],self.data[s]['remap']['flow'])

                    Cn2_corr = cv2.remap(Cn2.astype(np.float32), x_remap, y_remap, cv2.INTER_CUBIC)
                    Cn2_corr -= Cn2_corr.min()
                    Cn2_corr /= Cn2_corr.max()

                    ax_sShifted = plt.axes([0.75,0.11,0.2,0.325])
                    add_number(fig,ax_sShifted,order=6,offset=[-5,25])
                    im_col = np.zeros((512,512,3))
                    im_col[:,:,0] = Cn
                    im_col[:,:,1] = Cn2_corr
                    ax_sShifted.imshow(im_col,origin='lower')
                    ax_sShifted.text(125,510,'aligned sessions',bbox=props,fontsize=10)
                    ax_sShifted.set_xticks([])
                    ax_sShifted.set_yticks([])

                    ax_scatter = plt.axes([0.1,0.125,0.2,0.3])
                    add_number(fig,ax_scatter,order=4)
                    ax_scatter.scatter(com_silent[:,0],com_silent[:,1],s=0.7,c='k')
                    ax_scatter.scatter(com_active[:,0],com_active[:,1],s=0.7,c='tab:orange')
                    # x_ax = np.linspace(0,dims[0]-1,dims[0])
                    # y_ax = n[0]/n[1]*(p[0]-x_ax) + p[1] + n[2]/n[1]*p[2]
                    ax_scatter.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax,'-',color='tab:green')
                    # ax_scatter.plot(x_ax,y_ax,'k-')
                    ax_scatter.set_xlim([0,dims[0]])
                    ax_scatter.set_ylim([0,dims[0]])
                    ax_scatter.set_xlabel('x [$\mu$m]')
                    ax_scatter.set_ylabel('y [$\mu$m]')

                    # x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32),
                                                   # np.arange(0., dims[1]).astype(np.float32))

                    ax_hist = plt.axes([0.4,0.125,0.3,0.3])
                    add_number(fig,ax_hist,order=5,offset=[-50,25])
                    # ax_hist.hist(dist_mean,np.linspace(0,400,21),facecolor='k',alpha=0.5,density=True,label='all neurons')
                    ax_hist.hist(dist_silent,np.linspace(0,400,51),facecolor='k',alpha=0.5,density=True,label='silent')
                    ax_hist.hist(dist_active,np.linspace(0,400,51),facecolor='tab:orange',alpha=0.5,density=True,label='active')
                    ax_hist.legend(loc='lower left',fontsize=8)
                    ax_hist.set_ylabel('density')
                    ax_hist.set_yticks([])
                    ax_hist.set_xlabel('distance from axis [$\mu$m]')
                    ax_hist.set_xlim([0,400])
                    ax_hist.spines[['top','right']].set_visible(False)
            # except:
                # pass

        ax_p = plt.axes([0.525,0.325,0.125,0.125])
        ax_p.axhline(0.01,color='k',linestyle='--')
        ax_p.plot(np.where(sessions_bool)[0],p_vals[sessions_bool,0],'k',linewidth=0.5)
        ax_p.plot(np.where(sessions_bool)[0],p_vals[sessions_bool,1],'tab:orange',linewidth=0.5)
        # ax_p.plot(np.where(sessions_bool)[0],p_vals[sessions_bool],'b')
        #ax_p.plot(np.where(sessions_bool)[0],p_vals[sessions_bool,2],'--',color=[0.6,0.6,0.6])
        #ax_p.plot(np.where(sessions_bool)[0],p_vals[sessions_bool,3],'g--')
        ax_p.set_yscale('log')
        ax_p.xaxis.set_label_position("top")
        ax_p.yaxis.set_label_position("right")
        ax_p.tick_params(axis='y',which='both',left=False,right=True,labelright=True,labelleft=False)
        ax_p.tick_params(axis='x',which='both',top=True,bottom=False,labeltop=True,labelbottom=False)
        # ax_p.xaxis.tick_top()
        # ax_p.yaxis.tick_right()
        ax_p.set_xlabel('session')
        ax_p.set_ylim([10**(-4),1])
        # ax_p.set_ylim([1,0])
        ax_p.set_ylabel('p-value',fontsize=8,rotation='horizontal',labelpad=-5,y=-0.2)
        ax_p.spines[['bottom','left']].set_visible(False)
        # ax_p.tick_params(axis='x',which='both',top=True,bottom=False,labeltop=True,labelbottom=False)

        plt.tight_layout()
        plt.show(block=False)
        # if sv:
        #     pl_dat.save_fig('session_align')

      
    def plot_footprints(self,c,fp_color='r',ax_in=None,use_plotly=False):
        '''
            plots footprints of neuron c across all sessions in 3D view
        '''
        
        X = np.arange(0, self.params['dims'][0])
        Y = np.arange(0, self.params['dims'][1])
        X, Y = np.meshgrid(X, Y)

        use_opt_flow=True

        if ax_in is None:
            fig, ax = plt.subplots(ncols=1,subplot_kw={"projection": "3d"})
        else:
           ax = ax_in
        
        def plot_fp(ax,c):
            
            for s,path in enumerate(self.paths['sessions']):
                # if s > 20: break
                idx = self.results['assignments'][c,s]
                # print('footprint:',s,idx)
                if np.isfinite(idx):
                    file = h5py.File(path,'r')
                    # only load a single variable: A

                    data =  file['/A/data'][...]
                    indices = file['/A/indices'][...]
                    indptr = file['/A/indptr'][...]
                    shape = file['/A/shape'][...]
                    A = sp.sparse.csc_matrix((data[:], indices[:],
                        indptr[:]), shape[:])

                    A = A[:,int(idx)].reshape(self.params['dims']).todense()

                    if s>0:
                        ## use shift and flow to align footprints - apply reverse mapping
                        x_remap,y_remap = build_remap_from_shift_and_flow(self.params['dims'],self.data[s]['remap']['shift'],self.data[s]['remap']['flow'] if use_opt_flow else None)
                        A2 = cv2.remap(A,
                            x_remap, y_remap, # apply reverse identified shift and flow
                            cv2.INTER_CUBIC                 
                        )
                    else:
                        A2 = A
                    
                    A2 /= A2.max()
                    A2[A2<0.1*A2.max()] = np.NaN

                    # mask = A2>0.1*A2.max()
                    ax.plot_surface(X, Y, A2+s, linewidth=0, antialiased=False,rstride=5,cstride=5,color=fp_color)
                    # ax.plot_trisurf(X[mask], Y[mask], A2[mask]+s)
        plot_fp(ax,c)
        margin = 25
        com = np.nanmean(self.results['cm'][c,:],axis=0) / self.params['pxtomu']
        plt.setp(ax,
                xlim=[com[0]-margin,com[0]+margin],
                ylim=[com[1]-margin,com[1]+margin],
            )
        if ax_in:
            plt.show(block=False)       



    def calculate_RoC(self,steps,model='correlation',times=0):
        # key_counts = 'counts' if self.params['correlation_model']=='shifted' else 'counts_unshifted'

        counts = self.scale_counts(times)
        nbins = self.params['nbins']

        p_steps = np.linspace(0,1,steps+1)

        rates = {'tp':      {},
            'tn':      {},
            'fp':      {},
            'fn':      {},
            'cumfrac': {}
        }

        for key in rates.keys():
            rates[key] = {'joint':np.zeros(steps),
                'distance':np.zeros(steps),
                'correlation':np.zeros(steps)
            }
        

        nTotal = counts[...,0].sum()
        for i,p in enumerate(p_steps[:-1]):

            for key in ['joint','distance','correlation']:

                if key == 'joint':
                    idxes_negative = self.model['p_same']['joint'] < p
                    idxes_positive = self.model['p_same']['joint'] >= p

                    tp = counts[idxes_positive,1].sum()
                    tn = counts[idxes_negative,2].sum()
                    fp = counts[idxes_positive,2].sum()
                    fn = counts[idxes_negative,1].sum()

                    rates['cumfrac']['joint'][i] = counts[idxes_negative,0].sum()/nTotal
                elif key == 'distance':
                    idxes_negative = self.model['p_same']['single']['distance'] < p
                    idxes_positive = self.model['p_same']['single']['distance'] >= p

                    tp = counts[idxes_positive,:,1].sum()
                    tn = counts[idxes_negative,:,2].sum()
                    fp = counts[idxes_positive,:,2].sum()
                    fn = counts[idxes_negative,:,1].sum()

                    rates['cumfrac']['distance'][i] = counts[idxes_negative,:,0].sum()/nTotal
                else:
                    idxes_negative = self.model['p_same']['single']['correlation'] < p
                    idxes_positive = self.model['p_same']['single']['correlation'] >= p

                    tp = counts[:,idxes_positive,1].sum()
                    tn = counts[:,idxes_negative,2].sum()
                    fp = counts[:,idxes_positive,2].sum()
                    fn = counts[:,idxes_negative,1].sum()

                    rates['cumfrac']['correlation'][i] = counts[:,idxes_negative,0].sum()/nTotal

                rates['tp'][key][i] = tp/(fn+tp)
                rates['tn'][key][i] = tn/(fp+tn)
                rates['fp'][key][i] = fp/(fp+tn)
                rates['fn'][key][i] = fn/(fn+tp)

        return p_steps, rates
    

    def get_population_mean_and_var(self):
        '''
            function assumes correlation and distance model to be truncated 
            lognormal distributions
        '''
        mean = {
            'distance': {
                'NN': None,
                'nNN': None,
            },
            'correlation': {
                'NN': None,
                'nNN': None,
            }
        }
        var = {
            'distance': {
                'NN': None,
                'nNN': None,
            },
            'correlation': {
                'NN': None,
                'nNN': None,
            }
        }

        mean['correlation']['NN'], var['correlation']['NN'] = mean_of_trunc_lognorm(self.model['fit_parameter']['joint']['correlation']['NN'][:,2],self.model['fit_parameter']['joint']['correlation']['NN'][:,1],[0,1])
        mean['correlation']['NN'] = 1-mean['correlation']['NN']

        mean['distance']['NN'], var['distance']['NN'] = mean_of_trunc_lognorm(self.model['fit_parameter']['joint']['distance']['NN'][:,2],self.model['fit_parameter']['joint']['distance']['NN'][:,1],[0,1])

        if self.params['correlation_model'] == 'unshifted':
            a = self.model['fit_parameter']['joint']['correlation']['nNN'][:,1]
            b = self.model['fit_parameter']['joint']['correlation']['nNN'][:,2]
            #mean_corr_nNN = a/(a+b)
            #var_corr_nNN = a*b/((a+b)**2*(a+b+1))
            mean['correlation']['nNN'] = b
            var['correlation']['nNN'] = a
        else:
            #mean_corr_nNN, var_corr_nNN = mean_of_trunc_lognorm(self.model['fit_parameter']['joint']['correlation']['nNN'][:,1],self.model['fit_parameter']['joint']['correlation']['nNN'][:,0],[0,1])
            mean['correlation']['nNN'] = self.model['fit_parameter']['joint']['correlation']['nNN'][:,2]
            var['correlation']['nNN'] = self.model['fit_parameter']['joint']['correlation']['nNN'][:,1]
        #mean_corr_nNN = 1-mean_corr_nNN

        mean['distance']['nNN'] = self.model['fit_parameter']['joint']['distance']['nNN'][:,2]
        var['distance']['nNN'] = self.model['fit_parameter']['joint']['distance']['nNN'][:,1]
        return mean, var
  

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


def scale_down_counts(counts,times=1):

    '''
      scales down the whole matrix "counts" by a factor of 2^times
    '''

    if times==0:
       return counts
    
    assert counts.shape[0] > 8, 'No further scaling down allowed'
    
    cts = np.zeros(tuple((np.array(counts.shape[:2])/2).astype('int'))+(3,))
    # print(counts.shape,cts.shape)
    for d in range(counts.shape[2]):
        for i in range(2):
            for j in range(2):
              cts[...,d] += counts[i::2,j::2,d]
    
    # print(counts.sum(),cts.sum(),' - ',counts[...,0].sum(),cts[...,0].sum())
    return scale_down_counts(cts,times-1)




def fix_suffix(suffix):
    if suffix:
        if not suffix.startswith('_'):
            suffix = '_' + suffix
    return suffix

