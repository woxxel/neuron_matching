import cv2, sys, os, time, scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rc
from matplotlib_scalebar.scalebar import ScaleBar
import scipy.sparse
from tqdm import *
from scipy.io import loadmat, savemat
# from .utils import find_modes, pickleData, com, get_shift_and_flow, normalize_sparse_array, add_number

# sys.path.append('/home/wollex/Data/Science/PhD/Programs/CaImAn')

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.source_extraction.cnmf.initialization import hals

# from caiman.paths import caiman_datadir
# from caiman.motion_correction import MotionCorrect
# from past.utils import old_div

from .matching import *

### import packages from CNMF, etc

class test_silence_in_sessions:

  def __init__(self,mousePath,
            n_processes=0,
            session_classification=None,
            dataSet='OnACID',s_corr_min=0.3):

    '''

    '''
    
    #     requires:
    #         self.assignment['IDs']
    #         self.alignment['remap'] ('transpose','shift','flow')
    #         self.data (nC,nSes,)
    #         self.paths


    #pathMouse = pathcat([basePath,mouse])
    # self.dataSet = dataSet

    self.match = matching(mousePath)
    self.match.load_registration()

    self.params = {
       's_corr_min': 0.05,
       'dims': self.match.params['dims']
    }
    # self.cluster = cluster(basePath,mouse,nSes,dataSet=dataSet,s_corr_min=s_corr_min)

    # self.cluster.process_sessions(n_processes=n_processes,reprocess=True)
    # self.cluster.get_IDs()
    # self.cluster.get_stats(n_processes=n_processes,complete=False)

    # self.cluster.cluster_classification()
    #self.cluster.get_PC_fields()
    # self.cluster.update_status(complete=False)

    self.dims = self.match.params['dims']

  def run_all(self,s_process=None,n_processes=4,plt_bool=False,complete_new=False):
    
    if not s_process:
       s_process = [self.match.data[s]['filePath'] for s in self.match.data if isinstance(s,int)]
       
    for s,path in enumerate(s_process):

        ## test some stuff (such as session correlation and shift distance)
        if s==0:
            process = True
        else:
            process = self.match.data[s]['remap']['c_max'] > self.params['s_corr_min']
            print(path,self.match.data[s]['remap']['shifts'])
            print(np.sqrt(np.sum([x**2 for x in self.match.data[s]['remap']['shifts']])))
            process &= np.sqrt(np.sum([x**2 for x in self.match.data[s]['remap']['shifts']])) < self.params['max_shift']

        if process:
            print('process this!')
            pass

            t_start = time.time()
            self.obtain_footprints(s,complete_new=complete_new)
            t_end = time.time()
            print('### --------- footprints constructed - time took: %5.3fs ---------- ###'%(t_end-t_start))

            self.prepare_CNMF(n_processes)
            self.run_detection(n_processes,cleanup=True)
            t_end = time.time()
            print('### --------- rerun completed - time took: %5.3fs ---------- ###'%(t_end-t_start))

            self.analyze_traces(redo=True)
            self.save_results('mat')
            t_end = time.time()
            print('### --------- analysis and saving completed - time took: %5.3fs ---------- ###'%(t_end-t_start))

            if plt_bool:
                self.analyze_pre_plot(redo=True)
                self.plot_analysis()
                plt.close('all')


  def obtain_footprints(self,s,max_diff=None,complete_new=False):

    print('Run redetection of "silent" cells on session %d'%(s+1))

    self.s = s
    self.idxes = {}
    self.dataIn = {}
    self.dataOut = {}
    self.data = {}

    if max_diff is None:
      max_diff = self.cluster.meta['nSes']

    dims = self.dims
    T = 8989
    self.dataIn['A'] = scipy.sparse.csc_matrix((np.prod(self.dims),self.cluster.meta['nC']))
    self.dataIn['C'] = np.random.rand(self.cluster.meta['nC'],T)

    if complete_new:
      self.idxes['undetected'] = np.ones(self.cluster.meta['nC'],'bool')
      self.dataIn['b'] = np.random.rand(np.prod(dims),1)
      self.dataIn['f'] = np.random.rand(1,T)
    else:
      ## find silent neurons in session s
      self.idxes['undetected']= ~self.cluster.status[:,self.s,0]

      ## load footprints of active cells
      self.pathSession = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%(s+1)])
      pathData = pathcat([self.pathSession,'results_OnACID.mat'])
      ld = loadmat(pathData,variable_names=['A','C','b','f'])

      c_idx = np.where(~self.idxes['undetected'])[0]
      n_idx = self.cluster.IDs['neuronID'][c_idx,s,1].astype('int')

      if self.cluster.sessions['transpose'][s]:
          print('load transposed from session %d'%(s+1))
          self.dataIn['A'][:,c_idx] = scipy.sparse.hstack([(a/a.sum()).reshape(self.dims).transpose().reshape(-1,1) for a in ld['A'][:,n_idx].T])
      else:
          self.dataIn['A'][:,c_idx] = scipy.sparse.vstack([a/a.sum() for a in ld['A'][:,n_idx].T]).T

      #if not (ld['C'].shape[0] == ld['A'].shape[1]):
        #ld['C'] = ld['C'].transpose()
      T1 = ld['C'].shape[1]
      self.dataIn['C'][c_idx,:T1] = ld['C'][n_idx,:]
      if not (ld['b'].shape[0] == ld['A'].shape[0]):
        ld['b'] = ld['b'].transpose()
      self.dataIn['b'] = ld['b']
      if not (ld['f'].shape[1] == self.dataIn['C'].shape[1]):
        ld['f'] = ld['f'].transpose()
      self.dataIn['f'] = ld['f']

    # print(self.dataIn['b'].shape)
    # print(self.dataIn['f'].shape)

    ## construct footprint of silent cells (interpolation between adjacent sessions?) and adjust for shift & rotation
    s_ref = np.zeros((self.cluster.meta['nC'],2))*np.NaN
    n_ref = np.zeros((self.cluster.meta['nC'],2))*np.NaN

    for c in np.where(self.idxes['undetected'])[0]:
      s_pre = np.where(self.cluster.status[c,:s,0])[0]
      s_post = s+1+np.where(self.cluster.status[c,s+1:,0])[0]

      if len(s_pre)>0:
        s_ref[c,0] = s_pre[-1] if (s-s_pre[-1]) <= max_diff else np.NaN
        if not np.isnan(s_ref[c,0]):
          n_ref[c,0] = self.cluster.IDs['neuronID'][c,int(s_ref[c,0]),1]

      if len(s_post)>0:
        s_ref[c,1] = s_post[0] if (s_post[0] - s) <= max_diff else np.NaN
        if not np.isnan(s_ref[c,1]):
          n_ref[c,1] = self.cluster.IDs['neuronID'][c,int(s_ref[c,1]),1]

    s_load = np.unique(s_ref[~np.isnan(s_ref)])

    progress = tqdm(zip(s_load.astype('int'),range(len(s_load))),total=len(s_load),desc='Loading footprints for processing Session %d...'%(s+1))
    for (s_ld,i) in progress:

      pathSession = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%(s_ld+1)])
      pathData = pathcat([pathSession,'results_OnACID.mat'])
      ld = loadmat(pathData,variable_names=['A'])

      A_tmp = ld['A'].tocsc()
      if self.cluster.sessions['transpose'][s_ld]:
          print('load transposed from session %d'%(s_ld+1))
          A_tmp = scipy.sparse.hstack([img.reshape(self.dims).transpose().reshape(-1,1) for img in A_tmp.transpose()]).tocsc()
          # print(A_tmp)

      (x_shift,y_shift),flow,(x_grid,y_grid),_ = get_shift_and_flow(self.dataIn['A'],A_tmp,dims,projection=1,plot_bool=False)

      x_remap = (x_grid - x_shift + flow[:,:,0])
      y_remap = (y_grid - y_shift + flow[:,:,1])

      A_tmp = sp.sparse.hstack([sp.sparse.csc_matrix(cv2.remap(img.reshape(self.dims), x_remap,y_remap, cv2.INTER_CUBIC).reshape(-1,1)) for img in A_tmp[:,n_ref[s_ref==s_ld].astype('int')].toarray().T])

      self.dataIn['A'][:,np.where(s_ref==s_ld)[0]] += scipy.sparse.vstack([a/a.sum() for a in A_tmp.T]).T
      #print('%d neuron footprints taken from session %d'%(A_tmp.shape[1],s_ld+1))


    max_thr = 0.001
    self.dataIn['A'] = scipy.sparse.vstack([a.multiply(a>(max_thr*a.max()))/a.sum() for a in self.dataIn['A'].T]).T

  def prepare_CNMF(self,n_processes):

    fname = None
    for f in os.listdir(self.pathSession):
      if f.startswith("thy") or f.startswith("shank"):
        fname = pathcat([self.pathSession,f])
        if f.endswith('.h5'):
          break

    if not fname or not os.path.exists(fname):
      print("No file here to process :(")
      return

    #sv_dir = "/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/tmp"
    sv_dir = "/home/wollex/Data/Science/PhD/Data/tmp"
    #svname = self.pathSession + "results_OnACID.mat"
    #if os.path.exists(svname):
      #print("Processed file already present - skipping")
      #return

    svname_h5 = self.pathSession + "results_OnACID.hdf5"

    t_start = time.time()

    # set up CNMF parameters
    params_dict ={

            #general data
            'fnames': [fname],
            'fr': 15,
            'decay_time': 0.47,

            # init
            'ssub': 2,                          # spatial subsampling during initialization
            'tsub': 5,                          # temporal subsampling during initialization

            #online
            'init_batch': 300,                  # number of frames for initialization

            #preprocess
            #'compute_g': True,

            #quality
            'min_SNR': 2,                     # minimum SNR for accepting candidate components
            'SNR_lowest': 0.5,                  # minimum accepted SNR value
            'rval_thr': 0.85,                   # space correlation threshold for accepting a component
            'rval_lowest': 0,
            #'test_both': True,                 # use CNN and correlation to test for new components
            'use_cnn': True
    }

    self.opts = cnmf.params.CNMFParams(params_dict=params_dict)

    self.fname_memmap = pathcat([sv_dir,'memmap%s_%d__d1_512_d2_512_d3_1_order_C_frames_8989_.mmap'%(self.cluster.meta['mouse'],(self.s+1))])
    print(self.fname_memmap)
    if not os.path.exists(self.fname_memmap):
      print("Start writing memmapped file @t = " +  time.ctime())
      if n_processes > 1:
        c, self.dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, single_thread=False)
      else:
        self.dview=None

      self.fname_memmap = cm.save_memmap([fname], base_name='memmap%s_%d_'%(self.cluster.meta['mouse'],(self.s+1)), save_dir=sv_dir, n_chunks=100, order='C', dview=self.dview)  # exclude borders
      if n_processes > 1:
          cm.stop_server(dview=self.dview)      ## restart server to clean up memory
    #self.preprocessed = False


  def run_detection(self,n_processes,cleanup=True):

    #if not self.preprocessed:
    self.cnm = cnmf.CNMF(n_processes)

    Yr, dims, T = cm.load_memmap(self.fname_memmap)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    Y = np.reshape(Yr.T, [T] + list(self.cluster.meta['dims']), order='F')
    #Cn = Y.sum(0)
    #Cn_A = self.dataIn['A'][:,~self.idxes['undetected']].sum(1).reshape(self.cluster.meta['dims'])
    #print(Cn_A.shape)
    #return calculate_img_correlation(Cn,Cn_A),calculate_img_correlation(Cn,Cn_A.T)


    self.cnm.estimates.dims = self.cnm.dims = dims
    self.idxes['in'] = np.array(self.dataIn['A'].sum(0)>0).ravel()
    if self.cluster.sessions['transpose'][self.s]:
        print('transpose all footprints')
        self.dataIn['A'] = scipy.sparse.hstack([img.reshape(self.dims).transpose().reshape(-1,1) for img in self.dataIn['A'].transpose()]).tocsc()

    self.cnm.estimates.A = self.dataIn['A'][:,self.idxes['in']]


    self.cnm.estimates.C = self.dataIn['C'][self.idxes['in'],:]
    self.cnm.estimates.b = self.dataIn['b']
    self.cnm.estimates.f = self.dataIn['f']

    self.cnm.params.change_params({'p':1})

    self.cnm.params.set('online', {'init_batch': 1000})
    Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
    if np.isfortran(Yr):
        raise Exception('The file is in F order, it should be in C order (see save_memmap function')

    # Make sure filename is pointed correctly (numpy sets it to None sometimes)
    try:
        Y.filename = images.filename
        Yr.filename = images.filename
    except AttributeError:  # if no memmapping cause working with small data
        pass

    if self.cnm.estimates.sn is None:
      print('preprocessing...')
      Yr = self.cnm.preprocess(Yr)
      self.preprocessed = True
      print('done!')

    print('temporal and spatial update...')
    self.cnm.update_temporal(Yr)
    self.cnm.update_spatial(Yr)
    self.cnm.update_temporal(Yr)
    print('done!')

    self.cnm.deconvolve(optimize_g=5,s_min=0)
    self.cnm.estimates.evaluate_components(Y,self.opts)

    if cleanup:
      os.remove(self.fname_memmap)


  ## call cnmf to run
  ###   1. temporal trace updates on ROIs and background
  ###   2. spatial update on silent neurons (?) - rather not!

  ## evaluate results:
  ### significant calcium-activity? / spikes?
  ### correlation with overlapping/nearby neurons?

  def run_evaluation(self):
    Yr, dims, T = cm.load_memmap(self.fname_memmap)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    Y = np.reshape(Yr.T, [T] + list(self.cluster.meta['dims']), order='F')
    Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
    #print(Y.shape)
    #print(Yr.shape)

    #self.Y_proj = np.sum(Y,0)
    #self.Yr_proj = np.sum(Yr,1).reshape(dims)

    #plt.figure()
    #plt.subplot(121)
    #plt.imshow(self.Yr_proj,origin='lower')

    #plt.subplot(122)
    #plt.imshow(self.Yr_proj,origin='lower')
    #plt.show(block=False)


    #plt.figure()
    #plt.subplot(121)
    #plt.imshow(self.dataOut['A'].sum(1).reshape(dims),origin='lower')

    #plt.subplot(122)
    #plt.imshow(self.dataOut['A'].sum(1).reshape(dims),origin='lower')
    #plt.show(block=False)

    #plt.pause(1)

    #self.cnm.estimates.dims = self.cnm.dims = dims
    self.cnm.deconvolve(optimize_g=5,s_min=0)
    self.cnm.estimates.evaluate_components(Y,self.opts)

    self.analyze_traces(redo=True)
    #self.analyze_pre_plot(redo=True)
    #self.plot_analysis()



  def analyze_traces(self,redo=False):

    dims = self.cluster.meta['dims']
    nC = self.cluster.meta['nC']


    self.idxes['in'] = np.array(self.dataIn['A'].sum(0)>0).ravel()
    idxes_Ain = np.where(self.idxes['in'])[0]

    ### find proper IDs --- some are removed, some are weirdly remapped to completely other places
    print('Finding proper IDs...')
    nC_in = self.idxes['in'].sum()
    nC_out = self.cnm.estimates.A.shape[1]
    if nC_out < nC_in:
      print('\t %d neurons were removed ...'%(nC_in-nC_out))
    Aout = self.cnm.estimates.A

    ## find next footprints with low distance
    cm_in = com(self.dataIn['A'][:,self.idxes['in']],dims[0],dims[1])
    cm_out = com(Aout,dims[0],dims[1])

    idx_out_bool = np.zeros(nC_out,'bool')
    idx_in_bool = np.zeros(nC_in,'bool')

    idx_new = []
    dc = 0              ## difference in index
    for c in range(nC_out):
      d = np.linalg.norm(cm_in[c+dc,:]-cm_out[c,:])
      if d > 10:
        dist = np.linalg.norm(cm_in-cm_out[c,:],axis=1)
        dc_tmp = np.argmin(dist)-c
        if (dc_tmp>=dc) & (dc_tmp<=(nC_in-nC_out)):
          dc = dc_tmp
          idx_in_bool[c+dc] = True
          idx_out_bool[c] = True

          print('new match has been found: %d to %d with dc=%d'%(c,c+dc,dc))
        else:
          idx_new.append(c)
          print('no match could be found for neuron %d, minimum distance: %5.3g, dc = %d'%(c,dist.min(),dc_tmp))
      else:
        idx_in_bool[c+dc] = True
        idx_out_bool[c] = True

    nC_new = len(idx_new)

    print('%d new neurons detected'%nC_new)
    self.idxes['new'] = np.hstack([np.zeros(nC,'bool'),np.ones(nC_new,'bool')])
    self.idxes['previous'] = np.hstack([~self.idxes['undetected'],np.zeros(nC_new,'bool')])
    self.idxes['in'] = np.zeros(nC+nC_new,'bool')
    self.idxes['in'][idxes_Ain[idx_in_bool]] = True

    print('done!')

    idxes_Ain = np.where(self.idxes['in'])[0]

    idx_out_evaluate = np.zeros(nC_out,'bool')
    idx_out_evaluate[self.cnm.estimates.idx_components] = True
    idx_out_evaluate_new = idx_out_evaluate[~idx_out_bool]
    idx_out_evaluate_old = idx_out_evaluate[idx_out_bool]


    idx_evaluate = np.zeros(nC+nC_new,'bool')
    idx_evaluate[idxes_Ain[idx_out_evaluate_old]] = True
    idx_evaluate[nC:] = idx_out_evaluate_new

    self.dataOut = {}
    self.idxes['evaluate'] = idx_evaluate

    idx_bad = ~idx_evaluate
    idx_good = idx_evaluate

    self.idxes['active_good'] = self.idxes['previous'] & idx_good
    self.idxes['active_bad'] = self.idxes['previous'] & idx_bad

    self.idxes['silent_good'] = ~self.idxes['previous'] & idx_good
    self.idxes['silent_bad'] = ~self.idxes['previous'] & idx_bad

    print('active good: %d, \t active bad: %d'%(self.idxes['active_good'].sum(),self.idxes['active_bad'].sum()))
    print('silent good: %d, \t silent bad: %d'%(self.idxes['silent_good'].sum(),self.idxes['silent_bad'].sum()))


    self.dataOut['A'] = scipy.sparse.csc_matrix((np.prod(dims),nC+nC_new))
    self.dataOut['C'] = np.zeros((nC+nC_new,8989))
    self.dataOut['S'] = np.zeros((nC+nC_new,8989))

    self.dataOut['SNR'] = np.zeros(nC+nC_new)
    self.dataOut['r_values'] = np.zeros(nC+nC_new)
    self.dataOut['CNN'] = np.zeros(nC+nC_new)


    self.dataOut['A'][:,self.idxes['in']] = self.cnm.estimates.A.tocsc()[:,idx_out_bool]
    self.dataOut['A'][:,nC:] = self.cnm.estimates.A.tocsc()[:,~idx_out_bool]

    self.dataOut['C'][self.idxes['in'],:] = self.cnm.estimates.C[idx_out_bool,:]
    self.dataOut['C'][nC:,:] = self.cnm.estimates.C[~idx_out_bool,:]

    self.dataOut['S'][self.idxes['in'],:] = self.cnm.estimates.S[idx_out_bool,:]
    self.dataOut['S'][nC:,:] = self.cnm.estimates.S[~idx_out_bool,:]

    self.dataOut['SNR'][self.idxes['in']] = self.cnm.estimates.SNR_comp[idx_out_bool]
    self.dataOut['SNR'][nC:] = self.cnm.estimates.SNR_comp[~idx_out_bool]

    self.dataOut['r_values'][self.idxes['in']] = self.cnm.estimates.r_values[idx_out_bool]
    self.dataOut['r_values'][nC:] = self.cnm.estimates.r_values[~idx_out_bool]

    self.dataOut['CNN'][self.idxes['in']] = self.cnm.estimates.cnn_preds[idx_out_bool]
    self.dataOut['CNN'][nC:] = self.cnm.estimates.cnn_preds[~idx_out_bool]

    ### analyze active neurons: did they maintain same behavior? (correlation > 0.9)
    #self.Cout[self.idxes['in'],:] = np.vstack([Ca/Ca.max() for Ca in self.cnm.estimates.C])

    #t_start = time.time()
    #self.dataOut['fitness'] = np.zeros(nC+nC_new)
    #self.dataOut['fitness'][self.idxes['in']], erfc, sd_r, md = compute_event_exceptionality(self.dataOut['C'][self.idxes['in'],:])
    #if nC_new > 0:
      #self.dataOut['fitness'][nC:], erfc, sd_r, md = compute_event_exceptionality(self.dataOut['C'][nC:,:])
    #self.fitness[self.fitness<-10**4] = -10**4

    #t_end = time.time()
    #print('fitness computed - time took: %5.3g'%(t_end-t_start))




  def analyze_pre_plot(self,redo=False):

    idx_good = self.idxes['evaluate']
    idx_bad = ~self.idxes['evaluate']

    self.idxes['active_good'] = self.idxes['previous'] & idx_good
    self.idxes['active_bad'] = self.idxes['previous'] & idx_bad

    self.idxes['silent_good'] = ~self.idxes['previous'] & idx_good
    self.idxes['silent_bad'] = ~self.idxes['previous'] & idx_bad

    print('active good: %d, \t active bad: %d'%(self.idxes['active_good'].sum(),self.idxes['active_bad'].sum()))
    print('silent good: %d, \t silent bad: %d'%(self.idxes['silent_good'].sum(),self.idxes['silent_bad'].sum()))

    nC = self.idxes['previous'].shape[0]
    dims = self.cluster.meta['dims']

    self.dataOut['C_std'] = self.dataOut['C'].std(1)
    self.dataIn['C_std'] = self.dataIn['C'].std(1)

    self.data['nu'] = np.zeros(nC)*np.NaN
    for c in tqdm(range(nC),leave=False):
      if (self.dataOut['S'][c,:]>0).sum():
        spike_nr, md, sd_r = get_spikeNr(self.dataOut['S'][c,self.dataOut['S'][c,:]>0])
      else:
        spike_nr = 0
      self.data['nu'][c] = spike_nr/(8989/15)


    #if redo | (not hasattr(self,'A_norm_out')):
    #t_start = time.time()

    self.data['cm'] = com(self.dataOut['A'],dims[0],dims[1])
    self.data['cm'][np.all(self.data['cm']==0,1),:] = np.NaN
    self.data['D_ROIs'] = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.data['cm']))
    np.fill_diagonal(self.data['D_ROIs'],np.NaN)

    self.dataOut['A_norm'] = np.zeros(nC)*np.NaN
    self.dataIn['A_norm'] = np.zeros(nC)*np.NaN
    chunks = 500
    idx_Ain = np.where(self.idxes['in'])[0]
    nC_in = len(idx_Ain)
    steps = int(nC_in/chunks)
    for i in tqdm(range(steps+1),leave=False):
      c_start = i*chunks
      c_end = min(nC_in,(i+1)*chunks)
      self.dataOut['A_norm'][idx_Ain[c_start:c_end]] = np.linalg.norm(self.dataOut['A'][:,idx_Ain[c_start:c_end]].toarray(),axis=0)
      self.dataIn['A_norm'][idx_Ain[c_start:c_end]] = np.linalg.norm(self.dataIn['A'][:,idx_Ain[c_start:c_end]].toarray(),axis=0)
    #t_end = time.time()
    #print('Anorm computed - time took: %5.3g'%(t_end-t_start))

    #if redo | (not hasattr(self,'fp_corr')):
    self.data['corr'] = scipy.sparse.csc_matrix((nC,nC))#np.zeros(nC)*np.NaN
    self.data['fp_corr'] = scipy.sparse.csc_matrix((nC,nC))
    for c in tqdm(range(nC),leave=False):

      if self.idxes['previous'][c]:
        self.data['fp_corr'][c,c] = self.dataOut['A'][:,c].multiply(self.dataIn['A'][:,c]).sum()/(self.dataOut['A_norm'][c]*self.dataIn['A_norm'][c])
        self.data['corr'][c,c] = np.cov(self.dataIn['C'][c,:],self.dataOut['C'][c,:])[0,1]/(self.dataIn['C_std'][c]*self.dataOut['C_std'][c])

      idx_close = np.where(self.data['D_ROIs'][c,:]<10)[0]
      if len(idx_close)>0:
        for cc in idx_close:
          self.data['fp_corr'][c,cc] = self.dataOut['A'][:,c].multiply(self.dataOut['A'][:,cc]).sum()/(self.dataOut['A_norm'][c]*self.dataOut['A_norm'][cc])

          if self.idxes['previous'][cc]:
          #corr_tmp = np.cov(self.Cout[c,:],self.Cout[cc,:])[0,1]/(self.dataOut['C_std'][c]*self.dataOut['C_std'][cc])
            self.data['corr'][c,cc] = np.cov(self.dataOut['C'][c,:],self.dataIn['C'][cc,:])[0,1]/(self.dataOut['C_std'][c]*self.dataIn['C_std'][cc])

    #self.data['fp_corr'].tocsc()


  def save_results(self,ext='mat'):

    ### save everything important from running the detection
    #idx_evaluate = np.zeros(self.cluster.meta['nC']).astype('bool')
    #idx_evaluate[self.idxes['in']] = True
    #idx_evaluate[self.cnm.estimates.idx_components] = True

    results = {'A':self.dataOut['A'],
               'Ain':self.dataIn['A'],
               'C':self.dataOut['C'],
               'S':self.dataOut['S'],
               'b':self.cnm.estimates.b,
               'f':self.cnm.estimates.f,
               'Cin':self.dataIn['C'],
               'idx_previous':self.idxes['previous'],
               'idx_Ain':self.idxes['in'],
               'idx_evaluate':self.idxes['evaluate'],
               'SNR':self.dataOut['SNR'],
               'r_values':self.dataOut['r_values'],
               'CNN':self.dataOut['CNN']}
               #'fitness':self.dataOut['fitness']}

    svPath = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%(self.s+1),'results_redetect.%s'%ext])
    if ext == 'mat':
      savemat(svPath,results)
      print('Data saved in %s'%svPath)
    else:
      pickleData(results,svPath,'save')
    #something from evaluation

    ### analyze silent ones: do they show firing behavior at all? are they correlated with nearby active neurons?

  def load(self,s,ext='mat'):

    self.s = s-1
    pathSession = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%s])
    pathData = pathcat([pathSession,'results_redetect.%s'%ext])
    #pathData = pathcat([pathSession,'results_postSilent.%s'%ext])
    print('loading data from %s'%pathData)
    if ext == 'mat':
      ld = loadmat(pathData,squeeze_me=True)
    else:
      ld = pickleData([],pathData,'load')
    self.dataOut = {'A':ld['A'],
                    'C':ld['C'],
                    'S':ld['S'],
                    'SNR':ld['SNR'],
                    'r_values':ld['r_values'],
                    'CNN':ld['CNN']}
                    #'fitness':ld['fitness']}

    self.dataIn = {'A':ld['Ain'],
                   #'Cn':ld['Cn'],
                   'C':ld['Cin']}

    self.idxes = {'previous':ld['idx_previous'].astype('bool'),
                  'in':ld['idx_Ain'].astype('bool'),
                  'evaluate':ld['idx_evaluate'].astype('bool')}

    self.data = {}


  def plot_analysis(self,SNR_thr=2,r_val_thr=0,CNN_thr=0.6):

    ext = 'png'
    idx_good = (self.dataOut['SNR']>SNR_thr) & (self.dataOut['r_values']>r_val_thr) & (self.dataOut['CNN']>CNN_thr)
    idx_bad = ~idx_good

    self.idxes['active_good'] = self.idxes['previous'] & idx_good
    self.idxes['active_bad'] = self.idxes['previous'] & idx_bad

    self.idxes['silent_good'] = ~self.idxes['previous'] & idx_good
    self.idxes['silent_bad'] = ~self.idxes['previous'] & idx_bad

    print('active good: %d, \t active bad: %d'%(self.idxes['active_good'].sum(),self.idxes['active_bad'].sum()))
    print('silent good: %d, \t silent bad: %d'%(self.idxes['silent_good'].sum(),self.idxes['silent_bad'].sum()))


    idxs = [self.idxes['active_good'],self.idxes['active_bad'],self.idxes['silent_good'],self.idxes['silent_bad']]
    col_arr = ['k','b','r','g']

    ### plot changes from input to output
    acorr = np.diag(self.data['corr'].toarray())
    fp_acorr = np.diag(self.data['fp_corr'].toarray())

    corr = np.copy(self.data['corr'].toarray())
    fp_corr = np.copy(self.data['fp_corr'].toarray())

    np.fill_diagonal(corr,np.NaN)
    np.fill_diagonal(fp_corr,np.NaN)

    al=0.5

    plt.figure(figsize=(4,3))
    ax = plt.axes([0.1,0.2,0.25,0.4])
    for i in range(2):
      ax.hist(acorr[idxs[i]],np.linspace(0.5,1,21),facecolor=col_arr[i],alpha=al,orientation='horizontal')
    #ax.xlabel('$C_{Ca^{2+}}^{in-out}$',fontsize=14)
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax = plt.axes([0.4,0.65,0.35,0.3])
    for i in range(2):
      #if i==1:
        #continue
      plt.hist(fp_acorr[idxs[i]],np.linspace(0.5,1,21),facecolor=col_arr[i],alpha=al)
    #plt.xlabel('$C_{fp}^{in-out}$',fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax = plt.axes([0.4,0.2,0.35,0.4])
    for i in range(2):
      ax.scatter(fp_acorr[idxs[i]],acorr[idxs[i]],s=2,c=col_arr[i])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('$C_{fp}$',fontsize=14)
    ax.set_ylabel('$C_{Ca^{2+}}$',fontsize=14)
    ax.set_yticks(np.linspace(0,1,3))
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    plt.tight_layout()
    plt.show(block=False)
    pathSession = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%(self.s+1)])
    pathFigure = pathcat([pathSession,'find_silent_similarity.%s'%(ext)]);
    plt.savefig(pathFigure,format=ext,dpi=300)
    print('Figure saved as %s'%pathFigure)



    ### plot goodness of detected ROIs
    plt.figure(figsize=(4,3))

    ax = plt.axes([0.15,0.65,0.4,0.25])
    for i in range(4):
      ax.hist(self.dataOut['r_values'][idxs[i]],np.linspace(-1,1,21),facecolor=col_arr[i],alpha=al)
    #plt.xscale('log')
    ax.set_xlim([-1,1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.set_xlabel('r value',fontsize=14)

    ax = plt.axes([0.65,0.2,0.25,0.4])
    for i in range(4):
      ax.hist(self.dataOut['SNR'][idxs[i]],np.linspace(0,30,21),facecolor=col_arr[i],alpha=al,orientation='horizontal')
    #ax.invert_xaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.yaxis.tick_right()
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.yaxis.set_label_position("right")
    #ax.set_ylabel('SNR',fontsize=14)
    ax.set_ylim([0,30])
    ax.set_xlabel('occurence',fontsize=14)

    ax = plt.axes([0.15,0.2,0.4,0.4])
    for i in range(4):
      ax.plot(self.dataOut['r_values'][idxs[i]],self.dataOut['SNR'][idxs[i]],color=col_arr[i],markersize=1)
    ax.set_xlabel('r value',fontsize=14)
    ax.yaxis.tick_right()
    ax.set_ylabel('SNR',fontsize=14)
    ax.set_xlim([-1,1])
    ax.set_ylim([0,30])
    ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.set_yticks([])

    plt.tight_layout()
    plt.show(block=False)

    pathSession = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%(self.s+1)])
    pathFigure = pathcat([pathSession,'find_silent_evaluation.%s'%(ext)]);
    plt.savefig(pathFigure,format=ext,dpi=300)
    print('Figure saved as %s'%pathFigure)

  def plot_traces(self,idxs,n=3):

    ## some nice plots:
    ###   1. active vs silent ROI shapes
    ###   2. correlation vs activity

    plt.figure()
    idxs = np.where(idxs)[0]
    for i in range(n):

      c = idxs[np.random.randint(len(idxs))]
      print(c)
      plt.subplot(2,n,i+1)
      plt.imshow(self.dataIn['A'][:,c].reshape(512,512).toarray())
      plt.ylabel('$C_{out}$')

      plt.subplot(4,n,2*n+i+1)
      plt.plot(self.dataOut['C'][c,:])
      plt.ylabel('$C_{out}$')

      plt.subplot(4,n,3*n+i+1)
      plt.plot(self.dataOut['C'][c,:])
      plt.ylabel('$C_{in}$')

    #plt.subplot(3,2,6)
    #plt.plot(self.cnm.estimates.S[c,:])
    #plt.ylabel('$S_{out}$')

    #plt.subplot(315)
    #plt.plot(Cout[c,:])
    #if not self.idxes['undetected'][c]:
      #plt.plot(Cin[c,:])
    plt.suptitle('correlation: %5.3f'%self.data['corr'][c,c])
    plt.show(block=False)

  def plot_detection(self):

    ext = 'png'
    plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.imshow(self.dataOut['A'].sum(1).reshape(self.cluster.meta['dims']),origin='lower')

    [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=0.5) for a in self.dataIn['A'][:,~self.idxes['undetected']].T]
    #[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=0.5) for a in self.dataIn['A'][:,self.idxes['undetected']].T]
    plt.tight_layout()
    plt.show(block=False)

    pathSession = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%(self.s+1)])
    pathFigure = pathcat([pathSession,'find_silent_only_before.%s'%(ext)]);
    plt.savefig(pathFigure,format=ext,dpi=300)
    print('Figure saved as %s'%pathFigure)

    #plt.figure(figsize=(5,4))
    #ax = plt.subplot(111)
    #ax.imshow(self.dataOut['A'].sum(1).reshape(self.cluster.meta['dims']),origin='lower')

    #[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=0.5) for a in self.dataOut['A'][:,self.idxes['active_good']].T]
    #[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=0.5) for a in self.dataOut['A'][:,self.idxes['silent_good']].T]

    ##[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=1) for a in self.dataIn['A'][:,self.idxes['undetected']].T]
    #plt.tight_layout()
    #plt.show(block=False)
    #pathSession = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%(self.s+1)])
    #pathFigure = pathcat([pathSession,'find_silent_after.%s'%(ext)]);
    #plt.savefig(pathFigure,format=ext,dpi=300)
    #print('Figure saved as %s'%pathFigure)


class plot_test_undetected:

  def __init__(self,basePath,mouse):

    dataSet='redetect'  ## nothing else makes sense

    self.pathMouse = os.path.join(basePath,mouse)
    self.pathMatching = os.path.join(self.pathMouse,'matching/Sheintuch_registration_results_%s.pkl'%dataSet)


  def load_eval(self,ext='mat'):
      ld_dat = pickleData([],self.pathMatching,'load')

      self.eval_data = {}
      self.progress = tqdm(range(self.nSes))
      # self.matches = {}
      for s in self.progress:
          ## load results
          pathSession = os.path.join(self.pathMouse,'Session%02d'%(s+1))
          pathData = os.path.join(pathSession,'results_redetect.%s'%ext)
          self.progress.set_description('Now processing Session %d'%(s+1))
          if ext == 'mat':
            ld = loadmat(pathData,squeeze_me=True)
          else:
            ld = pickleData([],pathData,'load')

          self.eval_data[s] = {}#   {'SNR':       np.zeros((self.nC,self.nSes))*np.NaN,
                                #'CNN':       np.zeros((self.nC,self.nSes))*np.NaN,
                                #'r_values':  np.zeros((self.nC,self.nSes))*np.NaN}
          for key in ['SNR','r_values','CNN','idx_previous']:
              self.eval_data[s][key] = ld[key]


  def load(self,ext='mat'):

    ### load matching results
    ld_dat = pickleData([],self.pathMatching,'load')
    try:
      assignments = ld_dat['assignments']
    except:
      assignments = ld_dat['assignment']
    self.nC,self.nSes = assignments.shape

    ### preallocate arrays
    self.data = {'eval':        {'SNR':       np.zeros((self.nC,self.nSes))*np.NaN,
                                 'CNN':       np.zeros((self.nC,self.nSes))*np.NaN,
                                 'r_values':  np.zeros((self.nC,self.nSes))*np.NaN},
                 'nonMatched':{'SNR':           np.zeros((10000,self.nSes))*np.NaN,
                                'CNN':          np.zeros((10000,self.nSes))*np.NaN,
                                'r_values':     np.zeros((10000,self.nSes))*np.NaN},
                 'fp_corr':   np.zeros((self.nC,self.nSes))*np.NaN,
                 'C_corr':    np.zeros((self.nC,self.nSes))*np.NaN,
                 'idxes':       {'previous':  np.zeros((self.nC,self.nSes),'bool'),
                                 'in':        np.zeros((self.nC,self.nSes),'bool'),
                                 'detected':  np.zeros((self.nC,self.nSes),'bool')}
                 }

    ### for all s, store eval and idxes and calculate A-A and C-C correlation
    self.progress = tqdm(range(self.nSes))
    self.matches = {}
    for s in self.progress:

      ## load results
      pathSession = os.path.join(self.pathMouse,'Session%02d'%(s+1))
      pathData = os.path.join(pathSession,'results_redetect.%s'%ext)
      self.progress.set_description('Now processing Session %d'%(s+1))
      if ext == 'mat':
        ld = loadmat(pathData,squeeze_me=True)
      else:
        ld = pickleData([],pathData,'load')

      idx_c = np.where(~np.isnan(assignments[:,s]))[0]
      n_arr = assignments[idx_c,s].astype('int')

      ## store eval
      non_n_arr = np.zeros(10000,'bool')
      non_n_arr[:len(ld['SNR'])] = True
      non_n_arr[n_arr] = False
      for key in ['SNR','r_values','CNN']:
        self.data['eval'][key][idx_c,s] = ld[key][n_arr]
        self.data['nonMatched'][key][non_n_arr,s] = ld[key][np.where(non_n_arr)]

      # print(np.isfinite(self.data['nonMatched']['SNR']).sum())
      ## store indices
      self.data['idxes']['detected'][idx_c,s] = True
      self.data['idxes']['previous'][idx_c,s] = ld['idx_previous'][n_arr]
      self.data['idxes']['in'][idx_c,s] = ld['idx_Ain'][n_arr]

      dims = (512,512)
      ## find next footprints with low distance
      cm_in = com(ld['Ain'],dims[0],dims[1])
      cm_out = com(ld['A'],dims[0],dims[1])
      #return ld['A'], ld['Ain']

      nC_in = ld['Ain'].shape[1]
      nC_out = ld['A'].shape[1]

      D_ROIs = sp.spatial.distance.cdist(cm_in,cm_out)
      D_ROI_mask = np.ma.array(D_ROIs, mask = np.isnan(D_ROIs))

      #idx_out_bool = np.zeros(nC_out,'bool')
      #idx_in_bool = np.zeros(nC_in,'bool')

      self.matches[s] = np.zeros(cm_out.shape[0])*np.NaN
      idx_new = []
      nC_new = 0
      dc = 0              ## difference in index
      for c in np.where((ld['A']>0).sum(0)>50)[1]:
        idx_close = np.where(D_ROI_mask[:,c]<5)[0]
        if len(idx_close):
            d_idx = np.min(np.abs(idx_close-c))
            closeby = idx_close[np.argmin(np.abs(idx_close-c))]
            if np.isscalar(closeby) & (d_idx <= dc):
                self.matches[s][c] = closeby
            elif np.isscalar(closeby) & (d_idx < (dc+10)):
                dc = d_idx
                self.matches[s][c] = closeby

      C_std = ld['C'].std(1)
      Cin_std = ld['Cin'].std(1)

      chunks = 1000
      idx_Ain = np.where((ld['Ain']>0).sum(0)>50)[1]#np.where(self.data['idxes']['in'][:,s])[0]
      Ain_norm = np.zeros(ld['Ain'].shape[1])*np.NaN
      nC_in = len(idx_Ain)
      steps = int(nC_in/chunks)
      for i in tqdm(range(steps+(np.mod(nC_in,chunks)>0)),leave=False):
        c_start = i*chunks
        c_end = min(nC_in,(i+1)*chunks)
        Ain_norm[idx_Ain[c_start:c_end]] = np.linalg.norm(ld['Ain'][:,idx_Ain[c_start:c_end]].toarray(),axis=0)

      idx_A = np.where((ld['A']>0).sum(0)>50)[1]#np.where(self.data['idxes']['in'][:,s])[0]
      A_norm = np.zeros(ld['A'].shape[1])*np.NaN
      nC_out = len(idx_A)
      steps = int(nC_out/chunks)
      for i in tqdm(range(steps+(np.mod(nC_out,chunks)>0)),leave=False):
        c_start = i*chunks
        c_end = min(nC_out,(i+1)*chunks)
        A_norm[idx_A[c_start:c_end]] = np.linalg.norm(ld['A'][:,idx_A[c_start:c_end]].toarray(),axis=0)

      ## calculate correlation between input and output
      for (n2,n1) in tqdm(enumerate(self.matches[s]),total=len(self.matches[s]),leave=False):
        if ~np.isnan(n1):
            c = np.where(assignments[:,s]==n2)[0]
            n1 = int(n1)
            self.data['fp_corr'][c,s] = ld['A'][:,n2].multiply(ld['Ain'][:,n1]).sum()/(A_norm[n2]*Ain_norm[n1])
            self.data['C_corr'][c,s] = np.cov(ld['C'][n2,:],ld['Cin'][n1,:])[0,1]/(C_std[n2]*Cin_std[n1])


  def get_cluster(self,basePath,mouse,nSes,dataSet='redetect',n_processes=0,s_corr_min=0.3):

    self.cluster = cluster(basePath,mouse,nSes,dataSet=dataSet,s_corr_min=s_corr_min)
    self.cluster.load()

    # self.cluster.process_sessions(n_processes=n_processes,reprocess=True)
    # self.cluster.get_IDs()
    # self.cluster.get_stats(n_processes=n_processes,complete=False)

  def plot(self,SNR_thr=2,rval_thr=0,CNN_thr=0.6,pm_thr=0.5,sv=False):

    rc('font',size=10)
    rc('axes',labelsize=12)
    rc('xtick',labelsize=8)
    rc('ytick',labelsize=8)

    dims = (512,512)
    idx = self.data['idxes']['previous']

    idx_good = (self.data['eval']['SNR']>SNR_thr) & (self.data['eval']['r_values']>rval_thr) & (self.data['eval']['CNN']>CNN_thr)
    idx_bad = ~idx_good

    idxes = {}
    idxes['active_good'] = idx & idx_good
    idxes['active_bad'] = idx & idx_bad

    idxes['silent_good'] = ~idx & idx_good
    idxes['silent_bad'] = ~idx & idx_bad

    pathMatching = os.path.join(self.pathMouse,'matching/Sheintuch_registration_results_OnACID.pkl')
    dt = pickleData([],pathMatching,'load')
    assignments_OnACID = dt['assignments']

    pathMatching = os.path.join(self.pathMouse,'matching/Sheintuch_registration_results_redetect.pkl')
    ld_dat = pickleData([],pathMatching,'load')
    assignments = ld_dat['assignments']

    self.cluster.meta['SNR_thr'] = SNR_thr
    self.cluster.meta['rval_thr'] = rval_thr
    self.cluster.meta['CNN_thr'] = CNN_thr
    self.cluster.cluster_classification()

    idx_pm = (self.cluster.stats['match_score'][...,0]>0.5)# ((self.cluster.stats['match_score'][...,0]-self.cluster.stats['match_score'][...,1])>pm_thr) | (self.cluster.stats['match_score'][...,0]>0.95)

    if False:
        fig = plt.figure(figsize=(7,4),dpi=300)

        ax_ROIs = plt.axes([0.05,0.625,0.2,0.325])
        add_number(fig,ax_ROIs,order=1,offset=[-25,25])
        ax_ROIs2 = plt.axes([0.3,0.625,0.2,0.325])
        add_number(fig,ax_ROIs2,order=2,offset=[-25,25])

        s = 10

        n_arr = assignments_OnACID[np.where(~np.isnan(assignments_OnACID[:,s-1]))[0],s-1].astype('int')

        pathSession = os.path.join(self.pathMouse,'Session%02d'%(s))
        pathData = os.path.join(pathSession,'results_redetect.mat')
        ld = loadmat(pathData,squeeze_me=True)

        cm_in = com(ld['Ain'],dims[0],dims[1])
        cm_out = com(ld['A'],dims[0],dims[1])

        idx_A = ld['idx_previous'][:ld['Ain'].shape[1]].astype('bool')
        #idx_A = np.zeros(ld['Ain'].shape[1],'bool')
        #idx_A[n_arr] = True

        bounds = [[175,225],[175,225]]
        self.plot_detection(ax_ROIs,ld['Ain'][:,idx_A],ld['Ain'][:,~idx_A],bounds=bounds)
        ax_ROIs.set_xlim(bounds[0])
        ax_ROIs.set_ylim(bounds[1])
        ax_ROIs.set_xticks([])
        ax_ROIs.set_yticks([])
        sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
        ax_ROIs.add_artist(sbar)

        idx_A = ld['idx_previous'].astype('bool')# & idx_n
        # print(idx_A.shape)
        #idx_A = np.zeros(ld['A'].shape[1],'bool')
        #idx_A[n_arr] = True
        # print(ld['A'].shape)
        nC = self.data['eval']['SNR'].shape[0]
        nN = ld['A'].shape[1]
        c_arr = np.where(assignments[:,s-1]>=0)[0]
        n_arr = assignments[c_arr,s-1].astype('int')
        idx_n = np.in1d(range(nN),n_arr)
        idx_c = np.in1d(range(nC),c_arr)
        idx_mapping_cn = -np.ones(nC,'int')
        idx_mapping_cn[c_arr] = n_arr
        idx_mapping_nc = -np.ones(nN,'int')
        idx_mapping_nc[n_arr] = c_arr

        idx_good_n = np.in1d(range(nN),idx_mapping_cn[np.squeeze(idx_good[:,s-1])])

        # CM = com(ld['A'],512,512)
        # idx_bounds = (CM[:,0] > bounds[0][0]) & (CM[:,0] < bounds[0][1]) & (CM[:,1] > bounds[1][0]) & (CM[:,1] < bounds[1][1])

        self.plot_detection(ax_ROIs2,ld['A'][:,idx_A],ld['A'][:,~idx_A],bounds=bounds,lw=0.5,ls='dotted')
        self.plot_detection(ax_ROIs2,ld['A'][:,idx_A & idx_good_n],ld['A'][:,(~idx_A) & idx_good_n],bounds=bounds)

        ax_ROIs2.set_xlim(bounds[0])
        ax_ROIs2.set_ylim(bounds[1])
        ax_ROIs2.set_xticks([])
        ax_ROIs2.set_yticks([])

        # return
        #ax1 = ax.twinx()
        #ax2 = ax.twiny()
        #ax1.hist(self.data['fp_corr'][idx],np.linspace(0,1,101),facecolor='k',alpha=0.5)
        #ax1.hist(self.data['fp_corr'][~idx],np.linspace(0,1,101),facecolor='r',alpha=0.5)
        #ax1.set_yticks([])
        #ax1.invert_yaxis()

        #ax2.hist(self.data['C_corr'][idx],np.linspace(0,1,101),facecolor='k',alpha=0.5,orientation='horizontal')
        #ax2.hist(self.data['C_corr'][~idx],np.linspace(0,1,101),facecolor='r',alpha=0.5,orientation='horizontal')
        #ax2.set_xticks([])

        #plt.figure(figsize=(6,2.5))
        ax = plt.axes([0.55,0.65,0.17,0.3])
        add_number(fig,ax,order=5,offset=[-25,25])

        ax.plot(self.data['fp_corr'][idx].flat,self.data['C_corr'][idx].flat,'ko',markersize=0.5,markeredgewidth=0,alpha=0.5)
        ax.plot(self.data['fp_corr'][~idx].flat,self.data['C_corr'][~idx].flat,'o',color='tab:orange',markersize=0.5,markeredgewidth=0,alpha=0.5)
        ax.plot(-1,np.NaN,'ko',markersize=5,markeredgewidth=0,alpha=0.5,label='initial')
        ax.plot(-1,np.NaN,'o',color='tab:orange',markersize=5,markeredgewidth=0,alpha=0.5,label='inferred')

        ax.set_xlabel('$c_{fp}$')
        ax.set_ylabel('$c_{Ca}$')
        ax.set_xlim([-0.1,1])
        ax.tick_params(axis='y',which='both',left=False,right=True,labelright=True,labelleft=False)
        ax.yaxis.set_label_position("right")
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([-0.1,1])
        ax.set_yticks(np.linspace(0,1,3))

        ax.legend(loc='upper left',bbox_to_anchor=[-0.1,0.95],fontsize=8)

        SNR = []
        SNR_not = []
        r_val = []
        r_val_not = []
        for s in range(self.nSes):
            SNR.extend(self.eval_data[s]['SNR'][self.eval_data[s]['idx_previous'].astype('bool')])
            SNR_not.extend(self.eval_data[s]['SNR'][~self.eval_data[s]['idx_previous'].astype('bool')])
            r_val.extend(self.eval_data[s]['r_values'][self.eval_data[s]['idx_previous'].astype('bool')])
            r_val_not.extend(self.eval_data[s]['r_values'][~self.eval_data[s]['idx_previous'].astype('bool')])
        # r_val_not.extend(self.data['nonMatched']['r_values'].flat)
        # r_val_not.extend(self.data['eval']['r_values'][~idx])
        # SNR_not.extend(self.data['nonMatched']['SNR'].flat)
        # SNR_not.extend(self.data['eval']['SNR'][~idx])
        # print(SNR)
        # print(len(SNR))

        ax = plt.axes([0.55,0.125,0.17,0.3])
        add_number(fig,ax,order=8,offset=[-25,25])
        ax.plot(r_val_not,SNR_not,'o',color='tab:orange',markersize=0.25,markeredgewidth=0,alpha=0.5)
        ax.plot(r_val,SNR,'ko',markersize=0.25,markeredgewidth=0,alpha=0.5)
        # ax.plot(self.data['eval']['r_values'][idx],self.data['eval']['SNR'][idx],'ko',markersize=0.25,markeredgewidth=0,alpha=0.5)
        # ax.plot(self.data['nonMatched']['r_values'],self.data['nonMatched']['SNR'],'o',color='tab:orange',markersize=0.25,markeredgewidth=0,alpha=0.5)
        # ax.plot(self.data['eval']['r_values'][~idx],self.data['eval']['SNR'][~idx],'o',color='tab:orange',markersize=0.25,markeredgewidth=0,alpha=0.5)
        ax.plot([-1,1],[3,3],'k--',linewidth=0.5)
        ax.plot([0.5,0.5],[1,40],'k--',linewidth=0.5)

        ax.set_yscale('log')
        ax.set_ylabel('SNR')
        ax.set_xlim([-0.5,1])
        ax.set_ylim([1,40])
        ax.set_xlabel('r-value')
        ax.tick_params(axis='y',which='both',left=False,right=True,labelright=True,labelleft=False)
        ax.yaxis.set_label_position("right")
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax = plt.axes([0.85,0.85,0.125,0.1])
        add_number(fig,ax,order=6,offset=[-50,25])
        ax.hist(self.data['fp_corr'][idx],np.linspace(0.5,1,21),density=True,facecolor='k',alpha=0.8)
        ax.hist(self.data['fp_corr'][~idx],np.linspace(0.5,1,21),density=True,facecolor='tab:orange',alpha=0.8)
        ax.set_xlabel('$c_{fp}$')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = plt.axes([0.85,0.625,0.125,0.1])
        add_number(fig,ax,order=7,offset=[-50,25])
        ax.hist(self.data['C_corr'][idx],np.linspace(-0.2,1,25),density=True,facecolor='k',alpha=0.8)
        ax.hist(self.data['C_corr'][~idx],np.linspace(-0.2,1,25),density=True,facecolor='tab:orange',alpha=0.8)
        ax.set_xlabel('$c_{Ca}$')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = plt.axes([0.85,0.35,0.125,0.1])
        add_number(fig,ax,order=9,offset=[-50,25])
        # ax.hist(self.data['eval']['r_values'][idx],np.linspace(-0.5,1,31),density=True,facecolor='k',alpha=0.8)
        ax.hist(r_val,np.linspace(-0.5,1,31),density=True,facecolor='k',alpha=0.8)
        ax.hist(r_val_not,np.linspace(-0.5,1,31),density=True,facecolor='tab:orange',alpha=0.8)
        # ax.hist(self.data['eval']['r_values'][~idx],np.linspace(-0.5,1,31),density=True,facecolor='tab:orange',alpha=0.8)
        # ax.hist(self.data['nonMatched']['r_values'].flat,np.linspace(-0.5,1,31),density=True,facecolor='tab:orange',alpha=0.8)
        ax.plot([0.5,0.5],[0,ax.get_ylim()[1]],'k--')
        ax.set_xlabel('r-value')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = plt.axes([0.85,0.125,0.125,0.1])
        add_number(fig,ax,order=10,offset=[-50,25])
        # ax.hist(self.data['eval']['SNR'][idx],np.linspace(0,30,31),density=True,facecolor='k',alpha=0.8)
        ax.hist(SNR,np.linspace(0,30,31),density=True,facecolor='k',alpha=0.8)
        ax.hist(SNR_not,np.linspace(0,30,31),density=True,facecolor='tab:orange',alpha=0.8)
        # ax.hist(self.data['eval']['SNR'][~idx],np.linspace(0,30,31),density=True,facecolor='tab:orange',alpha=0.8)
        # ax.hist(self.data['nonMatched']['SNR'].flat,np.linspace(0,30,31),density=True,facecolor='tab:orange',alpha=0.8)
        ax.plot([3,3],[0,ax.get_ylim()[1]],'k--')
        ax.set_xlabel('SNR')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ax1.hist(self.data['eval']['r_values'][idx],np.linspace(-0.5,1,151),facecolor='k',alpha=0.5)
        # ax1.hist(self.data['eval']['r_values'][~idx],np.linspace(-0.5,1,151),facecolor='r',alpha=0.5)
        # ax1.set_yticks([])
        # ax1.invert_yaxis()
        #
        # ax2 = ax.twiny()
        # ax2.hist(self.data['eval']['SNR'][idx],np.linspace(0,40,101),facecolor='k',alpha=0.5,orientation='horizontal')
        # ax2.hist(self.data['eval']['SNR'][~idx],np.linspace(0,40,101),facecolor='r',alpha=0.5,orientation='horizontal')
        # ax2.set_xticks([])


        # plt.tight_layout()
        # plt.show(block=False)

        #ext = 'png'
        #path = pathcat([self.pathMouse,'complete_set_evaluation.%s'%ext])
        #plt.savefig(path,format=ext,dpi=300)

        ax_A = plt.axes([0.05,0.375,0.125,0.2])
        add_number(fig,ax_A,order=3,offset=[-25,25])
        ax_Ca = plt.axes([0.2,0.375,0.275,0.2])

        idx_good = np.where((self.data['C_corr']>0.8) & idx)
        i = np.random.randint(len(idx_bad[0]))
        c = idx_good[0][i]
        s = idx_good[1][i]
        s,c,n = (7,0,0)
        # ld_dat = pickleData([],self.pathMatching,'load')
        # assignments = ld_dat['assignments']
        n = int(assignments[c,s])
        #plt.draw()
        print(s,c,n)    ## (73,1441,2406)

        pathSession = os.path.join(self.pathMouse,'Session%02d'%(s+1))
        pathData = os.path.join(pathSession,'results_redetect.mat')
        ld = loadmat(pathData,variable_names=['Cin','C','Ain','A'],squeeze_me=True)

        dims=(512,512)
        ax_A.imshow(ld['A'].sum(1).reshape(dims))

        Cin = ld['Cin']/ld['Cin'].max(1)[:,np.newaxis]
        C = ld['C']/ld['C'].max(1)[:,np.newaxis]

        cm = com(ld['A'],dims[0],dims[1])

        # calculate CoMs and distance
        D_ROIs = sp.spatial.distance.cdist(cm,cm)
        n_close = np.where(D_ROIs[n,:]<10)[0]
        print(n_close)
        A_norm = np.linalg.norm(ld['A'].toarray(),axis=0)

        # plot all closeby ones, highlight overlapping ones
        t_arr = np.linspace(0,8989/15,8989)
        ax_Ca.plot(t_arr,Cin[n,:],'k',linewidth=0.5)
        ax_Ca.plot(t_arr,C[n,:]+1,'tab:green',linewidth=0.5)
        #ax_Ca.text(6000/15,1.5,'corr: %5.3g, SNR: %5.3g'%(self.data['fp_corr'][c,s],self.data['eval']['SNR'][c,s]))

        ax_A.contour((ld['Ain'][:,n]/ld['Ain'][:,n].max()).reshape(dims).toarray(), levels=[0.3], colors='w', linewidths=2)
        ax_A.contour((ld['A'][:,n]/ld['A'][:,n].max()).reshape(dims).toarray(), levels=[0.3], colors='tab:green', linewidths=2)

        i = 0
        for nn in n_close:
          A_corr = ld['A'][:,n].multiply(ld['A'][:,nn]).sum()/(A_norm[n]*A_norm[nn])
          if (A_corr>0.3) and not (n==nn):
            ax_A.contour((ld['A'][:,nn]/ld['A'][:,nn].max()).reshape(dims).toarray(), levels=[0.3], colors='tab:orange', linewidths=1.5)

            # cc = np.where(assignments[:,s] == nn)[0][0]
            #print(c)
            #print(self.data['corr'][cc,s])
            #ax_Ca.text(6000,i+2.5,'corr: %5.3g, SNR: %5.3g'%(A_corr,self.data['eval']['SNR'][cc,s]))
            ax_Ca.plot(t_arr,C[nn,:]+i+2,'tab:orange',linewidth=0.5)
            i+=1

        ax_A.set_xlim([cm[n,0]-15,cm[n,0]+15])
        ax_A.set_ylim([cm[n,1]-15,cm[n,1]+15])
        ax_A.set_xticks([])
        ax_A.set_yticks([])
        ax_A.axis('off')
        #ax_A.text(cm[n,0],cm[n,1]-14,'$c_{fp}=%4.2g$\n$c_{Ca}=%4.2g$'%(self.data['fp_corr'][c,s],self.data['C_corr'][c,s]),color='w',fontsize=12)
        ax_Ca.spines['right'].set_visible(False)
        ax_Ca.spines['top'].set_visible(False)
        ax_Ca.set_xticks([])
        ax_Ca.set_yticks([])
        #ax_Ca.set_xlabel('time [s]')

        #plt.tight_layout()
        #plt.show(block=False)
        #ext = 'png'
        #path = pathcat([self.pathMouse,'complete_set_evaluation_example_good.%s'%ext])
        #plt.savefig(path,format=ext,dpi=300)

        ax_A = plt.axes([0.05,0.125,0.125,0.2])
        add_number(fig,ax_A,order=4,offset=[-25,25])
        ax_Ca = plt.axes([0.2,0.125,0.275,0.2])

        idx_bad = np.where((self.data['C_corr']<0.5) & (self.data['fp_corr']<0.8) & idx)
        i = np.random.randint(len(idx_bad[0]))
        c = idx_bad[0][i]
        s = idx_bad[1][i]
        s,c,n = (5,2099,1853)
        # ld_dat = pickleData([],self.pathMatching,'load')
        # assignments = ld_dat['assignments']
        n = int(assignments[c,s])
        #plt.draw()
        print(s,c,n)    ## (73,1441,2406)

        pathSession = os.path.join(self.pathMouse,'Session%02d'%(s+1))
        pathData = os.path.join(pathSession,'results_redetect.mat')
        ld = loadmat(pathData,variable_names=['Cin','C','Ain','A'],squeeze_me=True)

        #plt.figure(figsize=(6,2.5))
        #dims=(512,512)
        #ax_A = plt.axes([0.1,0.2,0.35,0.75])
        ax_A.imshow(ld['A'].sum(1).reshape(dims))

        Cin = ld['Cin']/ld['Cin'].max(1)[:,np.newaxis]
        C = ld['C']/ld['C'].max(1)[:,np.newaxis]

        cm = com(ld['A'],dims[0],dims[1])

        # calculate CoMs and distance
        D_ROIs = sp.spatial.distance.cdist(cm,cm)
        n_close = np.where(D_ROIs[n,:]<10)[0]
        A_norm = np.linalg.norm(ld['A'].toarray(),axis=0)

        # plot all closeby ones, highlight overlapping ones
        t_arr = np.linspace(0,8989/15,8989)
        ax_Ca.plot(t_arr,Cin[n,:],'k',linewidth=0.5)
        ax_Ca.plot(t_arr,C[n,:]+1,'tab:green',linewidth=0.5)
        #ax_Ca.text(6000/15,1.5,'corr: %5.3g, SNR: %5.3g'%(self.data['fp_corr'][c,s],self.data['eval']['SNR'][c,s]))

        ax_A.contour((ld['Ain'][:,n]/ld['Ain'][:,n].max()).reshape(dims).toarray(), levels=[0.3], colors='w', linewidths=2)
        ax_A.contour((ld['A'][:,n]/ld['A'][:,n].max()).reshape(dims).toarray(), levels=[0.3], colors='tab:green', linewidths=2)

        i = 0
        for nn in n_close:
          A_corr = ld['A'][:,n].multiply(ld['A'][:,nn]).sum()/(A_norm[n]*A_norm[nn])
          if (A_corr>0.3) and not (n==nn):
            ax_A.contour((ld['A'][:,nn]/ld['A'][:,nn].max()).reshape(dims).toarray(), levels=[0.3], colors='tab:orange', linewidths=1.5)

            # cc = np.where(assignments[:,s] == nn)[0][0]
            #print(c)
            #print(self.data['corr'][cc,s])
            #ax_Ca.text(6000,i+2.5,'corr: %5.3g, SNR: %5.3g'%(A_corr,self.data['eval']['SNR'][cc,s]))
            ax_Ca.plot(t_arr,C[nn,:]+i+2,'tab:orange',linewidth=0.5)
            i+=1

        ax_A.set_xlim([cm[n,0]-15,cm[n,0]+15])
        ax_A.set_ylim([cm[n,1]-15,cm[n,1]+15])
        ax_A.set_xticks([])
        ax_A.set_yticks([])
        ax_A.axis('off')
        #ax_A.text(cm[n,0],cm[n,1]-14,'$c_{fp}=%4.2g$\n$c_{Ca}=%4.2g$'%(self.data['fp_corr'][c,s],self.data['C_corr'][c,s]),color='w',fontsize=12)
        ax_Ca.spines['right'].set_visible(False)
        ax_Ca.spines['top'].set_visible(False)
        ax_Ca.set_yticks([])
        ax_Ca.set_xlabel('time [s]')

        if sv:
            ext = 'png'
            path = pathcat([self.pathMouse,'Figures/complete_set_evaluation1.%s'%ext])
            plt.savefig(path,format=ext,dpi=300)

    if True:
        plt.figure(figsize=(3.5,4),dpi=300)
        dataSet = 'redetect'
        pathLoad = pathcat([self.pathMouse,'clusterStats_%s.pkl'%dataSet])
        ld = pickleData([],pathLoad,'load')
        idxes = (ld['SNR']>SNR_thr) & (ld['r_values']>rval_thr) & (ld['CNN']>CNN_thr) & (ld['firingrate']>0)
        idx_prev = self.data['idxes']['previous']

        colors = [(1,0,0,0),mcolors.to_rgba('tab:orange')]
        RedAlpha = mcolors.LinearSegmentedColormap.from_list('RedAlpha',colors,N=2)
        colors = [(0,0,0,0),(0,0,0,1)]
        BlackAlpha = mcolors.LinearSegmentedColormap.from_list('BlackAlpha',colors,N=2)
        colors = [(0,0,0,0),mcolors.to_rgba('tab:green')]
        GreenAlpha = mcolors.LinearSegmentedColormap.from_list('GreenAlpha',colors,N=2)

        nC,nS = assignments.shape
        ax_oc = plt.axes([0.2,0.1,0.45,0.55])
        ax_oc2 = ax_oc.twinx()
        # ax_oc.imshow((~np.isnan(assignments))&idxes,cmap=BlackAlpha,aspect='auto')
        # ax_oc2.imshow((~np.isnan(assignments))&(~idxes),cmap=RedAlpha,aspect='auto')
        ax_oc.imshow((idx_pm&idxes)[self.cluster.stats['cluster_bool'],:],cmap=BlackAlpha,aspect='auto')
        ax_oc.imshow((idx_pm&idxes&idx_prev)[self.cluster.stats['cluster_bool'],:],cmap=GreenAlpha,aspect='auto')
        ax_oc2.imshow((idx_pm&(~idxes))[self.cluster.stats['cluster_bool'],:],cmap=RedAlpha,aspect='auto')
        #ax_oc.imshow(self.results['p_matched'],cmap='binary',aspect='auto')
        ax_oc.set_xlabel('session')
        ax_oc.set_ylabel('neuron ID')


        ax = plt.axes([0.65,0.1,0.175,0.55])
        nC_clean = self.cluster.stats['cluster_bool'].sum()
        # ax.plot(((~np.isnan(assignments)) & idxes).sum(1),np.linspace(0,nC,nC),'ko',markersize=0.5)
        ax.plot((idx_pm & idxes)[self.cluster.stats['cluster_bool'],:].sum(1),np.linspace(1,nC_clean,nC_clean),'ko',markersize=1,mew=0)
        ax.invert_yaxis()
        ax.set_ylim([nC_clean,0])
        ax.set_yticks([])
        ax.set_xlabel('occ.')
        ax.spines['right'].set_visible(False)

        ax = plt.axes([0.65,0.65,0.175,0.3])
        # ax.hist((~np.isnan(assignments)).sum(1),np.linspace(0,nS,nS),color='r',cumulative=True,density=True,histtype='step')
        # ax.hist((ld_dat['p_matched']>pm_thr).sum(1),np.linspace(0,nS,nS),color='r',cumulative=True,density=True,histtype='step')
        ax.hist((idx_pm&idxes&idx_prev)[self.cluster.stats['cluster_bool'],:].sum(1),np.linspace(0,nS,nS),color='tab:green',cumulative=True,density=True,histtype='step')
        # ax.hist(((~np.isnan(assignments)) & idxes).sum(1),np.linspace(0,nS,nS),color='k',alpha=0.5,cumulative=True,density=True,histtype='step')
        ax.hist((idx_pm & idxes)[self.cluster.stats['cluster_bool'],:].sum(1),np.linspace(0,nS,nS),color='k',cumulative=True,density=True,histtype='step')
        ax.set_xticks([])
        #ax.set_yticks([])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylim([0,1])
        ax.set_yticks(np.linspace(0,1,3))
        ax.set_ylabel('cdf')
        ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)

        ax = plt.axes([0.2,0.65,0.45,0.3])
        # ax.plot(np.linspace(0,nS,nS),(~np.isnan(assignments_OnACID)).sum(0),'ro',markersize=1,markerfacecolor='None')
        # ax.plot(np.linspace(0,nS,nS),(~np.isnan(assignments)).sum(0),'ro',markersize=1)
        # ax.plot(np.linspace(0,nS,nS),((~np.isnan(assignments)) & idxes).sum(0),'ko',markersize=1)
        nROI_prev = (idx_pm&idxes&idx_prev)[self.cluster.stats['cluster_bool'],:].sum(0)
        nROI_total = (idx_pm&idxes)[self.cluster.stats['cluster_bool'],:].sum(0)
        frac_ROI = nROI_total/nROI_prev-1
        print(np.nanmean(frac_ROI[frac_ROI<1.5]))
        print(np.nanstd(frac_ROI[frac_ROI<1.5]))
        # print()

        ax.plot(np.linspace(0,nS,nS),(idx_pm&idxes&idx_prev)[self.cluster.stats['cluster_bool'],:].sum(0),'^',color='tab:green',markersize=2,label='initial')
        # ax.plot(np.linspace(0,nS,nS),idx_pm[self.cluster.stats['cluster_bool'],:].sum(0),'^',color='tab:orange',markersize=2,label='inferred')
        ax.plot(np.linspace(0,nS,nS),(idx_pm & idxes)[self.cluster.stats['cluster_bool'],:].sum(0),'k^',markersize=2,label='total')
        ax.set_xlim([0,nS])
        ax.set_ylim([0,1300])
        ax.set_xticks([])
        ax.set_ylabel('# neurons')
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=8,bbox_to_anchor=[0.0,0.0],loc='lower left')



        plt.tight_layout()
        plt.show(block=False)
        if sv:
            ext = 'png'
            path = pathcat([self.pathMouse,'Figures/complete_set_evaluation.%s'%ext])
            plt.savefig(path,format=ext,dpi=300)


  def plot_threshold(self):
      print('### plot test of thresholds for neuron number ###')

      nSes = self.cluster.meta['nSes']
      f_max = 5
      # SNR_thr = 1
      rval_thr = 0
      CNN_thr = 0.6

      A0_thr = 1
      A_thr = 3
      pmass_thr = 0.5
      Bayes_thr = 10
      rel_thr = 0.1

      # if PC is None:
      #
      #     active = {'SNR':        {},
      #         'r_values':     {},
      #         'CNN':          {}}
      #
      #     PC = {'Bayes_factor':   {},
      #         'reliability':  {},
      #         'A_0':          {},
      #         'A':            {},
      #         'p_mass':       {}}
      #
      #     for s in tqdm(range(nSes)):
      #         PC_para = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_para.mat'%(s+1),variable_names=['Bayes_factor','posterior_mass','parameter'],squeeze_me=True)
      #         firingstats = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/PC_fields_firingstats.mat'%(s+1),variable_names=['trial_map','map'],squeeze_me=True)
      #         CNMF_results = loadmat('/media/wollex/Analyze_AS1/linstop/762/Session%02d/results_redetect.mat'%(s+1),variable_names=['SNR','r_values','CNN'],squeeze_me=True)
      #
      #         active['SNR'][s] = CNMF_results['SNR']
      #         active['r_values'][s] = CNMF_results['r_values']
      #         active['CNN'][s] = CNMF_results['CNN']
      #         N = len(active['SNR'][s])
      #
      #         PC['Bayes_factor'][s] = PC_para['Bayes_factor'][...,0]
      #         PC['A_0'][s] = PC_para['parameter'][:,:,0,0]
      #         PC['A'][s] = PC_para['parameter'][:,:,1,0]
      #         PC['p_mass'][s] = PC_para['posterior_mass']
      #         PC['reliability'][s] = np.zeros((N,f_max))
      #         for n in range(N):
      #             for f in np.where(~np.isnan(PC['A'][s][n,:]))[0]:
      #                 PC['reliability'][s][n,f],_,_ = get_reliability(firingstats['trial_map'][n,...],firingstats['map'][n,...],PC_para['parameter'][n,...],f)


      # return active, PC
      ### get SNR dependence of detected neurons
      # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.05) | (cluster.stats['match_score'][...,0]>0.95)
      # idx_other = (cluster.stats['r_values'] > 0) & idx_pm
      # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.5) | (cluster.stats['match_score'][...,0]>0.95)
      # idx_other_certain = (cluster.stats['r_values'] > 0) & idx_pm
      r_thr = 0
      SNR_arr = np.linspace(1,10,10)
      nROI = np.zeros(SNR_arr.shape + (nSes,2))
      nPC = np.zeros(SNR_arr.shape + (nSes,2))
      plt.figure(figsize=(7,2.5))
      width = 0.4
      ax = plt.axes([0.1,0.2,0.25,0.65])
      for i,SNR_thr in enumerate(SNR_arr):

          for s in range(nSes):
              idx_active = (self.eval_data[s]['SNR'] > SNR_thr) & (self.eval_data[s]['r_values'] > r_thr) & (self.eval_data[s]['CNN'] > CNN_thr)
              nROI[i,s,0] = (idx_active).sum()

              morphed_A0_thr = A0_thr-PC['reliability'][s]/2
              idx_fields = (PC['A_0'][s]>morphed_A0_thr) & \
                          (PC['A'][s]>A_thr) & \
                          (PC['p_mass'][s]>pmass_thr) & \
                          (PC['Bayes_factor'][s]>Bayes_thr) & \
                          (PC['reliability'][s]>rel_thr)

              nPC[i,s,0] = (idx_active & np.any(idx_fields,1)).sum()
              # print('active neurons / place cells in session %d: %d / %d'%(s+1,nAct,nPC))

          # nROI[i,:,0] = ((cluster.stats['SNR']>SNR_thr) & idx_other).sum(0)
          # nROI[i,:,1] = ((cluster.stats['SNR']>SNR_thr) & idx_other_certain).sum(0)

          # nPC[i,:,0] = ((cluster.stats['SNR']>SNR_thr) & idx_other & cluster.status[...,2]).sum(0)
          # nPC[i,:,1] = ((cluster.stats['SNR']>SNR_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
          ax.scatter(SNR_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nROI[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
          ax.scatter(SNR_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nPC[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

      ax.plot(SNR_arr,nROI[:,self.cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')#,label='neurons ($p_m\geq0.05$)')
      # ax.plot(SNR_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')# ($p_m\geq0.95$)')
      ax.plot(SNR_arr,nPC[:,self.cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
      # ax.plot(SNR_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')

      ax.set_ylim([0,2700])
      ax.set_xlabel('$\Theta_{SNR}$')
      ax.set_ylabel('# neurons')
      ax.spines['top'].set_visible(False)

      ax2 = ax.twinx()
      # ax2.plot(SNR_arr,nPC[...,0].mean(1)/nROI[...,0].mean(1),'k:')
      ax2.plot(SNR_arr,(nPC[:,self.cluster.sessions['bool'],0]/nROI[:,self.cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
      ax2.plot([3,3],[0,1],'k--')
      ax2.set_ylim([0,0.54])
      # ax2.set_ylabel('PC fraction')
      ax2.set_yticklabels([])
      ax2.spines['top'].set_visible(False)
      ax.legend(fontsize=10,bbox_to_anchor=[0.3,1.15],loc='upper left')
      # plt.tight_layout()
      # plt.show(block=False)

      # if sv:
          # pl_dat.save_fig('neuronNumbers_SNR')


      ### get r_val dependence of detected neurons
      # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.05) | (cluster.stats['match_score'][...,0]>0.95)
      # idx_other = (cluster.stats['SNR'] > 2) & idx_pm
      # idx_pm = ((cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>0.5) | (cluster.stats['match_score'][...,0]>0.95)
      # idx_other_certain = (cluster.stats['SNR'] > 2) & idx_pm

      # idx_other = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.5)
      # idx_other_certain = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.95)
      SNR_thr = 3
      r_arr = np.linspace(-0.4,1,8)
      nROI = np.zeros(r_arr.shape + (nSes,2))
      nPC = np.zeros(r_arr.shape + (nSes,2))
      # plt.figure(figsize=(4,2.5))
      width = 0.08
      ax = plt.axes([0.375,0.2,0.25,0.65])
      for i,r_thr in enumerate(r_arr):
          for s in range(nSes):
              idx_active = (self.eval_data[s]['SNR'] > SNR_thr) & (self.eval_data[s]['r_values'] > r_thr) & (self.eval_data[s]['CNN'] > CNN_thr)
              nROI[i,s,0] = (idx_active).sum()

              morphed_A0_thr = A0_thr-PC['reliability'][s]/2
              idx_fields = (PC['A_0'][s]>morphed_A0_thr) & \
                          (PC['A'][s]>A_thr) & \
                          (PC['p_mass'][s]>pmass_thr) & \
                          (PC['Bayes_factor'][s]>Bayes_thr) & \
                          (PC['reliability'][s]>rel_thr)

              nPC[i,s,0] = (idx_active & np.any(idx_fields,1)).sum()

          # nROI[i,:,0] = ((cluster.stats['r_values']>r_thr) & idx_other).sum(0)
          # nROI[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain).sum(0)

          # nPC[i,:,0] = ((cluster.stats['r_values']>r_thr) & idx_other & cluster.status[...,2]).sum(0)
          # nPC[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
          ax.scatter(r_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nROI[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
          ax.scatter(r_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nPC[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

      ax.plot(r_arr,nROI[:,self.cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')
      # ax.plot(r_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5)
      ax.plot(r_arr,nPC[:,self.cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
      # ax.plot(r_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5)

      ax.set_ylim([0,2700])
      ax.set_xlabel('$\Theta_{r}$')
      # ax.set_ylabel('# neurons')
      ax.set_yticklabels([])
      # ax.legend(fontsize=10)
      ax.spines['top'].set_visible(False)

      ax2 = ax.twinx()
      # ax2.plot(r_arr,nPC[...,0].mean(1)/nROI[...,0].mean(1),'k:')
      ax2.plot(r_arr,(nPC[:,self.cluster.sessions['bool'],0]/nROI[:,self.cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
      ax2.plot([0,0],[0,1],'k--')
      ax2.set_ylim([0,0.54])
      ax2.set_yticklabels([])
      # ax2.set_ylabel('PC fraction')
      ax2.spines['top'].set_visible(False)
      plt.tight_layout()
      plt.show(block=False)
      # return
      ### get pm dependence of detected neurons
      idx_other = (self.cluster.stats['SNR'] > 3) & (cluster.stats['r_values']>0)
      # idx_other_certain = (cluster.stats['SNR'] > 2) & (cluster.stats['match_score'][...,0]>0.95)
      pm_arr = np.linspace(0,1,11)
      nROI = np.zeros(pm_arr.shape + (nSes,2))
      nPC = np.zeros(pm_arr.shape + (nSes,2))
      # plt.figure(figsize=(4,2.5))
      width = 0.04
      ax = plt.axes([0.65,0.2,0.25,0.65])
      for i,pm_thr in enumerate(pm_arr):
          # idx_pm = cluster.stats['match_score'][...,0]>pm_thr
          idx_pm = ((self.cluster.stats['match_score'][...,0]-self.cluster.stats['match_score'][...,1])>pm_thr) | (self.cluster.stats['match_score'][...,0]>0.95)
          # idx_pm = (cluster.stats['match_score'][...,0]-cluster.stats['match_score'][...,1])>pm_thr
          nROI[i,:,0] = (idx_pm & idx_other).sum(0)
          # nROI[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain).sum(0)

          nPC[i,:,0] = (idx_pm & idx_other & self.cluster.status[...,2]).sum(0)
          # nPC[i,:,1] = ((cluster.stats['r_values']>r_thr) & idx_other_certain & cluster.status[...,2]).sum(0)
          ax.scatter(pm_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nROI[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,0.8]])
          ax.scatter(pm_thr-width/2 + width*np.random.rand(nSes)[self.cluster.sessions['bool']],nPC[i,self.cluster.sessions['bool'],0],s=2,c=[[0.8,0.8,1]])

      ax.plot(pm_arr,nROI[:,self.cluster.sessions['bool'],0].mean(1),'k^',markersize=4,markeredgewidth=0.5,label='neurons')
      # ax.plot(pm_arr,nROI[:,cluster.sessions['bool'],1].mean(1),'k^',markersize=4,markeredgewidth=0.5)
      ax.plot(pm_arr,nPC[:,self.cluster.sessions['bool'],0].mean(1),'^',color='tab:blue',markersize=4,markeredgewidth=0.5,label='PCs')
      # ax.plot(pm_arr,nPC[:,cluster.sessions['bool'],1].mean(1),'r^',markersize=4,markeredgewidth=0.5)

      ax.set_ylim([0,2700])
      ax.set_xlabel('$\Theta_{p_m}$')
      # ax.set_ylabel('# neurons')
      ax.set_yticklabels([])
      # ax.legend(fontsize=10)
      ax.spines['top'].set_visible(False)

      ax2 = ax.twinx()
      ax2.plot(pm_arr,(nPC[:,self.cluster.sessions['bool'],0]/nROI[:,self.cluster.sessions['bool'],0]).mean(1),'-',color='tab:blue')
      # ax2.plot(pm_arr,nPC[...,1].mean(1)/nROI[...,1].mean(1),'k-')
      ax2.plot([0.5,0.5],[0,1],'k--')
      ax2.set_ylim([0,0.54])
      ax2.set_ylabel('PC fraction')
      ax2.spines['top'].set_visible(False)
      plt.tight_layout()
      plt.show(block=False)

      if sv:
          pl_dat.save_fig('neuronNumbers_test')


      print('whats with MI of place cells, only?')
      ### get SNR dependence of detected neurons
      idx_other = (self.cluster.stats['r_values'] > 0) & (self.cluster.stats['match_score'][...,0]>0.05)
      idx_other_certain = (self.cluster.stats['r_values'] > 0) & (self.cluster.stats['match_score'][...,0]>0.95)
      SNR_arr = np.linspace(2,10,5)
      MI = np.zeros(SNR_arr.shape + (2,))
      # nPC = np.zeros(SNR_arr.shape + (nSes,2))
      plt.figure(figsize=(4,2.5))
      width = 0.6
      ax = plt.axes([0.2,0.2,0.65,0.65])
      for i,SNR_thr in enumerate(SNR_arr):
          idx = (self.cluster.stats['SNR'] >= (SNR_thr-0.5)) & (self.cluster.stats['SNR'] < (SNR_thr+0.5)) & idx_other
          MI[i,0] = self.cluster.stats['MI_value'][idx].mean(0)
          idx = (self.cluster.stats['SNR'] >= (SNR_thr-0.5)) & (self.cluster.stats['SNR'] < (SNR_thr+0.5)) & idx_other_certain
          MI[i,1] = self.cluster.stats['MI_value'][idx].mean(0)

          ax.boxplot(self.cluster.stats['MI_value'][idx],positions=[SNR_thr],widths=width,whis=[5,95],notch=True,bootstrap=100,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))

      ax.plot(SNR_arr,MI[...,1],'k^',markersize=5,label='neurons')

      ax.set_ylim([0,0.6])
      ax.set_xlabel('$\Theta_{SNR}$')
      ax.set_ylabel('MI')
      ax.legend(fontsize=10)

      plt.tight_layout()
      plt.show(block=False)

      # if sv:
          # pl_dat.save_fig('MI_SNR')


  def plot_detection(self,ax,A1,A2,bounds=None,lw=0.7,ls='solid'):
    dims = (512,512)

    if bounds is None:
      bounds = [[0,dims[0]],[0,dims[1]]]

    A1_norm = normalize_sparse_array(A1)
    A2_norm = normalize_sparse_array(A2)

    ax.imshow(A1_norm.sum(1).reshape(dims)+A2_norm.sum(1).reshape(dims),origin='lower')

    CM = com(A1,dims[0],dims[1])
    idx_bounds = (CM[:,0] > bounds[0][0]) & (CM[:,0] < bounds[0][1]) & (CM[:,1] > bounds[1][0]) & (CM[:,1] < bounds[1][1])
    [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.5], colors='w', linewidths=0.5) for a in A1[:,idx_bounds].T]

    CM = com(A2,dims[0],dims[1])
    idx_bounds = (CM[:,0] > bounds[0][0]) & (CM[:,0] < bounds[0][1]) & (CM[:,1] > bounds[1][0]) & (CM[:,1] < bounds[1][1])
    [ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.5], colors='tab:orange', linewidths=lw,linestyles=ls) for a in A2[:,idx_bounds].T]


    #plt.figure(figsize=(5,4))
    #ax = plt.subplot(111)
    #ax.imshow(self.dataOut['A'].sum(1).reshape(self.cluster.meta['dims']),origin='lower')

    #[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='w', linewidths=0.5) for a in self.dataOut['A'][:,self.idxes['active_good']].T]
    #[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=0.5) for a in self.dataOut['A'][:,self.idxes['silent_good']].T]

    ##[ax.contour((a/a.max()).reshape(512,512).toarray(), levels=[0.3], colors='r', linewidths=1) for a in self.dataIn['A'][:,self.idxes['undetected']].T]
    #plt.tight_layout()
    #plt.show(block=False)
    #pathSession = pathcat([self.cluster.meta['pathMouse'],'Session%02d'%(self.s+1)])
    #pathFigure = pathcat([pathSession,'find_silent_after.%s'%(ext)]);
    #plt.savefig(pathFigure,format=ext,dpi=300)
    #print('Figure saved as %s'%pathFigure)


def compute_event_exceptionality(traces, robust_std=False, N=5, sigma_factor=3.):
    """
    Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Args:
        traces: ndarray
            Fluorescence traces

        N: int
            N number of consecutive events

        sigma_factor: float
            multiplicative factor for noise estimate (added for backwards compatibility)

    Returns:
        fitness: ndarray
            value estimate of the quality of components (the lesser the better)

        erfc: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise

        noise_est: ndarray
            the components ordered according to the fitness
    """

    T = np.shape(traces)[-1]

    md = find_modes(traces,axis=1)
    ff1 = traces - md[:,None]

    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:

        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0,1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, _ in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, -1)
        sd_r = np.sqrt(old_div(np.sum(ff1**2, -1), Ns))

    # compute z value
    z = old_div((traces - md[:,None]), (sigma_factor * sd_r[:,None]))

    # probability of observing values larger or equal to z given normal
    # distribution with mean md and std sd_r
    #erf = 1 - norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    #erf = np.log(erf)
    # compute with this numerically stable function
    erf = scipy.special.log_ndtr(-z)

    # moving sum
    erfc = np.cumsum(erf, 1)
    erfc[:, N:] -= erfc[:, :-N]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    return fitness, erfc, sd_r, md

def get_spikeNr(traces):
    md = find_modes(traces)
    ff1 = traces - md

    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)

    # compute 25 percentile
    ff1 = np.sort(ff1)
    ff1[ff1 == 0] = np.nan
    Ns = round((ff1>0).sum() * .5).astype('int')
    iqr_h = np.zeros(traces.shape[0])

    #for idx, _ in enumerate(ff1):
    iqr_h = ff1[-Ns]

    # approximate standard deviation as iqr/1.349
    sd_r = 2 * iqr_h / 1.349
    data_thr = md+2*sd_r;
    spikeNr = np.floor(traces/data_thr).sum();
    return spikeNr,md,sd_r
