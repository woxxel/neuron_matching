''' contains various useful program snippets:

  pickleData
        - wrapper for dumping or loading data to/from a pickle file
  com 
        - calculating the center of mass from a 2d matrix
  calculate_img_correlation
        - calculate correlation between two 2d matrices (e.g. for neuron footprints)
  get_shift_and_flow
        - calculate rigid shift and optical flow between two images 
  fun_wrapper
        - wrapper to pass array values as separate arguments to a function
        [should be possible via spread-operator - test that!]
'''

import pickle, cv2, os, tqdm
import scipy as sp
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import Any, Dict, List, Tuple, Union, Iterable



def pickleData(dat,path,mode='load',prnt=True):

  if mode=='save':
    f = open(path,'wb')
    pickle.dump(dat,f)
    f.close()
    if prnt:
        print('Data saved in %s'%path)
  else:
    f = open(path,'rb')
    dat = pickle.load(f)
    f.close()
    if prnt:
        print('Data loaded from %s'%path)
    return dat


def replace_relative_path(paths,newPath):
            prepath = os.path.commonpath(paths)
            return [os.path.join(newPath,os.path.relpath(path,prepath)) for path in paths]
        

def load_field_from_hdf5(filePath,field):

    '''
        currently only works for A
    '''

    file = h5py.File(filePath,'r')
    # only load a single variable: A

    path = '/'
    data =  file[os.path.join(path,field,'data')][...]
    indices = file[os.path.join(path,field,'indices')][...]
    indptr = file[os.path.join(path,field,'indptr')][...]
    shape = file[os.path.join(path,field,'shape')][...]
    
    val = sp.sparse.csc_matrix((data[:], indices[:],
        indptr[:]), shape[:])
    return val


def center_of_mass(A, d1, d2, d3=None,d1_offset=0,d2_offset=0,d3_offset=0,convert=1.):
  '''
    calculate center of mass of footprints A in 2D or 3D

    TODO:
      - implement offset in each dimension
  '''
  
  if 'csc_matrix' not in str(type(A)):
      A = sp.sparse.csc_matrix(A)

  if d3 is None:
      Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                        np.outer(np.arange(d2), np.ones(d1)).ravel()], dtype=A.dtype)
  else:
      Coor = np.matrix([
          np.outer(np.ones(d3), np.outer(np.ones(d2), np.arange(d1)).ravel()).ravel(),
          np.outer(np.ones(d3), np.outer(np.arange(d2), np.ones(d1)).ravel()).ravel(),
          np.outer(np.arange(d3), np.outer(np.ones(d2), np.ones(d1)).ravel()).ravel()],
          dtype=A.dtype)

  Anorm = normalize_sparse_array(A,relative_threshold=0.001,minimum_nonzero_entries=0)

  cm = (Coor * Anorm).T
  cm[np.squeeze(np.array((Anorm>0).sum(0)))==0,:] = np.NaN

  return np.array(cm)*convert


def calculate_img_correlation(A1,A2,dims=(512,512),crop=False,cm_crop=None,binary=False,shift=True,plot_bool=False):

  if shift:

    ## try with binary and continuous
    if binary == 'half':
      A1 = (A1>np.median(A1.data)).multiply(A1)
      A2 = (A2>np.median(A2.data)).multiply(A2)
    elif binary:
      A1 = A1>np.median(A1.data)
      A2 = A2>np.median(A2.data)

    #t_start = time.time()
    if not np.all(A1.shape == dims):
      A1 = A1.reshape(dims)
    if not np.all(A2.shape == dims):
      A2 = A2.reshape(dims)
    #t_end = time.time()
    #print('reshaping --- time taken: %5.3g'%(t_end-t_start))

    if crop:
      #t_start = time.time()
      row,col,tmp = sp.sparse.find(A1)
      A1 = A1.toarray()[row.min():row.max()+1,col.min():col.max()+1]
      row,col,tmp = sp.sparse.find(A2)
      A2 = A2.toarray()[row.min():row.max()+1,col.min():col.max()+1]
      #t_end = time.time()
      #print('cropping 1 --- time taken: %5.3g'%(t_end-t_start))

      #t_start = time.time()
      padding = np.subtract(A2.shape,A1.shape)
      if padding[0] > 0:
        A1 = np.pad(A1,[[padding[0],0],[0,0]],mode='constant',constant_values=0)
      else:
        A2 = np.pad(A2,[[-padding[0],0],[0,0]],mode='constant',constant_values=0)

      if padding[1] > 0:
        A1 = np.pad(A1,[[0,0],[padding[1],0]],mode='constant',constant_values=0)
      else:
        A2 = np.pad(A2,[[0,0],[-padding[1],0]],mode='constant',constant_values=0)
      #t_end = time.time()
      #print('cropping 2 --- time taken: %5.3g'%(t_end-t_start))
    else:
      if not (type(A1) is np.ndarray):
        A1 = np.array(A1)
        A2 = np.array(A2)

    dims = A1.shape

    #t_start = time.time()
    C = signal.convolve(A1-A1.mean(),A2[::-1,::-1]-A2.mean(),mode='same')/(np.prod(dims)*A1.std()*A2.std())
    #t_end = time.time()
    #print('corr-computation --- time taken: %5.3g'%(t_end-t_start))
    C_max = C.max()
    if np.isnan(C_max) | (C_max == 0):
      return np.NaN, np.ones(2)*np.NaN

    #if not crop:
    crop_half = ((dims[0]-np.mod(dims[0],2))/2,(dims[1]-np.mod(dims[1],2))/2)#tuple(int(d/2-1) for d in dims)
    idx_max = np.unravel_index(np.argmax(C),C.shape)
    img_shift = np.subtract(idx_max,crop_half)

    # plot_bool = True
    if (plot_bool):# | ((C_max>0.95)&(C_max<0.9999)):
      #idx_max = np.where(C.real==C_max)
      plt.figure()
      ax1 = plt.subplot(221)
      im = ax1.imshow(A1,origin='lower')
      plt.colorbar(im)
      plt.subplot(222,sharex=ax1,sharey=ax1)
      plt.imshow(A2,origin='lower')
      plt.colorbar()
      plt.subplot(223)
      plt.imshow(C,origin='lower')
      plt.plot(crop_half[1],crop_half[0],'ro')
      plt.colorbar()
      plt.suptitle('corr: %5.3g'%C_max)
      plt.show(block=False)

    return C_max, img_shift # C[crop_half],
  else:
    #if not (type(A1) is np.ndarray):
      #A1 = A1.toarray()
    #if not (type(A2) is np.ndarray):
      #A2 = A2.toarray()

    if not (cm_crop is None):

      cr = 20
      extent = np.array([cm_crop-cr,cm_crop+cr+1]).astype('int')
      extent = np.maximum(extent,0)
      extent = np.minimum(extent,dims)
      A1 = A1.reshape(dims)[extent[0,0]:extent[1,0],extent[0,1]:extent[1,1]]
      A2 = A2.reshape(dims)[extent[0,0]:extent[1,0],extent[0,1]:extent[1,1]]
    #else:
      #extent = [[0,0],[dims[0],dims[1]]]
    if plot_bool:
      #idx_max = np.where(C.real==C_max)
      plt.figure()
      plt.subplot(221)
      plt.imshow(A1.reshape(extent[1,0]-extent[0,0],extent[1,1]-extent[0,1]),origin='lower')
      plt.colorbar()
      plt.subplot(222)
      plt.imshow(A2.reshape(extent[1,0]-extent[0,0],extent[1,1]-extent[0,1]),origin='lower')
      plt.colorbar()
      #plt.subplot(223)
      #plt.imshow(C,origin='lower')
      #plt.plot(crop_half[1],crop_half[0],'ro')
      #plt.colorbar()
      plt.suptitle('corr: %5.3g'%np.corrcoef(A1.flat,A2.flat)[0,1])
      plt.show(block=False)
    return A1.multiply(A2).sum()/np.sqrt(A1.power(2).sum()*A2.power(2).sum()), None
    #return (A1*A2).sum()/np.sqrt((A1**2).sum()*(A2**2).sum()), None
    #return np.corrcoef(A1.flat,A2.flat)[0,1], None


def get_shift_and_flow(A1,A2,dims=(512,512),projection=-1,transpose_it=False,plot_bool=False):

  ## dims:          shape of the (projected) image
  ## projection:    axis, along which to project. If None, no projection needed

  if not (projection is None):
    A1 = np.array(A1.sum(projection))
    A2 = np.array(A2.sum(projection))
  A1 = A1.reshape(dims)
  A2 = A2.reshape(dims)

  if transpose_it:
      A2 = A2.T

  A1 = normalize_array(A1,'uint',8)
  A2 = normalize_array(A2,'uint',8)

  c,(y_shift,x_shift) = calculate_img_correlation(A1,A2,plot_bool=plot_bool)

  x_remap,y_remap = build_remap_from_shift_and_flow(dims,(x_shift,y_shift))

  A2 = cv2.remap(A2, x_remap, y_remap, interpolation=cv2.INTER_CUBIC)
  A2 = normalize_array(A2,'uint',8)

  flow = cv2.calcOpticalFlowFarneback(A1,A2,None,0.5,5,128,3,7,1.5,0)

  return (x_shift,y_shift), flow, c


def build_remap_from_shift_and_flow(dims,shifts,flow=None):

    '''
        returns remap arrays in 2D

        dims (2,) with (dim_x,dim_y)
        shifts (2,) with (x,y)
        flow (dim1,dim2,d) with d=dimension
    '''
    
    x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32), np.arange(0., dims[1]).astype(np.float32))
    
    if not (flow is None):
        x_remap = (x_grid - shifts[0] + flow[0,:,:]).astype(np.float32)
        y_remap = (y_grid - shifts[1] + flow[1,:,:]).astype(np.float32)
    else:
        x_remap = (x_grid - shifts[0]).astype(np.float32)
        y_remap = (y_grid - shifts[1]).astype(np.float32)
       
    return x_remap, y_remap


def plot_flow(flow,dims,idxes=15):
    x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32), np.arange(0., dims[1]).astype(np.float32))

    # print('shift:',[x_shift,y_shift])
    plt.figure()
    plt.quiver(x_grid[::idxes,::idxes], y_grid[::idxes,::idxes], flow[::idxes,::idxes,0], flow[::idxes,::idxes,1], angles='xy', scale_units='xy', scale=1, headwidth=4,headlength=4, width=0.002, units='width')
    plt.show(block=False)


def normalize_array(A,a_type='uint',a_bits=8,axis=None):
  A -= A.min()
  A = A/A.max()

  return (A*(A>A.mean(axis))*(2**a_bits-1)).astype('%s%d'%(a_type,a_bits))


def normalize_sparse_array(A,relative_threshold=0.001,minimum_nonzero_entries=50):
  '''
    normalizing sparse arrays with some thresholding
      - relative_threshold: float
          fraction of peak value, at below which entries are considered to be 0
      - minimum_nonzero_entries: int
          minimum number of nonzero entries of the footprint to be considered for further analyses
  '''
  return sp.sparse.vstack([
      # a.multiply(a>relative_threshold*a.max())/a.max()  # threshold footprint
      a.multiply(a>relative_threshold*a.max())/a[a>0.001*a.max()].sum()  # threshold footprint
      if (a>0).sum() > minimum_nonzero_entries          # require minimum non-zero entries ...
      else sp.sparse.csr_matrix(a.shape)                # ... otherwise return empty slice
      for a in A.T                                      # loop through all footprints
    ]).T


def nangauss_filter(X,sigma=None,mode='nearest',truncate=2):
  if (sigma is None) or not np.any(np.array(sigma)>0):
    return X
  else:
    V = X.copy()
    V[np.isnan(X)] = 0
    VV = sp.ndimage.gaussian_filter(V,sigma,truncate=truncate,mode=mode)

    W = 0*X.copy()+1
    W[np.isnan(X)] = 0
    WW = sp.ndimage.gaussian_filter(W,sigma,truncate=truncate,mode=mode)

  return VV/WW



def nanmedian_filter(X,footprint=None,mode='nearest'):
   
   return sp.ndimage.generic_filter(X,np.nanmedian,footprint=footprint,mode=mode)


def fun_wrapper(fun,x,p):
    '''
      
    '''
    if np.isscalar(p):
      return fun(x,p)
    if p.shape[-1] == 2:
      return fun(x,p[...,0],p[...,1])
    if p.shape[-1] == 3:
      return fun(x,p[...,0],p[...,1],p[...,2])
    if p.shape[-1] == 4:
      return fun(x,p[...,0],p[...,1],p[...,2],p[...,3])
    if p.shape[-1] == 5:
      return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4])
    if p.shape[-1] == 6:
      return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4],p[...,5])
    if p.shape[-1] == 7:
      return fun(x,p[...,0],p[...,1],p[...,2],p[...,3],p[...,4],p[...,5],p[...,6])


def replace_relative_path(paths,newPath):
    prepath = os.path.commonpath(paths)
    return [os.path.join(newPath,os.path.relpath(path,prepath)) for path in paths]

