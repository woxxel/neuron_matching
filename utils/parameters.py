# import os

matching_params = {

    'dims':         (512,512),  # dimensionality of the imaging window
    'pxtomu':       530.68/512, #200./512, # real distance between pixels in \mu m

    'nbins':        128,        # number of bins for the count histogram (should be n^2)
    'neighbor_distance': 15,    # \mu m distance up to which neurons are considered to be neighbors
    'model':        'shifted',  # ['shifted','unshifted','both'] - specifies whether to use maximum possible footprint correlation by shifting
    'binary':       False,

    'use_kde':      True,       # whether to use (True) or not (False) kernel density estimation to remove statistics from low/high density regions
    'qtl':          [0.05,0.95],# specifies range of densities from kde to include in statistics 

    ## evaluation thresholds
    # 'SNR_thr':      2.,         # signal-to-noise ratio
    # 'r_thr':        0.5,        # r_value
    # 'CNN_thr':      0.6,        # cnn-classifier value

    'min_SNR': 2.5,
    'SNR_lowest': 1.0,
    'rval_thr': 0.8,
    'rval_lowest': -1,
    'min_cnn_thr': 0.9,
    'cnn_lowest': 0.1,

    'min_session_correlation':   0.0,        # minimum value of correlation between session footprints to include data in matching
    'min_session_correlation_zscore':   3.,        # minimum value of correlation between session footprints to include data in matching

    'max_session_shift': 100,        # maximum shift between sessions to consider for matching
}

# class matchingParams:
#     def __init__(self,mousePath,paths,suffix=''):

        

        # return params