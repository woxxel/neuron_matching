import os

mousePath = 'data/555wt'; 
paths = [os.path.join(mousePath,sessionPath,'OnACID_results.hdf5') for sessionPath in os.listdir(mousePath) if 'Session' in sessionPath]
paths.sort()

matchingParams = {
    'paths':        paths,      # list of paths to CaImAn result files to be processed in order
    'pathData':     mousePath,   # path to which results are stored and loaded from

    'dims':         (512,512),  # dimensionality of the imaging window
    'pxtomu':       530.68/512, #200./512, # real distance between pixels in \mu m

    'nbins':        128,        # number of bins for the count histogram (should be n^2)
    'neighbor_distance': 12,    # \mu m distance up to which neurons are considered to be neighbors
    'model':        'shifted',  # ['shifted','unshifted','both'] - specifies whether to use maximum possible footprint correlation by shifting
    
    'use_kde':      True,       # whether to use (True) or not (False) kernel density estimation to remove statistics from low/high density regions
    'qtl':          [0.05,0.95],# specifies range of densities from kde to include in statistics 

    ## evaluation thresholds
    'SNR_thr':      2.,         # signal-to-noise ratio
    'r_thr':        0.5,        # r_value
    'CNN_thr':      0.6,        # cnn-classifier value


    'min_session_correlation':   0.1,        # minimum value of correlation between session footprints to include data in matching
}