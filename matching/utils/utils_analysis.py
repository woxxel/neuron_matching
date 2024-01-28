import tqdm
import numpy as np
import scipy as sp

from .utils import calculate_img_correlation, center_of_mass

def calculate_statistics(A,A_ref=None,idx_eval=None,idx_eval_ref=None,
            SNR_comp=None,C=None,
            binary='half',neighbor_distance=12,model='shifted',dims=(512,512)):
    '''
        function to calculate footprint correlations used for the matching procedure for 
          - nearest neighbour (NN) - closest center of mass to reference footprint
          - non-nearest neighbour (nNN) - other neurons with center of mass distance below threshold
        
        footprint correlations are calculated as shifted and non-shifted (depending on model used)

        this method is used both for comparing statistics 
          - referencing previous session (used for matching)
          - referencing itself (used for removal of duplicated footprints and kernel calculation)

    '''

    if A_ref is None:
        A_ref = A
        same = True
    else:
        same = False
    idx_remove = []     # only gets populated, when 'same' is True
    
    if idx_eval is None:
        idx_eval = np.array(A.sum(0)>0).ravel()
    
    if idx_eval_ref is None:
        idx_eval_ref = idx_eval if same else np.array(A_ref.sum(0)>0).ravel()
    
    cm = center_of_mass(A,*dims)
    cm_ref = center_of_mass(A_ref,*dims)
    # calculate distance between footprints and identify NN
    com_distance = sp.spatial.distance.cdist(cm_ref,cm)

    nA_ref = A_ref.shape[1]
    nA = A.shape[1]


    ## prepare arrays to hold statistics
    footprint_correlation = np.full((2,nA_ref,nA),np.NaN)
    
    c_rm = 0
    for i in tqdm.tqdm(range(nA_ref),desc='calculating footprint correlation of %d neurons'%idx_eval_ref.sum(),leave=False):
        if idx_eval_ref[i]:
            for j in np.where(com_distance[i,:]<neighbor_distance)[0]:
                if i in idx_remove or j in idx_remove: break   ## jump to next one
                
                if idx_eval[j]:
                    ## calculate pairwise correlation between reference and current set of neuron footprints
                    # if (model=='both') | (model=='unshifted'):
                    footprint_correlation[0,i,j],_ = calculate_img_correlation(A[:,j],A_ref[:,i],shift=False)

                    if (model=='both') | (model=='shifted'):
                        footprint_correlation[1,i,j],_ = calculate_img_correlation(A[:,j],A_ref[:,i],crop=True,shift=True,binary=binary)

                        # tag footprints for removal when calculating statistics for self-matching and they pass some criteria:
                        # 1) significant overlap with closeby neuron ("contestant")
                    if (same) & (i!=j) & (footprint_correlation[0,i,j] > 0.3):

                        if not (C is None) and not (SNR_comp is None):
                            
                            # 2) high correlation of neuronal activity ("C") (or very high footprint correlation!)
                            C_corr = np.corrcoef(C[i,:],C[j,:])[0,1]
                            if C_corr > 0.3 or footprint_correlation[0,i,j]>0.7:
                                
                                # 3) lower SNR than contestant
                                idx_remove.append(j if SNR_comp[i]>SNR_comp[j] else i)

                                c_rm += 1

                                # print('removing neuron %d (%d vs %d) from data (Acorr: %.3f, Ccorr: %.3f; SNR: %.2f vs %.2f)'%(idx_remove[-1],i,j,footprint_correlation[1,i,j],C_corr,SNR_comp[i],SNR_comp[j]))
                                # footprint_correlation[1,i,j] = np.NaN
                        

    # if c_rm:
    #     print('%d neurons removed'%c_rm)

    return com_distance, footprint_correlation, idx_remove



def calculate_p(d_ROIs,fp_corr,p_model,neighbor_distance=12):
        
    '''
        evaluates the probability of neuron footprints belonging to the same neuron. It uses an interpolated version of p_same

        This function requires the successful building of a matching model first
    '''

    ## evaluate probability-function for each of a neurons neighbors
    print(neighbor_distance)
    neighbors = d_ROIs < neighbor_distance
    p_same = np.zeros_like(d_ROIs)
    p_same[neighbors] = p_model.ev(d_ROIs[neighbors],fp_corr[neighbors])
    
    p_same[np.isnan(p_same)] = 0        # fill "bad" entries with zeros
    p_same = np.clip(p_same,0,1)        # function-shapes may allow for values exceeding [0,1]

    return sp.sparse.csc_matrix(p_same)