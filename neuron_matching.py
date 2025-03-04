"""
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
"""

import os, cv2, copy, time, logging, pickle, tqdm
from pathlib import Path
import numpy as np
from inspect import signature

import scipy as sp
from scipy.optimize import curve_fit, linear_sum_assignment
import scipy.stats as sstats

from matplotlib import (
    pyplot as plt,
    rc,
    colors as mcolors,
    patches as mppatches,
    lines as mplines,
    cm,
)


from .utils import *

logging.basicConfig(level=logging.INFO)


class matching:

    def __init__(
        self, mousePath, paths=None, suffix="", matlab=False, logLevel=logging.ERROR
    ):
        """
        TODO:
            * put match results into more handy shape:
                - list of filePaths
                - SNR, etc not twice (only in results)
                - remap in results, not data (shift=0,corr=1)
                - p_same containing best and best_without_match value and being in results
        """

        self.matlab = matlab
        ## make sure suffix starts with '_'
        if suffix and suffix[0] != "_":
            suffix = "_" + suffix

        assert not (
            paths is None
        ), "You have to provide a list of paths to the CaImAn results files to be processed!"
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
            "neuron_detection": paths,  # list of paths to CaImAn result files to be processed in order
            "data": mousePath,  # path to which results are stored and loaded from
            "suffix": suffix,
        }

        # mP = matching_params
        self.params = matching_params
        self.params["model"] = "correlation"
        self.params["correlation_model"] = "shifted"
        self.update_bins(self.params["nbins"])

        ## initialize the main data dictionaries
        self.data_blueprint = {
            "nA": np.zeros(2, "int"),
            "cm": None,
            "p_same": None,
            "idx_eval": None,
            "SNR_comp": None,
            "r_values": None,
            "cnn_preds": None,
            "filePath": None,
        }

        self.data = {}
        self.data_tmp = {}

        self.data_cross = {
            "D_ROIs": [],
            "fp_corr": [],
        }

        self.status = {
            "counts_calculated": False,
            "model_calculated": False,
            "neurons_matched": False,
        }

    def update_bins(self, nbins):

        self.params["nbins"] = nbins
        self.params["arrays"] = self.build_arrays(nbins)

    def build_arrays(self, nbins):

        ## create value arrays for distance and footprint correlation
        arrays = {}
        arrays["distance_bounds"] = np.linspace(
            0, self.params["neighbor_distance"], nbins + 1
        )  # [:-1]
        arrays["correlation_bounds"] = np.linspace(0, 1, nbins + 1)  # [:-1]

        arrays["distance_step"] = self.params["neighbor_distance"] / nbins
        arrays["correlation_step"] = 1.0 / nbins

        arrays["distance"] = (
            arrays["distance_bounds"][:-1] + arrays["distance_step"] / 2
        )
        arrays["correlation"] = (
            arrays["correlation_bounds"][:-1] + arrays["correlation_step"] / 2
        )

        ## update histogram counts (if nbins_new < nbins), merge counts, otherwise split evenly
        # self.log.warning('implement method to update count histograms!')
        return arrays

    def run_matching(self, p_thr=[0.3, 0.05], save_results=True):
        print("Now running matching procedure in %s" % self.paths["data"])

        print("Building model for matching ...")
        self.build_model(save_results=save_results)

        print("Matching neurons ...")
        self.register_neurons(save_results=save_results, p_thr=p_thr)

        print("Done!")

    def build_model(self, save_results=False):
        """
        Iterate through all sessions in chronological order to build a model for neuron matching.
        The model assumes that the closest neuron to a neuron in the reference session ("nearest neighbor", NN) has a high probability of being the same neuron. Neuron distance and footprint correlation are calculated and registered distinctly for NN and "non-nearest neighbours" (nNN, within the viscinity of each neuron) in a histogram, which is then used to build a 2D model of matching probability, assuming a characteristic function for NN and nNN.

        This function takes several steps applied to each session:
          - align data to reference session (rigid and non-rigid shift), or skip if not possible
          - remove highly correlated footprints within one session
          - calculate kernel density of neuron locations (if requested by "use_kde")
          - update count histogram
        Finally, the count histogram is used to create the model

        """

        self.update_bins(128)

        self.model = {
            "counts": np.zeros((self.params["nbins"], self.params["nbins"], 3), "int"),
            "counts_unshifted": np.zeros(
                (self.params["nbins"], self.params["nbins"], 3), "int"
            ),
            "counts_same": np.zeros(
                (self.params["nbins"], self.params["nbins"]), "int"
            ),
            "counts_same_unshifted": np.zeros(
                (self.params["nbins"], self.params["nbins"]), "int"
            ),
            "kernel": {"idxes": {}, "kde": {}},
        }

        self.nS = len(self.paths["neuron_detection"])
        self.progress = tqdm.tqdm(
            enumerate(self.paths["neuron_detection"]), total=self.nS
        )
        # print(self.paths['neuron_detection'])

        has_reference = False

        # s_ref = 0
        for s, self.currentPath in self.progress:
            self.data[s] = copy.deepcopy(self.data_blueprint)
            self.data[s]["filePath"] = self.currentPath
            self.data[s]["skipped"] = False

            self.A = self.load_footprints(self.currentPath, s)

            if isinstance(self.A, bool):
                self.data[s]["skipped"] = True
                continue

            self.progress.set_description(f"Aligning data from {self.currentPath}")

            ## prepare (and align) footprints
            prepared, self.data[s]["remap"] = self.prepare_footprints(
                align_to_reference=has_reference
            )
            self.calculate_footprint_data(s)

            if not prepared:
                self.data[s]["skipped"] = True
                continue

            if self.params["use_kde"]:
                # self.progress.set_description('Calculate kernel density for Session %d'%s0)
                self.position_kde(s)  # build kernel

            self.update_joint_model(s, s)

            if has_reference:
                self.progress.set_description(
                    "Calculate cross-statistics for %s" % self.currentPath
                )
                self.data_cross["D_ROIs"], self.data_cross["fp_corr"], _ = (
                    calculate_statistics(
                        self.A,
                        A_ref=self.A_ref,
                        idx_eval=self.data[s]["idx_eval"],
                        idx_eval_ref=self.data[s_ref]["idx_eval"],
                        binary=self.params["binary"],
                        neighbor_distance=self.params["neighbor_distance"],
                        convert=self.params["pxtomu"],
                        model=self.params["correlation_model"],
                        dims=self.params["dims"],
                    )
                )  # calculating distances and footprint correlations
                self.progress.set_description(
                    "Update model with data from %s" % self.currentPath
                )
                self.update_joint_model(s, s_ref)

            self.A_ref = self.A.copy()
            self.Cn_ref = self.data_tmp["Cn"]
            s_ref = s
            has_reference = True

        self.dynamic_fit()

        if save_results:
            self.save_model(suffix=self.paths["suffix"])

    def dynamic_fit(self):
        times = 0
        while times < 3:
            try:
                self.fit_model(times=times)
                success = True
            except:
                success = False

            if success and self.model["p_same"]["which"] == "joint":
                print(f"found proper fit @times={times}")
                break
            times += 1

    def register_neurons(self, p_thr=[0.3, 0.05], save_results=False, model="shifted"):
        """
        This function iterates through sessions chronologically to create a set of all detected neurons, matched to one another based on the created model

          p_thr - (list of floats)
              2 entry-list, specifying probability above which matches are accepted [0]
              and above which losing contenders for a match are removed from the data [1]
        """

        self.params["correlation_model"] = model

        self.nS = len(self.paths["neuron_detection"])

        self.data = {}

        # if not self.model['f_same']:
        assert self.status[
            "model_calculated"
        ], "Model not yet created - please run build_model first"

        ## load and prepare first set of footprints
        s = 0
        while True:
            self.data[s] = copy.deepcopy(self.data_blueprint)
            self.data[s]["filePath"] = self.paths["neuron_detection"][s]
            self.data[s]["skipped"] = False

            self.A = self.load_footprints(self.paths["neuron_detection"][s], s)
            if isinstance(self.A, bool):
                self.data[s]["skipped"] = True
                s += 1
                continue
            else:
                break

        self.progress = tqdm.tqdm(
            enumerate(self.paths["neuron_detection"][s + 1 :], start=s + 1),
            total=self.nS - (s + 1),
            leave=True,
        )

        prepared, self.data[s]["remap"] = self.prepare_footprints(
            align_to_reference=False
        )
        self.calculate_footprint_data(s)

        self.Cn_ref = self.data_tmp["Cn"]
        self.A_ref = self.A[:, self.data[s]["idx_eval"]]
        self.A0 = self.A.copy()  ## store initial footprints for common reference

        ## initialize reference session, containing the union of all neurons
        self.data["joint"] = copy.deepcopy(self.data_blueprint)
        self.data["joint"]["nA"][0] = self.A_ref.shape[1]
        self.data["joint"]["idx_eval"] = np.ones(self.data["joint"]["nA"][0], "bool")
        self.data["joint"]["cm"] = center_of_mass(
            self.A_ref,
            self.params["dims"][0],
            self.params["dims"][1],
            convert=self.params["pxtomu"],
        )

        ## prepare and initialize assignment- and p_matched-arrays for storing results
        self.results = {
            "assignments": np.full((self.data[s]["nA"][1], self.nS), np.NaN),
            "p_matched": np.full((self.data[s]["nA"][1], self.nS, 2), np.NaN),
            "Cn_corr": np.full((self.nS, self.nS), np.NaN),
        }
        self.results["assignments"][:, s] = np.where(self.data[s]["idx_eval"])[0]
        self.results["p_matched"][:, 0, 0] = 1

        for s, self.currentPath in self.progress:
            if not self.currentPath:
                continue

            session_name = os.path.dirname(self.paths["neuron_detection"][s]).split(
                "/"
            )[-1]
            self.data[s] = copy.deepcopy(self.data_blueprint)
            self.data[s]["filePath"] = self.currentPath
            self.data[s]["skipped"] = False

            self.A = self.load_footprints(self.currentPath, s)

            if isinstance(self.A, bool):
                self.data[s]["skipped"] = True
                continue

            for s0 in range(s):
                # self.progress.set_description('Calculate image correlations for Session %d'%s)
                if (
                    (self.paths["neuron_detection"][s0])
                    and ("Cn" in self.data[s0].keys())
                    and ("Cn" in self.data[s].keys())
                ):
                    c_max, _, _ = calculate_img_correlation(
                        self.data[s0]["Cn"], self.data[s]["Cn"], plot_bool=False
                    )
                    self.results["Cn_corr"][s0, s] = self.results["Cn_corr"][s, s0] = (
                        c_max
                    )

            self.progress.set_description(
                f"A union size: {self.data['joint']['nA'][0]}, Preparing footprints from {session_name}"
            )

            ## preparing data of current session
            prepared, self.data[s]["remap"] = self.prepare_footprints(A_ref=self.A0)
            self.calculate_footprint_data(s)

            if not prepared:
                self.data[s]["skipped"] = True
                continue

            ## calculate matching probability between each pair of neurons
            self.progress.set_description(
                f"A union size: {self.data['joint']['nA'][0]}, Calculate statistics for {session_name}"
            )
            self.data_cross["D_ROIs"], self.data_cross["fp_corr"], _ = (
                calculate_statistics(
                    self.A,
                    A_ref=self.A_ref,
                    idx_eval=self.data[s]["idx_eval"],
                    idx_eval_ref=self.data["joint"]["idx_eval"],
                    binary=self.params["binary"],
                    neighbor_distance=self.params["neighbor_distance"],
                    convert=self.params["pxtomu"],
                    model=self.params["correlation_model"],
                    dims=self.params["dims"],
                )
            )

            idx_fp = 1 if self.params["correlation_model"] == "shifted" else 0
            self.data[s]["p_same"] = calculate_p(
                self.data_cross["D_ROIs"],
                self.data_cross["fp_corr"][idx_fp, ...],
                self.model["f_same"],
                self.params["neighbor_distance"],
            )

            ## run hungarian algorithm (HA) with (1-p_same) as score
            self.progress.set_description(
                f"A union size: {self.data['joint']['nA'][0]}, Perform Hungarian matching on {session_name}"
            )
            matches = linear_sum_assignment(1 - self.data[s]["p_same"].toarray())
            p_matched = self.data[s]["p_same"].toarray()[matches]

            idx_TP = np.where(p_matched > p_thr[0])[
                0
            ]  ## thresholding results (HA matches all pairs, but we only want matches above p_thr)
            if len(idx_TP) > 0:
                matched_ref = matches[0][idx_TP]  # matched neurons in s_ref
                matched = matches[1][idx_TP]  # matched neurons in s
                # print(idx_TP,matched_ref)
            else:
                matched_ref = np.array([], "int")
                matched = np.array([], "int")

            ## find neurons which were not matched in current and reference session
            non_matched_ref = np.setdiff1d(
                list(range(self.data["joint"]["nA"][0])), matched_ref
            )
            non_matched = np.setdiff1d(
                list(np.where(self.data[s]["idx_eval"])[0]), matches[1][idx_TP]
            )
            non_matched = non_matched[self.data[s]["idx_eval"][non_matched]]

            # print(matched,non_matched)
            ## calculate number of matches found
            TP = np.sum(p_matched > p_thr[0]).astype("float32")
            self.A_ref = self.A_ref.tolil()

            ## update footprint shapes of matched neurons with A_ref = (1-p/2)*A_ref + p/2*A to maintain part or all of original shape, depending on p_matched
            self.A_ref[:, matched_ref] = self.A_ref[:, matched_ref].multiply(
                1 - p_matched[idx_TP] / 2
            ) + self.A[:, matched].multiply(p_matched[idx_TP] / 2)

            ## removing footprints from the data which were competing with another one
            ## to be matched and lost, but have significant probability to be the same
            ## this step ensures, that downstream session don't confuse this one and the
            ## 'winner', leading to arbitrary assignments between two clusters
            for nm in non_matched:
                p_all = self.data[s]["p_same"][:, nm].todense()
                if np.any(p_all > p_thr[1]):
                    #    print(f'!! neuron {nm} is removed, as it is nonmatched and has high match probability:',p_all)[p_all>0])
                    #    print(np.where(p_all>0))
                    non_matched = non_matched[non_matched != nm]

            ## append new neuron footprints to union
            self.A_ref = sp.sparse.hstack(
                [self.A_ref, self.A[:, non_matched]]
            ).asformat("csc")

            ## update union data
            self.data["joint"]["nA"][0] = self.A_ref.shape[1]
            self.data["joint"]["idx_eval"] = np.ones(
                self.data["joint"]["nA"][0], "bool"
            )
            self.data["joint"]["cm"] = center_of_mass(
                self.A_ref,
                self.params["dims"][0],
                self.params["dims"][1],
                convert=self.params["pxtomu"],
            )

            ## write neuron indices of neurons from this session
            self.results["assignments"][
                matched_ref, s
            ] = matched  # ... matched neurons are added

            ## ... and non-matched (new) neurons are appended
            N_add = len(non_matched)
            match_add = np.zeros((N_add, self.nS)) * np.NaN
            match_add[:, s] = non_matched
            self.results["assignments"] = np.concatenate(
                [self.results["assignments"], match_add], axis=0
            )

            ## write match probabilities to matched neurons and reshape array to new neuron number
            self.results["p_matched"][matched_ref, s, 0] = p_matched[idx_TP]

            ## write best non-matching probability
            p_all = self.data[s]["p_same"].toarray()
            self.results["p_matched"][matched_ref, s, 1] = [
                max(
                    p_all[
                        c,
                        np.where(p_all[c, :] != self.results["p_matched"][c, s, 0])[0],
                    ]
                )
                for c in matched_ref
            ]
            # self.results['p_matched'][non_matched,s,1] = [max(p_all[c,np.where(p_all[c,:]!=self.results['p_matched'][c,s,0])[0]]) for c in matched_ref]

            p_same_add = np.full((N_add, self.nS, 2), np.NaN)
            p_same_add[:, s, 0] = 1

            self.results["p_matched"] = np.concatenate(
                [self.results["p_matched"], p_same_add], axis=0
            )

            # if np.any(np.all(self.results['p_matched']>0.9,axis=2)):
            #     print('double match!')
            #     return

        ## some post-processing to create cluster-structures values / statistics
        self.store_results()

        # finally, save results
        if save_results:
            self.save_registration(suffix=self.paths["suffix"])
            self.save_data(suffix=self.paths["suffix"])

    def store_results(self):
        self.results["cm"] = np.zeros(self.results["assignments"].shape + (2,)) * np.NaN
        for key in ["SNR_comp", "r_values", "cnn_preds"]:
            self.results[key] = np.zeros_like(self.results["assignments"])

        # for s in range(self.nS):
        self.results["remap"] = {
            "shift": np.zeros((self.nS, 2)),
            "transposed": np.zeros(self.nS, "bool"),
            "corr": np.zeros(self.nS),
            "corr_zscored": np.zeros(self.nS),
        }
        self.results["filePath"] = [""] * self.nS

        has_reference = False
        for s in self.data:
            print(s, has_reference)
            if s == "joint":
                continue

            if "remap" in self.data[s].keys():
                self.results["remap"]["transposed"][s] = (
                    self.data[s]["remap"]["transposed"] if has_reference else False
                )
                self.results["remap"]["shift"][s, :] = (
                    self.data[s]["remap"]["shift"] if has_reference else [0, 0]
                )
                self.results["remap"]["corr"][s] = (
                    self.data[s]["remap"]["c_max"] if has_reference else 1
                )
                self.results["remap"]["corr_zscored"][s] = (
                    self.data[s]["remap"]["c_zscored"] if has_reference else np.inf
                )

            if self.data[s]["skipped"]:
                continue

            self.results["filePath"][s] = self.data[s]["filePath"]

            idx_c = np.where(~np.isnan(self.results["assignments"][:, s]))[0]
            idx_n = self.results["assignments"][idx_c, s].astype("int")

            for key in ["cm", "SNR_comp", "r_values", "cnn_preds"]:
                try:
                    self.results[key][idx_c, s, ...] = self.data[s][key][idx_n, ...]
                except:
                    pass

            has_reference = True

    def load_footprints(self, loadPath, s=None):
        """
        function to load results from neuron detection (CaImAn, OnACID) and store in according dictionaries

        TODO:
          * implement min/max thresholding (maybe shift thresholding to other part of code?)
        """

        if loadPath and os.path.exists(loadPath):
            ld = load_data(loadPath)

            if "Cn" in ld.keys():
                Cn = ld["Cn"].T
            else:
                self.log.warning(
                    "Cn not in result files. constructing own Cn from footprints!"
                )
                Cn = np.array(ld["A"].sum(axis=1).reshape(*self.params["dims"]))

            self.data_tmp["Cn"] = Cn
            self.data_tmp["C"] = ld["C"]

            if not (s is None):
                self.data[s]["Cn"] = Cn

                ## load some data necessary for further processing
                if np.all(
                    [key in ld.keys() for key in ["SNR_comp", "r_values", "cnn_preds"]]
                ):
                    ## if evaluation parameters are present, use them to define used neurons
                    for key in ["SNR_comp", "r_values", "cnn_preds"]:
                        self.data[s][key] = ld[key]

                    ## threshold neurons according to evaluation parameters
                    self.data[s]["idx_eval"] = (
                        (ld["SNR_comp"] > self.params["SNR_lowest"])
                        & (ld["r_values"] > self.params["rval_lowest"])
                        & (ld["cnn_preds"] > self.params["cnn_lowest"])
                    ) & (
                        (ld["SNR_comp"] > self.params["SNR_min"])
                        | (ld["r_values"] > self.params["rval_min"])
                        | (ld["cnn_preds"] > self.params["cnn_min"])
                    )
                else:
                    ## else, use all neurons
                    self.data[s]["idx_eval"] = np.ones(ld["A"].shape[1], "bool")

            return ld["A"]

        else:
            return False

    def prepare_footprints(
        self, A_ref=None, align_to_reference=True, use_opt_flow=True
    ):
        """
        Function to prepare footprints for calculation and matching:
          - casting to / ensuring sparse type
          - calculating and reverting rigid and non-rigid shift
          - normalizing
        """

        ## ensure footprints are stored as sparse matrices to lower computational costs and RAM usage
        if "csc_matrix" not in str(type(self.A)):
            self.A = sp.sparse.csc_matrix(self.A)

        remap = {
            "shift": np.full((2,), np.NaN),
            "flow": np.zeros((2,) + self.params["dims"]),
            "c_max": np.NaN,
            "c_zscored": np.NaN,  ## z-score of correlation
            "transposed": False,
        }

        if align_to_reference:
            ## align footprints A to reference set A_ref

            # if no reference set of footprints is specified, use current reference set
            if A_ref is None:
                A_ref = self.A_ref

            ## cast this one to sparse as well, if needed
            if "csc_matrix" not in str(type(A_ref)):
                A_ref = sp.sparse.csc_matrix(A_ref)

            ## test whether images might be transposed (sometimes happens...) and flip accordingly
            # Cn = np.array(self.A.sum(1).reshape(512,512))
            # Cn_ref = np.array(A_ref.sum(1).reshape(512,512))
            Cn = self.data_tmp["Cn"]
            Cn_ref = self.Cn_ref
            c_max, c_zscored, _ = calculate_img_correlation(Cn_ref, Cn, plot_bool=False)
            c_max_T, c_zscored_T, _ = calculate_img_correlation(
                Cn_ref, Cn.T, plot_bool=False
            )
            # print('z scores:',c_zscored,c_zscored_T)
            remap["c_max"] = c_max

            ##  if no good alignment is found, don't include this session in the matching procedure (e.g. if imaging window is shifted too much)
            if (c_zscored < self.params["min_session_correlation_zscore"]) & (
                c_zscored_T < self.params["min_session_correlation_zscore"]
            ):
                # if (c_max < self.params['min_session_correlation']) & \
                #   (c_max_T < self.params['min_session_correlation']):
                print(
                    f"Identified correlation {c_max} too low in session {self.currentPath}, skipping alignment, as sessions appear to have no common imaging window!"
                )
                return False, remap

            if (c_max > 0.95) | (c_max_T > 0.95):
                print(
                    f"High correlation {c_max} found in session {self.currentPath}, skipping alignment as it is likely to be the same imaging data"
                )
                return False, remap

            if (c_max_T > c_max) & (c_max_T > self.params["min_session_correlation"]):
                print("Transposed image")
                self.A = sp.sparse.hstack(
                    [
                        sp.sparse.csc_matrix(
                            img.reshape(self.params["dims"]).transpose().reshape(-1, 1)
                        )
                        for img in self.A.transpose()
                    ]
                )
                remap["transposed"] = True

            ## calculate rigid shift and optical flow from reduced (cumulative) footprint arrays
            remap["shift"], flow, remap["c_max"], remap["c_zscored"] = (
                get_shift_and_flow(
                    A_ref, self.A, self.params["dims"], projection=1, plot_bool=False
                )
            )
            # remap['shift'],flow,remap['c_max'],remap['c_zscored'] = get_shift_and_flow(Cn_ref,Cn,self.params['dims'],projection=None,plot_bool=False)
            remap["flow"][0, ...] = flow[..., 0]
            remap["flow"][1, ...] = flow[..., 1]

            total_shift = np.sum(
                np.sqrt(np.array([s**2 for s in remap["shift"]]).sum())
            )
            if total_shift > self.params["max_session_shift"]:
                print(
                    f"Large shift {total_shift} identified in session {self.currentPath}, skipping alignment as sessions appear to have no significantly common imaging window!"
                )
                return False, remap

            ## use shift and flow to align footprints - define reverse mapping
            x_remap, y_remap = build_remap_from_shift_and_flow(
                self.params["dims"],
                remap["shift"],
                remap["flow"] if use_opt_flow else None,
            )

            ## use shift and flow to align footprints - apply reverse mapping
            self.A = sp.sparse.hstack(
                [
                    sp.sparse.csc_matrix(  # cast results to sparse type
                        cv2.remap(
                            img.reshape(
                                self.params["dims"]
                            ),  # reshape image to original dimensions
                            x_remap,
                            y_remap,  # apply reverse identified shift and flow
                            cv2.INTER_CUBIC,
                        ).reshape(
                            -1, 1
                        )  # reshape back to allow sparse storage
                    )
                    for img in self.A.toarray().T  # loop through all footprints
                ]
            )

        ## finally, apply normalization (watch out! changed normalization to /sum instead of /max)
        # self.A_ref = normalize_sparse_array(self.A_ref)
        self.A = normalize_sparse_array(self.A)

        return True, remap

    def calculate_footprint_data(self, s):

        # print(f'Session: {s}, A: {self.A.shape}')

        ## calculating various statistics
        self.data[s]["idx_eval"] &= (
            np.diff(self.A.indptr) != 0
        )  ## finding non-empty rows in sparse array (https://mike.place/2015/sparse/)

        # print('\n idx evals:',self.data[s]['idx_eval'].sum())
        self.data[s]["nA"][0] = self.A.shape[1]  # number of neurons
        self.data[s]["cm"] = center_of_mass(
            self.A, *self.params["dims"], convert=self.params["pxtomu"]
        )

        self.progress.set_description(
            f"Calculate self-referencing statistics for {self.paths['neuron_detection'][s]}"
        )

        ## find and mark potential duplicates of neuron footprints in session
        self.data_cross["D_ROIs"], self.data_cross["fp_corr"], idx_remove = (
            calculate_statistics(
                self.A,
                idx_eval=self.data[s]["idx_eval"],
                SNR_comp=self.data[s]["SNR_comp"],
                C=self.data_tmp["C"],
                binary=self.params["binary"],
                neighbor_distance=self.params["neighbor_distance"],
                convert=self.params["pxtomu"],
                model=self.params["correlation_model"],
                dims=self.params["dims"],
            )
        )
        # print('\n idx remove:',idx_remove)
        self.data[s]["idx_eval"][idx_remove] = False
        self.data[s]["nA"][1] = self.data[s]["idx_eval"].sum()

    def update_joint_model(self, s, s_ref):
        """
        Function to update counts in the joint model

        inputs:
        - s,s_ref: int / string
            key of current (s) and reference (s_ref) session
        - use_kde: bool
            defines, whether kde (kernel density estimation) is used to ...

        """

        ## use all neurons or only those from "medium-dense regions", defined by kde
        idxes = (
            self.model["kernel"]["idxes"][s_ref]
            if self.params["use_kde"]
            else np.ones(self.data[s_ref]["nA"][0], "bool")
        )

        ## find all neuron pairs below a distance threshold
        neighbors = (
            self.data_cross["D_ROIs"][idxes, :] < self.params["neighbor_distance"]
        )

        NN_idx = np.zeros((self.data[s_ref]["nA"][0], self.data[s]["nA"][0]), "bool")
        if s != s_ref:
            ## identifying next-neighbours
            idx_NN = np.nanargmin(
                self.data_cross["D_ROIs"][self.data[s_ref]["idx_eval"], :], axis=1
            )

            NN_idx[self.data[s_ref]["idx_eval"], idx_NN] = True
            NN_idx = NN_idx[idxes, :][neighbors]
        else:
            np.fill_diagonal(NN_idx, True)
            # NN_idx[~idxes,:] = False
            NN_idx = NN_idx[idxes, :][neighbors]

        ## obtain distance and correlation values of close neighbors, only
        D_ROIs = self.data_cross["D_ROIs"][idxes, :][neighbors]
        fp_corr = self.data_cross["fp_corr"][0, idxes, :][neighbors]
        fp_corr_shifted = self.data_cross["fp_corr"][1, idxes, :][neighbors]

        ## update count histogram with data from current session pair
        for i in tqdm.tqdm(
            range(self.params["nbins"]), desc="updating joint model", leave=False
        ):

            ## find distance indices of values falling into the current bin
            idx_dist = (D_ROIs >= self.params["arrays"]["distance_bounds"][i]) & (
                D_ROIs < self.params["arrays"]["distance_bounds"][i + 1]
            )

            for j in range(self.params["nbins"]):

                ## differentiate between the two models for calculating footprint-correlation
                if (self.params["correlation_model"] == "unshifted") | (
                    self.params["correlation_model"] == "both"
                ):
                    ## find correlation indices of values falling into the current bin
                    idx_fp = (
                        fp_corr > self.params["arrays"]["correlation_bounds"][j]
                    ) & (self.params["arrays"]["correlation_bounds"][j + 1] > fp_corr)
                    idx_vals = idx_dist & idx_fp

                    if s == s_ref:  # for self-comparing
                        self.model["counts_same_unshifted"][i, j] += np.count_nonzero(
                            idx_vals & ~NN_idx
                        )

                    else:  # for cross-comparing
                        self.model["counts_unshifted"][i, j, 0] += np.count_nonzero(
                            idx_vals
                        )
                        self.model["counts_unshifted"][i, j, 1] += np.count_nonzero(
                            idx_vals & NN_idx
                        )
                        self.model["counts_unshifted"][i, j, 2] += np.count_nonzero(
                            idx_vals & ~NN_idx
                        )

                if (self.params["correlation_model"] == "shifted") | (
                    self.params["correlation_model"] == "both"
                ):
                    idx_fp = (
                        fp_corr_shifted > self.params["arrays"]["correlation_bounds"][j]
                    ) & (
                        self.params["arrays"]["correlation_bounds"][j + 1]
                        > fp_corr_shifted
                    )
                    idx_vals = idx_dist & idx_fp
                    if s == s_ref:  # for self-comparing
                        # self.model['counts_same'][i,j] += np.count_nonzero(idx_vals)
                        self.model["counts_same"][i, j] += np.count_nonzero(
                            idx_vals & ~NN_idx
                        )
                    else:  # for cross-comparing
                        self.model["counts"][i, j, 0] += np.count_nonzero(idx_vals)
                        self.model["counts"][i, j, 1] += np.count_nonzero(
                            idx_vals & NN_idx
                        )
                        self.model["counts"][i, j, 2] += np.count_nonzero(
                            idx_vals & ~NN_idx
                        )

    def position_kde(self, s, plot_bool=False):
        """
        function to calculate kernel density estimate of neuron density in session s
        this is optional, but can be used to exclude highly dense and highly sparse regions from statistics in order to not skew statistics

        """
        self.log.info("calculating kernel density estimates for session %d" % s)

        ## calculating kde from center of masses
        x_grid, y_grid = np.meshgrid(
            np.linspace(
                0,
                self.params["dims"][0] * self.params["pxtomu"],
                self.params["dims"][0],
            ),
            np.linspace(
                0,
                self.params["dims"][1] * self.params["pxtomu"],
                self.params["dims"][1],
            ),
        )
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        kde = sp.stats.gaussian_kde(self.data[s]["cm"][self.data[s]["idx_eval"], :].T)
        self.model["kernel"]["kde"][s] = np.reshape(kde(positions), x_grid.shape)

        cm_px = (
            self.data[s]["cm"][self.data[s]["idx_eval"], :] / self.params["pxtomu"]
        ).astype("int")
        kde_at_com = np.zeros(self.data[s]["nA"][0]) * np.NaN
        kde_at_com[self.data[s]["idx_eval"]] = self.model["kernel"]["kde"][s][
            cm_px[:, 1], cm_px[:, 0]
        ]
        self.model["kernel"]["idxes"][s] = (
            kde_at_com
            > np.quantile(self.model["kernel"]["kde"][s], self.params["qtl"][0])
        ) & (
            kde_at_com
            < np.quantile(self.model["kernel"]["kde"][s], self.params["qtl"][1])
        )

        if plot_bool:
            plt.figure()
            h_kde = plt.imshow(
                self.model["kernel"]["kde"][s],
                cmap=plt.cm.gist_earth_r,
                origin="lower",
                extent=[
                    0,
                    self.params["dims"][0] * self.params["pxtomu"],
                    0,
                    self.params["dims"][1] * self.params["pxtomu"],
                ],
            )
            # if s>0:
            # col = self.data_cross['D_ROIs'].min(1)
            # else:
            # col = 'w'
            plt.scatter(
                self.data[s]["cm"][:, 0],
                self.data[s]["cm"][:, 1],
                c="w",
                s=5 + 10 * self.model["kernel"]["idxes"][s],
                clim=[0, 10],
                cmap="YlOrRd",
            )
            plt.xlim([0, self.params["dims"][0] * self.params["pxtomu"]])
            plt.ylim([0, self.params["dims"][1] * self.params["pxtomu"]])

            # cm_px = (self.data['cm'][s]/self.params['pxtomu']).astype('int')
            # kde_at_cm = self.model['kernel']['kde'][s][cm_px[:,1],cm_px[:,0]]
            plt.colorbar(h_kde)
            plt.show(block=False)

    def fit_model(self, correlation_model="shifted", times=0):
        """ """

        self.params["correlation_model"] = correlation_model

        neighbor_dictionary = {"NN": [], "nNN": [], "all": []}

        self.model = self.model | {
            "fit_function": {
                "single": {
                    "distance": copy.deepcopy(neighbor_dictionary),
                    "correlation": copy.deepcopy(neighbor_dictionary),
                },
                "joint": {
                    "distance": copy.deepcopy(neighbor_dictionary),
                    "correlation": copy.deepcopy(neighbor_dictionary),
                },
            },
            "fit_parameter": {
                "single": {
                    "distance": copy.deepcopy(neighbor_dictionary),
                    "correlation": copy.deepcopy(neighbor_dictionary),
                },
                "joint": {},
            },
            "pdf": {"single": {"distance": [], "correlation": []}, "joint": []},
            "p_same": {"single": {}, "joint": np.zeros(0)},
            "f_same": False,
        }

        # print(self.params['correlation_model'])
        if (not (self.params["correlation_model"] == "unshifted")) & (
            not (self.params["correlation_model"] == "shifted")
        ):
            raise Exception(
                'Please specify model to be either "shifted" or "unshifted"'
            )

        key_counts = (
            "counts"
            if self.params["correlation_model"] == "shifted"
            else "counts_unshifted"
        )
        counts = scale_down_counts(
            self.model[key_counts].astype("float32"), times=times
        )

        nbins = counts.shape[0]
        self.update_bins(nbins)

        def build_single_model(model, times=0):

            m = 0 if model == "correlation" else 1
            max_val = (
                1.0 if model == "correlation" else self.params["neighbor_distance"]
            )

            fit_fun, fit_bounds = self.set_functions(model, "single")

            ## obtain fit parameters from fit to count histograms
            for p, pop in zip((1, 2, 0), ["NN", "nNN", "all"]):

                if pop == "all":
                    p0 = (
                        (
                            self.model[key_counts][..., 1].sum()
                            / self.model[key_counts][..., 0].sum(),
                        )
                        + tuple(self.model["fit_parameter"]["single"][model]["NN"])
                        + tuple(self.model["fit_parameter"]["single"][model]["nNN"])
                    )
                else:
                    p0 = None

                failed = True
                nbins = self.model[key_counts].shape[0]
                # while failed==True and nbins>10:
                #     try:
                self.log.debug(model, pop)

                counts = scale_down_counts(
                    self.model[key_counts].astype("float32"), times=times
                )
                nbins = counts.shape[0]
                self.update_bins(nbins)

                normed_counts = (
                    counts[..., p].sum(m)
                    / counts[..., p].sum()
                    / self.params["arrays"][f"{model}_step"]
                )
                plot = False
                if plot:
                    plt.figure()
                    plt.plot(self.params["arrays"][model], normed_counts)

                    if pop == "all":
                        plt.plot(
                            self.params["arrays"][model],
                            fit_fun[pop](self.params["arrays"][model], *p0),
                            "k--",
                        )

                self.model["fit_parameter"]["single"][model][pop] = curve_fit(
                    fit_fun[pop],
                    self.params["arrays"][model],
                    normed_counts,
                    bounds=fit_bounds[pop],
                    p0=p0,
                )[0]

                if plot:
                    plt.plot(
                        self.params["arrays"][model],
                        fit_fun[pop](
                            self.params["arrays"][model],
                            *self.model["fit_parameter"]["single"][model][pop],
                        ),
                    )
                    plt.show(block=False)

                failed = False
                # except:
                #     times += 1
                #     print(f'Finding solution failed. Scaling down counts by factor {times} ({nbins} bins)')

            ## build functions for NN population and overall
            ## using "all model", as it doesn't overestimate wrongly assigned tails of NN distribution

            fun_NN = (
                fun_wrapper(
                    fit_fun["NN"],
                    self.params["arrays"][model],
                    self.model["fit_parameter"]["single"][model]["all"][
                        1 : self.model["fit_function"]["single"][model]["NN_Nparams"]
                    ],
                )
                * self.model["fit_parameter"]["single"][model]["all"][0]
            )

            fun_nNN = fun_wrapper(
                fit_fun["nNN"],
                self.params["arrays"][model],
                self.model["fit_parameter"]["single"][model]["all"][
                    self.model["fit_function"]["single"][model]["NN_Nparams"] :
                ],
            ) * (1 - self.model["fit_parameter"]["single"][model]["all"][0])

            if model == "correlation":
                cdf_NN = np.cumsum(fun_NN) / np.sum(fun_NN)
                cdf_nNN = np.cumsum(fun_nNN[::-1])[::-1] / np.sum(fun_nNN)
            else:
                cdf_NN = np.cumsum(fun_NN[::-1])[::-1] / np.sum(fun_NN)
                cdf_nNN = np.cumsum(fun_nNN) / np.sum(fun_nNN)

            self.model["p_same"]["single"][model] = cdf_NN / (cdf_NN + cdf_nNN)

        def build_joint_model(model, times=0, counts_thr=20):

            # 'model' defines the single model functions which are used to calculate the joint model,
            # while the 'weight model' merely specifies the weighting between NN and nNN functions

            counts = scale_down_counts(
                self.model[key_counts].astype("float32"), times=times
            )
            nbins = counts.shape[0]
            self.update_bins(nbins)

            m = 1 if model == "correlation" else 0

            fit_fun, fit_bounds = self.set_functions(model, "joint")

            # normalized_histogram = counts/counts.sum(m,keepdims=True)/self.params['arrays'][f'{model}_step']# * nbins/max_val
            # normalized_histogram[np.isnan(normalized_histogram)] = 0
            # print(normalized_histogram.shape)

            self.model["fit_parameter"]["joint"][model] = {
                "NN": np.full(
                    (
                        nbins,
                        self.model["fit_function"]["single"][model]["NN_Nparams"] - 1,
                    ),
                    np.NaN,
                ),
                "nNN": np.full(
                    (
                        nbins,
                        self.model["fit_function"]["single"][model]["nNN_Nparams"] - 1,
                    ),
                    np.NaN,
                ),
                #'all':np.zeros((nbins,len(self.model['fit_parameter']['single']['correlation']['all'])))*np.NaN}
            }

            for p, pop in enumerate(["NN", "nNN"], start=1):
                fitted = False
                for n in range(nbins):
                    ### to fp-correlation: NN - reverse lognormal, nNN - reverse lognormal
                    fit_data = np.take(counts[..., p], n, axis=abs(m - 1))
                    # print(f'n data (@bin {n}):',fit_data.sum())
                    if fit_data.sum() > counts_thr:
                        fit_data /= fit_data.sum()
                        # print(fit_data)
                        self.log.debug(
                            f"data for {pop} {model}-distribution: ", fit_data
                        )

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
                        plot = False
                        if plot:
                            plt.figure()
                            plt.plot(self.params["arrays"][model], fit_data)

                        try:
                            self.model["fit_parameter"]["joint"][model][pop][n, :] = (
                                curve_fit(
                                    fit_fun[pop],
                                    self.params["arrays"][model],
                                    fit_data,
                                    bounds=fit_bounds[pop],
                                    # p0=self.model['fit_parameter']['joint'][model][pop][i-1,:] if fitted else None
                                )[0]
                            )
                            self.log.debug(
                                f"fit results @bin {n}:",
                                self.model["fit_parameter"]["joint"][model][pop][n, :],
                            )
                        except:
                            print(
                                f"fit failed for {model} model and population {pop} @bin {n}/{nbins}"
                            )
                            if plot:
                                plt.plot(
                                    self.params["arrays"][model],
                                    fit_fun[pop](
                                        self.params["arrays"][model],
                                        *self.model["fit_parameter"]["joint"][model][
                                            pop
                                        ][n, :],
                                    ),
                                )
                                plt.title(
                                    f"{model} model, {pop} population, bin {n} ({nbins} bins)"
                                )
                                plt.show(block=False)
                            continue

                            #     break
                            # except:
                            #     scale_times += 1
                            #     self.log.debug(f'Finding solution failed. Scaling down counts by factor {scale_times} ({nbins/2} bins)')

                        if plot:
                            plt.plot(
                                self.params["arrays"][model],
                                fit_fun[pop](
                                    self.params["arrays"][model],
                                    *self.model["fit_parameter"]["joint"][model][pop][
                                        n, :
                                    ],
                                ),
                            )
                            plt.title(
                                f"{model} model, {pop} population, bin {n} ({nbins} bins)"
                            )
                            plt.show(block=False)

                        # nbins = normalized_histogram.shape[0]
                        # self.update_bins(nbins)
                        fitted = True
                    else:
                        fitted = False

            extrapolate = True
            postprocess = True
            ## running smoothing, intra- and extrapolation
            filter_footprint = (int(nbins / 10), 0)
            # print('blabla')
            if postprocess:
                # print('smoothing')
                # print(filter_footprint)
                for pop in ["NN", "nNN"]:  # ,'all']
                    # print(pop)
                    # print('params before smoothing:',self.model['fit_parameter']['joint'][model][pop])
                    # for ax in range(self.model['fit_parameter']['joint'][key][pop].shape(1)):
                    # self.model['fit_parameter']['joint'][model][pop] = nanmedian_filter(self.model['fit_parameter']['joint'][model][pop],np.ones(filter_footprint))
                    self.model["fit_parameter"]["joint"][model][pop] = nanmedian_filter(
                        self.model["fit_parameter"]["joint"][model][pop],
                        np.ones((3, 1)),
                    )
                    # print('params after median:',self.model['fit_parameter']['joint'][model][pop])
                    self.model["fit_parameter"]["joint"][model][pop] = nangauss_filter(
                        self.model["fit_parameter"]["joint"][model][pop],
                        filter_footprint,
                    )
                    # print(self.model['fit_parameter']['joint'][key][pop])
                    # print('params after smoothing:',self.model['fit_parameter']['joint'][model][pop])

                    if extrapolate:
                        # print('extrapolating data...')

                        for ax in range(
                            self.model["fit_parameter"]["joint"][model][pop].shape[1]
                        ):
                            ## find first/last index, at which parameter has a non-nan value
                            nan_idx = np.isnan(
                                self.model["fit_parameter"]["joint"][model][pop][:, ax]
                            )

                            if (
                                nan_idx[0] and (~nan_idx).sum() > 1
                            ):  ## extrapolate beginning
                                idx = np.where(~nan_idx)[0][: int(nbins / 5)]
                                y_arr = self.model["fit_parameter"]["joint"][model][
                                    pop
                                ][idx, ax]
                                f_interp = np.polyfit(
                                    self.params["arrays"][model][idx], y_arr, 1
                                )
                                poly_fun = np.poly1d(f_interp)

                                self.model["fit_parameter"]["joint"][model][pop][
                                    : idx[0], ax
                                ] = poly_fun(self.params["arrays"][model][: idx[0]])

                            if nan_idx[-1] and (~nan_idx).sum() > 1:  ## extrapolate end
                                idx = np.where(~nan_idx)[0][-int(nbins / 5) :]
                                y_arr = self.model["fit_parameter"]["joint"][model][
                                    pop
                                ][idx, ax]
                                f_interp = np.polyfit(
                                    self.params["arrays"][model][idx], y_arr, 1
                                )
                                poly_fun = np.poly1d(f_interp)

                                self.model["fit_parameter"]["joint"][model][pop][
                                    idx[-1] + 1 :, ax
                                ] = poly_fun(
                                    self.params["arrays"][model][idx[-1] + 1 :]
                                )

        build_single_model("distance", times)
        build_single_model("correlation", times)

        ## build both models first
        build_joint_model("distance", times)
        build_joint_model("correlation", times)

        ## define probability density functions from "major" model
        weight_model = (
            "distance" if self.params["model"] == "correlation" else "correlation"
        )

        fit_fun, _ = self.set_functions(
            self.params["model"], "joint"
        )  # could also be distance
        # self.model['pdf']['joint'] = np.zeros((2,nbins,nbins))

        self.model["p_same"]["joint"] = np.zeros((nbins, nbins))
        for n in range(nbins):

            fun_NN = fun_wrapper(
                fit_fun["NN"],
                self.params["arrays"][self.params["model"]],
                self.model["fit_parameter"]["joint"][self.params["model"]]["NN"][n, :],
            )

            fun_nNN = fun_wrapper(
                fit_fun["nNN"],
                self.params["arrays"][self.params["model"]],
                self.model["fit_parameter"]["joint"][self.params["model"]]["nNN"][n, :],
            )

            # for p,pop in enumerate(['NN','nNN']):
            #     # if not np.any(np.isnan(self.model['fit_parameter']['joint'][self.params['model']][pop][n,:])):
            #         f_pop = fun_wrapper(
            #             fit_fun[pop],
            #             self.params['arrays'][self.params['model']],
            #             self.model['fit_parameter']['joint'][self.params['model']][pop][n,:]
            #         )

            weight = self.model["p_same"]["single"][weight_model][n]
            # if pop=='nNN':
            #     weight = 1 - weight
            # if self.params['model']=='correlation':
            #     self.model['pdf']['joint'][p,n,:] = f_pop*weight
            # else:
            #     self.model['pdf']['joint'][p,:,n] = f_pop*weight

            if self.params["model"] == "correlation":
                cdf_NN = np.cumsum(fun_NN) / np.sum(fun_NN) * weight
                cdf_nNN = (
                    np.cumsum(fun_nNN[::-1])[::-1] / np.sum(fun_nNN) * (1 - weight)
                )
                self.model["p_same"]["joint"][n, :] = cdf_NN / (cdf_NN + cdf_nNN)
            else:
                cdf_NN = np.cumsum(fun_NN[::-1])[::-1] / np.sum(fun_NN) * weight
                cdf_nNN = np.cumsum(fun_nNN) / np.sum(fun_nNN) * (1 - weight)
                self.model["p_same"]["joint"][:, n] = cdf_NN / (cdf_NN + cdf_nNN)

        ## obtain probability of being same neuron

        # self.model['p_same']['joint'] = 1-self.model['pdf']['joint'][1,...]/np.nansum(self.model['pdf']['joint'],0)

        # self.model['p_same']['joint'] = nangauss_filter(self.model['p_same']['joint'],2)
        if np.any(np.isnan(self.model["p_same"]["joint"])):
            self.model["p_same"]["which"] = "single"
        else:
            self.model["p_same"]["which"] = "joint"

        # if count_thr > 0:
        # self.model['p_same']['joint'] *= np.minimum(self.model[key_counts][...,0],count_thr)/count_thr
        # sp.ndimage.filters.gaussian_filter(self.model['p_same']['joint'],2,output=self.model['p_same']['joint'])
        self.create_model_evaluation()
        self.status["model_calculated"] = True
        # return self.model['pdf']['joint']

    def create_model_evaluation(self):

        if self.model["p_same"]["which"] == "single":
            fun = sp.interpolate.interp1d(
                self.params["arrays"]["distance"],
                self.model["p_same"]["single"]["distance"],
                fill_value="extrapolate",
            )
            self.model["f_same"] = lambda distance, correlation=None: fun(distance)
        else:
            self.model["f_same"] = lambda distance, correlation: sp.interpolate.interpn(
                (
                    self.params["arrays"]["distance"],
                    self.params["arrays"]["correlation"],
                ),
                self.model["p_same"]["joint"],
                (distance, correlation),
                bounds_error=False,
                fill_value=None,
            )

    def set_functions(self, model, model_type="joint"):
        """
        set up functions for model fitting to count histogram for both, single and joint model.
        This function defines both, the functional shape (from a set of predefined functions), and the boundaries for function parameters

        REMARK:
          * A more flexible approach with functions to be defined on class creation could be feasible, but makes everything a lot more complex for only minor pay-off. Could be implemented in future changes.
        """

        fit_fun = {}
        fit_bounds = {}
        bounds_p = np.array([(0, 1)]).T  # weight 'p' between NN and nNN function

        if model == "distance":

            ## set functions for model
            fit_fun["NN"] = functions["lognorm"]
            fit_bounds["NN"] = np.array(
                [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]
            ).T
            # fit_fun['NN'] = functions['gamma']
            # fit_bounds['NN'] = np.array([(0,np.inf),(0,np.inf),(0,np.inf)]).T

            fit_fun["nNN"] = functions["linear_sigmoid"]
            fit_bounds["nNN"] = np.array(
                [
                    (0, np.inf),
                    (0, np.inf),
                    (0, self.params["neighbor_distance"] * 2 / 3),
                ]
            ).T

        elif model == "correlation":

            ## set functions for model
            fit_fun["NN"] = functions["lognorm_reverse"]
            fit_bounds["NN"] = np.array(
                [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]
            ).T

            # fit_fun['NN'] = functions['gamma_reverse']
            # fit_bounds['NN'] = np.array([(0,np.inf),(0,np.inf),(0,np.inf)]).T

            if self.params["correlation_model"] == "shifted":
                fit_fun["nNN"] = functions["gauss"]
                # fit_fun['nNN'] = functions['skewed_gauss']
                fit_bounds["nNN"] = np.array([(0, np.inf), (0, np.inf), (0, 1)]).T

            else:
                fit_fun["nNN"] = functions["beta"]
                fit_bounds["nNN"] = np.array([(-np.inf, np.inf), (-np.inf, np.inf)]).T

        sig_NN = signature(fit_fun["NN"])
        NN_parameters = len(sig_NN.parameters) - 1

        ## set bounds for fit-parameters
        fit_fun["all"] = lambda x, p, *args: p * fit_fun["NN"](
            x, *args[:NN_parameters]
        ) + (1 - p) * fit_fun["nNN"](x, *args[NN_parameters:])
        fit_bounds["all"] = np.hstack([bounds_p, fit_bounds["NN"], fit_bounds["nNN"]])

        for pop in ["NN", "nNN", "all"]:
            if pop in fit_fun:
                self.model["fit_function"][model_type][model][pop] = fit_fun[pop]

                sig = signature(fit_fun[pop])
                self.model["fit_function"][model_type][model][f"{pop}_Nparams"] = len(
                    sig.parameters
                )

        return fit_fun, fit_bounds

    ### -------------------- SAVING & LOADING ---------------------- ###

    def save_model(self, suffix="", matlab=None):

        fix_suffix(suffix)
        pathMatching = os.path.join(self.paths["data"], "matching")
        if ~os.path.exists(pathMatching):
            os.makedirs(pathMatching, exist_ok=True)

        results = {}
        for key in [
            "p_same",
            "fit_parameter",
            "pdf",
            "counts",
            "counts_unshifted",
            "counts_same",
            "counts_same_unshifted",
        ]:  # ,'f_same']:
            results[key] = self.model[key]

        pathSv = os.path.join(pathMatching, f"match_model{suffix}")
        pathSv += ".mat" if (matlab is None and self.matlab) or matlab else ".pkl"
        save_data(results, pathSv)

    def load_model(self, suffix=None, matlab=None):

        if suffix is None:
            suffix = self.paths["suffix"]
        suffix = fix_suffix(suffix)

        ext = "mat" if (matlab is None and self.matlab) or matlab else "pkl"
        pathLd = Path(self.paths["data"]) / f"matching/match_model{suffix}.{ext}"

        print(pathLd)
        results = load_data(pathLd)

        self.model = {}
        for key in results.keys():
            self.model[key] = results[key]

        self.update_bins(self.model["p_same"]["joint"].shape[0])

        self.create_model_evaluation()
        self.model["model_calculated"] = True

    def save_registration(self, suffix=None, matlab=None):

        if suffix is None:
            suffix = self.paths["suffix"]
        suffix = fix_suffix(suffix)

        pathMatching = Path(self.paths["data"]) / "matching"
        if not pathMatching.exists():
            os.makedirs(pathMatching, exist_ok=True)

        ext = "mat" if (matlab is None and self.matlab) or matlab else "pkl"
        pathSv = Path(pathMatching) / f"neuron_registration{suffix}.{ext}"
        save_data(self.results, pathSv)

    def save_data(self, suffix=None, matlab=None):

        if suffix is None:
            suffix = self.paths["suffix"]
        suffix = fix_suffix(suffix)

        pathMatching = Path(self.paths["data"]) / "matching"
        if not pathMatching.exists():
            os.makedirs(pathMatching, exist_ok=True)

        ext = "mat" if (matlab is None and self.matlab) or matlab else "pkl"
        pathSv = Path(pathMatching) / f"matching_data{suffix}.{ext}"

        save_data(self.data, pathSv)

    def load_registration(self, suffix=None, matlab=None):

        if suffix is None:
            suffix = self.paths["suffix"]
        suffix = fix_suffix(suffix)

        ext = "mat" if (matlab is None and self.matlab) or matlab else "pkl"
        pathLd = (
            Path(self.paths["data"]) / f"matching/neuron_registration{suffix}.{ext}"
        )
        self.results = load_data(pathLd)
        print(pathLd)

        # self.paths['neuron_detection'] = replace_relative_path(self.paths['neuron_detection'],self.paths['data'])
        # try:
        # self.results['assignments'] = self.results['assignment']
        # except:
        # 1

    def load_data(self, suffix=None, matlab=None):

        if suffix is None:
            suffix = self.paths["suffix"]
        suffix = fix_suffix(suffix)

        self.data = {}
        if (matlab is None and self.matlab) or matlab:
            pathLd = Path(self.paths["data"]) / f"matching/matching_data{suffix}.mat"
            dataLd = load_data(pathLd)
            for key in dataLd.keys():
                try:
                    self.data[int(key)] = dataLd[key]
                except:
                    self.data[key] = dataLd[key]
        else:
            pathLd = Path(self.paths["data"]) / f"matching/matching_data{suffix}.pkl"
            self.data = load_data(pathLd)
        print(pathLd)
        # self.paths['neuron_detection'] = replace_relative_path(self.paths['neuron_detection'],self.paths['data'])

    def find_confusion_candidates(self, confusion_distance=5):

        # cm = self.results['com']

        cm_mean = np.nanmean(self.results["cm"], axis=1)
        cm_dists = sp.spatial.distance.squareform(sp.spatial.distance.pdist(cm_mean))

        confusion_candidates = np.where(
            np.logical_and(cm_dists > 0, cm_dists < confusion_distance)
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
        print(len(confusion_candidates[0]) / 2, " candidates found")
        ct = 0
        for i, j in zip(*confusion_candidates):

            if i < j:
                assignments = self.results["assignments"][(i, j), :].T
                occ = np.isfinite(assignments)
                nOcc = occ.sum(axis=0)
                print("clusters:", i, j, "occurences:", nOcc)

                # print(assignments)

                ### confusion can occur if in one session two footprints are competing for a match
                confused_sessions = np.where(np.all(np.isfinite(assignments), axis=1))[
                    0
                ]
                if len(confused_sessions) > 0:
                    confused_session = confused_sessions[0]
                    print(assignments[: confused_session + 1])

                    # print(self.data[confused_session]['p_same'][i,:])
                    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
                    self.plot_footprints(i, fp_color="k", ax_in=ax, use_plotly=True)
                    self.plot_footprints(j, fp_color="r", ax_in=ax, use_plotly=True)
                    plt.show(block=False)

                    ct += 1

                ### or can occur when matching probability is quite low
                if ct > 3:
                    break
        return

    def scale_counts(self, times=0):

        key_counts = (
            "counts"
            if self.params["correlation_model"] == "shifted"
            else "counts_unshifted"
        )
        counts = scale_down_counts(self.model[key_counts], times)
        self.params["nbins"] = counts.shape[0]
        self.update_bins(self.params["nbins"])

        if self.model["p_same"]["joint"].shape[0] != self.params["nbins"]:
            self.fit_model(times=times)

        return counts

    ### ------------------- PLOTTING FUNCTIONS --------------------- ###

    def calculate_RoC(self, steps, model="correlation", times=0):
        # key_counts = 'counts' if self.params['correlation_model']=='shifted' else 'counts_unshifted'

        counts = self.scale_counts(times)
        nbins = self.params["nbins"]

        p_steps = np.linspace(0, 1, steps + 1)

        rates = {"tp": {}, "tn": {}, "fp": {}, "fn": {}, "cumfrac": {}}

        for key in rates.keys():
            rates[key] = {
                "joint": np.zeros(steps),
                "distance": np.zeros(steps),
                "correlation": np.zeros(steps),
            }

        nTotal = counts[..., 0].sum()
        for i, p in enumerate(p_steps[:-1]):

            for key in ["joint", "distance", "correlation"]:

                if key == "joint":
                    idxes_negative = self.model["p_same"]["joint"] < p
                    idxes_positive = self.model["p_same"]["joint"] >= p

                    tp = counts[idxes_positive, 1].sum()
                    tn = counts[idxes_negative, 2].sum()
                    fp = counts[idxes_positive, 2].sum()
                    fn = counts[idxes_negative, 1].sum()

                    rates["cumfrac"]["joint"][i] = (
                        counts[idxes_negative, 0].sum() / nTotal
                    )
                elif key == "distance":
                    idxes_negative = self.model["p_same"]["single"]["distance"] < p
                    idxes_positive = self.model["p_same"]["single"]["distance"] >= p

                    tp = counts[idxes_positive, :, 1].sum()
                    tn = counts[idxes_negative, :, 2].sum()
                    fp = counts[idxes_positive, :, 2].sum()
                    fn = counts[idxes_negative, :, 1].sum()

                    rates["cumfrac"]["distance"][i] = (
                        counts[idxes_negative, :, 0].sum() / nTotal
                    )
                else:
                    idxes_negative = self.model["p_same"]["single"]["correlation"] < p
                    idxes_positive = self.model["p_same"]["single"]["correlation"] >= p

                    tp = counts[:, idxes_positive, 1].sum()
                    tn = counts[:, idxes_negative, 2].sum()
                    fp = counts[:, idxes_positive, 2].sum()
                    fn = counts[:, idxes_negative, 1].sum()

                    rates["cumfrac"]["correlation"][i] = (
                        counts[:, idxes_negative, 0].sum() / nTotal
                    )

                rates["tp"][key][i] = tp / (fn + tp)
                rates["tn"][key][i] = tn / (fp + tn)
                rates["fp"][key][i] = fp / (fp + tn)
                rates["fn"][key][i] = fn / (fn + tp)

        return p_steps, rates

    def get_population_mean_and_var(self):
        """
        function assumes correlation and distance model to be truncated
        lognormal distributions
        """
        mean = {
            "distance": {
                "NN": None,
                "nNN": None,
            },
            "correlation": {
                "NN": None,
                "nNN": None,
            },
        }
        var = {
            "distance": {
                "NN": None,
                "nNN": None,
            },
            "correlation": {
                "NN": None,
                "nNN": None,
            },
        }

        mean["correlation"]["NN"], var["correlation"]["NN"] = mean_of_trunc_lognorm(
            self.model["fit_parameter"]["joint"]["correlation"]["NN"][:, 2],
            self.model["fit_parameter"]["joint"]["correlation"]["NN"][:, 1],
            [0, 1],
        )
        mean["correlation"]["NN"] = 1 - mean["correlation"]["NN"]

        mean["distance"]["NN"], var["distance"]["NN"] = mean_of_trunc_lognorm(
            self.model["fit_parameter"]["joint"]["distance"]["NN"][:, 2],
            self.model["fit_parameter"]["joint"]["distance"]["NN"][:, 1],
            [0, 1],
        )

        if self.params["correlation_model"] == "unshifted":
            a = self.model["fit_parameter"]["joint"]["correlation"]["nNN"][:, 1]
            b = self.model["fit_parameter"]["joint"]["correlation"]["nNN"][:, 2]
            # mean_corr_nNN = a/(a+b)
            # var_corr_nNN = a*b/((a+b)**2*(a+b+1))
            mean["correlation"]["nNN"] = b
            var["correlation"]["nNN"] = a
        else:
            # mean_corr_nNN, var_corr_nNN = mean_of_trunc_lognorm(self.model['fit_parameter']['joint']['correlation']['nNN'][:,1],self.model['fit_parameter']['joint']['correlation']['nNN'][:,0],[0,1])
            mean["correlation"]["nNN"] = self.model["fit_parameter"]["joint"][
                "correlation"
            ]["nNN"][:, 2]
            var["correlation"]["nNN"] = self.model["fit_parameter"]["joint"][
                "correlation"
            ]["nNN"][:, 1]
        # mean_corr_nNN = 1-mean_corr_nNN

        mean["distance"]["nNN"] = self.model["fit_parameter"]["joint"]["distance"][
            "nNN"
        ][:, 2]
        var["distance"]["nNN"] = self.model["fit_parameter"]["joint"]["distance"][
            "nNN"
        ][:, 1]
        return mean, var


def mean_of_trunc_lognorm(mu, sigma, trunc_loc):

    alpha = (trunc_loc[0] - mu) / sigma
    beta = (trunc_loc[1] - mu) / sigma

    phi = lambda x: 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * x**2)
    psi = lambda x: 1 / 2 * (1 + sp.special.erf(x / np.sqrt(2)))

    trunc_mean = mu + sigma * (phi(alpha) - phi(beta)) / (psi(beta) - psi(alpha))
    trunc_var = np.sqrt(
        sigma**2
        * (
            1
            + (alpha * phi(alpha) - beta * phi(beta)) / (psi(beta) - psi(alpha))
            - ((phi(alpha) - phi(beta)) / (psi(beta) - psi(alpha))) ** 2
        )
    )

    return trunc_mean, trunc_var


def norm_nrg(a_):

    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order="F")
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx] ** 2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order="F")


def scale_down_counts(counts, times=1):
    """
    scales down the whole matrix "counts" by a factor of 2^times
    """

    if times == 0:
        return counts

    assert counts.shape[0] > 8, "No further scaling down allowed"

    cts = np.zeros(tuple((np.array(counts.shape[:2]) / 2).astype("int")) + (3,))
    # print(counts.shape,cts.shape)
    for d in range(counts.shape[2]):
        for i in range(2):
            for j in range(2):
                cts[..., d] += counts[i::2, j::2, d]

    # print(counts.sum(),cts.sum(),' - ',counts[...,0].sum(),cts[...,0].sum())
    return scale_down_counts(cts, times - 1)


def fix_suffix(suffix):
    if suffix:
        if not suffix.startswith("_"):
            suffix = "_" + suffix
    return suffix
