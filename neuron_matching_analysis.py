import time, cv2, tqdm, h5py
from pathlib import Path
import numpy as np
import scipy as sp
import scipy.stats as sstats
from scipy import signal

from matplotlib import (
    pyplot as plt,
    rc,
    colors as mcolors,
    patches as mppatches,
    lines as mplines,
    cm,
)
from matplotlib.widgets import Slider


from .neuron_matching import matching
from .utils import (
    load_data,
    fun_wrapper,
    add_number,
    build_remap_from_shift_and_flow,
    plot_with_confidence,
)


class neuron_matching_analysis(matching):
    """
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
    """

    def plot_fit_results(self, times=1, sv=False, suffix=""):
        """
        TODO:
        * build plot function to include widgets, with which parameter space can be visualized, showing fit results and histogram slice (bars), each (2 subplots: correlation & distance)
        """

        counts = self.scale_counts(times)
        nbins = self.params["nbins"]

        model = self.params["model"]

        # print(joint_hist_norm_dist.sum(axis=1)*self.params['arrays']['distance_step'])

        fig, ax = plt.subplots(3, 2)

        mean, var = self.get_population_mean_and_var()
        model = "correlation"

        for axx, model in zip([ax[0][1], ax[1][0]], ["correlation", "distance"]):

            weight_model = "distance" if model == "correlation" else "correlation"

            axx.plot(
                self.params["arrays"][weight_model],
                mean[model]["NN"],
                "g",
                label="lognorm $\mu$",
            )
            axx.plot(
                self.params["arrays"][weight_model],
                mean[model]["nNN"],
                "r",
                label="$A$ sigm.",
            )
            axx.set_title(f"{model} models")
            # plt.plot(self.params['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['all'][:,2],'g--')
            # plt.plot(self.params['arrays']['correlation'],self.model['fit_parameter']['joint']['distance']['all'][:,5],'r--')
            axx.legend()
            plt.setp(axx, ylim=[0, self.params["arrays"][model][-1]])

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

        ax[0][0].axis("off")

        h = {
            "distance": {},
            "distance_hist": {},
            "correlation": {},
            "correlation_hist": {},
        }

        for axx, model in zip([ax[1][1], ax[2][0]], ["correlation", "distance"]):
            for pop, col in zip(["NN", "nNN"], ["g", "r"]):
                (h[model][pop],) = axx.plot(
                    self.params["arrays"][model], np.zeros(nbins), c=col
                )

                h[f"{model}_hist"][pop] = axx.bar(
                    self.params["arrays"][model],
                    np.zeros(nbins),
                    width=self.params["arrays"][f"{model}_step"],
                    facecolor=col,
                    alpha=0.5,
                )
        plt.setp(ax[1][1], xlim=self.params["arrays"]["correlation"][[0, -1]])
        plt.setp(ax[2][0], xlim=self.params["arrays"]["distance"][[0, -1]])

        ax[2][1].imshow(
            counts[..., 0],
            origin="lower",
            aspect="auto",
            extent=tuple(self.params["arrays"]["correlation"][[0, -1]])
            + tuple(self.params["arrays"]["distance"][[0, -1]]),
        )

        h["distance_marker"] = ax[2][1].axhline(0, c="r")
        h["correlation_marker"] = ax[2][1].axvline(0, c="r")
        plt.setp(
            ax[2][1],
            xlim=self.params["arrays"]["correlation"][[0, -1]],
            ylim=self.params["arrays"]["distance"][[0, -1]],
        )

        slider_h = 0.02
        slider_w = 0.3
        axamp_dist = plt.axes([0.15, 0.85, slider_w, slider_h])
        axamp_corr = plt.axes([0.15, 0.75, slider_w, slider_h])

        counts_norm_along_corr = counts / counts.sum(
            1, keepdims=True
        )  # / self.params['arrays']['distance_step']# * nbins/self.params['neighbor_distance']
        counts_norm_along_dist = counts / counts.sum(
            0, keepdims=True
        )  # / self.params['arrays']['correlation_step']

        def update_plot(ax, n, weight_model="correlation", model_type="joint"):

            # print('somehow, joint model functions are still not working nicely - fits look aweful! is it an issue of displaying, or of construction?')
            model = "distance" if weight_model == "correlation" else "correlation"
            m = 0 if model == "correlation" else 1
            joint_hist = (
                counts_norm_along_corr
                if model == "correlation"
                else counts_norm_along_dist
            )

            fit_fun, _ = self.set_functions(model, model_type)

            max_val = 0
            for p, pop in enumerate(["NN", "nNN"], start=1):

                weight = self.model["p_same"]["single"][weight_model][n]
                if pop == "nNN":
                    weight = 1 - weight

                # print(self.params['arrays'][model])

                fun_pop = (
                    fun_wrapper(
                        fit_fun[pop],
                        self.params["arrays"][model],
                        self.model["fit_parameter"][model_type][model][pop][n, :],
                    )
                    * weight
                    / self.params["arrays"][f"{model}_step"]
                )

                # print(fun_pop)
                # print('parameter:',self.model['fit_parameter'][model_type][model][pop][n,:])
                # print('mean&var:',mean[model][pop][n],var[model][pop][n])

                norm_counts = (
                    np.take(joint_hist[..., p], n, axis=m)
                    / self.params["arrays"][f"{model}_step"]
                )
                # print(f'counts: ',norm_counts.sum())
                h[model][pop].set_ydata(fun_pop)
                max_val = max(max_val, fun_pop.max())

                for rect, height in zip(h[f"{model}_hist"][pop], norm_counts):
                    rect.set_height(height)
                    max_val = max(max_val, height)

                marker_value = self.params["arrays"][weight_model][n]
                if weight_model == "distance":
                    h[f"{weight_model}_marker"].set_ydata(marker_value)
                else:
                    h[f"{weight_model}_marker"].set_xdata(marker_value)

            plt.setp(ax, ylim=[0, max_val * 1.1])
            fig.canvas.draw_idle()

        distance_init = int(nbins * 0.1)
        correlation_init = int(nbins * 0.9)

        self.slider = {}
        self.slider["distance"] = Slider(
            axamp_dist,
            r"d_{com}",
            0,
            nbins - 1,
            valinit=distance_init,
            orientation="horizontal",
            valstep=range(nbins),
        )
        self.slider["correlation"] = Slider(
            axamp_corr,
            r"c_{com}",
            0,
            nbins - 1,
            valinit=correlation_init,
            orientation="horizontal",
            valstep=range(nbins),
        )

        update_plot_correlation = lambda n: update_plot(
            ax[2][0], n=n, weight_model="correlation"
        )
        update_plot_distance = lambda n: update_plot(
            ax[1][1], n=n, weight_model="distance"
        )

        self.slider["distance"].on_changed(update_plot_distance)
        self.slider["correlation"].on_changed(update_plot_correlation)

        update_plot_distance(distance_init)
        update_plot_correlation(correlation_init)

        plt.show(block=False)

    def plot_model(self, model="correlation", sv=False, suffix="", times=1):

        rc("font", size=10)
        rc("axes", labelsize=12)
        rc("xtick", labelsize=8)
        rc("ytick", labelsize=8)

        counts = self.scale_counts(times)
        nbins = self.params["nbins"]

        arrays = self.params["arrays"]
        X, Y = np.meshgrid(arrays["correlation"], arrays["distance"])

        fig = plt.figure(figsize=(7, 4), dpi=150)
        ax_phase = plt.axes([0.3, 0.13, 0.2, 0.4])
        add_number(fig, ax_phase, order=1, offset=[-250, 200])
        # ax_phase.imshow(self.model[key_counts][:,:,0],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',clim=[0,0.25*self.model[key_counts][:,:,0].max()],origin='lower')
        NN_ratio = counts[:, :, 1] / counts[:, :, 0]
        cmap = plt.cm.RdYlGn
        NN_ratio = cmap(NN_ratio)
        NN_ratio[..., -1] = np.minimum(counts[..., 0] / (np.max(counts) / 3.0), 1)

        im_ratio = ax_phase.imshow(
            NN_ratio,
            extent=[0, 1, 0, self.params["neighbor_distance"]],
            aspect="auto",
            clim=[0, 0.5],
            origin="lower",
        )
        nlev = 3
        # col = (np.ones((nlev,3)).T*np.linspace(0,1,nlev)).T
        p_levels = ax_phase.contour(
            X,
            Y,
            self.model["p_same"]["joint"],
            levels=[0.05, 0.5, 0.95],
            colors="k",
            linestyles=[":", "--", "-"],
            linewidths=1.0,
        )
        plt.setp(
            ax_phase,
            xlim=[0, 1],
            ylim=[0, self.params["neighbor_distance"]],
            xlabel="correlation",
            ylabel="distance",
        )
        ax_phase.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=True,
            labelbottom=False,
            labeltop=True,
        )
        ax_phase.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelright=False,
            labelleft=True,
        )
        ax_phase.yaxis.set_label_position("right")
        # ax_phase.xaxis.tick_top()
        ax_phase.xaxis.set_label_coords(0.5, -0.15)
        ax_phase.yaxis.set_label_coords(1.15, 0.5)

        im_ratio.cmap = cmap
        if self.params["correlation_model"] == "unshifted":
            cbaxes = plt.axes([0.41, 0.47, 0.07, 0.03])
            # cbar.ax.set_xlim([0,0.5])
        else:
            cbaxes = plt.axes([0.32, 0.47, 0.07, 0.03])
        cbar = plt.colorbar(im_ratio, cax=cbaxes, orientation="horizontal")
        # cbar.ax.set_xlabel('NN ratio')
        cbar.ax.set_xticks([0, 0.5])
        cbar.ax.set_xticklabels(["nNN", "NN"])

        # cbar.ax.set_xticks(np.linspace(0,1,2))
        # cbar.ax.set_xticklabels(np.linspace(0,1,2))

        ax_dist = plt.axes([0.05, 0.13, 0.2, 0.4])

        ax_dist.barh(
            arrays["distance"],
            counts[..., 0].sum(1).flat,
            self.params["neighbor_distance"] / nbins,
            facecolor="k",
            alpha=0.5,
            orientation="horizontal",
        )
        ax_dist.barh(
            arrays["distance"],
            counts[..., 2].sum(1),
            arrays["distance_step"],
            facecolor="salmon",
            alpha=0.5,
        )
        ax_dist.barh(
            arrays["distance"],
            counts[..., 1].sum(1),
            arrays["distance_step"],
            facecolor="lightgreen",
            alpha=0.3,
        )
        ax_dist.invert_xaxis()
        # h_d_move = ax_dist.bar(arrays['distance'],np.zeros(nbins),arrays['distance_step'],facecolor='k')

        model_distance_all = (
            fun_wrapper(
                self.model["fit_function"]["single"]["distance"]["NN"],
                arrays["distance"],
                self.model["fit_parameter"]["single"]["distance"]["NN"],
            )
            * counts[..., 1].sum()
            + fun_wrapper(
                self.model["fit_function"]["single"]["distance"]["nNN"],
                arrays["distance"],
                self.model["fit_parameter"]["single"]["distance"]["nNN"],
            )
            * counts[..., 2].sum()
        ) * arrays["distance_step"]

        NN_params = self.model["fit_function"]["single"]["distance"]["NN_Nparams"]

        ax_dist.plot(
            fun_wrapper(
                self.model["fit_function"]["single"]["distance"]["all"],
                arrays["distance"],
                self.model["fit_parameter"]["single"]["distance"]["all"],
            )
            * counts[..., 0].sum()
            * arrays["distance_step"],
            arrays["distance"],
            "k:",
        )
        ax_dist.plot(model_distance_all, arrays["distance"], "k")

        ax_dist.plot(
            fun_wrapper(
                self.model["fit_function"]["single"]["distance"]["NN"],
                arrays["distance"],
                self.model["fit_parameter"]["single"]["distance"]["all"][1:NN_params],
            )
            * self.model["fit_parameter"]["single"]["distance"]["all"][0]
            * counts[..., 0].sum()
            * arrays["distance_step"],
            arrays["distance"],
            "g",
        )
        ax_dist.plot(
            fun_wrapper(
                self.model["fit_function"]["single"]["distance"]["NN"],
                arrays["distance"],
                self.model["fit_parameter"]["single"]["distance"]["NN"],
            )
            * counts[..., 1].sum()
            * arrays["distance_step"],
            arrays["distance"],
            "g:",
        )

        ax_dist.plot(
            fun_wrapper(
                self.model["fit_function"]["single"]["distance"]["nNN"],
                arrays["distance"],
                self.model["fit_parameter"]["single"]["distance"]["all"][NN_params:],
            )
            * (1 - self.model["fit_parameter"]["single"]["distance"]["all"][0])
            * counts[..., 0].sum()
            * arrays["distance_step"],
            arrays["distance"],
            "r",
        )
        ax_dist.plot(
            fun_wrapper(
                self.model["fit_function"]["single"]["distance"]["nNN"],
                arrays["distance"],
                self.model["fit_parameter"]["single"]["distance"]["nNN"],
            )
            * counts[..., 2].sum()
            * arrays["distance_step"],
            arrays["distance"],
            "r",
            linestyle=":",
        )
        ax_dist.set_ylim([0, self.params["neighbor_distance"]])
        ax_dist.set_xlabel("counts")
        ax_dist.spines["left"].set_visible(False)
        ax_dist.spines["top"].set_visible(False)
        ax_dist.tick_params(
            axis="y",
            which="both",
            left=False,
            right=True,
            labelright=False,
            labelleft=False,
        )

        ax_corr = plt.axes([0.3, 0.63, 0.2, 0.325])
        ax_corr.bar(
            arrays["correlation"],
            counts[..., 0].sum(0).flat,
            1 / nbins,
            facecolor="k",
            alpha=0.5,
        )
        ax_corr.bar(
            arrays["correlation"],
            counts[..., 2].sum(0),
            1 / nbins,
            facecolor="salmon",
            alpha=0.5,
        )
        ax_corr.bar(
            arrays["correlation"],
            counts[..., 1].sum(0),
            1 / nbins,
            facecolor="lightgreen",
            alpha=0.3,
        )

        f_NN = fun_wrapper(
            self.model["fit_function"]["single"]["correlation"]["NN"],
            arrays["correlation"],
            self.model["fit_parameter"]["single"]["correlation"]["NN"],
        )
        f_nNN = fun_wrapper(
            self.model["fit_function"]["single"]["correlation"]["nNN"],
            arrays["correlation"],
            self.model["fit_parameter"]["single"]["correlation"]["nNN"],
        )
        model_fp_correlation_all = (
            f_NN * counts[..., 1].sum() + f_nNN * counts[..., 2].sum()
        ) * arrays["correlation_step"]
        # ax_corr.plot(arrays['correlation'],fun_wrapper(self.model['fit_function']['correlation']['all'],arrays['correlation'],self.model['fit_parameter']['single']['correlation']['all'])*counts[...,0].sum()*arrays['correlation_step'],'k')
        ax_corr.plot(arrays["correlation"], model_fp_correlation_all, "k")

        ax_corr.plot(
            arrays["correlation"],
            fun_wrapper(
                self.model["fit_function"]["single"]["correlation"]["NN"],
                arrays["correlation"],
                self.model["fit_parameter"]["single"]["correlation"]["NN"],
            )
            * counts[..., 1].sum()
            * arrays["correlation_step"],
            "g",
        )

        # ax_corr.plot(arrays['correlation'],fun_wrapper(self.model['fit_function']['correlation']['nNN'],arrays['correlation'],self.model['fit_parameter']['single']['correlation']['all'][3:])*(1-self.model['fit_parameter']['single']['correlation']['all'][0])*counts[...,0].sum()*arrays['correlation_step'],'r')
        ax_corr.plot(
            arrays["correlation"],
            fun_wrapper(
                self.model["fit_function"]["single"]["correlation"]["nNN"],
                arrays["correlation"],
                self.model["fit_parameter"]["single"]["correlation"]["nNN"],
            )
            * counts[..., 2].sum()
            * arrays["correlation_step"],
            "r",
        )

        ax_corr.set_ylabel("counts")
        ax_corr.set_xlim([0, 1])
        ax_corr.spines["right"].set_visible(False)
        ax_corr.spines["top"].set_visible(False)
        ax_corr.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=False,
            labeltop=False,
        )

        # ax_parameter =
        p_steps, rates = self.calculate_RoC(100)

        ax_cum = plt.axes([0.675, 0.7, 0.3, 0.225])
        add_number(fig, ax_cum, order=2)

        uncertain = {}
        idx_low = np.where(p_steps > 0.05)[0][0]
        idx_high = np.where(p_steps < 0.95)[0][-1]
        for key in rates["cumfrac"].keys():

            if key == "joint" and (rates["cumfrac"][key][idx_low] > 0.01) & (
                rates["cumfrac"][key][idx_high] < 0.99
            ):
                ax_cum.fill_between(
                    [rates["cumfrac"][key][idx_low], rates["cumfrac"][key][idx_high]],
                    [0, 0],
                    [1, 1],
                    facecolor="y",
                    alpha=0.5,
                )
            uncertain[key] = (
                rates["cumfrac"][key][idx_high] - rates["cumfrac"][key][idx_low]
            )  # /(1-rates['cumfrac'][key][idx_high+1])

        ax_cum.plot([0, 1], [0.05, 0.05], "b", linestyle=":")
        ax_cum.plot([0.5, 1], [0.95, 0.95], "b", linestyle="-")

        ax_cum.plot(rates["cumfrac"]["joint"], p_steps[:-1], "grey", label="Joint")
        # ax_cum.plot(rates['cumfrac']['distance'],p_steps[:-1],'k',label='Distance')
        if self.params["correlation_model"] == "unshifted":
            ax_cum.plot(
                rates["cumfrac"]["correlation"],
                p_steps[:-1],
                "lightgrey",
                label="Correlation",
            )
        ax_cum.set_ylabel("$p_{same}$")
        ax_cum.set_xlabel("cumulative fraction")
        # ax_cum.legend(fontsize=10,frameon=False)
        ax_cum.spines["right"].set_visible(False)
        ax_cum.spines["top"].set_visible(False)

        ax_uncertain = plt.axes([0.75, 0.825, 0.05, 0.1])
        # ax_uncertain.bar(2,uncertain['distance'],facecolor='k')
        ax_uncertain.bar(3, uncertain["joint"], facecolor="k")
        if self.params["correlation_model"] == "unshifted":
            ax_uncertain.bar(1, uncertain["correlation"], facecolor="lightgrey")
            ax_uncertain.set_xlim([1.5, 3.5])
            ax_uncertain.set_xticks(range(1, 4))
            ax_uncertain.set_xticklabels(
                ["Corr.", "Dist.", "Joint"], rotation=60, fontsize=10
            )
        else:
            ax_uncertain.set_xticks([])
            ax_uncertain.set_xlim([2.5, 3.5])
            # ax_uncertain.set_xticklabels(['Dist.','Joint'],rotation=60,fontsize=10)
            ax_uncertain.set_xticklabels([])
        ax_uncertain.set_ylim([0, 0.2])
        ax_uncertain.spines["right"].set_visible(False)
        ax_uncertain.spines["top"].set_visible(False)
        ax_uncertain.set_title("uncertain fraction", fontsize=10)

        # ax_rates = plt.axes([0.83,0.6,0.15,0.3])
        # ax_rates.plot(rates['fp']['joint'],p_steps[:-1],'r',label='false positive rate')
        # ax_rates.plot(rates['tp']['joint'],p_steps[:-1],'g',label='true positive rate')

        # ax_rates.plot(rates['fp']['distance'],p_steps[:-1],'r--')
        # ax_rates.plot(rates['tp']['distance'],p_steps[:-1],'g--')

        # ax_rates.plot(rates['fp']['correlation'],p_steps[:-1],'r:')
        # ax_rates.plot(rates['tp']['correlation'],p_steps[:-1],'g:')
        # ax_rates.legend()
        # ax_rates.set_xlabel('rate')
        # ax_rates.set_ylabel('$p_{same}$')

        idx = np.where(p_steps == 0.3)[0]

        ax_RoC = plt.axes([0.675, 0.13, 0.125, 0.3])
        add_number(fig, ax_RoC, order=3)
        ax_RoC.plot(rates["fp"]["joint"], rates["tp"]["joint"], "k", label="Joint")
        # ax_RoC.plot(rates['fp']['distance'],rates['tp']['distance'],'k',label='Distance')
        if self.params["correlation_model"] == "unshifted":
            ax_RoC.plot(
                rates["fp"]["correlation"],
                rates["tp"]["correlation"],
                "lightgrey",
                label="Correlation",
            )
        ax_RoC.plot(rates["fp"]["joint"][idx], rates["tp"]["joint"][idx], "kx")
        # ax_RoC.plot(rates['fp']['distance'][idx],rates['tp']['distance'][idx],'kx')
        if self.params["correlation_model"] == "unshifted":
            ax_RoC.plot(
                rates["fp"]["correlation"][idx], rates["tp"]["correlation"][idx], "kx"
            )
        ax_RoC.set_ylabel("true positive")
        ax_RoC.set_xlabel("false positive")
        ax_RoC.spines["right"].set_visible(False)
        ax_RoC.spines["top"].set_visible(False)
        ax_RoC.set_xlim([0, 0.1])
        ax_RoC.set_ylim([0.6, 1])

        ax_fp = plt.axes([0.925, 0.13, 0.05, 0.1])
        # ax_fp.bar(2,rates['fp']['distance'][idx],facecolor='k')
        ax_fp.bar(3, rates["fp"]["joint"][idx], facecolor="k")
        ax_fp.set_xticks([])

        if self.params["correlation_model"] == "unshifted":
            ax_fp.bar(1, rates["fp"]["correlation"][idx], facecolor="lightgrey")
            ax_fp.set_xlim([1.5, 3.5])
            ax_fp.set_xticks(range(1, 4))
            ax_fp.set_xticklabels(["Corr.", "Dist.", "Joint"], rotation=60, fontsize=10)
        else:
            ax_fp.set_xticks([])
            ax_fp.set_xlim([2.5, 3.5])
            # ax_fp.set_xticklabels(['Dist.','Joint'],rotation=60,fontsize=10)
            ax_fp.set_xticklabels([])

        ax_fp.set_ylim([0, 0.05])
        ax_fp.spines["right"].set_visible(False)
        ax_fp.spines["top"].set_visible(False)
        ax_fp.set_ylabel("false pos.", fontsize=10)

        ax_tp = plt.axes([0.925, 0.33, 0.05, 0.1])
        add_number(fig, ax_tp, order=4, offset=[-100, 25])
        # ax_tp.bar(2,rates['tp']['distance'][idx],facecolor='k')
        ax_tp.bar(3, rates["tp"]["joint"][idx], facecolor="k")
        ax_tp.set_xticks([])
        if self.params["correlation_model"] == "unshifted":
            ax_tp.bar(1, rates["tp"]["correlation"][idx], facecolor="lightgrey")
            ax_tp.set_xlim([1.5, 3.5])
        else:
            ax_tp.set_xlim([2.5, 3.5])
            ax_fp.set_xticklabels([])
        ax_tp.set_ylim([0.7, 1])
        ax_tp.spines["right"].set_visible(False)
        ax_tp.spines["top"].set_visible(False)
        # ax_tp.set_ylabel('fraction',fontsize=10)
        ax_tp.set_ylabel("true pos.", fontsize=10)

        # plt.tight_layout()
        plt.show(block=False)
        if sv:
            ext = "png"
            path = Path(
                self.params["pathMouse"],
                f"Sheintuch_matching_{self.params['correlation_model']}{suffix}.{ext}",
            )
            plt.savefig(path, format=ext, dpi=150)
        # return
        # ax_cvc = plt.axes([0.65,0.1,0.2,0.4])
        # idx = self.data_cross['fp_corr_max']>0
        # ax_cvc.scatter(self.data_cross['fp_corr_max'][idx].toarray().flat,self.data_cross['fp_corr'][idx].toarray().flat,c='k',marker='.')
        # ax_cvc.plot([0,1],[0,1],'r--')
        # ax_cvc.set_xlim([0,1])
        # ax_cvc.set_ylim([0,1])
        # ax_cvc.set_xlabel('shifted correlation')
        # ax_cvc.set_ylabel('unshifted correlation')

        # plt.show(block=False)
        # return

    def plot_count_histogram(self, times=0):

        counts = self.scale_counts(times)
        nbins = self.params["nbins"]

        arrays = self.params["arrays"]

        plt.figure(figsize=(12, 9))
        # plt.subplot(221)
        ax = plt.subplot(221, projection="3d")
        X, Y = np.meshgrid(arrays["correlation"], arrays["distance"])
        NN_ratio = counts[:, :, 1] / counts[:, :, 0]
        cmap = plt.cm.RdYlGn
        NN_ratio = cmap(NN_ratio)
        ax.plot_surface(X, Y, counts[:, :, 0], facecolors=NN_ratio)
        ax.view_init(30, -120)
        ax.set_xlabel("footprint correlation", fontsize=14)
        ax.set_ylabel("distance", fontsize=14)
        ax.set_zlabel("# pairs", fontsize=14)

        # plt.imshow(counts[...,0],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',clim=[0,counts[...,0].max()],origin='lower')
        # nlev = 3
        # col = (np.ones((nlev,3)).T*np.linspace(0,1,nlev)).T
        # p_levels = plt.contour(X,Y,self.model['p_same']['joint'],levels=[0.05,0.5,0.95],colors=col)
        # plt.colorbar(p_levels)
        # plt.imshow(self.model[key_counts][...,0],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',origin='lower')
        ax2 = plt.subplot(222)
        ax2.imshow(
            counts[..., 0],
            extent=[0, 1, 0, self.params["neighbor_distance"]],
            aspect="auto",
            origin="lower",
        )
        ax2.set_title("all counts", y=1, pad=-14, color="white", fontweight="bold")

        ax3 = plt.subplot(223)
        ax3.imshow(
            counts[..., 1],
            extent=[0, 1, 0, self.params["neighbor_distance"]],
            aspect="auto",
            origin="lower",
        )
        ax3.set_title(
            "nearest neighbour counts", y=1, pad=-14, color="white", fontweight="bold"
        )

        ax4 = plt.subplot(224)
        ax4.imshow(
            counts[..., 2],
            extent=[0, 1, 0, self.params["neighbor_distance"]],
            aspect="auto",
            origin="lower",
        )
        ax4.set_title(
            "non-nearest neighbour counts",
            y=1,
            pad=-14,
            color="white",
            fontweight="bold",
        )
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

    def plot_p_same(self, times=0, animate=False):

        counts = self.scale_counts(times)
        nbins = counts.shape[0]
        self.update_bins(nbins)

        X, Y = np.meshgrid(
            self.params["arrays"]["correlation"], self.params["arrays"]["distance"]
        )

        fig = plt.figure(figsize=(8, 10))
        ax_corr = [fig.add_subplot(4, 2, 1), fig.add_subplot(4, 2, 3)]
        ax_dist = [fig.add_subplot(4, 2, 2), fig.add_subplot(4, 2, 4)]

        for m, (ax, model) in enumerate(
            zip([ax_corr, ax_dist], ["correlation", "distance"])
        ):

            fit_fun, _ = self.set_functions(model, "single")
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

            fun_total = fun_NN + fun_nNN

            ax[0].plot(
                self.params["arrays"][model],
                counts[..., 0].sum(m)
                / counts[..., 0].sum()
                / self.params["arrays"][f"{model}_step"],
                "g",
                label="data",
            )
            # for p,pop in enumerate(['NN','nNN'],start=1):
            ax[0].plot(self.params["arrays"][model], fun_NN, "k:")
            ax[0].plot(self.params["arrays"][model], fun_nNN, "k:")
            ax[0].plot(self.params["arrays"][model], fun_total, "k--", label="model")
            plt.setp(ax[0], xlim=[0, self.params["arrays"][model][-1]], yticks=[])
            ax[0].spines[["top", "right"]].set_visible(False)
            ax[0].legend()

            ax[0].set_title(f"single model: {model}")

            ax[1].plot(
                self.params["arrays"][model], self.model["p_same"]["single"][model], "r"
            )
            plt.setp(
                ax[1],
                xlim=[0, self.params["arrays"][model][-1]],
                xlabel=model,
                ylabel="$p_{same}$",
            )
            ax[1].spines[["top", "right"]].set_visible(False)
            # plt.show()

        ax_p_same = fig.add_subplot(2, 2, 4, projection="3d")
        prob = ax_p_same.plot_surface(X, Y, self.model["p_same"]["joint"], cmap="jet")
        prob.set_clim(0, 1)
        ax_p_same.set_zlim([0, 1])
        ax_p_same.set_xlabel("correlation")
        ax_p_same.set_ylabel("distance")
        ax_p_same.set_zlabel("$p_{same}$")

        # ax = plt.subplot(224,projection='3d')
        ax_counts = fig.add_subplot(2, 2, 3, projection="3d")
        prob = ax_counts.plot_surface(X, Y, counts[..., 0], cmap="jet")
        # prob = ax.bar3d(X.flatten(),Y.flatten(),np.zeros((nbins,nbins)).flatten(),np.ones((nbins,nbins)).flatten()*self.params['arrays']['correlation_step'],np.ones((nbins,nbins)).flatten()*self.params['arrays']['distance_step'],self.model[key_counts][...,0].flatten(),cmap='jet')
        # prob.set_clim(0,1)
        ax_counts.set_xlabel("correlation")
        ax_counts.set_ylabel("distance")
        ax_counts.set_zlabel("counts")
        ax_counts.set_title(f"joint model")
        plt.tight_layout()

        def rotate_view(i, axes, fixed_angle=30):
            for axx in axes:
                axx.view_init(fixed_angle, (i * 2) % 360)
            # return ax

        rotate_view(100, [ax_p_same, ax_counts], fixed_angle=30)
        plt.show(block=False)

        print("proper weighting of bin counts")
        print("smoothing by gaussian")

    def plot_matches(
        self, s_ref, s, color_s_ref="coral", color_s="lightgreen", level=[0.05]
    ):
        """

        TODO:
        * rewrite function, such that it calculates and plots footprint matching for 2 arbitrary sessions (s,s_ref)
        * write function description and document code properly
        * optimize plotting, so it doesn't take forever
        """

        matched1 = self.results["assignments"]
        matched_c = np.all(
            np.isfinite(self.results["assignments"][:, (s_ref, s)]), axis=1
        )
        matched1 = self.results["assignments"][matched_c, s_ref].astype(int)
        matched2 = self.results["assignments"][matched_c, s].astype(int)
        # print('matched: ',matched_c.sum())
        nMatched = matched_c.sum()

        non_matched_c = np.isfinite(self.results["assignments"][:, s_ref]) & np.isnan(
            self.results["assignments"][:, s]
        )
        non_matched1 = self.results["assignments"][non_matched_c, s_ref].astype(int)
        # print('non_matched 1: ',non_matched_c.sum())
        nNonMatched1 = non_matched_c.sum()

        non_matched_c = np.isnan(self.results["assignments"][:, s_ref]) & np.isfinite(
            self.results["assignments"][:, s]
        )
        non_matched2 = self.results["assignments"][non_matched_c, s].astype(int)
        # print('non_matched 1: ',non_matched_c.sum())
        nNonMatched2 = non_matched_c.sum()

        print("plotting...")
        t_start = time.time()

        def load_and_align(s):

            ld = load_data(self.paths["neuron_detection"][s])
            A = ld["A"]
            Cn = ld["Cn"].T

            if "remap" in self.data[s].keys():

                x_remap, y_remap = build_remap_from_shift_and_flow(
                    self.params["dims"],
                    self.data[s]["remap"]["shift"],
                    self.data[s]["remap"]["flow"],
                )

                Cn = cv2.remap(
                    Cn,  # reshape image to original dimensions
                    x_remap,
                    y_remap,  # apply reverse identified shift and flow
                    cv2.INTER_CUBIC,
                )

                A = sp.sparse.hstack(
                    [
                        sp.sparse.csc_matrix(  # cast results to sparse type
                            cv2.remap(
                                fp.reshape(
                                    self.params["dims"]
                                ),  # reshape image to original dimensions
                                x_remap,
                                y_remap,  # apply reverse identified shift and flow
                                cv2.INTER_CUBIC,
                            ).reshape(
                                -1, 1
                            )  # reshape back to allow sparse storage
                        )
                        for fp in A.toarray().T  # loop through all footprints
                    ]
                )

            lp, hp = np.nanpercentile(Cn, [25, 99])
            Cn -= lp
            Cn /= hp - lp

            Cn = np.clip(Cn, a_min=0, a_max=1)

            return A, Cn

        Cn = np.zeros(self.params["dims"] + (3,))

        A_ref, Cn[..., 0] = load_and_align(s_ref)
        A, Cn[..., 1] = load_and_align(s)

        plt.figure(figsize=(15, 12))

        ax_matches = plt.subplot(111)
        ax_matches.imshow(np.transpose(Cn, (1, 0, 2)))  # , origin='lower')

        [
            ax_matches.contour(
                np.reshape(a.todense(), self.params["dims"]).T,
                levels=level,
                colors=color_s_ref,
                linewidths=2,
            )
            for a in A_ref[:, matched1].T
        ]
        [
            ax_matches.contour(
                np.reshape(a.todense(), self.params["dims"]).T,
                levels=level,
                colors=color_s_ref,
                linewidths=2,
                linestyles="--",
            )
            for a in A_ref[:, non_matched1].T
        ]

        print("first half done: %5.3f" % (time.time() - t_start))
        [
            ax_matches.contour(
                np.reshape(a.todense(), self.params["dims"]).T,
                levels=level,
                colors=color_s,
                linewidths=2,
            )
            for a in A[:, matched2].T
        ]
        [
            ax_matches.contour(
                np.reshape(a.todense(), self.params["dims"]).T,
                levels=level,
                colors=color_s,
                linewidths=2,
                linestyles="--",
            )
            for a in A[:, non_matched2].T
        ]

        ax_matches.legend(
            handles=[
                mppatches.Patch(color=color_s_ref, label="reference session"),
                mppatches.Patch(color=color_s, label="session"),
                mplines.Line2D(
                    [0], [0], color="k", linestyle="-", label=f"matched ({nMatched})"
                ),
                mplines.Line2D(
                    [0],
                    [0],
                    color="k",
                    linestyle="--",
                    label=f"non-matched ({nNonMatched1}/{nNonMatched2})",
                ),
            ],
            loc="lower left",
            framealpha=0.9,
        )
        plt.setp(ax_matches, xlabel="x [px]", ylabel="y [px]")
        print("done. time taken: %5.3f" % (time.time() - t_start))
        plt.show(block=False)

    def plot_neuron_numbers(self):

        ### plot occurence of neurons
        colors = [(1, 0, 0, 0), (1, 0, 0, 1)]
        RedAlpha = mcolors.LinearSegmentedColormap.from_list("RedAlpha", colors, N=2)
        colors = [(0, 0, 0, 0), (0, 0, 0, 1)]
        BlackAlpha = mcolors.LinearSegmentedColormap.from_list(
            "BlackAlpha", colors, N=2
        )

        idxes = np.ones(self.results["assignments"].shape, "bool")

        fig = plt.figure(figsize=(5, 8))
        ### plot occurence of neurons
        ax_oc = fig.add_subplot([0.1, 0.4, 0.6, 0.5])
        # ax_oc2 = ax_oc.twinx()
        ax_oc.imshow(
            (~np.isnan(self.results["assignments"])) & idxes,
            cmap=BlackAlpha,
            aspect="auto",
            interpolation="none",
        )
        # ax_oc2.imshow((~np.isnan(self.results['assignments']))&(~idxes),cmap=RedAlpha,aspect='auto')
        # ax_oc.imshow(self.results['p_matched'],cmap='binary',aspect='auto')
        ax_oc.set_xlabel("session")
        ax_oc.set_ylabel("neuron ID")

        ax_per_session = fig.add_subplot([0.1, 0.1, 0.6, 0.3])
        ax_per_session.plot(
            (~np.isnan(self.results["assignments"])).sum(0), "ko", markersize=3
        )

        ax_per_cluster = fig.add_subplot([0.7, 0.4, 0.25, 0.5])
        nC, nS = self.results["assignments"].shape
        ax_per_cluster.plot(
            (~np.isnan(self.results["assignments"])).sum(1),
            np.linspace(0, nC, nC),
            "ko",
            markersize=3,
        )

        nS = ((~np.isnan(self.results["assignments"])).sum(0) > 0).sum()
        plt.setp(
            ax_per_cluster, xlim=[0, nS], ylim=[nC, 0], yticks=[], xlabel="# sessions"
        )

        ax_per_cluster_histogram = fig.add_subplot([0.7, 0.9, 0.25, 0.075])
        ax_per_cluster_histogram.hist(
            (~np.isnan(self.results["assignments"])).sum(1),
            np.linspace(0, nS, nS),
            color="k",
            cumulative=True,
            density=True,
            histtype="step",
        )
        plt.setp(ax_per_cluster_histogram, xlim=[0, nS])

        plt.tight_layout()
        plt.show(block=False)

        # ext = 'png'
        # path = pathcat([self.params['pathMouse'],'Figures/Sheintuch_registration_score_stats_raw_%s_%s.%s'%(self.params['correlation_model'],suffix,ext)])
        # plt.savefig(path,format=ext,dpi=300)

    def plot_registration(self, suffix="", sv=False):

        rc("font", size=10)
        rc("axes", labelsize=12)
        rc("xtick", labelsize=8)
        rc("ytick", labelsize=8)

        # fileName = 'clusterStats_%s.pkl'%dataSet
        idxes = np.ones(self.results["assignments"].shape, "bool")

        fileName = f"cluster_stats_{suffix}.pkl"
        pathLoad = Path(self.paths["data"], fileName)
        if pathLoad.exists():
            ld = load_data(pathLoad)
            if ~np.all(np.isnan(ld["SNR_comp"])):
                idxes = (
                    (ld["SNR_comp"] > 2)
                    & (ld["r_values"] > 0)
                    & (ld["cnn_preds"] > 0.3)
                    & (ld["firingrate"] > 0)
                )

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

        plt.figure(figsize=(7, 1.5))
        ax_sc1 = plt.axes([0.1, 0.3, 0.35, 0.65])

        ax = ax_sc1.twinx()
        ax.hist(
            self.results["p_matched"][idxes, 1].flat,
            np.linspace(0, 1, 51),
            facecolor="tab:red",
            alpha=0.3,
        )
        # ax.invert_yaxis()
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = ax_sc1.twiny()
        ax.hist(
            self.results["p_matched"][idxes, 0].flat,
            np.linspace(0, 1, 51),
            facecolor="tab:blue",
            orientation="horizontal",
            alpha=0.3,
        )
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax_sc1.plot(
            self.results["p_matched"][idxes, 1].flat,
            self.results["p_matched"][idxes, 0].flat,
            ".",
            markeredgewidth=0,
            color="k",
            markersize=1,
        )
        ax_sc1.plot([0, 1], [0, 1], "--", color="tab:red", lw=0.5)
        ax_sc1.plot([0, 0.45], [0.5, 0.95], "--", color="tab:orange", lw=1)
        ax_sc1.plot([0.45, 1], [0.95, 0.95], "--", color="tab:orange", lw=1)
        ax_sc1.set_ylabel("$p^{\\asterisk}$")
        ax_sc1.set_xlabel("max($p\\backslash p^{\\asterisk}$)")
        ax_sc1.spines["top"].set_visible(False)
        ax_sc1.spines["right"].set_visible(False)

        # match vs max
        # idxes &= idx_pm

        # avg matchscore per cluster, min match score per cluster, ...
        ax_sc2 = plt.axes([0.6, 0.3, 0.35, 0.65])
        # plt.hist(np.nanmean(self.results['p_matched'],1),np.linspace(0,1,51))
        ax = ax_sc2.twinx()
        ax.hist(
            np.nanmin(self.results["p_matched"][..., 0], 1),
            np.linspace(0, 1, 51),
            facecolor="tab:red",
            alpha=0.3,
        )
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = ax_sc2.twiny()
        ax.hist(
            np.nanmean(self.results["p_matched"][..., 0], axis=1),
            np.linspace(0, 1, 51),
            facecolor="tab:blue",
            orientation="horizontal",
            alpha=0.3,
        )
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax_sc2.plot(
            np.nanmin(self.results["p_matched"][..., 0], 1),
            np.nanmean(self.results["p_matched"][..., 0], axis=1),
            ".",
            markeredgewidth=0,
            color="k",
            markersize=1,
        )
        ax_sc2.set_xlabel("min($p^{\\asterisk}$)")
        ax_sc2.set_ylabel("$\left\langle p^{\\asterisk} \\right\\rangle$")
        ax_sc2.spines["top"].set_visible(False)
        ax_sc2.spines["right"].set_visible(False)

        ### plot positions of neurons
        plt.tight_layout()
        plt.show(block=False)

        if sv:
            ext = "png"
            path = Path(
                self.params["pathMouse"],
                f"Figures/Sheintuch_registration_score_stats_{self.params['correlation_model']}_{suffix}.{ext}",
            )
            plt.savefig(path, format=ext, dpi=300)

    def plot_cluster_stats(self):

        print("### Plotting ROI and cluster statistics of matching ###")

        # idx_unsure = cluster.stats['match_score'][...,0]<(cluster.stats['match_score'][...,1]+0.5)

        nC, nSes = self.results["assignments"].shape
        active = ~np.isnan(self.results["assignments"])

        idx_unsure = self.results["p_matched"][..., 0] < 0.95

        fig = plt.figure(figsize=(7, 4), dpi=300)

        nDisp = 20
        ax_3D = plt.subplot(221, projection="3d")
        # ax_3D.set_position([0.2,0.5,0.2,0.3])
        ##fig.gca(projection='3d')
        # a = np.arange(30)
        # for c in range(30):

        n_arr = np.random.choice(np.where(active.sum(1) > 10)[0], nDisp)
        # n_arr = np.random.randint(0,cluster.meta['nC'],nDisp)
        cmap = cm.get_cmap("tab20")
        ax_3D.set_prop_cycle(color=cmap.colors)
        # print(self.results['cm'][n_arr,:,0],self.results['cm'][n_arr,:,0].shape)
        for n in n_arr:
            ax_3D.scatter(
                self.results["cm"][n, :, 0],
                self.results["cm"][n, :, 1],
                np.arange(nSes),
                s=0.5,
            )  # linewidth=2)
        ax_3D.set_xlim([0, 512 * self.params["pxtomu"]])
        ax_3D.set_ylim([0, 512 * self.params["pxtomu"]])

        ax_3D.set_xlabel("x [$\mu$m]")
        ax_3D.set_ylabel("y [$\mu$m]")
        ax_3D.invert_zaxis()
        # ax_3D.zaxis._axinfo['label']['space_factor'] = 2.8
        ax_3D.set_zlabel("session")

        ax_proxy = plt.axes([0.1, 0.925, 0.01, 0.01])
        add_number(fig, ax_proxy, order=1, offset=[-50, 25])
        ax_proxy.spines[["top", "right", "bottom", "left"]].set_visible(False)
        # pl_dat.remove_frame(ax_proxy)
        ax_proxy.set_xticks([])
        ax_proxy.set_yticks([])

        # ax = plt.subplot(243)
        ax = plt.axes([0.65, 0.65, 0.125, 0.275])
        add_number(fig, ax, order=2, offset=[-50, 25])
        dx = np.diff(self.results["cm"][..., 0], axis=1) * self.params["pxtomu"]
        ax.hist(
            dx.flatten(), np.linspace(-10, 10, 101), facecolor="tab:blue", alpha=0.5
        )
        ax.hist(
            dx[idx_unsure[:, 1:]].flatten(),
            np.linspace(-10, 10, 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        ax.set_xlabel("$\Delta$x [$\mu$m]")
        ax.set_ylabel("density")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_ylim([0, 10000])
        ax.set_yticks([])

        # ax = plt.subplot(244)
        ax = plt.axes([0.8, 0.65, 0.125, 0.275])
        dy = np.diff(self.results["cm"][..., 1], axis=1) * self.params["pxtomu"]
        ax.hist(
            dy.flatten(), np.linspace(-10, 10, 101), facecolor="tab:blue", alpha=0.5
        )
        ax.hist(
            dy[idx_unsure[:, 1:]].flatten(),
            np.linspace(-10, 10, 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        ax.set_xlabel("$\Delta$y [$\mu$m]")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_ylim([0, 10000])
        ax.set_yticks([])

        ax = plt.axes([0.73, 0.85, 0.075, 0.05])
        ax.hist(
            dx.flatten(), np.linspace(-10, 10, 101), facecolor="tab:blue", alpha=0.5
        )
        ax.hist(
            dx[idx_unsure[:, 1:]].flatten(),
            np.linspace(-10, 10, 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        # ax.set_xlabel('$\Delta$x [$\mu$m]',fontsize=10)
        ax.set_yticks([])
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_ylim([0, 500])

        ax = plt.axes([0.88, 0.85, 0.075, 0.05])
        ax.hist(
            dy.flatten(), np.linspace(-10, 10, 101), facecolor="tab:blue", alpha=0.5
        )
        ax.hist(
            dy[idx_unsure[:, 1:]].flatten(),
            np.linspace(-10, 10, 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        # ax.set_xlabel('$\Delta$y [$\mu$m]',fontsize=10)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_ylim([0, 500])
        ax.set_yticks([])

        ROI_diff = np.full((nC, nSes, 2), np.NaN)
        com_ref = np.full((nC, 2), np.NaN)
        for n in range(nC):
            s_ref = np.where(active[n, :])[0]
            if len(s_ref) > 0:
                com_ref[n, :] = self.results["cm"][n, s_ref[0], :]
                ROI_diff[n, : nSes - s_ref[0], :] = (
                    self.results["cm"][n, s_ref[0] :, :] - com_ref[n, :]
                )
                # print('neuron %d, first session: %d, \tposition: (%.2f,%.2f)'%(n,s_ref[0],com_ref[n,0],com_ref[n,1]))

        ax_mv = plt.axes([0.1, 0.11, 0.35, 0.3])
        add_number(fig, ax_mv, order=3, offset=[-75, 50])
        # ROI_diff = (self.results['cm'].transpose(1,0,2)-self.results['cm'][:,0,:]).transpose(1,0,2)#*cluster.para['pxtomu']
        # for n in range(nC):
        # ROI_diff[n,:]
        # ROI_diff = (self.results['cm'].transpose(1,0,2)-com_ref).transpose(1,0,2)#*cluster.para['pxtomu']
        ROI_diff_abs = np.array(
            [np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) for x in ROI_diff]
        )
        # ROI_diff_abs[~cluster.status[...,1]] = np.NaN

        for n in n_arr:
            ax_mv.plot(
                range(nSes), ROI_diff_abs[n, :], linewidth=0.5, color=[0.6, 0.6, 0.6]
            )
        ax_mv.plot(
            range(nSes),
            ROI_diff_abs[n, :] * np.NaN,
            linewidth=0.5,
            color=[0.6, 0.6, 0.6],
            label="displacement",
        )

        plot_with_confidence(
            ax_mv,
            range(nSes),
            np.nanmean(ROI_diff_abs, 0),
            np.nanstd(ROI_diff_abs, 0),
            col="tab:red",
            ls="-",
            label="average",
        )
        ax_mv.set_xlabel("session")
        ax_mv.set_ylabel("$\Delta$d [$\mu$m]")
        ax_mv.set_ylim([0, 11])
        ax_mv.legend(fontsize=10)
        ax_mv.spines[["top", "right"]].set_visible(False)

        idx_c_unsure = idx_unsure.any(1)

        ax_mv_max = plt.axes([0.6, 0.11, 0.35, 0.325])
        add_number(fig, ax_mv_max, order=4, offset=[-75, 50])
        ROI_max_mv = np.nanmax(ROI_diff_abs, 1)
        ax_mv_max.hist(
            ROI_max_mv,
            np.linspace(0, 20, 41),
            facecolor="tab:blue",
            alpha=0.5,
            label="certain",
        )
        ax_mv_max.hist(
            ROI_max_mv[idx_c_unsure],
            np.linspace(0, 20, 41),
            facecolor="tab:red",
            alpha=0.5,
            label="uncertain",
        )
        ax_mv_max.set_xlabel("max($\Delta$d) [$\mu$m]")
        ax_mv_max.set_ylabel("# cluster")
        ax_mv_max.legend(fontsize=10)

        ax_mv_max.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.show(block=False)

        # if sv:
        # pl_dat.save_fig('ROI_positions')

    def plot_match_statistics(self):

        print("### Plotting matching score statistics ###")

        print(
            "now add example how to calculate footprint correlation(?), sketch how to fill cost-matrix"
        )

        s = 15
        margins = 20

        nC, nSes = self.results["assignments"].shape
        sessions_bool = np.ones(nSes, "bool")
        active = ~np.isnan(self.results["assignments"])

        D_ROIs = sp.spatial.distance.squareform(
            sp.spatial.distance.pdist(self.results["cm"][:, s, :])
        )
        np.fill_diagonal(D_ROIs, np.NaN)

        idx_dense = np.where(
            (np.sum(D_ROIs < margins, 1) <= 8) & active[:, s] & active[:, s + 1]
        )[0]
        c = np.random.choice(idx_dense)
        # c = idx_dense[0]
        # c = 375
        # print(c)
        # print(cluster.IDs['neuronID'][c,s,1])
        n = int(self.results["assignments"][c, s])
        # n = 328
        print(c, n)
        fig = plt.figure(figsize=(7, 4), dpi=150)
        props = dict(boxstyle="round", facecolor="w", alpha=0.8)

        ## plot ROIs from a single session

        # c = np.where(cluster.IDs['neuronID'][:,s,1] == n)[0][0]
        # idx_close = np.where(D_ROIs[c,:]<margins*2)[0]

        n_close = self.results["assignments"][D_ROIs[c, :] < margins * 1.5, s].astype(
            "int"
        )

        print("load from %s" % self.paths["neuron_detection"][s])
        A = self.load_footprints(self.paths["neuron_detection"][s], None)
        Cn = self.data_tmp["Cn"]

        x, y = self.results["cm"][c, s, :].astype("int")
        print(x, y)

        # x = int(self.results['cm'][c,s,0])#+cluster.sessions['shift'][s,0])
        # y = int(self.results['cm'][c,s,1])#+cluster.sessions['shift'][s,1])
        # x = int(cm[0])#-cluster.sessions['shift'][s,0])
        # y = int(cm[1])#-cluster.sessions['shift'][s,1])

        ax_ROIs1 = plt.axes([0.05, 0.55, 0.25, 0.4])
        add_number(fig, ax_ROIs1, order=1, offset=[-25, 25])

        # margins = 10
        Cn_tmp = Cn[y - margins : y + margins, x - margins : x + margins]
        Cn -= Cn_tmp.min()
        Cn_tmp -= Cn_tmp.min()
        Cn /= Cn_tmp.max()

        ax_ROIs1.imshow(Cn, origin="lower", clim=[0, 1])
        An = A[..., n].reshape(self.params["dims"]).toarray()
        for nn in n_close:
            cc = np.where(self.results["assignments"][:, s] == nn)[0]
            print(cc, nn)
            # print('SNR: %.2g'%cluster.stats['SNR'][cc,s])
            ax_ROIs1.contour(
                A[..., nn].reshape(self.params["dims"]).toarray(),
                [0.2 * A[..., nn].max()],
                colors="w",
                linestyles="--",
                linewidths=1,
            )
        ax_ROIs1.contour(An, [0.2 * An.max()], colors="w", linewidths=3)
        # ax_ROIs1.plot(cluster.sessions['com'][c,s,0],cluster.sessions['com'][c,s,1],'kx')

        # sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
        # ax_ROIs1.add_artist(sbar)
        ax_ROIs1.set_xlim([x - margins, x + margins])
        ax_ROIs1.set_ylim([y - margins, y + margins])
        ax_ROIs1.text(
            x - margins + 3, y + margins - 5, "Session s", bbox=props, fontsize=10
        )
        ax_ROIs1.set_xticklabels([])
        ax_ROIs1.set_yticklabels([])

        # plt.show(block=False)
        # return
        D_ROIs_cross = sp.spatial.distance.cdist(
            self.results["cm"][:, s, :], self.results["cm"][:, s + 1, :]
        )
        n_close = self.results["assignments"][
            D_ROIs_cross[c, :] < margins * 2, s + 1
        ].astype("int")

        A = self.load_footprints(self.paths["neuron_detection"][s + 1], None)

        ## plot ROIs of session 2 compared to one of session 1

        # Cn = cv2.remap(Cn,x_remap,y_remap, interpolation=cv2.INTER_CUBIC)

        shift = np.array(self.data[s + 1]["remap"]["shift"]) - np.array(
            self.data[s]["remap"]["shift"]
        )
        # y_shift = self.data[s+1]['remap']['shift'][0] - self.data[s]['remap']['shift'][0]
        print("shift", shift)

        x_remap, y_remap = build_remap_from_shift_and_flow(self.params["dims"], shift)

        # x_grid, y_grid = np.meshgrid(np.arange(0., cluster.meta['dims'][0]).astype(np.float32), np.arange(0., cluster.meta['dims'][1]).astype(np.float32))
        # x_remap = (x_grid - \
        #               cluster.sessions['shift'][s+1,0] + cluster.sessions['shift'][s,0] + \
        #               cluster.sessions['flow_field'][s+1,:,:,0] - cluster.sessions['flow_field'][s,:,:,0]).astype('float32')
        # y_remap = (y_grid - \
        #               cluster.sessions['shift'][s+1,1] + cluster.sessions['shift'][s,1] + \
        #               cluster.sessions['flow_field'][s+1,:,:,1] - cluster.sessions['flow_field'][s,:,:,1]).astype('float32')
        # # Cn = cv2.remap(Cn,x_remap,y_remap, interpolation=cv2.INTER_CUBIC)

        ax_ROIs2 = plt.axes([0.35, 0.55, 0.25, 0.4])
        add_number(fig, ax_ROIs2, order=2, offset=[-25, 25])
        ax_ROIs2.imshow(Cn, origin="lower", clim=[0, 1])
        n_match = int(self.results["assignments"][c, s + 1])
        for nn in n_close:
            cc = np.where(self.results["assignments"][:, s + 1] == nn)
            # print('SNR: %.2g'%cluster.stats['SNR'][cc,s+1])
            if not (nn == n_match):  # & (cluster.stats['SNR'][cc,s+1]>3):
                A_tmp = cv2.remap(
                    A[..., nn].reshape(self.params["dims"]).toarray(),
                    x_remap,
                    y_remap,
                    interpolation=cv2.INTER_CUBIC,
                )
                # A_tmp = A[...,nn].reshape(self.params['dims']).toarray()
                ax_ROIs2.contour(
                    A_tmp,
                    [0.2 * A_tmp.max()],
                    colors="r",
                    linestyles="--",
                    linewidths=1,
                )
        ax_ROIs2.contour(An, [0.2 * An.max()], colors="w", linewidths=3)
        A_tmp = cv2.remap(
            A[..., n_match].reshape(self.params["dims"]).toarray(),
            x_remap,
            y_remap,
            interpolation=cv2.INTER_CUBIC,
        )
        # A_tmp = A[...,n_match].reshape(self.params['dims']).toarray()
        ax_ROIs2.contour(A_tmp, [0.2 * A_tmp.max()], colors="g", linewidths=3)

        ax_ROIs2.set_xlim([x - margins, x + margins])
        ax_ROIs2.set_ylim([y - margins, y + margins])
        ax_ROIs2.text(
            x - margins + 3, y + margins - 5, "Session s+1", bbox=props, fontsize=10
        )
        ax_ROIs2.set_xticklabels([])
        ax_ROIs2.set_yticklabels([])

        ax_zoom1 = plt.axes([0.075, 0.125, 0.225, 0.275])
        add_number(fig, ax_zoom1, order=3, offset=[-50, 25])
        ax_zoom1.hist(
            D_ROIs.flatten(), np.linspace(0, 15, 31), facecolor="k", density=True
        )
        ax_zoom1.set_xlabel("distance [$\mu$m]")
        ax_zoom1.spines[["top", "left", "right"]].set_visible(False)
        ax_zoom1.set_yticks([])
        ax_zoom1.set_ylabel("counts")

        ax = plt.axes([0.1, 0.345, 0.075, 0.125])
        plt.hist(
            D_ROIs.flatten(),
            np.linspace(0, np.sqrt(2 * 512**2), 101),
            facecolor="k",
            density=True,
        )
        ax.set_xlabel("d [$\mu$m]", fontsize=10)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        D_matches = np.copy(D_ROIs_cross.diagonal())
        np.fill_diagonal(D_ROIs_cross, np.NaN)

        ax_zoom2 = plt.axes([0.35, 0.125, 0.225, 0.275])
        add_number(fig, ax_zoom2, order=4, offset=[-50, 25])
        ax_zoom2.hist(
            D_ROIs_cross.flatten(),
            np.linspace(0, 15, 31),
            facecolor="tab:red",
            alpha=0.5,
        )
        ax_zoom2.hist(
            D_ROIs.flatten(),
            np.linspace(0, 15, 31),
            facecolor="k",
            edgecolor="k",
            histtype="step",
        )
        ax_zoom2.hist(
            D_matches, np.linspace(0, 15, 31), facecolor="tab:green", alpha=0.5
        )
        ax_zoom2.set_xlabel("distance [$\mu$m]")
        ax_zoom2.spines[["top", "left", "right"]].set_visible(False)
        ax_zoom2.set_yticks([])

        ax = plt.axes([0.38, 0.345, 0.075, 0.125])
        ax.hist(
            D_ROIs_cross.flatten(),
            np.linspace(0, np.sqrt(2 * 512**2), 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        ax.hist(
            D_matches,
            np.linspace(0, np.sqrt(2 * 512**2), 101),
            facecolor="tab:green",
            alpha=0.5,
        )
        ax.set_xlabel("d [$\mu$m]", fontsize=10)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        plt.show(block=False)
        return

        ax = plt.axes([0.7, 0.775, 0.25, 0.125])  # ax_sc1.twinx()
        add_number(fig, ax, order=5, offset=[-75, 50])
        ax.hist(
            cluster.stats["match_score"][:, :, 0].flat,
            np.linspace(0, 1, 51),
            facecolor="tab:blue",
            alpha=1,
            label="$p^*$",
        )
        ax.hist(
            cluster.stats["match_score"][:, :, 1].flat,
            np.linspace(0, 1, 51),
            facecolor="tab:orange",
            alpha=1,
            label="max($p\\backslash p^*$)",
        )
        # ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.set_yticks([])
        ax.set_xlabel("p")
        ax.legend(
            fontsize=8, bbox_to_anchor=[0.3, 0.2], loc="lower left", handlelength=1
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax_sc1 = plt.axes([0.7, 0.45, 0.25, 0.125])
        add_number(fig, ax_sc1, order=6, offset=[-75, 50])
        # ax = plt.axes([0.925,0.85,0.225,0.05])#ax_sc1.twiny()
        # ax.set_xticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        ax_sc1.plot(
            cluster.stats["match_score"][:, :, 1].flat,
            cluster.stats["match_score"][:, :, 0].flat,
            ".",
            markeredgewidth=0,
            color="k",
            markersize=1,
        )
        ax_sc1.plot([0, 1], [0, 1], "--", color="tab:red", lw=1)
        # ax_sc1.plot([0,0.45],[0.5,0.95],'--',color='tab:blue',lw=2)
        # ax_sc1.plot([0.45,1],[0.95,0.95],'--',color='tab:blue',lw=2)
        ax_sc1.set_ylabel("$p^{\\asterisk}$")
        ax_sc1.set_xlabel("max($p\\backslash p^*$)")
        ax_sc1.set_xlim([0, 1])
        ax_sc1.set_ylim([0.5, 1])
        ax_sc1.spines["top"].set_visible(False)
        ax_sc1.spines["right"].set_visible(False)

        ax_sc2 = plt.axes([0.7, 0.125, 0.25, 0.125])
        add_number(fig, ax_sc2, order=7, offset=[-75, 50])
        # plt.hist(np.nanmean(self.results['p_matched'],1),np.linspace(0,1,51))
        ax = ax_sc2.twinx()
        ax.hist(
            np.nanmin(cluster.stats["match_score"][:, :, 0], 1),
            np.linspace(0, 1, 51),
            facecolor="tab:red",
            alpha=0.3,
        )
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = ax_sc2.twiny()
        ax.hist(
            np.nanmean(cluster.stats["match_score"][:, :, 0], axis=1),
            np.linspace(0, 1, 51),
            facecolor="tab:blue",
            orientation="horizontal",
            alpha=0.3,
        )
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax_sc2.plot(
            np.nanmin(cluster.stats["match_score"][:, :, 0], 1),
            np.nanmean(cluster.stats["match_score"][:, :, 0], axis=1),
            ".",
            markeredgewidth=0,
            color="k",
            markersize=1,
        )
        ax_sc2.set_xlabel("min($p^{\\asterisk}$)")
        ax_sc2.set_ylabel("$\left\langle p^{\\asterisk} \\right\\rangle$")
        ax_sc2.set_xlim([0.5, 1])
        ax_sc2.set_ylim([0.5, 1])
        ax_sc2.spines["top"].set_visible(False)
        ax_sc2.spines["right"].set_visible(False)

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
            pl_dat.save_fig("match_stats")

    def plot_alignment_statistics(self, s_compare):

        print("### Plotting session alignment procedure and statistics ###")

        nC, nSes = self.results["assignments"].shape
        sessions_bool = np.ones(nSes, "bool")
        active = ~np.isnan(self.results["assignments"])

        # s = s-1

        dims = self.params["dims"]
        com_mean = np.nanmean(self.results["cm"], 1)

        W = sstats.norm.pdf(range(dims[0]), dims[0] / 2, dims[0] / (0.5 * 1.96))
        W /= W.sum()
        W = np.sqrt(np.diag(W))
        # x_w = np.dot(W,x)

        y = np.hstack([np.ones((512, 1)), np.arange(512).reshape(512, 1)])
        y_w = np.dot(W, y)
        x = np.hstack([np.ones((512, 1)), np.arange(512).reshape(512, 1)])
        x_w = np.dot(W, x)

        # pathSession1 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%1])
        # ROIs1_ld = loadmat(pathSession1)
        s_ = 0
        print(self.paths["neuron_detection"][s_])
        ROIs1_ld = load_data(self.paths["neuron_detection"][s_])
        print(ROIs1_ld.keys())
        Cn = np.array(ROIs1_ld["A"].sum(1).reshape(dims))
        # Cn = ROIs1_ld['Cn'].T
        Cn -= Cn.min()
        Cn /= Cn.max()
        # if self.data[_s]['remap']['transposed']:
        # Cn2 = Cn2.T
        # dims = Cn.shape

        # p_vals = np.zeros((cluster.meta['nSes'],4))*np.NaN
        p_vals = np.zeros((nSes, 2)) * np.NaN
        # fig1 = plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        fig = plt.figure(figsize=(7, 5), dpi=150)
        for s in tqdm.tqdm(np.where(sessions_bool)[0][1:]):  # cluster.meta['nSes'])):

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
                x_remap, y_remap = build_remap_from_shift_and_flow(
                    self.params["dims"],
                    self.data[s]["remap"]["shift"],
                    self.data[s]["remap"]["flow"],
                )

                flow_w_y = np.dot(self.data[s]["remap"]["flow"][0, ...], W)
                y0, res, rank, tmp = np.linalg.lstsq(y_w, flow_w_y)
                dy = -y0[0, :] / y0[1, :]
                idx_out = (dy > 512) | (dy < 0)
                r_y = sstats.linregress(np.where(~idx_out), dy[~idx_out])
                tilt_ax_y = r_y.intercept + r_y.slope * range(512)

                # print((res**2).sum())
                res_y = np.sqrt(((tilt_ax_y - dy) ** 2).sum()) / dims[0]
                # print('y: %.3f'%(np.sqrt(((tilt_ax_y-dy)**2).sum())/dims[0]))

                flow_w_x = np.dot(self.data[s]["remap"]["flow"][1, ...], W)
                x0, res, rank, tmp = np.linalg.lstsq(x_w, flow_w_x)
                dx = -x0[0, :] / x0[1, :]
                idx_out = (dx > 512) | (dx < 0)
                r_x = sstats.linregress(np.where(~idx_out), dx[~idx_out])
                tilt_ax_x = r_x.intercept + r_x.slope * range(512)
                # print(r_x)
                # print('x:')
                # print((res**2).sum())
                # print('x: %.3f'%(np.sqrt(((tilt_ax_x-dx)**2).sum())/dims[0]))
                res_x = np.sqrt(((tilt_ax_x - dx) ** 2).sum()) / dims[0]
                r = r_y if (res_y < res_x) else r_x
                d = dy if (res_y < res_x) else dx
                tilt_ax = r.intercept + r.slope * range(512)

                com_silent = com_mean[~active[:, s], :]
                com_active = com_mean[active[:, s], :]
                # com_PCs = com_mean[cluster.status[cluster.stats['cluster_bool'],s,2],:]

                dist_mean = np.abs(
                    (r.slope * com_mean[:, 0] - com_mean[:, 1] + r.intercept)
                    / np.sqrt(r.slope**2 + 1**2)
                )
                dist_silent = np.abs(
                    (r.slope * com_silent[:, 0] - com_silent[:, 1] + r.intercept)
                    / np.sqrt(r.slope**2 + 1**2)
                )
                dist_active = np.abs(
                    (r.slope * com_active[:, 0] - com_active[:, 1] + r.intercept)
                    / np.sqrt(r.slope**2 + 1**2)
                )
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
                r_silent = sstats.ks_2samp(dist_silent, dist_mean)
                r_active = sstats.ks_2samp(dist_active, dist_mean)
                # r_cross = sstats.ks_2samp(dist_active,dist_silent)
                # r_PCs = sstats.ks_2samp(dist_PCs,dist_mean)
                # p_vals[s,:] = [r_silent.pvalue,r_active.pvalue,r_cross.pvalue,r_PCs.pvalue]
                # p_vals[s] = r_cross.pvalue
                p_vals[s, :] = [r_silent.statistic, r_active.statistic]
            except:
                pass
            # print('time (KS): %.3f'%(time.time()-t_start))
            if s == s_compare:

                ROIs2_ld = load_data(self.paths["neuron_detection"][s])

                Cn2 = np.array(ROIs2_ld["A"].sum(1).reshape(dims))
                # Cn2 = ROIs2_ld['Cn'].T
                Cn2 -= Cn2.min()
                Cn2 /= Cn2.max()
                # if self.data[s]['remap']['transposed']:
                #     Cn2 = Cn2.T

                props = dict(boxstyle="round", facecolor="w", alpha=0.8)

                ax_im1 = plt.axes([0.1, 0.625, 0.175, 0.35])
                add_number(fig, ax_im1, order=1, offset=[-50, -5])
                im_col = np.zeros((512, 512, 3))
                im_col[:, :, 0] = Cn2
                ax_im1.imshow(im_col, origin="lower")
                ax_im1.text(50, 430, "Session %d" % (s + 1), bbox=props, fontsize=8)
                ax_im1.set_xticks([])
                ax_im1.set_yticks([])

                im_col = np.zeros((512, 512, 3))
                im_col[:, :, 1] = Cn

                ax_im2 = plt.axes([0.05, 0.575, 0.175, 0.35])
                ax_im2.imshow(im_col, origin="lower")
                ax_im2.text(50, 430, "Session %d" % 1, bbox=props, fontsize=8)
                ax_im2.set_xticks([])
                ax_im2.set_yticks([])
                # ax_im2.set_xlabel('x [px]',fontsize=14)
                # ax_im2.set_ylabel('y [px]',fontsize=14)
                # sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
                # ax_im2.add_artist(sbar)

                ax_sShift = plt.axes([0.4, 0.575, 0.175, 0.35])
                add_number(fig, ax_sShift, order=2)
                cbaxes = plt.axes([0.4, 0.88, 0.05, 0.02])

                C = signal.convolve(
                    Cn - Cn.mean(), Cn2[::-1, ::-1] - Cn2.mean(), mode="same"
                ) / (np.prod(dims) * Cn.std() * Cn2.std())
                C -= np.percentile(C, 95)
                C /= C.max()
                im = ax_sShift.imshow(
                    C,
                    origin="lower",
                    extent=[-dims[0] / 2, dims[0] / 2, -dims[1] / 2, dims[1] / 2],
                    cmap="jet",
                    clim=[0, 1],
                )

                cb = fig.colorbar(im, cax=cbaxes, orientation="horizontal")
                cbaxes.xaxis.set_label_position("top")
                cbaxes.xaxis.tick_top()
                cb.set_ticks([0, 1])
                cb.set_ticklabels(["low", "high"])
                cb.set_label("corr.", fontsize=10)
                ax_sShift.arrow(
                    0,
                    0,
                    float(self.data[s]["remap"]["shift"][0]),
                    float(self.data[s]["remap"]["shift"][1]),
                    head_width=1.5,
                    head_length=2,
                    color="k",
                    width=0.1,
                    length_includes_head=True,
                )
                ax_sShift.text(
                    -13,
                    -13,
                    "shift: (%d,%d)"
                    % (
                        self.data[s]["remap"]["shift"][0],
                        self.data[s]["remap"]["shift"][1],
                    ),
                    size=10,
                    ha="left",
                    va="bottom",
                    color="k",
                    bbox=props,
                )

                # ax_sShift.colorbar()
                ax_sShift.set_xlim([-15, 15])
                ax_sShift.set_ylim([-15, 15])
                ax_sShift.set_xlabel("$\Delta x [\mu m]$")
                ax_sShift.set_ylabel("$\Delta y [\mu m]$")

                ax_sShift_all = plt.axes([0.54, 0.79, 0.1, 0.15])
                for ss in range(1, nSes):
                    if sessions_bool[ss]:
                        col = [0.6, 0.6, 0.6]
                    else:
                        col = "tab:red"
                    try:
                        ax_sShift_all.arrow(
                            0,
                            0,
                            self.data[ss]["remap"]["shift"][0],
                            self.data[ss]["remap"]["shift"][1],
                            color=col,
                            linewidth=0.5,
                        )
                    except:
                        pass
                ax_sShift_all.arrow(
                    0,
                    0,
                    self.data[s]["remap"]["shift"][0],
                    self.data[s]["remap"]["shift"][1],
                    color="k",
                    linewidth=0.5,
                )
                ax_sShift_all.yaxis.set_label_position("right")
                ax_sShift_all.yaxis.tick_right()
                ax_sShift_all.xaxis.set_label_position("top")
                ax_sShift_all.xaxis.tick_top()
                ax_sShift_all.set_xlim([-25, 50])

                ax_sShift_all.set_ylim([-25, 50])
                # ax_sShift_all.set_xlabel('x [px]',fontsize=10)
                # ax_sShift_all.set_ylabel('y [px]',fontsize=10)

                idxes = 50
                # tx = dims[0]/2 - 1
                # ty = tilt_ax_y[int(tx)]
                ax_OptFlow = plt.axes([0.8, 0.625, 0.175, 0.25])
                add_number(fig, ax_OptFlow, order=3)

                x_grid, y_grid = np.meshgrid(
                    np.arange(0.0, dims[0]).astype(np.float32),
                    np.arange(0.0, dims[1]).astype(np.float32),
                )

                ax_OptFlow.quiver(
                    x_grid[::idxes, ::idxes],
                    y_grid[::idxes, ::idxes],
                    self.data[s]["remap"]["flow"][0, ::idxes, ::idxes],
                    self.data[s]["remap"]["flow"][1, ::idxes, ::idxes],
                    angles="xy",
                    scale_units="xy",
                    scale=0.1,
                    headwidth=4,
                    headlength=4,
                    width=0.002,
                    units="width",
                )  # ,label='x-y-shifts')
                ax_OptFlow.plot(
                    np.linspace(0, dims[0] - 1, dims[0]), d, ":", color="tab:green"
                )
                # ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),dx,'g:')
                # ax_OptFlow.plot(np.linspace(0,dims[0]-1,dims[0]),tilt_ax,'g-')
                ax_OptFlow.plot(
                    np.linspace(0, dims[0] - 1, dims[0]),
                    tilt_ax,
                    "-",
                    color="tab:green",
                )

                ax_OptFlow.set_xlim([0, dims[0]])
                ax_OptFlow.set_ylim([0, dims[1]])
                ax_OptFlow.set_xlabel("$x [\mu m]$")
                ax_OptFlow.set_ylabel("$y [\mu m]$")

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
                print("shift:", self.data[s]["remap"]["shift"])
                x_remap, y_remap = build_remap_from_shift_and_flow(
                    self.params["dims"],
                    self.data[s]["remap"]["shift"],
                    self.data[s]["remap"]["flow"],
                )

                Cn2_corr = cv2.remap(
                    Cn2.astype(np.float32), x_remap, y_remap, cv2.INTER_CUBIC
                )
                Cn2_corr -= Cn2_corr.min()
                Cn2_corr /= Cn2_corr.max()

                ax_sShifted = plt.axes([0.75, 0.11, 0.2, 0.325])
                add_number(fig, ax_sShifted, order=6, offset=[-5, 25])
                im_col = np.zeros((512, 512, 3))
                im_col[:, :, 0] = Cn
                im_col[:, :, 1] = Cn2_corr
                ax_sShifted.imshow(im_col, origin="lower")
                ax_sShifted.text(125, 510, "aligned sessions", bbox=props, fontsize=10)
                ax_sShifted.set_xticks([])
                ax_sShifted.set_yticks([])

                ax_scatter = plt.axes([0.1, 0.125, 0.2, 0.3])
                add_number(fig, ax_scatter, order=4)
                ax_scatter.scatter(com_silent[:, 0], com_silent[:, 1], s=0.7, c="k")
                ax_scatter.scatter(
                    com_active[:, 0], com_active[:, 1], s=0.7, c="tab:orange"
                )
                # x_ax = np.linspace(0,dims[0]-1,dims[0])
                # y_ax = n[0]/n[1]*(p[0]-x_ax) + p[1] + n[2]/n[1]*p[2]
                ax_scatter.plot(
                    np.linspace(0, dims[0] - 1, dims[0]),
                    tilt_ax,
                    "-",
                    color="tab:green",
                )
                # ax_scatter.plot(x_ax,y_ax,'k-')
                ax_scatter.set_xlim([0, dims[0]])
                ax_scatter.set_ylim([0, dims[0]])
                ax_scatter.set_xlabel("x [$\mu$m]")
                ax_scatter.set_ylabel("y [$\mu$m]")

                # x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32),
                # np.arange(0., dims[1]).astype(np.float32))

                ax_hist = plt.axes([0.4, 0.125, 0.3, 0.3])
                add_number(fig, ax_hist, order=5, offset=[-50, 25])
                # ax_hist.hist(dist_mean,np.linspace(0,400,21),facecolor='k',alpha=0.5,density=True,label='all neurons')
                ax_hist.hist(
                    dist_silent,
                    np.linspace(0, 400, 51),
                    facecolor="k",
                    alpha=0.5,
                    density=True,
                    label="silent",
                )
                ax_hist.hist(
                    dist_active,
                    np.linspace(0, 400, 51),
                    facecolor="tab:orange",
                    alpha=0.5,
                    density=True,
                    label="active",
                )
                ax_hist.legend(loc="lower left", fontsize=8)
                ax_hist.set_ylabel("density")
                ax_hist.set_yticks([])
                ax_hist.set_xlabel("distance from axis [$\mu$m]")
                ax_hist.set_xlim([0, 400])
                ax_hist.spines[["top", "right"]].set_visible(False)
        # except:
        # pass

        ax_p = plt.axes([0.525, 0.325, 0.125, 0.125])
        ax_p.axhline(0.01, color="k", linestyle="--")
        ax_p.plot(
            np.where(sessions_bool)[0], p_vals[sessions_bool, 0], "k", linewidth=0.5
        )
        ax_p.plot(
            np.where(sessions_bool)[0],
            p_vals[sessions_bool, 1],
            "tab:orange",
            linewidth=0.5,
        )
        # ax_p.plot(np.where(sessions_bool)[0],p_vals[sessions_bool],'b')
        # ax_p.plot(np.where(sessions_bool)[0],p_vals[sessions_bool,2],'--',color=[0.6,0.6,0.6])
        # ax_p.plot(np.where(sessions_bool)[0],p_vals[sessions_bool,3],'g--')
        ax_p.set_yscale("log")
        ax_p.xaxis.set_label_position("top")
        ax_p.yaxis.set_label_position("right")
        ax_p.tick_params(
            axis="y",
            which="both",
            left=False,
            right=True,
            labelright=True,
            labelleft=False,
        )
        ax_p.tick_params(
            axis="x",
            which="both",
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
        )
        # ax_p.xaxis.tick_top()
        # ax_p.yaxis.tick_right()
        ax_p.set_xlabel("session")
        ax_p.set_ylim([10 ** (-4), 1])
        # ax_p.set_ylim([1,0])
        ax_p.set_ylabel(
            "p-value", fontsize=8, rotation="horizontal", labelpad=-5, y=-0.2
        )
        ax_p.spines[["bottom", "left"]].set_visible(False)
        # ax_p.tick_params(axis='x',which='both',top=True,bottom=False,labeltop=True,labelbottom=False)

        plt.tight_layout()
        plt.show(block=False)
        # if sv:
        #     pl_dat.save_fig('session_align')

    def plot_footprints(self, c, fp_color="r", ax_in=None, use_plotly=False):
        """
        plots footprints of neuron c across all sessions in 3D view
        """
        print(
            "uhm... appears to be broken: are footprint locations not corrected for shift?"
        )

        X = np.arange(0, self.params["dims"][0])
        Y = np.arange(0, self.params["dims"][1])
        X, Y = np.meshgrid(X, Y)

        use_opt_flow = True

        if ax_in is None:
            fig, ax = plt.subplots(ncols=1, subplot_kw={"projection": "3d"})
        else:
            ax = ax_in

        def plot_fp(ax, c):

            for s, path in enumerate(self.paths["neuron_detection"]):
                # if s > 20: break
                idx = self.results["assignments"][c, s]
                # print('footprint:',s,idx)
                if np.isfinite(idx):
                    file = h5py.File(path, "r")
                    # only load a single variable: A

                    data = file["/A/data"][...]
                    indices = file["/A/indices"][...]
                    indptr = file["/A/indptr"][...]
                    shape = file["/A/shape"][...]
                    A = sp.sparse.csc_matrix((data[:], indices[:], indptr[:]), shape[:])

                    A = A[:, int(idx)].reshape(self.params["dims"]).todense()

                    if s > 0:
                        ## use shift and flow to align footprints - apply reverse mapping
                        x_remap, y_remap = build_remap_from_shift_and_flow(
                            self.params["dims"],
                            self.data[s]["remap"]["shift"],
                            self.data[s]["remap"]["flow"] if use_opt_flow else None,
                        )
                        A2 = cv2.remap(
                            A,
                            x_remap,
                            y_remap,  # apply reverse identified shift and flow
                            cv2.INTER_CUBIC,
                        )
                    else:
                        A2 = A

                    A2 /= A2.max()
                    A2[A2 < 0.1 * A2.max()] = np.NaN

                    # mask = A2>0.1*A2.max()
                    ax.plot_surface(
                        X,
                        Y,
                        A2 + s,
                        linewidth=0,
                        antialiased=False,
                        rstride=5,
                        cstride=5,
                        color=fp_color,
                    )
                    # ax.plot_trisurf(X[mask], Y[mask], A2[mask]+s)

        plot_fp(ax, c)
        margin = 25
        com = np.nanmean(self.results["cm"][c, :], axis=0) / self.params["pxtomu"]
        plt.setp(
            ax,
            xlim=[com[0] - margin, com[0] + margin],
            ylim=[com[1] - margin, com[1] + margin],
        )
        if ax_in:
            plt.show(block=False)
