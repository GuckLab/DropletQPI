import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import qpimage
import qpsphere
import skimage.filters as fil
import skimage.measure as ms

from dropletqpi.plotting.plotting import plot_scatter
from dropletqpi.data_processing.parameters import data_path, bg_path, meta_data,\
    file_list, evaluated_repetition, keywords
from dropletqpi.data_models.qpi_fit import QpiResult


def process_RI_dataset(folder, savefig=True, export_plot_data=True,
                       ri_valid_interval=None):

    result_list = list(folder.glob('*_results.h5'))
    if len(result_list) == 0:
        raise FileNotFoundError('no result files to evaluate')

    plot_data = []
    outlier_list = []

    # TODO put the reading part in a function
    for n, result in enumerate(result_list):
        print('\n')
        print('file', n+1, 'of', len(result_list), result.name)
        with h5py.File(result, "r") as r:
            n = r['fitting results']['RI'][()]
            radius = r['fitting results']['radius'][()]
            center = r['fitting results']['center'][()]
            phase_im = r['data']['phase image [rad]'][()]
            residuals = r['data']['Rytov-SC fit residuals'][()]
            print('radius in m: ', round(radius, 10),
                  'refractive index: ', round(n, 5),
                  list(r.attrs.items())[3])

            qpi_fit = QpiResult(result.name, radius, center, n,
                                phase_im, residuals, ri_valid_interval)
            qpi_fit.identify_bad_fit()

            # If ri_valid_interval is given, only append data if refractive index is within it
            # if ri_valid_interval:
            #     if n > ri_valid_interval[0] and n < ri_valid_interval[1]:
            #         plot_data.append(
            #             [result.name, radius, n])
            #     else:
            #         print(result.name, ' ri is not in the given interval')
            #         outlier_list.append(
            #         [result.name, radius, n, 'out of ' + str(ri_valid_interval)])
            # else:
            if not qpi_fit.bad_fit:
                plot_data.append(
                    [result.name, radius, n])
            else:
                print(result.name, 'QPI fit is bad')
                outlier_list.append(
                [result.name, radius, n, qpi_fit.bad_fit_criterion])

    export_data = pd.DataFrame(plot_data,
                               columns=['sample_name',
                                        'radius [m]',
                                        'refractive index'])

    if not export_data.empty:
        fig, ax = plot_scatter(export_data['radius [m]'], export_data['refractive index'])
        ax.set_title('radius: ' + str(export_data['radius [m]'].median().round(10)) + 'm'
                     '   RI: ' + str(export_data['refractive index'].median().round(4)))

        if savefig:
            fig.savefig(str(folder.parent) + "\\" + folder.parts[-1] + '_'+ "RI_vs_r"
                        + "_" + "full" + ".png", dpi=300,
                        transparent=True, bbox_inches='tight')

    if export_plot_data:
        with pd.ExcelWriter(str(folder.parent) + '\\' + folder.parts[-2] + '_'
                             + 'RI_all_fill' + '.xlsx') as writer:
            export_data.to_excel(writer, sheet_name='valid_fits', index=False)

            if len(outlier_list) > 0:
                outliers = pd.DataFrame(outlier_list, columns=['sample_name',
                                                               'radius [m]',
                                                               'refractive index',
                                                               'bad fit criterion'])
                outliers.to_excel(writer, sheet_name='outliers ', index=False)
                                                # + str(ri_valid_interval).strip('[]'),



def run_analysis(file_list, bg_path, save_path, meta_data=meta_data ,**kwargs):

    debug_mode = kwargs['debug_mode']
    prefilter = kwargs['prefilter']
    rel_frame_width = kwargs['rel_frame_width']
    filter_radius_low = kwargs['filter_radius_low']
    filter_radius_high = kwargs['filter_radius_high']
    area_filter_pix_low = np.pi * filter_radius_low ** 2 / meta_data['pixel size'] ** 2
    area_filter_pix_up = np.pi * filter_radius_high ** 2 / meta_data['pixel size'] ** 2

    if len(file_list) == 0:
        warnings.warn("nothing to evaluate")

    for file in file_list:
        # close the previous figures for more than 3 files
        if len(file_list) > 3:
            plt.close('all')

        print(file)
        with h5py.File(file, "r") as h5_sample, h5py.File(bg_path, "r") as h5_bg:
            print('evaluating repetition:', evaluated_repetition,
                  'available repetitions:', list(h5_sample['ODT'].keys()))

            # h5_sample['ODT'][evaluated_repetition] refers to measurements with 150 images from 0°
            try:
                data = h5_sample['ODT'][evaluated_repetition]['payload']['data']
            except KeyError:
                warnings.warn('repetition '
                              + evaluated_repetition + ' is not available')
                continue

            bg_data = h5_bg['ODT']['0']['payload']['data']['48'].__array__()[0]
            sample_data = np.array([data[key].__array__()[0] for key in
                     data.__iter__()]
                )

            if len(sample_data) <= 40:
                # this line was not tested yet
                sample_holo = np.median(sample_data[:10], axis=0)
                # sample_holo = np.median(
                #     np.array(
                #         [data[key].__array__()[0] for n, key in
                #          zip(range(10), data.__iter__())]
                #     ),
                #     axis=0)
            else:
                sample_holo = data['48'].__array__()[0]

            # if np.median(abs(data['148'] - data['10'].__array__())) > 5:
            # if np.median(abs(data['99'].__array__() - data['10'].__array__())) > 5:
            #     # if True:
            #
            #     sample_holo = data['48'].__array__()[0]
            #     warnings.warn('Images probably not measured from the same angle')

            qpi = qpimage.QPImage(data=sample_holo,
                                  which_data='hologram',
                                  holo_kw={'sideband': -1},
                                  bg_data=bg_data,
                                  meta_data=meta_data)

            qpi.compute_bg(which_data=["amplitude", "phase"],
                           fit_offset="fit",
                           # fit_profile="tilt",
                           fit_profile="poly2o",
                           border_px=5,
                           )

            fig, ax = plt.subplots()
            qpi_im = ax.imshow(qpi.pha)
            plt.colorbar(qpi_im, ax=ax)

            if prefilter:
                filter_idx = int(qpi.pha.flatten().size * prefilter)
                prefilter_tresh = np.sort(qpi.pha.flatten())[filter_idx]
                premask = ma.array(qpi.pha, mask=[qpi.pha < prefilter_tresh])
                # treshold = fil.threshold_multiotsu(premask.compressed())[-1]
                treshold = fil.threshold_otsu(premask.compressed())
            # treshold = fil.threshold_otsu(qpi.pha)
            else:
                treshold = fil.threshold_multiotsu(qpi.pha)[-1]

            mask = qpi.pha > treshold

            qpi.compute_bg(which_data=["phase"],
                           fit_offset="fit",
                           fit_profile="poly2o",
                           from_mask=~mask
                           )

            labeled_mask, n_labels = ms.label(mask, return_num=True)
            region_props = ms.regionprops_table(labeled_mask,
                                                properties=['label', 'bbox', 'area',
                                                            'solidity', 'filled_area',
                                                            'image'])

            if debug_mode:
                print('treshold:', treshold)
                fig_d, axs = plt.subplots(1, 2, figsize=(9, 3))
                axs[0].imshow(qpi.pha)
                axs[0].imshow(mask, cmap='gray', alpha=0.3)
                if prefilter:
                    axs[1].hist(premask.compressed(), 50)
                else:
                    axs[1].hist(qpi.pha.flatten(), 50)
                axs[1].axvline(treshold, linewidth=1.5, color='gray')
                plt.show()
                fig_d.savefig(str(save_path) + "\\" + file.name.strip(".h5")
                              + "_" + evaluated_repetition
                              + "_" + "mask" + ".png", dpi=300)

            if len(region_props['label']) > 430:
                print(len(region_props['label']), 'regions found')
                return qpi, sample_holo, region_props

            # (min_row, min_col, max_row, max_col)
            bboxes = [region_props['bbox-0'], region_props['bbox-1'],
                      region_props['bbox-2'], region_props['bbox-3']]

            # add an extra frame to the bbox
            bboxes_row_frame = abs((bboxes[2] - bboxes[0])
                                   * rel_frame_width).astype(int)
            bboxes_col_frame = abs((bboxes[3] - bboxes[1])
                                   * rel_frame_width).astype(int)

            bboxes_big = [abs(bboxes[0] - bboxes_row_frame),
                          abs(bboxes[2] + bboxes_row_frame),
                          abs(bboxes[1] - bboxes_col_frame),
                          abs(bboxes[3] + bboxes_col_frame)]

        droplet_frames = []
        droplet_holo_frame = []

        for n, box in enumerate(region_props['label']):
            droplet_frames.append(qpi[bboxes_big[0][n]: bboxes_big[1][n],
                                  bboxes_big[2][n]: bboxes_big[3][n]])

            droplet_holo_frame.append(
                sample_holo[bboxes_big[0][n]: bboxes_big[1][n],
                bboxes_big[2][n]: bboxes_big[3][n]])

        # filter out bigger drops (potential drops with a radius > 5* min radius are
        # considered as an artifact)
        big_drops_index = [n - 1 for n in region_props['label'] if
                           (area_filter_pix_low <
                            region_props['filled_area'][n - 1]
                            < area_filter_pix_up
                            and region_props['solidity'][n - 1] > 0.4)]

        # analyze only the first drop for code testing
        for i, sample_idx in enumerate(big_drops_index):
            print('drop', i + 1, 'of', len(big_drops_index), 'object index', sample_idx)

            save_name_base = (str(save_path) + "\\"
                              + file.name.strip(".h5")
                              + "_" + evaluated_repetition
                              + "_" + str(sample_idx))
            # sample_idx = big_drops_index[0]

            # indicate the location of the analyzed drop in the phase image
            ax.scatter(
                bboxes_big[2][sample_idx] +
                (bboxes_big[3][sample_idx] - bboxes_big[2][sample_idx]) / 2,
                bboxes_big[0][sample_idx] +
                (bboxes_big[1][sample_idx] - bboxes_big[0][sample_idx]) / 2,
                marker='+', color='w', s=100, alpha=0.8)

            # calculate the approximate droplet radius
            approx_radius = meta_data['pixel size'] * np.sqrt(
                region_props['filled_area'][sample_idx] / np.pi)

            try:
                n_fit, r_fit, c_fit, qpi_fit = qpsphere.analyze(
                    qpi=droplet_frames[sample_idx],
                    r0=approx_radius,
                    method='image',
                    model='rytov-sc',
                    ret_center=True,
                    ret_qpi=True
                )
            except:  # qpsphere.models.mod_rytov_sc.RefractiveIndexLowerThanMediumError:
                print('Evaluation aborted, droplet was '
                      'probably not correctly segmented and is marked with a red cross')

                ax.scatter(
                    bboxes_big[2][sample_idx] +
                    (bboxes_big[3][sample_idx] - bboxes_big[2][sample_idx]) / 2,
                    bboxes_big[0][sample_idx] +
                    (bboxes_big[1][sample_idx] - bboxes_big[0][sample_idx]) / 2,
                    marker='+', color='red', s=100, alpha=0.9)

                continue

            circle = plt.Circle((c_fit[0], c_fit[1]), r_fit / qpi["pixel size"],
                                color='w', fill=False, ls="dashed", lw=2, alpha=.5)
            info = "n={:.4F}\nr={:.2f}µm".format(n_fit, r_fit * 1e6)
            holkw = {"cmap": "gray"}
            # extent=(0, 1024, 0, 1024) *
            rytov_sc_residuals = droplet_frames[sample_idx].pha.T - qpi_fit.pha.T

            fig_single_sample = plt.figure(figsize=(12, 12))

            ax0 = plt.subplot(221, title="drop hologram")
            map0 = ax0.imshow(droplet_holo_frame[sample_idx].T, **holkw)
            plt.colorbar(map0, ax=ax0, fraction=.048, pad=0.04)

            ax1 = plt.subplot(222, title="phase image [rad]")
            map1 = ax1.imshow(droplet_frames[sample_idx].pha.T)
            plt.colorbar(map1, ax=ax1, fraction=.048, pad=0.04)

            ax1.text(.8, .8, info, color="w", fontsize="12", verticalalignment="top")

            ax2 = plt.subplot(223, title="phase fit [rad]")
            map2 = ax2.imshow(qpi_fit.pha.T)
            ax2.text(.8, .8, info, color="w", fontsize="12", verticalalignment="top")
            ax2.add_artist(circle)
            plt.colorbar(map2, ax=ax2, fraction=.048, pad=0.04)

            ax3 = plt.subplot(224, title="Rytov-SC fit residuals")
            map3 = ax3.imshow(rytov_sc_residuals, cmap="seismic")
            plt.colorbar(map3, ax=ax3, fraction=.046, pad=0.04,
                         label="phase error [rad]")
            plt.tight_layout()

            # don't show the plots, just save them
            plt.close()
            fig_single_sample.savefig(save_name_base + ".png", dpi=300)

            result_filename = (save_name_base + "_results.h5")

            with h5py.File(result_filename, "w") as results:
                # TODO there is probably a smarter way of exporting the whole qpi-object
                data = results.create_group("data")

                data.create_dataset("drop hologram",
                                    data=droplet_holo_frame[sample_idx].T)
                # dset = results.get("drop hologram")
                # dset.attrs['CLASS'] = np.string_('IMAGE')
                data.create_dataset("phase image [rad]",
                                    data=droplet_frames[sample_idx].pha.T)
                # dset2 = results.get("phase image [rad]")
                # dset2.attrs['CLASS'] = np.string_('IMAGE')
                data.create_dataset("phase fit [rad]", data=qpi_fit.pha.T)
                # dset3 = results.get("phase image [rad]")
                # dset3.attrs['CLASS'] = np.string_('IMAGE')
                data.create_dataset("Rytov-SC fit residuals", data=rytov_sc_residuals)
                # dset4 = results.get("Rytov-SC fit residuals")
                # dset4.attrs['CLASS'] = np.string_('IMAGE')

                fitting = results.create_group("fitting results")
                fitting.create_dataset("RI", data=n_fit)
                fitting.create_dataset("radius", data=r_fit)
                fitting.create_dataset("center", data=c_fit)
                # meta_data = results.create_group("meta-data")
                results.attrs['sample_file'] = str(file)
                results.attrs['evaluated_repetition'] = evaluated_repetition
                results.attrs['background_file'] = str(bg_path)
                results.attrs['sample_label'] = sample_idx
                results.attrs['lower_radius_treshold'] = filter_radius_low
                results.attrs['upper_radius_treshold'] = filter_radius_high
                results.attrs['treshold_phase'] = treshold
                results.attrs['prefilter'] = prefilter
                for attr in meta_data.keys():
                    results.attrs[attr] = meta_data[attr]

        with h5py.File(str(save_path) + "\\"
                          + file.name.removesuffix(".h5") + "_fov.h5", "w") as fov:
            data = fov.create_group("Full field of view image data")
            data.create_dataset("hologram",
                                data=sample_holo.T)
            data.create_dataset("phase image [rad]",
                                data=qpi.pha.T)
            data.create_dataset("phase mask [rad]",
                                data=mask.T)
            fov.attrs['treshold_phase'] = treshold
            fov.attrs['background_file'] = str(bg_path)
            fov.attrs['prefilter'] = prefilter
            for attr in meta_data.keys():
                fov.attrs[attr] = meta_data[attr]

        fig.savefig(str(save_path) + "\\" + file.name.strip(".h5")
                    + "_" + evaluated_repetition
                    + "_" + "full" + ".png", dpi=300)
        # plt.show()
        print('analyis finished')


def main():
    save_path = data_path.parent.joinpath('QPI_analysis')
    save_path.mkdir(exist_ok=True)
    # warnings.filterwarnings("error")

    output = run_analysis(file_list, bg_path, save_path, meta_data=meta_data, **keywords)

    if output:
        raise Exception('Segmentation probably went wrong due to bad quality '
                        'of the QPI image')
    plt.close()

    process_RI_dataset(save_path, ri_valid_interval=[1.37, 1.45])

if __name__ == "__main__":
    main()