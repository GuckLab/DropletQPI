import scipy.io as sio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# from skimage import segmentation
import skimage.filters as fil
import skimage.measure as ms
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d

# PIXEL_SIZE = 84.2e-9  # m
plt.close()
PIXEL_AREA = 0.253 * 0.253  # lateral pixel area in µm^2
VOXEL_VOLUME = 0.253 * 0.253 * (2 * 0.253)  # µm^3
data_path = Path(r'*')
file_list = list(data_path.glob('Tomogram*rep0.mat'))

# odt_data = sio.loadmat(path)
# RI = odt_data['Reconimg']
# RI_max_proj = RI.max(axis=2)
# 
# plt.figure()
# plt.imshow(RI_max_proj)
# plt.colorbar()

fig_hist, ax_hist = plt.subplots()
ax_hist.set_xlabel('RI')
fig_scat, ax_scat = plt.subplots()
ax_scat.set_title(
    'refractive index - volume dependency of individual droplets')
ax_scat.set_xlabel('Evaluated Voxel-Volume in µm^3')
ax_scat.set_ylabel('RI')
# plt.figure()
n_bins = 400
hist_total = np.zeros(n_bins)


for file in file_list[:]:
    min_tresh = 1.375
    print(file)
    odt_data = sio.loadmat(file)
    ri = odt_data['Reconimg']
    ri_max_proj = ri.max(axis=2)
    ri = ri.max(axis=2)

    # fig, ax = plt.subplots()
    # im = ax.imshow(ri_max_proj)
    # fig.colorbar(im)

    # plt.figure()
    # plt.imshow(ri_max_proj)
    # plt.colorbar()
    # plt.figure()
    # plt.hist(ri_max_proj.flatten(), 100)

    # test_image = RI_max_proj
    try:
        min_tresh_fil = fil.threshold_minimum(ri)
        if min_tresh_fil > min_tresh:
            min_tresh = min_tresh_fil
    except:
        print('Unable to set treshold, using default value')
    print(min_tresh)
    # tresh = fil.threshold_niblack(ri, window_size=7, k=0.4)

    # all_tresh = fil.try_all_threshold(test_image)
    # mask = ri < tresh
    mask_min = ri_max_proj > min_tresh
    hist = ax_hist.hist(ri_max_proj[mask_min], n_bins,
                        range=(1.36, 1.46), histtype="step", alpha=0.6, lw=2)
    hist_total += hist[0]

    labeled_mask, n_labels = ms.label(mask_min, return_num=True)
    region_props = ms.regionprops_table(labeled_mask,
                                        properties=['label', 'area',
                                                    'filled_area', 'coords',
                                                    'image'])
    # mean ri 
    label_vol_meanri_drop = pd.DataFrame([
        [label, vol, ri_max_proj[tuple(co.T)].mean()] for label, vol, co in
        zip(region_props['label'],
            region_props['filled_area'],
            region_props['coords'])
    ], columns=['drop_label', 'volume', 'mean_ri'])

    # plt.figure()
    ax_scat.scatter(label_vol_meanri_drop.volume * VOXEL_VOLUME,
                    label_vol_meanri_drop.mean_ri, alpha=0.7, s=10)

    # plt.figure()
    # plt.imshow(labeled_mask.max(2), cmap="Set3")
    # plt.imshow(labeled_mask, cmap="Set3")
    # plt.colorbar()

    # plt.figure()
    # plt.hist(tresh.flatten(), bins=200)
    # plt.show()
    # # obtain buffer peak
    # # hist = plt.hist(ri.flatten(), 400)
    # # idx = int(np.where(np.max(hist[0])==hist[0])[0])
    # # ri_med = (hist[1][idx + 1] + hist[1][idx]) / 2
bins = hist[1]
bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2
hist_total_smooth = uniform_filter1d(hist_total, size=9)
hist_total_max = bin_centers[np.argmax(hist_total_smooth)]

ax_hist.step(bin_centers, hist_total, alpha=0.8, lw=1.5,
             color=(0.3, 0.3, 0.4))

ax_hist.plot(bin_centers, hist_total_smooth,
             alpha=0.7, lw=2,
             color=(0.2, 0.2, 0.3))
ax_hist.bar(hist_total_max,
            np.max(hist_total_smooth),
            width=(bins[1] - bins[0])*2, alpha=0.9,
            color=(0.2, 0.2, 0.3), ls=':')

ax_hist.set_title('RI distribution (RI max:{:.4f})'.format(hist_total_max))
fig_scat.savefig(str(data_path), dpi=300)
fig_hist.savefig(str(data_path) + 'hist', dpi=300)
# ri_max_ind = ri.argmax(axis=2)
# ri_max_ind_up = ri.argmax(axis=2) +1
# ri_max_ind_down = ri.argmax(axis=2) -1
# only first element is correct
# ri_max = ri[np.unravel_index(np.argmax(ri, axis=2), ri.shape)]

# a = np.random.rand(342,342,86)
# idx = a.argmax(axis=2)
# #idx += 1
# results = np.array([[a[i, j, idx[i,j]] for j in range(idx.shape[1])] for i in range(idx.shape[0])])
#
# np.testing.assert_array_equal(a.max(axis=2),results)
