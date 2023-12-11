from skimage.draw import disk
import numpy as np
from dropletqpi.data_processing.parameters import meta_data, keywords
# import matplotlib.pyplot as plt
import numpy.ma as ma


class QpiResult:

    def __init__(self, name, r, center, ri, phase_im, residuals, valid_ri_interval,
                 valid_r_interval=None):
        self.name = name
        self.r = r
        self.center = center
        self.ri = ri
        self.phase_im = phase_im
        self.residuals = residuals
        self.valid_ri_interval = valid_ri_interval
        self.valid_r_interval = QpiResult._set_r_interval(valid_r_interval)
        self.bad_fit = False
        self.bad_fit_criterion = []

        self.shape_xy = np.shape(self.phase_im.T)
        self.r_px = QpiResult.meter_to_px(r)

        self.drop_mask = np.zeros(np.shape(self.phase_im)).astype('bool')
        fit_coords = disk(self.center, self.r_px, shape=np.shape(self.phase_im))
        self.drop_mask[fit_coords] = True

    @classmethod
    def meter_to_px(cls, r):
        return r / meta_data['pixel size']

    @classmethod
    def _set_r_interval(cls, valid_r_interval):
        if not valid_r_interval:
            return [keywords['filter_radius_low'], keywords['filter_radius_high']]
        else:
            return valid_r_interval

    def _evaluate_residuals(self, tresh=0.7):
        masked_res = ma.array(self.residuals, mask=~self.drop_mask)
        masked_phase = ma.array(self.phase_im, mask=~self.drop_mask)
        value = np.std(masked_res.compressed()) / np.mean(masked_phase.compressed())
        print(value, np.std(masked_res.compressed()), np.mean(masked_phase.compressed()))
        if value > tresh:
            print('residuals fucked up')
            self.bad_fit_criterion.append('residuals')
            return True

    def _unique_value_frac(self, tresh=0.95):
        phase_fit = self.phase_im - self.residuals
        x_proj = np.mean(phase_fit, axis=0)
        y_proj = np.mean(phase_fit, axis=1)
        unique_val_frac_x = len(np.unique(x_proj)) / len(x_proj)
        unique_val_frac_y = len(np.unique(y_proj)) / len(y_proj)
        if unique_val_frac_x < tresh or unique_val_frac_y < tresh:
            print('unique_value_frac x,y', unique_val_frac_x, unique_val_frac_y)
            self.bad_fit_criterion.append('consant value stripe')
            return True

    def _check_center_dist(self):
        rel_xdist = abs(self.shape_xy[0] / 2 - self.center[0]) / (self.shape_xy[0] / 2)
        rel_ydist = abs(self.shape_xy[1] / 2 - self.center[1]) / (self.shape_xy[1] / 2)
        if rel_ydist > 1 or rel_xdist > 1:
            print('fit center out of ROI', rel_xdist, rel_ydist)
            self.bad_fit_criterion.append('fit center out of ROI')
            return True

    def _check_fit_radius_size(self, min_area=0.2, max_radius=0.9):

        # fit_too_big = 2 * self.r_px > max_radius * np.sqrt(self.shape_xy[0]
        #                                         * self.shape_xy[1])
        fit_too_big_x = 2 * self.r_px > max_radius * self.shape_xy[0]
        fit_too_big_y = 2 * self.r_px > max_radius * self.shape_xy[1]
        print(2 * self.r_px, max_radius * self.shape_xy[0], max_radius * self.shape_xy[1])
        if fit_too_big_x or fit_too_big_y:
            print('fit radius too big')
            self.bad_fit_criterion.append('fit radius too big')
            return True

        fit_too_small = min_area > np.size(np.where(self.drop_mask)) / np.size(
            self.drop_mask)
        if fit_too_small:
            print('fit radius too small')
            print(np.size(np.where(self.drop_mask)) / np.size(self.drop_mask))
            self.bad_fit_criterion.append('fit radius too small')
            return True

    def _ri_out_of_valid_interval(self):
        if self.ri < self.valid_ri_interval[0] or self.ri > self.valid_ri_interval[1]:
            print('ri out of ', self.valid_ri_interval)
            self.bad_fit_criterion.append('ri out of ' + str(self.valid_ri_interval))
            return True

    def _r_out_of_valid_interval(self):
        if self.r < self.valid_r_interval[0] or self.r > self.valid_r_interval[1]:
            print(self.r, ' out of ', self.valid_r_interval)
            self.bad_fit_criterion.append('r out of '
                                          + str(self.valid_r_interval[0])
                                          + '-'
                                          + str(self.valid_r_interval[1]))
            return True

    def identify_bad_fit(self):

        out_of_roi = self._check_center_dist()
        contains_stripe = self._unique_value_frac()
        large_residuals = self._evaluate_residuals()
        fit_size = self._check_fit_radius_size()
        out_of_ri_interval = self._ri_out_of_valid_interval()
        out_of_r_interval = self._r_out_of_valid_interval()

        if (out_of_roi or contains_stripe or large_residuals or fit_size
                or out_of_ri_interval or out_of_r_interval):
            self.bad_fit = True

