from pathlib import Path

data_path = Path(r'\\foo\baz')

bg_path = Path(r'\\foo\baz')

meta_data = {'medium index': 1.336,
             'pixel size': 4.8 / (90.4762 * 63 / 100) * 1e-6,
             'wavelength': 532e-9}

file_list = list(data_path.glob('*.h5'))

evaluated_repetition = '0'

keywords = {
# if debug_mode == True the segmented mask is plotted as well
'debug_mode': True,
# if prefilter can be set to None or a value [0,1], in the later case a highpass filter
# is applied to the phase image and the given fraction is filtered out. This helps to
# obtain a better result for the image segmentation
'prefilter': 0.01,
'rel_frame_width': 0.35,
'filter_radius_low': 1.5e-6,
'filter_radius_high': 8e-6,
}

