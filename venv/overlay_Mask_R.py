import sys
import pickle
import h5py
import numpy as np
import cv2
import openslide
from memory_profiler import profile
import gc
import pyvips
import time


# global variable
cv2.setUseOptimized(True)

# smallest possible difference
EPSILON = sys.float_info.epsilon

# file name of prediction mask
input_file_name = ["4", "22", "45", "48", "55", "56", "73", "84", "98",
                   "112", "136", "143", "146", "157", "176", "189", "213",
                   "217", "283", "284", "292", "293", "333", "338", "367",
                   "393", "397", "441", "480", "519", "595", "679", "731",
                   "809", "817", "870", "880", "880", "958", "974", "992",
                   "997", "1017", "1034", "1097", "1104", "1118", "1121",
                   "1137", "1174", "1198", "1277", "1351", "1352", "1500",
                   "1542", "1676", "1784", "1994", "2090"]

# folder of prediction mask and indices
mask_folder_address = "/Users/hz32/Desktop/Mask_RCNN/"

# folder of svs file
slide_folder_address = "/Users/hz32/Desktop/wholeslides/"


# range for category 0, 1, 2
a_range_min = 0.8
a_range_max = 1

b_range_min = 0.4
b_range_max = 0.8

# color input for convert_to_rgb function
colors = [(255, 255, 255), (255, 0, 0)]

'''
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if gc.isenabled() is False:
    gc.enable()
'''


# map np dtypes to vips
# oh dear str lookup ... is there a better way?
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


def prob_to_category(mask, a_range_min, a_range_max, b_range_min, b_range_max):
    category = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    category[:, :] = 0
    category[np.where(np.logical_and(mask >= a_range_min, mask <= a_range_max))] = 1
    category[np.where(np.logical_and(mask >= b_range_min, mask < b_range_max))] = 2

    return category


def convert_to_rgb(minval, maxval, value_for_change, input_colors):
    # colors specifies a series of points deliniating color ranges
    # determine where val falls within the entire range
    fi = float(value_for_change-minval) / float(maxval-minval) * (len(input_colors)-1)
    # determine between which color points val falls
    i = int(fi)
    # determine where val falls within that range
    f = fi - i
    # does it fall on one of the color points?
    if f < EPSILON:
        return input_colors[i]
    else:
        # otherwise return the color within the range it corresponds
        (r1, g1, b1), (r2, g2, b2) = input_colors[i], input_colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))


@profile
def change_color(mask):
    return np.array([rgb(value) for i, value in np.ndenumerate(mask)])


@profile
def prediction_data_input(data_folder_address, slide_folder_address, input_file_name):

    # load the location tile indices for current
    with open(data_folder_address + "locations/" +input_file_name + ".pkl", "rb") as f:
        tile_locations = pickle.load(f, encoding='bytes')

    # Get the prediction mask for current slide
    slide_results = h5py.File(data_folder_address + "Output/" +input_file_name + ".hdf5", "r")

    # read in original slide
    slide = openslide.OpenSlide(slide_folder_address + input_file_name + '.svs')
    # get if slide is 20x or 40x
    slide_mg = int(slide.properties['openslide.objective-power'])

    # prediction result dict list, set to white
    pred_result = np.ndarray((slide.dimensions[1], slide.dimensions[0], 3), dtype=np.uint8)
    pred_result[:, :] = [255, 255, 255]
    # pred_counter = np.zeros((slide.dimensions[1], slide.dimensions[0]), dtype=np.int8)

    del slide
    gc.collect()

    for tile_id, cur_loc in tile_locations.items():

        tile_result = slide_results['%d_det' % int(tile_id)][:]
        if slide_mg == 40:
            tile_result_color = np.ndarray((1024, 1024, 3), dtype=np.uint8)

            for index, value in np.ndenumerate(tile_result):
                tile_result_color[index[0]][index[1]] = convert_to_rgb(0.0, 3.0, value, colors)
                tile_result_color = cv2.UMat(tile_result_color)
                tile_result_color = cv2.resize(tile_result_color.get(), None, fx=2, fy=2)

            del tile_result
            gc.collect()

            x = int(cur_loc[1])
            y = int(cur_loc[0])
            pred_result[x:x + 1024, y:y + 1024] = tile_result_color
            # pred_counter[x: x + 1024, y: y + 1024] += 1
            tile_result_color.release()
        else:
            tile_result_color = np.ndarray((512, 512, 3), dtype=np.uint8)
            for index, value in np.ndenumerate(tile_result):
                tile_result_color[index[0]][index[1]] = convert_to_rgb(0.0, 3.0, value, colors)
                tile_result_color = cv2.UMat(tile_result_color)
                tile_result_color = cv2.resize(tile_result_color.get(), None, fx=2, fy=2)

            del tile_result
            gc.collect()

            x = int(cur_loc[1])
            y = int(cur_loc[0])
            pred_result[x:x + 512, y:y + 512] = tile_result_color
            # pred_counter[x: x + 512, y: y + 512] += 1

    # pred_counter[np.where(pred_counter == 0)[0], np.where(pred_counter == 0)[1]] = 1
    # pred_result /= pred_counter

    return pred_result


'''
for i in input_file_name:

    t0 = time.time()
    a = prediction_data_input(mask_folder_address, slide_folder_address, i)
    t1 = time.time()
    print(t1-t0)

    height, width, bands = a.shape
    linear = a.reshape(width*height*bands)
    img = pyvips.Image.new_from_memory(linear.data, width, height, bands, dtype_to_format[str(a.dtype)])
    img.dzsave(mask_folder_address + 'prediction_dzi/'+ i + '_pred')

    del a
    del img
    gc.collect()
'''

t0 = time.time()
a = prediction_data_input(mask_folder_address, slide_folder_address, input_file_name[0])
t1 = time.time()
print(t1-t0)

height, width, bands = a.shape
linear = a.reshape(width*height*bands)
img = pyvips.Image.new_from_memory(linear.data, width, height, bands, dtype_to_format[str(a.dtype)])
img.dzsave(mask_folder_address + 'prediction_dzi/'+ i + '_pred')

del a
del img
gc.collect()