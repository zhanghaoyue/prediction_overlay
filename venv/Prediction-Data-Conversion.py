
import h5py
import numpy as np
import cv2
import openslide
from memory_profiler import profile
import gc
import pyvips
import time

input_file_name = "1302"
data_folder_address = "/home/zhanghaoyue/Desktop/PyHDF5/"
tile_size = 1200

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

@profile
def change_color(mask):

    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[:, :] = [255, 255, 255]
    colored[np.where(a == 1)] = [0, 0, 255]
    colored[np.where(a == 2)] = [0, 255, 255]

    return colored


@profile
def prediction_data_input(data_folder_address, input_file_name, tile_size):

    # load the location tile indices for current
    tile_locations = np.load(data_folder_address + input_file_name + '.npy', encoding='bytes')
    # Get the prediction mask for current slide
    slide_results = h5py.File(data_folder_address + input_file_name + ".hdf5", "r")['/predictions']
    # prediction result dict list
    slide = openslide.OpenSlide(data_folder_address + input_file_name + '.svs')
    prediction_result = np.ndarray((slide.dimensions[0], slide.dimensions[1]), dtype=np.int8)

    for tile_id in range(1, slide_results.shape[0]):
        tile_result = np.squeeze(slide_results[tile_id - 1, :, :])
        tile_result[tile_result > 2] = 0
        tile_loc = tile_locations.item().get(tile_id)
        x = int(tile_loc[0])
        y = int(tile_loc[1])
        prediction_result[x:x + tile_size, y:y + tile_size] = tile_result

    '''
    prediction_result = sparse.csr_matrix(prediction_result)

    with open(data_folder_address+input_file_name+'.json','w+') as fp:
        json.dump(prediction_result,fp,cls=MyEncoder)
    '''
    return prediction_result


t0 = time.time()
a = prediction_data_input(data_folder_address, input_file_name, tile_size)
t1 = time.time()
print(t1-t0)


t0 = time.time()
b = change_color(a)
t1 = time.time()
print(t1-t0)

del a

filename = data_folder_address+input_file_name+'pred.png'

cv2.imwrite(filename, b)

del b


img = pyvips.Image.new_from_file(data_folder_address+input_file_name+'pred.png')
img.dzsave(data_folder_address+input_file_name)

del img
gc.collect()

