import h5py
import numpy as np
import openslide
import scipy


input_file_name = "1302"
data_folder_address = "/home/zhanghaoyue/Desktop/PyHDF5/"
tile_size = 1200

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


def prediction_data_input(data_folder_address,input_file_name,tile_size):
    
    # load the location tile indices for current
    tile_locations = np.load(data_folder_address + input_file_name + '.npy',encoding='bytes')
    # Get the prediction mask for current slide
    slide_results = h5py.File(data_folder_address + input_file_name+".hdf5","r")['/predictions']
    # prediction result dict list
    prediction_result = np.zeros[]

    for tile_id in range(1,  len(tile_locations.item())):
        tile_result = np.squeeze(slide_results[tile_id - 1, :, :])
        tile_loc = tile_locations.item().get(tile_id)
        for i in range(0, tile_size):
            for j in range(0,tile_size):
                if tile_result[i][j] in (1.0,2.0):
                    prediction_result.append({'x': i + int(tile_loc[0]),'y': j + int(tile_loc[1]),'p':int(tile_result[i][j])})
                else:
                    prediction_result.append({'x': i + int(tile_loc[0]),'y': j + int(tile_loc[1]),'p':0})

    '''
    with open(data_folder_address+input_file_name+'.json','w+') as fp:
        json.dump(prediction_result,fp,cls=MyEncoder)
    '''
    return prediction_result

a = prediction_data_input(data_folder_address,input_file_name,tile_size)