import h5py
import numpy as np
import os


# with h5py.File("dataset.hdf5", 'a') as file:
    # try:
        # kvadra_g = file["kvadra"]
        # sony_g = file["sony"]
    # except KeyError as e:
        # kvadra_g = file.create_group('kvadra')
        # sony_g = file.create_group('sony')
    # kvadra = kvadra_g.create_dataset('pixels', shape=(1, 3, 100, 100), maxshape=(None, 3, 100, 100), dtype='uint8', compression="gzip", compression_opts=9)
    # sony = sony_g.create_dataset('pixels', shape=(1, 3, 100, 100), maxshape=(None, 3, 100, 100), dtype='uint8', compression="gzip", compression_opts=9)
    # print(kvadra.name)
    # file['kvadra/pixels'][0] = np.random.rand(100, 100)
    # print(file['kvadra/pixels'].)

class Dataset(h5py.File):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __enter__(self):
        try:
            kvadra_g = self["kvadra"]
            sony_g = self["sony"]
        except KeyError as e:
            kvadra_g = self.create_group('kvadra')
            sony_g = self.create_group('sony')
        
        try:
            self.kvadra = kvadra_g['pixels']
            self.sony = sony_g['pixels']
        except KeyError as e:
            self.sony = sony_g.create_dataset('pixels', shape=(1, 3, 100, 100), maxshape=(None, 3, 100, 100), dtype='uint8', compression="gzip", compression_opts=9)
            self.kvadra = kvadra_g.create_dataset('pixels', shape=(1, 3, 100, 100), maxshape=(None, 3, 100, 100), dtype='uint8', compression="gzip", compression_opts=9)

        return self

    def _append(self, dest, arg):
        length = (dest.shape[0])
        dest.resize(length+1, 0)
        dest[length] = arg

    def print(self):
        print(self.sony[-1])

    def insert(self, kvadra, sony):
        self._append(self.sony, sony)
        self._append(self.kvadra, kvadra)