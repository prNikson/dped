import h5py

with h5py.File("data.h5", 'r') as file:
   print(len(file['kvadra']['pixels']))