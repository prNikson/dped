from multiprocessing import Queue, Process
from concurrent.futures import ProcessPoolExecutor, as_completed
from h5 import Dataset
from pathlib import Path
from process_image import process_image


filename = 'data.h5'
def writer_process(queue):
    with Dataset(filename, 'a') as dataset:
        while True:
            data = queue.get()
            if data == 'stop':
                break
            if data is not None:
                [dataset.insert(pair[0], pair[1]) for pair in data]

queue = Queue()
writer = Process(target=writer_process, args=(queue,))
writer.start()

image_paths = [(str(path.name), i) for i, path in enumerate(Path('/home/miriteam/Desktop/A/JPEG').glob('*'))]

with ProcessPoolExecutor(max_workers=2) as executor:
    # results = list(executor.map(process_image, image_paths))
    futures = [
        executor.submit(process_image, filename)
        for filename in image_paths
    ]

    for future in as_completed(futures):
        chunk = future.result()
        queue.put(chunk)

queue.put('stop')
writer.join()