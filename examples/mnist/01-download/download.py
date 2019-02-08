from PIL import Image
import struct, numpy
import os, gzip, tarfile, shutil, glob
import urllib, urllib.parse, urllib.request

filenames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]


def download_files(base_url, base_dir):
    """ Download required data """
    global filenames
    
    os.makedirs(base_dir, exist_ok=True)
    for file in filenames:
        print(f"Started downloading {file}", flush=True)
        download_url = urllib.parse.urljoin(base_url, file)
        download_path = os.path.join(base_dir, file)
        local_file, _ = urllib.request.urlretrieve(download_url, download_path)
        unpack_archive(local_file, base_dir)


def unpack_archive(file, base_dir):
    """ Unpack the compressed file. """

    print(f"Unpacking archive {file}", flush=True)
    with gzip.open(file, 'rb') as f_in, open(file[:-3],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file)


def process_images(path, dataset):
    """ Preprocess downloaded MNIST datasets. """
    
    print(f"Processing images {os.path.join(path, dataset)}", flush=True)
    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = numpy.fromfile(file, dtype=numpy.int8) #int8
        new_labels = numpy.zeros((num, 10))
        new_labels[numpy.arange(num), labels] = 1

    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = numpy.fromfile(file, dtype=numpy.uint8).reshape(num, rows, cols) #uint8
        imgs = imgs.astype(numpy.float32) / 255.0

    os.remove(label_file); os.remove(img_file)
    print(f"Saving files under {os.path.join(path, dataset)} path", flush=True)
    numpy.savez_compressed(os.path.join(path, dataset), imgs=imgs, labels=labels)


def download_mnist(base_url, base_dir):
    """ Donwload original MNIST structs and unpack them into .png files """

    download_files(base_url, base_dir)
    process_images(base_dir, "train")
    process_images(base_dir, "t10k") 


if __name__ == "__main__": 
    download_mnist(
        base_url="http://yann.lecun.com/exdb/mnist/",
        base_dir=os.environ.get("MNIST_DATA_DIR", "data/mnist"))


       
        


