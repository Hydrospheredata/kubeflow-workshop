from PIL import Image
import struct, numpy
import os, gzip, tarfile, shutil, glob
import urllib, urllib.parse, urllib.request


def download_files(base_url, base_dir, files):
    """ Download required data """

    downloaded = []
    os.makedirs(base_dir, exist_ok=True)

    for file in files:
        print(f"Started downloading {file}")
        download_url = urllib.parse.urljoin(base_url, file)
        download_path = os.path.join(base_dir, file)
        local_file, _ = urllib.request.urlretrieve(download_url, download_path)
        unpack_file(local_file, base_dir)
    
    return downloaded


def unpack_file(file, base_dir):
    """ Unpacking all compressed files. """

    print(f"Unpacking {file}")
    if os.path.split(base_dir)[-1] == "mnist":
        with gzip.open(file, 'rb') as f_in, open(file[:-3],'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if os.path.split(base_dir)[-1] == "notmnist":
        with tarfile.open(file) as f_tar: 
            f_tar.extractall(base_dir)
    os.remove(file)


def preprocess_mnist_files(path, dataset):
    """ Preprocess downloaded MNIST datasets. """
    
    print(f"Preprocessing {os.path.join(path, dataset)}")
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
    numpy.savez_compressed(os.path.join(path, dataset), imgs=imgs, labels=labels)


if __name__ == "__main__": 
    mnist_dir = os.environ.get("MNIST_DATA_DIR", "data/mnist")
    mnist_files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']
    
    downloaded = download_files(
        base_url="http://yann.lecun.com/exdb/mnist/", 
        base_dir=mnist_dir, 
        files=mnist_files)
    preprocess_mnist_files(mnist_dir, "train")
    preprocess_mnist_files(mnist_dir, "t10k")    
        