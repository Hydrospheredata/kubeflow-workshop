from PIL import Image
import struct, numpy
import os, gzip, tarfile, shutil, glob
import urllib, urllib.parse, urllib.request
import datetime, argparse


filenames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]


def download_files(base_url, base_dir, filenames=None):
    """ Download required data """
    if not filenames: 
        # if not any filenames provided, use global instead
        filenames = globals()["filenames"]
    
    os.makedirs(base_dir, exist_ok=True)
    for file in filenames:
        print("Started downloading {}".format(file), flush=True)
        download_url = urllib.parse.urljoin(base_url, file)
        download_path = os.path.join(base_dir, file)
        local_file, _ = urllib.request.urlretrieve(download_url, download_path)
        unpack_archive(local_file, base_dir)


def unpack_archive(file, base_dir):
    """ Unpack compressed file """

    print("Unpacking archive {}".format(file), flush=True)
    with gzip.open(file, 'rb') as f_in, open(file[:-3],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file)


def process_images(path, dataset):
    """ Preprocess downloaded MNIST datasets """
    
    print("Processing images {}".format(os.path.join(path, dataset)), flush=True)
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
    return imgs, labels


def download_mnist(base_url, base_dir):
    """ Download original MNIST structs and pack them into numpy arrays """

    download_files(base_url, base_dir)
    train_imgs, train_labels = process_images(base_dir, "train")
    test_imgs, test_labels = process_images(base_dir, "t10k") 

    print("Saving train data under {}.npz path".format(os.path.join(base_dir, "train")), flush=True)
    numpy.savez_compressed(os.path.join(base_dir, "train"), imgs=train_imgs, labels=train_labels)
    
    print("Saving test data under {}.npz path".format(os.path.join(base_dir, "test")), flush=True)
    numpy.savez_compressed(os.path.join(base_dir, "test"), imgs=test_imgs, labels=test_labels)



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mount-path',
        help = 'Path to PersistentVolumeClaim, deployed on the cluster',
        required = True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    mount_path = arguments["mount_path"]
    data_path = os.path.join(
        mount_path, "data", "mnist", str(round(datetime.datetime.now().timestamp())))
    pipeline_dataset_path = "/data_path.txt" if mount_path != "./" else "./data_path.txt"

    download_mnist(
        base_dir=data_path, 
        base_url="http://yann.lecun.com/exdb/mnist/")

    # Dump dataset location
    with open(pipeline_dataset_path, "w+") as file:
        file.write(data_path)