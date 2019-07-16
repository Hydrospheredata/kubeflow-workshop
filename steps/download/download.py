from PIL import Image
import struct, numpy
import os, gzip, tarfile, shutil, glob
import urllib, urllib.parse, urllib.request
import datetime, argparse
from decouple import Config, RepositoryEnv

from storage import * 
from orchestrator import *


config = Config(RepositoryEnv("config.env"))
MNIST_URL = config('MNIST_URL')


filenames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]


def download_files(base_url, storage_path, filenames=None):
    """ Download required data """

    if not filenames: 
        # if not any filenames provided, use global instead
        filenames = globals()["filenames"]
    
    for file in filenames:
        print(f"Started downloading {file}", flush=True)
        download_url = urllib.parse.urljoin(base_url, file)
        download_path = os.path.join(storage_path, file)
        local_file, _ = urllib.request.urlretrieve(download_url, download_path)
        unpack_archive(local_file)


def unpack_archive(file):
    """ Unpack compressed file """

    print(f"Unpacking archive {file}", flush=True)
    with gzip.open(file, 'rb') as f_in, open(file[:-3],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file)


def process_images(dataset, storage_path):
    """ Preprocess downloaded MNIST datasets """
    
    print(f"Processing images {dataset}", flush=True)
    label_file = os.path.join(storage_path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = numpy.fromfile(file, dtype=numpy.int8) #int8
        new_labels = numpy.zeros((num, 10))
        new_labels[numpy.arange(num), labels] = 1

    img_file = os.path.join(storage_path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = numpy.fromfile(file, dtype=numpy.uint8).reshape(num, rows, cols)
        imgs = imgs.astype(numpy.float32) / 255.0

    os.remove(label_file); os.remove(img_file)
    return imgs, labels


def download_mnist(base_url, storage_path):
    """ Download and preprocess train/test datasets """

    download_files(base_url, storage_path)
    train_imgs, train_labels = process_images("train", storage_path)
    test_imgs, test_labels = process_images("t10k", storage_path) 

    train_path = os.path.join(storage_path, "train.npz")
    test_path = os.path.join(storage_path, "test.npz")

    numpy.savez_compressed(train_path, imgs=train_imgs, labels=train_labels)
    numpy.savez_compressed(test_path, imgs=test_imgs, labels=test_labels)
    return [train_path, test_path]


def main(bucket_name, storage_path="/"):
    
    # Define helper classes
    storage = Storage(bucket_name=bucket_name)
    orchestrator = Orchestrator(storage_path=storage_path)

    # Define path, where to store files
    data_path = os.path.join("data", str(round(datetime.datetime.now().timestamp())))
    
    # Download and process MNIST files
    processed_files = download_mnist(MNIST_URL, storage_path)

    # Upload files to the cloud
    for filename in processed_files:
        source_path = os.path.join(storage_path, filename)
        destination_path = os.path.join(data_path, os.path.basename(filename))
        storage.upload_file(source_path, destination_path)

    # Export parameters for orchestrator
    orchestrator.export_meta(
        "data_path", os.path.join(storage.full_name, data_path), "txt")


def aws_lambda(event, context):
    return main(
        bucket_name=event["bucket_name"],
        storage_path="/tmp",
    )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage-path', default='/')
    parser.add_argument('--bucket-name', required=True)

    args = parser.parse_args()
    main(
        storage_path=args.storage_path,
        bucket_name=args.bucket_name,
    )
