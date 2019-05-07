from PIL import Image
import struct, numpy, boto3
import os, gzip, tarfile, shutil, glob
import urllib, urllib.parse, urllib.request
import datetime, argparse


filenames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]


def download_files(base_url, filenames=None):
    """ Download required data """

    if not filenames: 
        # if not any filenames provided, use global instead
        filenames = globals()["filenames"]
    
    for file in filenames:
        print(f"Started downloading {file}", flush=True)
        download_url = urllib.parse.urljoin(base_url, file)
        download_path = os.path.join('./', file)
        local_file, _ = urllib.request.urlretrieve(download_url, download_path)
        unpack_archive(local_file)


def unpack_archive(file):
    """ Unpack compressed file """

    print(f"Unpacking archive {file}", flush=True)
    with gzip.open(file, 'rb') as f_in, open(file[:-3],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file)


def process_images(dataset):
    """ Preprocess downloaded MNIST datasets """
    
    print(f"Processing images {dataset}", flush=True)
    label_file = dataset + '-labels-idx1-ubyte'
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = numpy.fromfile(file, dtype=numpy.int8) #int8
        new_labels = numpy.zeros((num, 10))
        new_labels[numpy.arange(num), labels] = 1

    img_file = dataset + '-images-idx3-ubyte'
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = numpy.fromfile(file, dtype=numpy.uint8).reshape(num, rows, cols)
        imgs = imgs.astype(numpy.float32) / 255.0

    os.remove(label_file); os.remove(img_file)
    return imgs, labels


def download_mnist(base_url):

    # Download files
    download_files(base_url)

    # Transform images into numpy arrays
    train_imgs, train_labels = process_images("train")
    test_imgs, test_labels = process_images("t10k") 

    # Save numpy arrays

    numpy.savez_compressed("train", imgs=train_imgs, labels=train_labels)
    numpy.savez_compressed("test", imgs=test_imgs, labels=test_labels)

    return ["train.npz", "test.npz"]


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument(
        '--dev', help='Flag for development mode', action="store_true")
    
    args = parser.parse_args()
    s3 = boto3.resource('s3')

    # Define the path, where to store files
    namespace = urllib.parse.urlparse(args.hydrosphere_address).netloc.split(".")[0]
    data_path = os.path.join(namespace, "data", "mnist", 
        str(round(datetime.datetime.now().timestamp())))

    # Download and process MNIST files
    processed_files = download_mnist("http://yann.lecun.com/exdb/mnist/")

    # Upload files to S3 
    for filename in processed_files:
        print(f"Uploading {filename} to S3", flush=True)
        s3.meta.client.upload_file(
            filename, "odsc-workshop", os.path.join(data_path, filename))
    
    # Dump dataset path for Kubeflow Pipelines
    with open("./data_path.txt" if args.dev else "/data_path.txt", "w+") as file:
        file.write(data_path)