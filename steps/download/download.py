import os, sys, gzip, tarfile, logging
import shutil, glob, struct, hashlib
import urllib, urllib.parse, urllib.request
import datetime, argparse, numpy
from PIL import Image
from cloud import CloudHelper


logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("download.log")])
logger = logging.getLogger(__name__)


filenames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]


def md5(filenames: list):
    """ Get md5 hash of the given files """

    hash_md5 = hashlib.md5()
    for filename in filenames:
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

    return hash_md5.hexdigest()


def download_files(base_url, filenames=None):
    """ Download required data """

    if not filenames: 
        # if not any filenames provided, use global instead
        filenames = globals()["filenames"]
    
    for file in filenames:
        logger.info(f"Started downloading {file}")
        download_url = urllib.parse.urljoin(base_url, file)
        local_file, _ = urllib.request.urlretrieve(download_url, file)
        unpack_archive(local_file)


def unpack_archive(file):
    """ Unpack compressed file """

    logger.info(f"Unpacking archive {file}")
    with gzip.open(file, 'rb') as f_in, open(file[:-3],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file)


def process_images(dataset):
    """ Preprocess downloaded MNIST datasets """
    
    logger.info(f"Processing images {dataset}")
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


def write_data(imgs: numpy.ndarray, labels: numpy.ndarray, directory: str):
    """ Write data and return md5 checksum """

    os.makedirs(directory, exist_ok=True)
    numpy.savez_compressed(os.path.join(directory, "imgs.npz"), imgs=imgs)
    numpy.savez_compressed(os.path.join(directory, "labels.npz"), labels=labels)
    
    return md5([os.path.join(directory, "imgs.npz"), os.path.join(directory, "labels.npz")])


def main(uri):
    """ Download MNIST data, process it and upload it to the cloud. """

    download_files(uri)
    imgs, labels = process_images("train")
    train_md5 = write_data(imgs, labels, "data/train")
    imgs, labels = process_images("t10k")
    test_md5 = write_data(imgs, labels, "data/t10k")

    return {
        "sample_version": CloudHelper._md5_string(train_md5 + test_md5)
    }


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-path", required=True)
    parser.add_argument("--dev", action="store_true", default=False)
    args = parser.parse_args()

    cloud = CloudHelper(
        default_logs_path="mnist/logs",
        default_config_map_params={
            "uri.mnist": "http://yann.lecun.com/exdb/mnist/",
        },
    )
    config = cloud.get_kube_config_map()
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")
    
    try:

        # Initialize runtime variables
        pass 

        # Execute main script
        result = main(uri=cloud.get_kube_config_map()["uri.mnist"])
        output_data_path = os.path.join(
            args.output_data_path, f"sample-version={result['sample_version']}")

        # Prepare variables for logging
        pass 
        
    except Exception as e:
        logger.exception("Main execution script failed.")
    
    finally: 
        cloud.log_execution(
            outputs={
                "output_data_path": output_data_path,
            },
            logs_bucket=cloud.get_bucket_from_uri(args.output_data_path).full_uri,
            logs_file="download.log",
            dev=args.dev,
        )
