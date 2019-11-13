import logging, sys, os

os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s.%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/step_download.log")])
logger = logging.getLogger(__name__)

import os, gzip, tarfile, wo
import shutil, glob, struct, hashlib
import urllib, urllib.parse, urllib.request
import datetime, argparse, numpy
from PIL import Image

INPUTS_DIR, OUTPUTS_DIR = "inputs/", "outputs/"
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
    
    return wo.utils.io.md5_files([
        os.path.join(directory, "imgs.npz"), 
        os.path.join(directory, "labels.npz")
    ])


def main(uri):
    """ Download MNIST data, process it and upload it to the cloud. """

    download_files(uri)

    imgs, labels = process_images("train")
    train_md5 = write_data(imgs, labels, "data/train")
    imgs, labels = process_images("t10k")
    test_md5 = write_data(imgs, labels, "data/t10k")

    sample_version = wo.utils.io.md5_string(train_md5 + test_md5)
    output_directory = os.path.join(OUTPUTS_DIR, f"sample-version={sample_version}")
    if os.path.exists(output_directory):
        logger.warning(f"Directory {output_directory} already exists")
        logger.warning(f"Cleaning {output_directory} from the old files")
        shutil.rmtree(os.path.join(OUTPUTS_DIR, f"sample-version={sample_version}"))
    
    shutil.move("data/train", os.path.join(OUTPUTS_DIR, f"sample-version={sample_version}", "train"))
    shutil.move("data/t10k", os.path.join(OUTPUTS_DIR, f"sample-version={sample_version}", "t10k"))
    return {"sample_version": sample_version}


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-path", required=True)
    parser.add_argument("--dev", action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")

    with wo.Orchestrator(
        outputs=[(OUTPUTS_DIR, args.output_data_path)],
        default_params={"uri.mnist": "http://yann.lecun.com/exdb/mnist/"},
        logs_file="logs/step_download.log",
        logs_bucket=wo.parse_bucket(args.output_data_path, with_scheme=True),
        dev=args.dev,
    ) as orchestrator:

        # Main script execution
        config = orchestrator.get_config()
        result = main(uri=config["uri.mnist"])

        # Execution logging
        orchestrator.log_execution(
            outputs={"output_data_path": os.path.join(args.output_data_path, f"sample-version={result['sample_version']}")})
