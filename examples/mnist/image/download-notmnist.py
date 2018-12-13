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


def preprocess_notmnist_files(path, dataset):
    """ Preprocess downloaded notMNIST datasets. """

    print(f"Preprocessing {os.path.join(path, dataset)}")
    files = glob.glob(os.path.join(path, dataset, "**", "*.png"))
    imgs = numpy.zeros((len(files), 28, 28))
    labels = numpy.zeros((len(files),))
    for index, file in enumerate(files): 
        try: 
            label = os.path.split(os.path.dirname(file))[-1]
            imgs[index,:,:] = numpy.array(Image.open(file))
            labels[index] = ord(label) - ord("A")
        except: pass

    shutil.rmtree(os.path.join(path, dataset), ignore_errors=True)
    numpy.savez_compressed(os.path.join(path, dataset), imgs=imgs, labels=labels)


if __name__ == "__main__": 
    notmnist_dir = os.environ.get("notMNIST_DATA_DIR", "data/notmnist")
    notmnist_files = [
        "notMNIST_small.tar.gz",]
        # "notMNIST_large.tar.gz"]

    download_files(
        base_url="http://yaroslavvb.com/upload/notMNIST/",
        base_dir=notmnist_dir,
        files=notmnist_files)
    preprocess_notmnist_files(notmnist_dir, "notMNIST_small")
    # preprocess_notmnist_files(notmnist_dir, "notMNIST_large")
    
        