import urllib.parse, urllib.request
import gzip, os, struct, numpy, shutil


def download_file(base_url, filename):
    download_url = urllib.parse.urljoin(base_url, filename)
    return urllib.request.urlretrieve(download_url, filename)


def unpack_file(filename):
    with gzip.open(filename, 'rb') as f_in, open(filename[:-3],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filename)
    return filename[:-3]


def process_images(filename):
    with open(filename, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = numpy.fromfile(file, dtype=numpy.uint8).reshape(num, rows, cols)
        return imgs.astype(numpy.float32) / 255.0
        

def process_labels(filename):
    with open(filename, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        return numpy.fromfile(file, dtype=numpy.int8)
