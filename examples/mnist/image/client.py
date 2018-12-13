import os
import time
import requests
import numpy as np


host_address = os.environ.get("HOST_ADDRESS", "http://localhost")
application_name = os.environ.get("APPLICATION_NAME", "mnist-app")
signature_name = os.environ.get("SIGNATURE_NAME", "predict")
warmup_images_count = os.environ.get("WARMUP_IMAGES_AMOUNT", 1000)

mnist_base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
test_file = "t10k.npz"

notmnist_base_path = os.environ.get("notMNIST_DATA_DIR", "data/notmnist")
ntest_file = "notMNIST_small.npz"


# Import MNIST data
with np.load(os.path.join(mnist_base_path, test_file)) as data:
    imgs, labels = data["imgs"], data["labels"]
np.random.shuffle(imgs)
imgs = imgs[:warmup_images_count//2]

# Import notMNIST data
with np.load(os.path.join(notmnist_base_path, ntest_file)) as data:
    nimgs, nlabels = data["imgs"], data["labels"]
np.random.shuffle(nimgs)
nimgs = nimgs[:warmup_images_count//100]


data = np.concatenate((imgs, nimgs))
np.random.shuffle(data)
for image in data:
    # Warm up application
    image = [image.tolist()]
    r = requests.post(
        url=f"{host_address}/gateway/applications/{application_name}/{signature_name}", 
        json={'imgs': image})
    print("predicted class", r.json()["class_ids"])
    time.sleep(0.6)
