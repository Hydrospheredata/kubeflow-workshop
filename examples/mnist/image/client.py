import os, sys, time, json
import requests
import numpy as np
from sklearn.metrics import accuracy_score

host_address = os.environ.get("HOST_ADDRESS", "http://localhost")
application_name = os.environ.get("APPLICATION_NAME", "mnist-app")
signature_name = os.environ.get("SIGNATURE_NAME", "predict")
warmup_images_count = int(os.environ.get("WARMUP_IMAGES_AMOUNT", 100))

mnist_base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
test_file = "t10k.npz"

# Import MNIST data
with np.load(os.path.join(mnist_base_path, test_file)) as data:
    imgs, labels = data["imgs"], data["labels"]
    imgs, labels = imgs[:warmup_images_count], labels[:warmup_images_count]
    clean_images = len(imgs)

noisy_imgs, noisy_labels = np.copy(imgs), np.copy(labels)
noisy_imgs, noisy_labels = noisy_imgs[:warmup_images_count//2], noisy_labels[:warmup_images_count//2]
noise = np.random.uniform(size=noisy_imgs.shape)

data = np.concatenate((imgs, noisy_imgs+noise))
labels = np.concatenate((labels, noisy_labels))

link = f"{host_address}/gateway/applications/{application_name}/{signature_name}"
print(f"Using URL :: {link}", flush=True)

predicted = []
for index, image in enumerate(data):
    try:
        image = [image.tolist()]
        response = requests.post(url=link, json={'imgs': image})
        print(f"{index+1}/{len(data)} :: predicted class " \
              f"{response.json()['class_ids'][0][0]}", flush=True)
        predicted.append(response.json()["class_ids"][0][0])
    except Exception as e:
        predicted.append(-1)
        print(e, flush=True)
        time.sleep(5)
    time.sleep(0.6)

print(accuracy_score(labels[:clean_images], predicted[:clean_images]), flush=True)