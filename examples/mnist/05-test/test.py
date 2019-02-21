import os, time, requests
import numpy as np
from sklearn.metrics import accuracy_score


host_address = os.environ.get("CLUSTER_ADDRESS", "http://localhost")
application_name = os.environ.get("APPLICATION_NAME", "mnist-app")
signature_name = os.environ.get("SIGNATURE_NAME", "predict")
warmup_images_count = int(os.environ.get("WARMUP_IMAGES_AMOUNT", 100))
acceptable_accuracy = float(os.environ.get("ACCEPTABLE_ACCURACY", 0.90))

mnist_base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
test_file = "t10k.npz"

service_link = f"{host_address}/gateway/applications/{application_name}/{signature_name}"
print(f"Using URL :: {service_link}", flush=True)


def generate_data(mnist_base_path, test_file, warmup_images_count):
    with np.load(os.path.join(mnist_base_path, test_file)) as data:
        imgs, labels = data["imgs"], data["labels"]
        return imgs[:warmup_images_count], labels[:warmup_images_count]


if __name__ == "__main__": 
    predicted = []

    data, labels = generate_data(mnist_base_path, test_file, warmup_images_count)
    for index, image in enumerate(data):
        response = requests.post(url=service_link, json={'imgs': [image.tolist()]})
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(4)
        
    assert accuracy_score(
       labels[:warmup_images_count], predicted[:warmup_images_count]
    ) > acceptable_accuracy, "Accuracy is not acceptable"