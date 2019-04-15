import os, time, requests, sys
import numpy as np
from sklearn.metrics import accuracy_score


host_address = os.environ.get("CLUSTER_ADDRESS", "http://localhost")
application_name = os.environ.get("APPLICAITON_NAME", "mnist-app")
application_name = sys.argv[1] if len(sys.argv) > 1 else application_name
signature_name = os.environ.get("SIGNATURE_NAME", "predict")
warmup_images_count = int(os.environ.get("WARMUP_IMAGES_AMOUNT", 50))
acceptable_accuracy = float(os.environ.get("ACCEPTABLE_ACCURACY", 0.90))
requests_delay = float(os.environ.get("REQUEST_DELAY", 0.5))
recurring_run = int(os.environ.get("RECURRING_RUN", "0"))
mount_path = os.environ.get("MOUNT_PATH", "./")
data_path = os.path.join(mount_path, "data", "mnist")

if recurring_run:
    test_file = "subsample-test.npz"
else: 
    test_file = "t10k.npz"

service_link = f"{host_address}/gateway/application/{application_name}"
print(f"Using URL :: {service_link}", flush=True)
print(f"Using file :: {os.path.join(data_path, test_file)}", flush=True)


def generate_data(data_path, test_file, warmup_images_count):
    with np.load(os.path.join(data_path, test_file)) as data:
        imgs, labels = data["imgs"], data["labels"]
        return imgs[:warmup_images_count], labels[:warmup_images_count]


if __name__ == "__main__": 
    predicted = []

    data, labels = generate_data(data_path, test_file, warmup_images_count)
    for index, image in enumerate(data):
        response = requests.post(url=service_link, json={'imgs': [image.tolist()]})
        print(f"{index} | {index / len(data)}\n{response.text}", flush=True)
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(requests_delay)
    
    assert accuracy_score(
       labels[:warmup_images_count], predicted[:warmup_images_count]
    ) > acceptable_accuracy, "Accuracy is not acceptable"