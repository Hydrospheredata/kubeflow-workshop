import os, time, requests, sys
import numpy as np

host_address = os.environ.get("CLUSTER_ADDRESS", "http://localhost")
application_name = os.environ.get("APPLICAITON_NAME", "mnist-app")
application_name = sys.argv[1] if len(sys.argv) > 1 else application_name
signature_name = os.environ.get("SIGNATURE_NAME", "predict")
test_amount = int(os.environ.get("TEST_AMOUNT", 50))
acceptable_accuracy = float(os.environ.get("ACCEPTABLE_ACCURACY", 0.90))
requests_delay = float(os.environ.get("REQUEST_DELAY", 0.5))
recurring_run = int(os.environ.get("RECURRING_RUN", "0"))
mount_path = os.environ.get("MOUNT_PATH", "./")
data_path = os.path.join(mount_path, "data", "mnist")

if recurring_run:
    test_file = "subsample-test.npz"
else: 
    test_file = "t10k.npz"

service_link = "{}/gateway/application/{}".format(host_address, application_name)
print("Using URL :: {}".format(service_link), flush=True)
print("Using file :: {}".format(os.path.join(data_path, test_file)), flush=True)


def generate_data(data_path, test_file, test_amount):
    with np.load(os.path.join(data_path, test_file)) as data:
        imgs, labels = data["imgs"], data["labels"]
        return imgs[:test_amount], labels[:test_amount]


if __name__ == "__main__": 
    predicted = []

    data, labels = generate_data(data_path, test_file, test_amount)
    for index, image in enumerate(data):
        response = requests.post(url=service_link, json={'imgs': [image.tolist()]})
        print("{} | {}%\n{}".format(str(index), str(index / len(data) * 100), response.text), flush=True)
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(requests_delay)
    
    assert np.sum(labels == np.array(predicted)) / len(labels) > acceptable_accuracy, \
        "Accuracy is not acceptable"