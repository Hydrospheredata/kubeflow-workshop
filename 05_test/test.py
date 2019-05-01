import os, time, requests, sys
import numpy as np
import argparse

def generate_data(data_path, test_file, test_amount):
    with np.load(os.path.join(data_path, test_file)) as data:
        imgs, labels = data["imgs"], data["labels"]
        return imgs[:test_amount], labels[:test_amount]


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data-path', 
        help='Path, where the current run\'s data was stored',
        required=True)
    parser.add_argument(
        '--mount-path',
        help='Path to PersistentVolumeClaim, deployed on the cluster',
        required=True)
    parser.add_argument('--stage-app-name', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--acceptable-accuracy', type=float, default=0.9)
    parser.add_argument('--model-name', required=True)
    
    args = parser.parse_args()
    arguments = args.__dict__

    requests_delay = 2
    application_name = "{}-stage-app".format(arguments['model_name'])
    service_link = "{}/gateway/application/{}".format(
        arguments["hydrosphere_address"], application_name)

    print("Using URL :: {}".format(service_link), flush=True)
    print("Using file :: {}".format(
        os.path.join(arguments["data_path"], "test.npz")), flush=True)

    predicted = []

    data, labels = generate_data(arguments["data_path"], "test.npz", test_amount=100)
    for index, image in enumerate(data):
        response = requests.post(url=service_link, json={'imgs': [image.reshape(1, 28, 28, 1).tolist()]})
        print("{} | {}%\n{}".format(str(index), str(round(index / len(data) * 100)), response.text), flush=True)
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(requests_delay)
    
    accuracy = np.sum(labels == np.array(predicted)) / len(labels)
    assert accuracy > arguments["acceptable_accuracy"], \
        "Accuracy is not acceptable ({} < {})".format(accuracy, arguments["acceptable_accuracy"])