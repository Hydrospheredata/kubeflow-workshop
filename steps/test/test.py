import os, time, requests, sys
import numpy as np
import argparse
from decouple import Config, RepositoryEnv

from storage import * 


config = Config(RepositoryEnv("config.env"))
HYDROSPHERE_LINK = config('HYDROSPHERE_LINK')


def main(data_path, acceptable_accuracy, application_name, bucket_name, storage_path="/"):

    # Define helper classes
    storage = Storage(bucket_name)

    # Download testing data
    storage.download_file(os.path.join(data_path, "test.npz"), os.path.join(storage_path, "test.npz"))
    
    # Prepare data inputs
    with np.load(os.path.join(storage_path, "test.npz")) as data:
        images = data["imgs"][:100]
        labels = data["labels"].astype(int)[:100]
    
    # Define variables 
    requests_delay = 0.2
    service_link = f"{HYDROSPHERE_LINK}/gateway/application/{application_name}"
    print(f"Using URL :: {service_link}", flush=True)

    # Collect responses
    predicted = []
    for index, image in enumerate(images):
        response = requests.post(
            url=service_link, json={'imgs': [image.reshape((1, 28, 28, 1)).tolist()]})
        print(f"{index} | {round(index / len(images) * 100)}% \n{response.text}", flush=True)
        
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(requests_delay)
    
    accuracy = np.sum(labels == np.array(predicted)) / len(labels)
    print(f"Achieved accuracy of {accuracy}", flush=True)

    assert accuracy > acceptable_accuracy, \
        f"Accuracy is not acceptable ({accuracy} < {acceptable_accuracy})"
    

def aws_lambda(event, context):
    return main(
        data_path=event["data_path"],
        acceptable_accuracy=event["acceptable_accuracy"],
        application_name=event["application_name"],
        bucket_name=event["bucket_name"],
        storage_path="/tmp/",
    )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--acceptable-accuracy', type=float, required=True)
    parser.add_argument('--application-name', required=True)
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        acceptable_accuracy=args.acceptable_accuracy,
        application_name=args.application_name,
        bucket_name=args.bucket_name,
    )

    