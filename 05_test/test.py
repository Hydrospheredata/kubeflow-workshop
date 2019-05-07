import os, time, requests, sys
import numpy as np
import argparse, boto3


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--acceptable-accuracy', type=float, required=True)
    parser.add_argument('--model-name', required=True)
    
    args = parser.parse_args()
    s3 = boto3.resource('s3')

    # Download testing data
    s3.Object('odsc-workshop', os.path.join(args.data_path, "test.npz")).download_file('./test.npz')
    
    # Prepare data inputs
    with np.load("./test.npz") as data:
        images = data["imgs"][:100]
        labels = data["labels"].astype(int)[:100]
    
    requests_delay = 0.2
    application_name = f"{args.model_name}-stage-app"
    service_link = f"{args.hydrosphere_address}/gateway/application/{application_name}"

    print(f"Using URL :: {service_link}", flush=True)

    predicted = []
    for index, image in enumerate(images):
        image = image.reshape((1, 28, 28, 1))
        response = requests.post(url=service_link, json={'imgs': [image.tolist()]})
        print(
            f"{index} | {round(index / len(images) * 100)}% \n{response.text}", 
            flush=True
        )
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(requests_delay)
    
    accuracy = np.sum(labels == np.array(predicted)) / len(labels)
    print(f"Achieved accuracy of {accuracy}", flush=True)

    assert accuracy > args.acceptable_accuracy, \
        f"Accuracy is not acceptable ({accuracy} < {args.acceptable_accuracy})"