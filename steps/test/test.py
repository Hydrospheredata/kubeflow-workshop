import os, time, requests, sys
import numpy as np
import argparse, logging
from cloud import CloudHelper


logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("test.log")])
logger = logging.getLogger(__name__)


def main(hydrosphere_uri, application_name, acceptable_accuracy, sample_size):
    # Prepare data inputs
    with np.load(os.path.join("data", "imgs.npz")) as data:
        images = data["imgs"][:sample_size]
    with np.load(os.path.join("data", "labels.npz")) as data:
        labels = data["labels"].astype(int)[:sample_size]
    
    # Define variables 
    requests_delay = 0.2
    service_link = f"{hydrosphere_uri}/gateway/application/{application_name}"
    logger.info(f"Using URL :: {service_link}")

    # Collect responses
    predicted = []
    for index, image in enumerate(images):
        response = requests.post(
            url=service_link, json={'imgs': [image.reshape((1, 28, 28, 1)).tolist()]})
        logger.info(f"{index} | {round(index / len(images) * 100)}% \n{response.text}")
        
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(requests_delay)
    
    accuracy = np.sum(labels == np.array(predicted)) / len(labels)
    logger.info(f"Achieved accuracy of {accuracy}")
    
    return {
        "accuracy": accuracy,
    }    


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--acceptable-accuracy', type=float, required=True)
    parser.add_argument('--application-name', required=True)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--dev', action='store_true', default=False)
    args = parser.parse_args()

    # Prepare environment
    cloud = CloudHelper(
        default_config_map_params={"uri.hydrosphere": "https://dev.k8s.hydrosphere.io"})
    cloud.download_prefix(os.path.join(args.data_path, "t10k"), "data/")

    result = main(
        acceptable_accuracy=args.acceptable_accuracy,
        application_name=args.application_name,
        hydrosphere_uri=cloud.get_kube_config_map()["uri.hydrosphere"],
        sample_size=args.sample_size,
    )

    cloud.log_execution(
        outputs={"integration_test_accuracy": result["accuracy"]}, 
        logs_bucket=cloud.get_bucket_from_uri(args.data_path).full_uri,
        logs_file="test.log",
        logs_path="mnist/logs",
        dev=args.dev)

    assert result["accuracy"] > args.acceptable_accuracy, \
        f"Accuracy is not acceptable ({result['accuracy']} < {args.acceptable_accuracy})"

    