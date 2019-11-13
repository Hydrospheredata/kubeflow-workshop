import logging, sys, os

os.makedirs("logs", exist_ok=True)
logs_file = "logs/step_test.log"
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s.%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logs_file)])
logger = logging.getLogger(__name__)

import os, time, requests, argparse
import numpy as np
import wo

INPUTS_DIR, OUTPUTS_DIR = "inputs", "outputs"
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def main(hydrosphere_uri, application_name, acceptable_accuracy, sample_size, *args, **kwargs):
    # Prepare data inputs
    with np.load(os.path.join(INPUTS_DIR, "imgs.npz")) as data:
        images = data["imgs"][:sample_size]
    with np.load(os.path.join(INPUTS_DIR, "labels.npz")) as data:
        labels = data["labels"].astype(int)[:sample_size]
    
    # Define variables 
    requests_delay = 0.2
    service_link = f"{hydrosphere_uri}/gateway/application/{application_name}"
    logger.info(f"Using URL :: {service_link}")

    # Collect responses
    retries = 10
    predicted = []
    for index, image in enumerate(images):
        try: 
            response = requests.post(
                url=service_link, json={'imgs': [image.reshape((1, 28, 28, 1)).tolist()]})
            logger.info(f"{index} | {round(index / len(images) * 100)}% \n{response.text}")
            predicted.append(response.json()["class_ids"][0][0])
        except Exception as e:
            if retries: 
                logger.warning(str(e))
                retries -= 1 
                predicted.append(-1)
                time.sleep(5)
            else: 
                raise
        finally: 
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
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")

    inputs = [(os.path.join(args.data_path, "t10k"), INPUTS_DIR)]
    logs_bucket = wo.parse_bucket(args.data_path, with_scheme=True)
    params = {"uri.hydrosphere": "http://localhost"}

    with wo.Orchestrator(inputs=inputs,
        logs_file=logs_file, logs_bucket=logs_bucket,
        default_params=params, dev=args.dev
    ) as w:

        # Execute main script
        config = w.get_config()
        result = main(**vars(args), hydrosphere_uri=config["uri.hydrosphere"])

        # Execution logging 
        w.log_execution(
            outputs={"integration_test_accuracy": result["accuracy"]},
        )

        assert result["accuracy"] >= args.acceptable_accuracy, \
            f"Accuracy is not acceptable ({result['accuracy']} < {args.acceptable_accuracy})"
