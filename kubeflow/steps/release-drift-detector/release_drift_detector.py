import logging, sys, os

os.makedirs("logs", exist_ok=True)
logs_file = "logs/step_release_drift_detector.log"
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s.%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logs_file)])
logger = logging.getLogger(__name__)

import argparse, wo, json
import os, urllib.parse, pprint
from hydrosdk import sdk


INPUTS_DIR, OUTPUTS_DIR = "inputs", "outputs"
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def main(model_name, runtime, payload, metadata, hydrosphere_uri, *args, **kwargs):
    logger.info("Creating a Model object")
    model = sdk.Model()
    logger.info("Adding payload")
    model = model.with_payload(payload)
    logger.info("Adding runtime")
    model = model.with_runtime(runtime)
    logger.info("Adding metadata")
    model = model.with_metadata(metadata)
    logger.info("Assigning name")
    model = model.with_name(model_name)
    
    logger.info(f"Uploading model to the cluster {hydrosphere_uri}")
    result = model.apply(hydrosphere_uri)

    logger.info(pprint.pformat(result))
    return result


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--steps', required=True)
    parser.add_argument('--loss', required=True)
    parser.add_argument('--dev', action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")

    inputs = [(args.model_path, INPUTS_DIR)]
    logs_bucket = wo.parse_bucket(args.data_path, with_scheme=True)
    params = {
        "default.tensorflow_runtime": "hydrosphere/serving-runtime-tensorflow-1.13.1:dev",
        "uri.hydrosphere": "http://localhost",
    }

    with wo.Orchestrator(inputs=inputs, 
        logs_file=logs_file, logs_bucket=logs_bucket,
        default_params=params, dev=args.dev) as w:

        # Initialize runtime variables
        config = w.get_config()
        payload = list(map(lambda a: os.path.join(INPUTS_DIR, a), os.listdir(INPUTS_DIR)))

        # Execute main script
        kwargs = vars(args)
        dev = kwargs.pop("dev")
        result = main(**kwargs, 
            runtime=config["default.tensorflow_runtime"],
            hydrosphere_uri=config["uri.hydrosphere"],
            payload=payload,
            metadata=kwargs,
            dev=dev,
        )

        kwargs.update({
            "model_version": result["modelVersion"],
            "model_uri": urllib.parse.urljoin(
                config["uri.hydrosphere"], f"/models/{result['model']['id']}/{result['id']}/details"),
        })
        w.log_execution(outputs=kwargs)
        