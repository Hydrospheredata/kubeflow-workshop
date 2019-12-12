import logging, sys, os

os.makedirs("logs", exist_ok=True)
logs_file = "logs/step_release_model.log"
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s.%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logs_file)])
logger = logging.getLogger(__name__)

import argparse, os, urllib.parse, pprint, wo
from hydrosdk import sdk


INPUTS_DIR, OUTPUTS_DIR = "inputs", "outputs"
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def main(drift_detector_app, model_name, runtime, payload, metadata, hydrosphere_uri, *args, **kwargs):
    monitoring = [
        sdk.Monitoring('Drift Detector').with_health(True) \
            .with_spec(
                kind='CustomModelMetricSpec', 
                threshold=0.15, 
                operator="<=",
                application=drift_detector_app
            )
    ]

    logger.info("Creating a Model object")
    model = sdk.Model()
    logger.info(f"Adding payload\n{payload}")
    model = model.with_payload(payload)
    logger.info(f"Adding runtime\n{runtime}", )
    model = model.with_runtime(runtime)
    logger.info(f"Adding metadata\n{metadata}")
    model = model.with_metadata(metadata)
    model = model.with_monitoring(monitoring)
    signature = sdk.Signature('predict') \
        .with_input('imgs', 'float32', [-1, 28, 28, 1], 'image') \
        .with_output('probabilities', 'float32', [-1, 10]) \
        .with_output('class_ids', 'int64', [-1, 1]) \
        .with_output('logits', 'float32', [-1, 10]) \
        .with_output('classes', 'string', [-1, 1])
    model.with_signature(signature)
    logger.info(f"Assigning name\n{model_name}")
    model = model.with_name(model_name)
    logger.info(f"Uploading model to the cluster {hydrosphere_uri}")
    result = model.apply(hydrosphere_uri)
    logger.info(pprint.pformat(result))

    return result


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--drift-detector-app', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--accuracy', required=True)
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--average-loss', required=True)
    parser.add_argument('--loss', required=True)
    parser.add_argument('--global-step', required=True)
    parser.add_argument('--dev', action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")

    inputs = [(os.path.join(args.model_path, "saved_model"), INPUTS_DIR)]
    logs_bucket = wo.parse_bucket(args.model_path, with_scheme=True)
    params = {
        "default.tensorflow_runtime": "hydrosphere/serving-runtime-tensorflow-1.13.1:dev",
        "uri.hydrosphere": "http://localhost",
    }

    with wo.Orchestrator(inputs=inputs,
        logs_file=logs_file, logs_bucket=logs_bucket,
        default_params=params, dev=args.dev) as w:
    
        # Initialize runtime variables
        kwargs = vars(args)
        config = w.get_config()
        dev = kwargs.pop("dev")
        
        # Execute main script
        result = main(**kwargs,
            runtime=config["default.tensorflow_runtime"],
            hydrosphere_uri=config["uri.hydrosphere"],
            payload=list(map(lambda a: os.path.join(INPUTS_DIR, a), os.listdir(INPUTS_DIR))),
            metadata=kwargs,
        )

        # Prepare variables for logging
        kwargs["model_version"] = result["modelVersion"]
        kwargs["model_uri"] =  urllib.parse.urljoin(
            config["uri.hydrosphere"], f"/models/{result['model']['id']}/{result['id']}/details")
        w.log_execution(outputs=kwargs)
