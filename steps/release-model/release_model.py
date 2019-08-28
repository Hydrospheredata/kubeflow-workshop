import argparse, os, urllib.parse, logging, sys, pprint
from hydrosdk import sdk
from cloud import CloudHelper


logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("release_model.log")])
logger = logging.getLogger(__name__)


def main(drift_detector_app, model_name, runtime, payload, metadata, hydrosphere_uri):

    monitoring = [
        sdk.Monitoring('Requests').with_spec('CounterMetricSpec', interval=15),
        sdk.Monitoring('Latency').with_spec('LatencyMetricSpec', interval=15),
        sdk.Monitoring('Accuracy').with_spec('AccuracyMetricSpec'),
        sdk.Monitoring('Drift Detector').with_health(True) \
            .with_spec(
                kind='ImageAEMetricSpec', 
                threshold=0.15, 
                application=drift_detector_app
            )
    ]

    logger.info("Creating a Model object")
    model = sdk.Model()
    logger.info("Adding payload")
    model = model.with_payload(payload)
    logger.info("Adding runtime")
    model = model.with_runtime(runtime)
    logger.info("Adding metadata")
    model = model.with_metadata(metadata)
    logger.info("Adding monitoring")
    model = model.with_monitoring(monitoring)
    logger.info("Assigning name")
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
    args = parser.parse_args()
    kwargs = dict(vars(args))

    # Prepare environment
    cloud = CloudHelper(default_config_map_params={
        "default.tensorflow_runtime": "hydrosphere/serving-runtime-tensorflow-1.13.1:dev", 
        "uri.hydrosphere": "https://dev.k8s.hydrosphere.io"
    })
    config = cloud.get_kube_config_map()
    cloud.download_prefix(os.path.join(args.model_path, "saved_model"), args.model_path) 

    # Prepare deployment essentials
    dev = kwargs.pop("dev")
    model_name = kwargs.pop("model_name")
    drift_detector_app = kwargs.pop("drift_detector_app")
    runtime = config["default.tensorflow_runtime"]
    hydrosphere_uri = config["uri.hydrosphere"]
    model_path = cloud.get_relative_path_from_uri(args.model_path)
    payload = list(map(lambda a: os.path.join(model_path, a), os.listdir(model_path)))

    # Release the model
    result = main(drift_detector_app, model_name, runtime, payload, kwargs, hydrosphere_uri)

    # Export metadata
    kwargs["model_version"] = result["modelVersion"]
    kwargs["model_uri"] =  urllib.parse.urljoin(
        config["uri.hydrosphere"], f"/models/{result['model']['id']}/{result['id']}/details")
    cloud.log_execution(
        outputs=kwargs, 
        logs_bucket=cloud.get_bucket_from_uri(args.model_path).full_uri,
        logs_file="release_model.log",
        logs_path="mnist/logs",
        dev=dev,
    )
     