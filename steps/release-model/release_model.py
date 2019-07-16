import argparse
import os, urllib.parse
from hydrosdk import sdk
from decouple import Config, RepositoryEnv

from storage import *
from orchestrator import * 


config = Config(RepositoryEnv("config.env"))
TENSORFLOW_RUNTIME = config('TENSORFLOW_RUNTIME')
HYDROSPHERE_LINK = config('HYDROSPHERE_LINK')


def main(drift_detector_app, model_name, classes, bucket_name, storage_path="/", **kwargs):
    
    storage = Storage(bucket_name)
    orchestrator = Orchestrator(storage_path=storage_path)

    # Download model 
    working_dir = os.path.join(storage_path, "model")
    storage.download_prefix(os.path.join(kwargs["model_path"], "saved_model"), working_dir)

    # Build servable
    payload = [
        os.path.join(storage_path, 'model', 'saved_model.pb'),
        os.path.join(storage_path, 'model', 'variables')
    ]

    metadata = {
        'learning_rate': kwargs["learning_rate"],
        'batch_size': kwargs["batch_size"],
        'epochs': kwargs["epochs"], 
        'accuracy': kwargs["accuracy"],
        'average_loss': kwargs["average_loss"],
        'loss': kwargs["loss"],
        'global_step': kwargs["global_step"],
        'mlflow_link': kwargs["mlflow_link"], 
        'data': kwargs["data_path"],
        'model_path': kwargs["model_path"],
    }

    signature = sdk.Signature('predict') \
        .with_input('imgs', 'float32', [-1, 28, 28, 1], 'image') \
        .with_output('probabilities', 'float32', [-1, classes]) \
        .with_output('class_ids', 'int64', [-1, 1]) \
        .with_output('logits', 'float32', [-1, classes]) \
        .with_output('classes', 'string', [-1, 1])

    monitoring = [
        sdk.Monitoring('Requests').with_spec('CounterMetricSpec', interval=15),
        sdk.Monitoring('Latency').with_spec('LatencyMetricSpec', interval=15),
        sdk.Monitoring('Accuracy').with_spec('AccuracyMetricSpec'),
        sdk.Monitoring('Drift Detector') \
            .with_health(True) \
            .with_spec(
                kind='ImageAEMetricSpec', 
                threshold=0.15, 
                application=drift_detector_app
            )
    ]

    model = sdk.Model() \
        .with_name(model_name) \
        .with_runtime(TENSORFLOW_RUNTIME) \
        .with_metadata(metadata) \
        .with_payload(payload) \
        .with_signature(signature) \
        .with_monitoring(monitoring)

    result = model.apply(HYDROSPHERE_LINK)
    print(result)

    orchestrator.export_meta("model_version", result["modelVersion"], "txt")
    orchestrator.export_meta("model_link", urllib.parse.urljoin(
        HYDROSPHERE_LINK, f"/models/{result['model']['id']}/{result['id']}/details"), "txt")

def aws_lambda(event, context):
    return main(
        drift_detector_app=event["drift_detector_app"],
        model_name=event["model_name"],
        classes=event["classes"],
        bucket_name=event["bucket_name"],
        data_path=event["data_path"],
        model_path=event["model_path"],
        accuracy=event["accuracy"],
        average_loss=event["average_loss"],
        loss=event["loss"],
        learning_rate=event["learning_rate"],
        batch_size=event["batch_size"],
        epochs=event["epochs"],
        global_step=event["global_step"],
        mlflow_link=event["mlflow_link"],
        storage_path="/tmp/",
    )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--drift-detector-app', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--classes', type=int, required=True)
    parser.add_argument('--bucket-name', required=True)

    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--accuracy', required=True)
    parser.add_argument('--average-loss', required=True)
    parser.add_argument('--loss', required=True)
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--global-step', required=True)
    parser.add_argument('--mlflow-link', required=True)
    
    args = parser.parse_args()
    main(
        drift_detector_app=args.drift_detector_app,
        model_name=args.model_name,
        classes=args.classes,
        bucket_name=args.bucket_name,
        data_path=args.data_path,
        model_path=args.model_path,
        accuracy=args.accuracy,
        average_loss=args.average_loss,
        loss=args.loss,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        global_step=args.global_step,
        mlflow_link=args.mlflow_link,
    )
