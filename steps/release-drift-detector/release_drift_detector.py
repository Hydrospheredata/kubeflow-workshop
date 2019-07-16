import argparse
import os, urllib.parse
from hydrosdk import sdk
from decouple import Config, RepositoryEnv

from storage import *
from orchestrator import * 


config = Config(RepositoryEnv("config.env"))
TENSORFLOW_RUNTIME = config('TENSORFLOW_RUNTIME')
HYDROSPHERE_LINK = config('HYDROSPHERE_LINK')


def main(
    data_path, model_path, model_name, loss, learning_rate, 
    steps, classes, batch_size, bucket_name, storage_path="/"
):

    storage = Storage(bucket_name)
    orchestrator = Orchestrator(storage_path=storage_path)

    # Download model 
    working_dir = os.path.join(storage_path, "model")
    storage.download_prefix(model_path, working_dir)
    
    # Build servable
    payload = [
        os.path.join(storage_path, 'model', 'saved_model.pb'),
        os.path.join(storage_path, 'model', 'variables')
    ]

    metadata = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'data': data_path,
        'model': model_path,
        'loss': loss,
        'steps': steps,
    }

    signature = sdk.Signature('predict') \
        .with_input('imgs', 'float32', [-1, 28, 28, 1], 'image') \
        .with_input('probabilities', 'float32', [-1, classes]) \
        .with_input('class_ids', 'int64', [-1, 1]) \
        .with_input('logits', 'float32', [-1, classes]) \
        .with_input('classes', 'string', [-1, 1]) \
        .with_output('score', 'float32', [-1, 1])

    model = sdk.Model() \
        .with_name(model_name) \
        .with_runtime(TENSORFLOW_RUNTIME) \
        .with_metadata(metadata) \
        .with_payload(payload) \
        .with_signature(signature)

    result = model.apply(HYDROSPHERE_LINK)
    print(result)

    orchestrator.export_meta("model_version", result["modelVersion"], "txt")
    orchestrator.export_meta("model_link", urllib.parse.urljoin(
        HYDROSPHERE_LINK, f"/models/{result['model']['id']}/{result['id']}/details"), "txt")


def aws_lambda(event, context):
    return main(
        data_path=event["data_path"],
        model_path=event["model_path"],
        model_name=event["model_name"],
        loss=event["loss"],
        learning_rate=event["learning_rate"],
        steps=event["steps"],
        classes=event["classes"],
        batch_size=event["batch_size"],
        storage_path="/tmp/",
        bucket_name=event["bucket_name"],
    )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--steps', required=True),
    parser.add_argument('--loss', required=True)
    parser.add_argument('--classes', type=int, required=True),
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        model_path=args.model_path,
        model_name=args.model_name,
        loss=args.loss,
        learning_rate=args.learning_rate,
        steps=args.steps,
        classes=args.classes,
        batch_size=args.batch_size,
        bucket_name=args.bucket_name,
    )



    
