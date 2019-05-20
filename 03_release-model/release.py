import argparse
import os, boto3
from hydrosdk import sdk

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--models-path', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--autoencoder-app', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--accuracy')
    parser.add_argument('--loss')
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--epochs'),
    parser.add_argument('--steps'),
    parser.add_argument('--classes', type=int, required=True),
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--dev', help='Flag for development purposes', action="store_true")
    
    args = parser.parse_args()
    s3 = boto3.resource('s3')

    # Download model 
    os.makedirs("model", exist_ok=True)
    for file in s3.Bucket('odsc-workshop').objects.filter(Prefix=args.models_path):
        relevant_folder = file.key.split("/")[4:]

        # Create nested folders if necessary
        if len(relevant_folder) > 1:
            os.makedirs(os.path.join('model', *relevant_folder[:-1]), exist_ok=True)
        
        s3.Object(file.bucket_name, file.key).download_file(os.path.join('model', *relevant_folder))

    # Build servable
    payload = [
        os.path.join('model', 'saved_model.pb'),
        os.path.join('model', 'variables')
    ]

    metadata = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'data': args.data_path,
        'model': args.models_path
    }

    if args.epochs:
        metadata["epochs"] = args.epochs
    if args.accuracy:
        metadata["accuracy"] = args.accuracy
    if args.loss:
        metadata["loss"] = args.loss
    if args.steps:
        metadata["steps"] = args.steps

    signature = sdk.Signature('predict')\
        .with_input('imgs', 'float32', [-1, 28, 28, 1], 'image')\
        .with_output('probabilities', 'float32', [-1, args.classes])\
        .with_output('class_ids', 'int64', [-1, 1])\
        .with_output('logits', 'float32', [-1, args.classes])\
        .with_output('classes', 'string', [-1, 1])

    monitoring = [
        sdk.Monitoring('Requests').with_spec('CounterMetricSpec', interval=15),
        sdk.Monitoring('Latency').with_spec('LatencyMetricSpec', interval=15),
        sdk.Monitoring('Accuracy').with_spec('AccuracyMetricSpec'),
        sdk.Monitoring('Autoencoder') \
            .with_health(True) \
            .with_spec(
                kind='ImageAEMetricSpec', 
                threshold=0.15, 
                application=args.autoencoder_app
            )
    ]

    model = sdk.Model() \
        .with_name(args.model_name) \
        .with_runtime('hydrosphere/serving-runtime-tensorflow-1.13.1:latest') \
        .with_metadata(metadata) \
        .with_payload(payload) \
        .with_signature(signature) \
        .with_monitoring(monitoring)

    result = model.apply(args.hydrosphere_address)
    print(result)

    # Dump built model version 
    with open("./model_version.txt" if args.dev else "/model_version.txt", 'w+') as file:
        file.write(str(result['modelVersion']))
