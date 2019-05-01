import argparse
import os
from hydrosdk import sdk

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data-path', 
        help='Path, where the current run\'s data was stored',
        required=True)
    parser.add_argument(
        '--mount-path',
        help='Path to PersistentVolumeClaim, deployed on the cluster',
        required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--accuracy', type=float, default=0.9)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--batch-size', required=True)
    
    args = parser.parse_args()
    arguments = args.__dict__

    payload = [
        os.path.join(arguments["model_path"], 'saved_model.pb'),
        os.path.join(arguments["model_path"], 'variables')
    ]

    metadata = {
        'learning_rate': arguments['learning_rate'],
        'epochs': arguments['epochs'],
        'batch_size': arguments['batch_size'],
        'accuracy': str(arguments['accuracy']),
        'data': arguments['data_path']
    }

    signature = sdk.Signature('predict')\
        .with_input('imgs', 'float32', [-1, 28, 28, 1], 'image')\
        .with_output('probabilities', 'float32', [-1, 10])\
        .with_output('class_ids', 'int64', [-1, 1])\
        .with_output('logits', 'float32', [-1, 10])\
        .with_output('classes', 'string', [-1, 1])

    monitoring = [
        sdk.Monitoring('Requests').with_spec('CounterMetricSpec', interval = 15),
        sdk.Monitoring('Latency').with_spec('LatencyMetricSpec', interval = 15),
        sdk.Monitoring('Accuracy').with_spec('AccuracyMetricSpec'),
        sdk.Monitoring('Autoencoder').with_health(True).with_spec('ImageAEMetricSpec', threshold=0.15, application='mnist-concept-app')
    ]

    model = sdk.Model()\
        .with_name(arguments['model_name'])\
        .with_runtime('hydrosphere/serving-runtime-tensorflow-1.13.1:latest')\
        .with_metadata(metadata)\
        .with_payload(payload)\
        .with_signature(signature)

    result = model.apply(arguments['hydrosphere_address'])
    print(result)

# i.  Upload the model to Hydrosphere Serving
# ii. Parse the status of the model uploading, retrieve the built 
#     model version and write it to the `/model_version.txt` file. 
    with open('/model-version.txt', 'w') as f:
        f.write(str(result['modelVersion']))
