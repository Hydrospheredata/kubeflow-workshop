import argparse, datetime
import os, csv, logging, sys
from cloud import CloudHelper


logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("download.log")])
logger = logging.getLogger(__name__)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-artifacts-path', required=True)
    parser.add_argument('--drift-detector-artifacts-path', required=True)
    parser.add_argument('--model-uri', required=True)
    parser.add_argument('--drift-detector-uri', required=True)
    parser.add_argument('--model-application-uri', required=True)
    parser.add_argument('--drift-detector-application-uri', required=True)
    parser.add_argument('--integration-test-accuracy', required=True)
    parser.add_argument('--dev', action="store_true", default=False)
    
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Parsed unknown args: {unknown}")
    
    with open('output.csv', 'w+', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['key', 'value'])
        writer.writerows([
            {'key': 'data-path', 'value': args.data_path},
            {'key': 'model-artifacts-path', 'value': args.model_artifacts_path},
            {'key': 'drift-detector-artifacts-path', 'value': args.drift_detector_artifacts_path},
            {'key': 'model-uri', 'value': args.model_uri},
            {'key': 'drift-detector-uri', 'value': args.drift_detector_uri},
            {'key': 'model-application-uri', 'value': args.model_application_uri},
            {'key': 'drift-detector-application-uri', 'value': args.drift_detector_application_uri},
            {'key': 'integration-test-accuracy', 'value': args.integration_test_accuracy},
        ])

    cloud = CloudHelper()
    bucket = cloud.get_bucket_from_uri(args.data_path)
    output_path = os.path.join(
        bucket.full_uri, "mnist/run", str(round(datetime.datetime.now().timestamp())))
    cloud.upload_file('output.csv', os.path.join(output_path, 'output.csv'))
    cloud.log_execution(
        outputs={
            "mlpipeline-ui-metadata.json": {
                'outputs': [{
                    'type': 'table',
                    'storage': bucket.scheme,
                    'format': 'csv',
                    'source': os.path.join(output_path, 'output.csv'),
                    'header': ['key', 'value']
                }]
            },
        },
        dev=args.dev,
    )