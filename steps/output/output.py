import logging, sys, os

os.makedirs("logs", exist_ok=True)
logs_file = "logs/step_output.log"
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s.%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logs_file)])
logger = logging.getLogger(__name__)

import argparse, datetime
import os, csv, wo 

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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
    
    bucket = wo.parse_bucket(args.data_path, with_scheme=True)
    timestamp = datetime.datetime.now().isoformat("T")
    outputs = [(OUTPUTS_DIR, os.path.join(bucket, "run", timestamp))]
    
    with wo.Orchestrator(
        outputs=outputs, dev=args.dev, logs_file=logs_file, 
        logs_bucket=wo.parse_bucket(args.data_path, with_scheme=True),
    ) as w:

        # Execute main script
        with open(os.path.join(OUTPUTS_DIR, 'output.csv'), 'w+', newline='') as file:
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

        w.log_execution(outputs={
            "mlpipeline-ui-metadata.json": {
                'outputs': [{
                    'type': 'table',
                    'storage': 's3',
                    'format': 'csv',
                    'source': os.path.join(bucket, 'run', timestamp, 'output.csv'),
                    'header': ['key', 'value']
                }]
            },
        })
