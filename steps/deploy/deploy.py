import logging, sys

logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("deploy.log")])
logger = logging.getLogger(__name__)

import argparse, datetime
import os, urllib.parse, pprint
from hydrosdk import sdk
import wo


def main(model_name, model_version, application_name, hydrosphere_uri):
    logger.info(f"Referencing existing model `{model_name}:{model_version}`")
    model = sdk.Model.from_existing(model_name, model_version)
    logger.info(f"Creating singular application `{application_name}`")
    application = sdk.Application.singular(application_name, model)
    logger.info(f"Applying application to the cluster `{hydrosphere_uri}`")
    result = application.apply(hydrosphere_uri)
    logger.info(pprint.pformat(result))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)  # Required for inferring bucket, where to store logs
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--application-name-postfix', required=True)
    parser.add_argument('--dev', action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")

    w = wo.Orchestrator(
        default_logs_path="mnist/logs",
        default_params={
            "uri.hydrosphere": "https://dev.k8s.hydrosphere.io"
        },
        dev=args.dev,
    )
    config = w.get_config()
    
    try:

        # Download artifacts
        pass

        # Initialize runtime variables
        hydrosphere_uri = config["uri.hydrosphere"]
        application_name = f"{args.model_name}{args.application_name_postfix}"

        # Execute main script
        main(
            model_name=args.model_name,
            model_version=args.model_version,
            application_name=application_name, 
            hydrosphere_uri=hydrosphere_uri,
        )

        # Prepare variables for logging
        application_uri = urllib.parse.urljoin(
            config["uri.hydrosphere"], f"applications/{application_name}")

        # Upload artifacts
        pass 
        
    except Exception as e:
        logger.exception("Main execution script failed")
    
    finally: 
        scheme, bucket, path = w.parse_uri(args.data_path)
        w.log_execution(
            outputs={
                "application_name": application_name,
                "application_uri": application_uri,
            },
            logs_bucket=f"{scheme}://{bucket}",
            logs_file="deploy.log",
        )
