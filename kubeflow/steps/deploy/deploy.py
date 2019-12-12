import logging, sys, os

os.makedirs("logs", exist_ok=True)
logs_file = "logs/step_deploy.log"
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s.%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logs_file)])
logger = logging.getLogger(__name__)

import argparse, datetime
import os, urllib.parse, pprint
from hydrosdk import sdk
import wo


def main(model_name, model_version, application_name, hydrosphere_uri, *args, **kwargs):
    logger.info(f"Referencing existing model `{model_name}:{model_version}`")
    model = sdk.Model.from_existing(model_name, model_version)
    logger.info(f"Creating singular application `{application_name}`")
    application = sdk.Application.singular(application_name, model)
    logger.info(f"Applying application to the cluster `{hydrosphere_uri}`")
    result = application.apply(hydrosphere_uri)
    logger.info(pprint.pformat(result))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--application-name-postfix', required=True)
    parser.add_argument('--dev', action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")

    logs_bucket = wo.parse_bucket(args.data_path, with_scheme=True)
    params = {"uri.hydrosphere": "http://localhost"}

    with wo.Orchestrator(
        logs_file=logs_file, logs_bucket=logs_bucket,
        default_params=params, dev=args.dev,
    ) as w: 
    
        
        # Execute main script
        config = w.get_config()
        application_name = f"{args.model_name}{args.application_name_postfix}"
        main(
            **vars(args), 
            hydrosphere_uri=config["uri.hydrosphere"],
            application_name=application_name,
        )

        # Execution logging 
        w.log_execution(
            outputs={
                "application_name": f"{args.model_name}{args.application_name_postfix}",
                "application_uri": urllib.parse.urljoin(
                    config["uri.hydrosphere"], f"applications/{application_name}"),
            }
        )
