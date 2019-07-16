import argparse, csv, datetime
import os, urllib.parse
from hydrosdk import sdk
from decouple import Config, RepositoryEnv

from storage import * 
from orchestrator import *


config = Config(RepositoryEnv("config.env"))
HYDROSPHERE_LINK = config("HYDROSPHERE_LINK")


def main(
    model_version, model_name, application_name_postfix, 
    bucket_name, storage_path="/", **kwargs
):
    
    # Define helper class
    storage = Storage(bucket_name)
    orchestrator = Orchestrator(storage_path=storage_path)

    # Create and deploy endpoint application
    application_name = f"{model_name}{application_name_postfix}"
    model = sdk.Model.from_existing(model_name, model_version)
    
    application = sdk.Application.singular(application_name, model)
    result = application.apply(HYDROSPHERE_LINK)
    print(result)

    # Export meta to the orchestrator
    application_link = urllib.parse.urljoin(HYDROSPHERE_LINK, f"applications/{application_name}")
    orchestrator.export_meta("application_name", application_name, "txt")
    orchestrator.export_meta("application_link", application_link, "txt")
    
    if kwargs.get("mlflow_model_link"):

        with open(os.path.join(storage_path, 'output.csv'), 'w+', newline='') as file:
            fieldnames = ['key', 'value']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({'key': 'mlflow-model-link', 'value': kwargs["mlflow_model_link"]})
            writer.writerow({'key': 'mlflow-drift-detector-link', 'value': kwargs["mlflow_drift_detector_link"]})
            writer.writerow({'key': 'application-link', 'value': application_link})
            writer.writerow({'key': 'data-path', 'value': kwargs["data_path"]})
            writer.writerow({'key': 'model-path', 'value': kwargs["model_path"]})
            writer.writerow({'key': 'model-drift-detector-path', 'value': kwargs["model_drift_detector_path"]})

        run_path = os.path.join("run", str(round(datetime.datetime.now().timestamp())))
        output_cloud_path = storage.upload_file(
            os.path.join(storage_path, 'output.csv'), 
            os.path.join(run_path, "output.csv"))

        orchestrator.export_meta(
            key="mlpipeline-ui-metadata", 
            value={
                'outputs': [{
                    'type': 'table',
                    'storage': storage.prefix,
                    'format': 'csv',
                    'source': output_cloud_path,
                    'header': ['key', 'value'],
                }]
            }, 
            extension="json"
        )


def aws_lambda(event, context):
    return main(
        model_version=event["model_version"],
        model_name=event["model_name"],
        application_name_postfix=event["application_name_postfix"],
        bucket_name=event["bucket_name"],
        storage_path="/tmp/"
    )


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--application-name-postfix', required=True)
    parser.add_argument('--bucket-name', required=True)
    parser.add_argument('--mlflow-model-link')
    parser.add_argument('--mlflow-drift-detector-link')
    parser.add_argument('--data-path')
    parser.add_argument('--model-path')
    parser.add_argument('--model-drift-detector-path')
    
    args = parser.parse_args()
    main(
        model_version=args.model_version,
        model_name=args.model_name,
        application_name_postfix=args.application_name_postfix, 
        bucket_name=args.bucket_name,
        mlflow_model_link=args.mlflow_model_link,
        mlflow_drift_detector_link=args.mlflow_drift_detector_link,
        data_path=args.data_path,
        model_path=args.model_path,
        model_drift_detector_path=args.model_drift_detector_path,
    )
