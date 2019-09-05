import boto3, botocore, sagemaker as sagemaker_library
import os, json, datetime, argparse


def main(
    timestamp: str, 
    role_arn: str, 
    data_path: str, 
    training_image: str, 
    learning_rate: tuple, 
    batch_size: tuple,
    epochs: int,
    instance_type="ml.m4.xlarge", 
    instance_count=2, 
    max_number_training_jobs=12, 
    max_parallel_training_jobs=4,
    max_runtime_seconds=3600, 
):
    """ Start Hyper-Parameter Training Job """

    cloud = CloudHelper()
    sagemaker_client = boto3.resource('sagemaker')
    timestamp = cloud.get_timestamp_from_path(args.data_path)
    bucket_name = cloud.get_bucket_from_path(args.data_path)
    
    tuning_job_config = {
        "HyperParameterTuningJobObjective": { 
            "MetricName": "loss",
            "Type": "Minimize"
        },
        "ParameterRanges": {
            "CategoricalParameterRanges": [],
            "ContinuousParameterRanges": [
                {
                    "Name": "learning_rate",
                    "MinValue": learning_rate[0],
                    "MaxValue" : learning_rate[1],
                    "ScalingType": "Auto"
                }
            ],
            "IntegerParameterRanges": [
                {
                    "Name": "batch_size",
                    "MinValue": batch_size[0],
                    "MaxValue": batch_size[1],
                    "ScalingType": "Auto"
                }
            ]
        },
        "ResourceLimits": { 
            "MaxNumberOfTrainingJobs": max_number_training_jobs,
            "MaxParallelTrainingJobs": max_parallel_training_jobs
        },
        "Strategy": "Bayesian",
        "TrainingJobEarlyStoppingType": "Off"
    }
    training_job_definition = {
        "AlgorithmSpecification": { 
            "MetricDefinitions": [ 
                {
                    "Name": "loss",
                    "Regex": "loss = (.*?),"
                }
            ],
            "TrainingImage": training_image,
            "TrainingInputMode": "File"
        },
        "InputDataConfig": [ 
            { 
                "ChannelName": "train",
                "ContentType": "npz",
                "DataSource": { 
                    "S3DataSource": { 
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": os.path.join(data_path, "train")
                    }
                }
            },
            { 
                "ChannelName": "t10k",
                "ContentType": "npz",
                "DataSource": { 
                    "S3DataSource": { 
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": os.path.join(data_path, "t10k")
                    }
                }
            }
        ],
        "OutputDataConfig": { 
            "S3OutputPath": os.path.join(bucket_name, "hyper-parameter-tuning", timestamp)
        },
        "ResourceConfig": {
            "InstanceCount": instance_count,
            "InstanceType": instance_type,
            "VolumeSizeInGB": 10
            },
        "RoleArn": role_arn,
        "StaticHyperParameters": { 
            "epochs" : epochs
        },
        "StoppingCondition": { 
            "MaxRuntimeInSeconds": max_runtime_seconds
        }
    }

    tuning_job_name = f"mnist-tuning-job-{timestamp}"
    sagemaker_client.create_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name,
        HyperParameterTuningJobConfig=tuning_job_config,
        TrainingJobDefinition=training_job_definition)
    sagemaker_library.tuner.HyperparameterTuner.attach(tuning_job_name).wait()
    
    best_job_summary = sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name)["BestTrainingJob"]
    
    # cloud.export_metas(best_job_summary["TunedHyperParameters"], key_prefix="best_")
    # cloud.export_metas(best_job_summary["FinalHyperParameterTuningJobObjectiveMetric"], key_prefix="objective_")
    # cloud.export_metas({
    #     "tuning_job_name": tuning_job_name, 
        
    #     "best_training_job_name": best_job_summary["TrainingJobName"],
    #     "creation_time": best_job_summary["CreationTime"],
    #     "training_start_time": best_job_summary["TrainingStartTime"],
    #     "training_end_time": best_job_summary["TrainingEndTime"],
    # })


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()

    
