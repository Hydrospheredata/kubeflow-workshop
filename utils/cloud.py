import os, json, urllib.parse, hashlib
import itertools, datetime, logging, sys
from collections import namedtuple

__all__ = ["CloudHelper"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


Bucket = namedtuple('Bucket', ['scheme', 'name', 'full_uri'])

class MLflowMixin:
    """ Mixin for working with the MLflow instance. """

    def set_mlflow_endpoint(self, mlflow_uri, mlflow_experiment="Default"):
        import mlflow 
        self.mlflow = mlflow
        
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment = mlflow_experiment
        self.mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow.set_experiment(self.mlflow_experiment)
        logger.debug(f"Set MLflow to use {self.mlflow_uri} endpoint")

    def set_mlflow_experiment(self, name):
        self.experiment = name
        self.mlflow.set_experiment(self.experiment)

    def log_params(self, params: dict, **kwargs):
        self.mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict, step=None, **kwargs):
        self.mlflow.log_metrics(metrics, step)


class KubeflowMixin:
    """ Mixin for working with the Kubeflow orchestrator. """

    def export_output(self, key: str, value: object, as_root: bool = False, **kwargs):
        if as_root and os.path.dirname(key) != "/": 
            key = os.path.join("/", key)
        with open(key, "w+") as file:
            if isinstance(value, dict): 
                json.dump(value, file)
            else: file.write(str(value))

    def export_outputs(self, outputs: dict, as_root: bool = False, **kwargs): 
        for key, value in outputs.items():
            self.export_output(key, value, as_root=as_root)


class StorageMixin:
    """ Mixin for working with the cloud storage like S3, Google Storage, etc. """ 
    
    @staticmethod
    def _md5_file(filename):
        hash_md5 = hashlib.md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def _md5_string(string):
        return hashlib.md5(string.encode("utf-8")).hexdigest()

    @staticmethod
    def _upload_file_s3(source_path: str, destination_path: str, bucket: Bucket):
        import boto3 
        s3 = boto3.resource('s3')
        
        try:
            head = s3.meta.client.head_object(Bucket=bucket.name, Key=destination_path)
            if head.get("Metadata", {}).get("md5") == StorageMixin._md5_file(source_path):
                return logger.debug("Local and cloud objects are the same, skipping upload")
        except boto3.exceptions.botocore.exceptions.ClientError as e:
            logger.debug(e)
            logger.debug("Uploading new version of the file")
            
        s3.meta.client.upload_file(
            Filename=source_path, 
            Bucket=bucket.name, 
            Key=destination_path, 
            
            # Even though Etag, which is calculated automatically, can be interpreted 
            # as md5 checksum, this does not always retain. In case when the file is 
            # encrypted with AWS KMS, Etag differs from md5. This also relates to multipart 
            # uploads. For this reason, we store md5 checksum on our own. 
            ExtraArgs={
                "Metadata": {
                    "md5": StorageMixin._md5_file(source_path)
                }, 
            },
        )

    @staticmethod
    def _upload_file_gs(source_path: str, destination_path: str, bucket: Bucket):
        from google.cloud import storage
        storage_client = storage.Client()
        storage_client.get_bucket(bucket.name).blob(destination_path).upload_from_filename(source_path)

    @staticmethod
    def _download_file_s3(source_path, destination_path: str, bucket: Bucket):
        import boto3
        s3 = boto3.resource('s3')
        if os.path.exists(destination_path):
            head = s3.meta.client.head_object(Bucket=bucket.name, Key=source_path)
            if StorageMixin._md5_file(destination_path) == head.get("Metadata", {}).get("md5"):
                return logger.debug(
                    "MD5 checksums of the local and the cloud objects are the same, skipping download")
        s3.Object(bucket.name, source_path).download_file(destination_path)

    @staticmethod
    def _download_file_gs(source_path, destination_path: str, bucket: Bucket):
        from google.cloud import storage
        storage_client = storage.Client()
        storage_client.get_bucket(bucket.name).blob(source_path).download_to_filename(destination_path)

    @staticmethod
    def _list_folder_gs(source_folder: str, bucket: Bucket): 
        from google.cloud import storage
        storage_client = storage.Client()
        for blob in storage_client.get_bucket(bucket.name).list_blobs(prefix=source_folder):
            yield '/'.join((bucket.full_uri, blob.name.strip('/'))), \
                os.path.relpath(blob.name, source_folder) 

    @staticmethod
    def _list_folder_s3(source_folder: str, bucket: Bucket):
        import boto3
        s3 = boto3.resource('s3')
        result = s3.meta.client.list_objects_v2(Bucket=bucket.name, Prefix=source_folder)

        if result["IsTruncated"]: 
            raise ValueError("Too much records found under specified path, please specify a prefix " \
                "with a smaller amount of objects.")
        if not result.get("Contents"):
            raise ValueError("Could not find any contents under specified path")

        for path in result["Contents"]:
            yield '/'.join((bucket.full_uri, path["Key"].strip('/'))), \
                os.path.relpath(path["Key"], source_folder)
        
    def get_bucket_from_uri(self, path: str) -> Bucket or None:
        logger.debug(f"Parsing bucket from {path}")
        result = urllib.parse.urlparse(path)
        if result.scheme: return Bucket(result.scheme, result.netloc, f"{result.scheme}://{result.netloc}")
        raise ValueError("`path` must contain full URI")
    
    def get_relative_path_from_uri(self, path: str) -> str:
        result = urllib.parse.urlparse(path)
        return result.path.strip('/')

    def upload_file(self, source_path, destination_path):
        """ Upload a specific file into a destination path. """
        logger.info(f"Uploading {source_path} to {destination_path}")
        assert os.path.isfile(source_path), f"'{source_path}' must be file"

        bucket = self.get_bucket_from_uri(destination_path)
        relative_destination_path = self.get_relative_path_from_uri(destination_path)
        
        if bucket.scheme == 's3': 
            return StorageMixin._upload_file_s3(
                source_path, relative_destination_path, bucket)
        if bucket.scheme == 'gs': 
            return StorageMixin._upload_file_gs(
                source_path, relative_destination_path, bucket)
        raise NotImplementedError("Only aws and gcp clouds are supported")

    def upload_prefix(self, source_folder, destination_folder):
        """ Upload an entire folder into a destination path. """
        assert os.path.isdir(source_folder), "`source_folder` must be directory"

        for root, _, files in os.walk(source_folder):
            for file in files:
                source_path = os.path.relpath(os.path.join(root, file), source_folder)
                destination_path = os.path.join(destination_folder, source_path)
                self.upload_file(os.path.join(root, file), destination_path)

    def download_file(self, source_path, destination_path):
        """ Download a specific file into a desired location. """

        bucket = self.get_bucket_from_uri(source_path)
        relative_source_path = self.get_relative_path_from_uri(source_path)
        relative_destination_path = self.get_relative_path_from_uri(destination_path)

        os.makedirs(os.path.dirname(relative_destination_path), exist_ok=True)
        logger.info(f"Downloading {source_path} to {relative_destination_path}")
        if bucket.scheme == 's3': 
            return StorageMixin._download_file_s3(
                relative_source_path, relative_destination_path, bucket)
        if bucket.scheme == 'gs': 
            return StorageMixin._download_file_gs(
                relative_source_path, relative_destination_path, bucket)
        raise NotImplementedError("Only aws and gcp clouds are supported")

    def download_prefix(self, source_folder, destination_folder):
        """ Download an entire `folder` from the cloud to a desired location. """ 

        for fullpath, relpath in self.list_folder(source_folder):
            download_path = os.path.join(destination_folder, relpath)
            os.makedirs(self.get_relative_path_from_uri(
                os.path.dirname(download_path)), exist_ok=True)
            self.download_file(fullpath, download_path)
    
    def list_folder(self, source_folder):
        """ Yield fullpath, relpath of each item in the folder. """
        
        bucket = self.get_bucket_from_uri(source_folder)
        source_folder = self.get_relative_path_from_uri(source_folder)
        if bucket.scheme == 's3': return iter(StorageMixin._list_folder_s3(source_folder, bucket))
        if bucket.scheme == 'gs': return iter(StorageMixin._list_folder_gs(source_folder, bucket))
        raise NotImplementedError("Only aws and gcp clouds are supported")


class DefaultParamDict(dict):

    def __init__(self, defaults: dict, *args, **kwargs):
        self.defaults = defaults
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        try: 
            return self.__dict__[key]
        except KeyError:
            return self.defaults[key]


class CloudHelper(KubeflowMixin, StorageMixin, MLflowMixin): 

    def __init__(self, default_config_map_params=None, default_logs_path="logs/"):  
        """
        Helps to leverage the underlying infrastructure and minimize the amount of boilerplate 
        code that is needed to be written to track experiments and control training environment.  

        Parameters
        ----------
        default_config_map_params: dict
            Default fallback parameters for ConfigMap.
        default_logs_path: str
            Default path, where logs will be written on the specified bucket. 
        """
        self.default_params = default_config_map_params
        self.default_logs_path = default_logs_path

    def log_execution(self, outputs=None, params=None, metrics=None, 
        logs_file=None, logs_bucket=None, logs_path=None, dev=False, **kwargs
    ):
        """ 
        Logs all produced information of the current execution to the underlying infrastructure. 

        Parameters
        ----------
        outputs: dict
            Information on which the following steps of the workflow will rely on. For example, 
            it might be the cloud path of the artifacts of the trained model, etc.  
        params: dict
            Parameters used in the current execution step to adjust computation. For example, 
            a learning rate value, or a batch size used by the model. 
        metrics: dict 
            Metrics, generated by the current execution step to track experiments. In order to 
            use this functionality, you must set up MLflow endpoint first. See MLflowMixin for 
            more information. 
        logs_file: str
        logs_bucket: str
        logs_path: str 
            Upload logs of the current execution to the specified bucket under a specified path.
            If `logs_path` is not specified, use `self.default_logs_path` instead. Each file is 
            uploaded under ISO UTC timestamp. `outputs` is enriched with the path to the logs. 
        dev: bool
            Flag, symbolizing, whether this is a dev environment. In that case, some functionality
            will behave differently. For example, logs will not be uploaded to the cloud, outputs
            will be exported in the current dir, instead of the root dir. 
        """

        if logs_file and not dev:
            assert logs_bucket, "`logs_bucket` must be provided along with the `logs_file`"
            timestamp = datetime.datetime.utcnow().isoformat("T")
            logs_path = logs_path or self.default_logs_path
            destination_path = os.path.join(
                logs_bucket, logs_path, timestamp, logs_file)
            self.upload_file(logs_file, destination_path)
            if outputs: outputs["logs_path"] = destination_path
            else: outputs = {"logs_path": destination_path}
        
        if outputs: self.export_outputs(outputs, as_root=not dev, **kwargs)
        if metrics: self.log_metrics(metrics, **kwargs)
        if params: self.log_params(params, **kwargs)

    def get_kube_config_map(self, mount_path="/etc/config"):
        """ 
        Reads mounted ConfigMap into Python dictionary.   

        Parameters
        ----------
        mount_path: str
            Path, where ConfigMap was mounted. 
        
        Returns
        -------
        dict
            A dictionary with a key corresponding to a filename and a value corresponding 
            to the file contents. 
        """

        config_map = DefaultParamDict(self.default_params)
        for root, _, files in os.walk(mount_path):
            for file in files:
                key = os.path.join(os.path.relpath(root, mount_path), file)
                with open(os.path.join(root, file), "r") as value: 
                    config_map[os.path.basename(key)] = value.read()
        return config_map