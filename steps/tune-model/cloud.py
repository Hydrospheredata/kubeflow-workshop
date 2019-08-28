import os, json, urllib.parse, itertools, datetime

__all__ = ["CloudHelper"]


class KubeflowMixin:
    """ Mixin for working with the Kubeflow orchestrator. """

    def export_meta(self, key, value, extension=None):
        filename = key if not extension else f"{key}.{extension}"
        with open(filename, "w+") as file:
            json.dump(value, file)

    def export_metas(self, metas, key_prefix=None): 
        key = f"{key_prefix}{key}" if key_prefix else key
        for key, value in metas.items():
            self.export_meta(key, value)

    
class StorageMixin:
    """ Mixin for working with the cloud storage like S3, Google Storage, etc. """ 

    @property
    def cloud(self):
        assert self.bucket_name_uri, "Please, specify the bucket <object>.set_bucket(...)"

        if self.scheme == 's3': return 'aws'
        if self.scheme == 'gs': return 'gcp'

    def set_bucket(self, bucket_name_uri):
        result = urllib.parse.urlparse(bucket_name_uri)
        assert result.scheme, "Full URI to the bucket must be provided"
        assert result.scheme in ('s3', 'gs', 'https'), "URI scheme must be either s3, gs, or https"
        # Azure URI https://myaccount.blob.core.windows.net/mycontainer/myblob   

        self.scheme = result.scheme
        self.bucket_name = result.netloc
        self.bucket_name_uri = bucket_name_uri
        return self

    def upload_file(self, source_path, destination_path):
        """ Upload a specific file into a destination path. """

        if self.cloud == 'aws': return self._upload_file_s3(source_path, destination_path)
        if self.cloud == 'gcp': return self._upload_file_gs(source_path, destination_path)

    def upload_prefix(self, source_folder, destination_folder):
        """ Upload an entire folder into a destination path. """

        for root, _, files in os.walk(source_folder):
            for file in files:
                source_path = os.path.relpath(os.path.join(root, file), source_folder)
                destination_path = os.path.join(destination_folder, source_path)
                self.upload_file(os.path.join(root, file), destination_path)

    def download_file(self, source_path, destination_path):
        """ Download a specific file into a desired location. """

        result = urllib.parse.urlparse(source_path)
        if result.scheme: source_path = result.path[1:]  # uri -> related_path
        if self.cloud == 'aws': return self._download_file_s3(source_path, destination_path)
        if self.cloud == 'gcp': return self._download_file_gs(source_path, destination_path)
    
    def download_prefix(self, source_folder, destination_folder):
        """ Download an entire `folder` from the cloud to a desired location. """ 

        result = urllib.parse.urlparse(source_folder)
        if result.scheme: source_folder = result.path[1:]  # uri -> related_path
        for fullpath, relpath in self.list_folder(source_folder):
            download_path = os.path.join(destination_folder, relpath)
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            self.download_file(fullpath, download_path)
    
    def list_folder(self, source_folder):
        """ Yield fullpath, relpath of each item in the folder. """

        if self.cloud == 'aws': return iter(self._list_folder_s3(source_folder))
        if self.cloud == 'gcp': return iter(self._list_folder_gs(source_folder))

    def _upload_file_s3(self, source_path, destination_path):
        import boto3 
        s3 = boto3.resource('s3')
        s3.meta.client.upload_file(Filename=source_path, Bucket=self.bucket_name, Key=destination_path)
        return f"{self.bucket_name_uri}/{destination_path}"

    def _upload_file_gs(self, source_path, destination_path):
        from google.cloud import storage
        storage_client = storage.Client()
        storage_client.get_bucket(self.bucket_name).blob(destination_path).upload_from_filename(source_path)    
        return f"{self.bucket_name_uri}/{destination_path}"

    def _download_file_s3(self, source_path, destination_path):
        import boto3
        s3 = boto3.resource('s3')
        s3.Object(self.bucket_name, source_path).download_file(destination_path)

    def _download_file_gs(self, source_path, destination_path):
        from google.cloud import storage
        storage_client = storage.Client()
        storage_client.get_bucket(self.bucket_name).blob(source_path).download_to_filename(destination_path)

    def _list_folder_gs(self, source_folder): 
        from google.cloud import storage
        storage_client = storage.Client()
        for blob in storage_client.get_bucket(self.bucket_name).list_blobs(prefix=source_folder):
            yield blob.name, os.path.relpath(blob.name, source_folder) 
    
    def _list_folder_s3(self, source_folder):
        import boto3
        s3 = boto3.resource('s3')
        result = s3.meta.client.list_objects_v2(Bucket=self.bucket_name, Prefix=source_folder)
        assert not result["IsTruncated"], "Response is truncated due to hitting 1000 objects limit. " \
            "Please, specify the prefix with a smaller amount of objects."
        for path in result["Contents"]:
            yield path["Key"], os.path.relpath(path["Key"], source_folder)


class DefaultParamDict(dict):

    def __init__(self, defaults: dict, *args, **kwargs):
        self.defaults = defaults
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        try: 
            return self.__dict__[key]
        except KeyError:
            return self.defaults[key]


class CloudHelper(KubeflowMixin, StorageMixin): 
    """ This help class is dedicated to ease the work with Kubeflow and the Cloud """

    def __init__(self, default_params=None, *args, **kwargs):
        self.default_params = default_params

    def get_kube_config_map(self, mount_path="/etc/config"):
        """ Reads every file in the `mount_path` and places the contents into
            a dictionary with the key corresponding to a filename. """

        config_map = DefaultParamDict(self.default_params)
        for root, _, files in os.walk(mount_path):
            for file in files:
                key = os.path.join(os.path.relpath(root, mount_path), file)
                with open(os.path.join(root, file), "r") as value: 
                    config_map[os.path.basename(key)] = value.read()
        return config_map
    
    def get_timestamp_from_path(self, path: str) -> str:
        for part in path.split("/"):
            try: 
                datetime.datetime.fromtimestamp(int(part))
                return part
            except (ValueError, OverflowError):
                continue
        raise ValueError("Cannot find any integer in the path")
    
    def get_bucket_from_path(self, path: str, strict=True) -> str:
        result = urllib.parse.urlparse(path)
        if strict: assert result.scheme, "In strict=True mode full bucket URI must be provided."
        if strict: assert result.scheme in ('s3', 'gs'), "In strict=True mode full bucket URI must be provided."
        return f"{result.scheme}://{result.netloc}"


if __name__ == "__main__": 
    cloud = CloudHelper(defaults={"uri.mnist": "http://yann.lecun.com/exdb/mnist/"})
    cloud.set_bucket("s3://odsc-workshop")
    cloud.download_prefix("dev/data/mnist/1558442049", "data")