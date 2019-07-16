import urllib.parse
import os, itertools

__all__ = ["Storage"]


class Storage:

    def __init__(self, bucket_name):
        result = urllib.parse.urlparse(bucket_name)
        assert result.scheme, "Full URI to the bucket must be provided"
        assert result.scheme in ('s3', 'gs'), "URI scheme must be either s3 or gs"

        self.prefix = result.scheme
        self.bucket_name = result.netloc
        self.full_name = bucket_name

        if self.prefix == 's3': self.cloud = 'aws' 
        if self.prefix == 'gs': self.cloud = 'gcp' 

        print(f"Initialized {self}")
    
    def __repr__(self):
        return f"Storage(cloud={self.cloud}, bucket_name={self.bucket_name})"

    def upload_file(self, source_path, destination_path):
        upload_path = None

        if self.cloud == 'aws':
            upload_path = _upload_file_s3(source_path, destination_path, self.bucket_name)
        if self.cloud == 'gcp':
            upload_path = _upload_file_gs(source_path, destination_path, self.bucket_name)

        print('File {} has been uploaded to {}'.format(source_path, upload_path), flush=True)
        return upload_path

    def download_file(self, source_path, destination_path):
        result = urllib.parse.urlparse(source_path)
        if result.scheme: source_path = result.path[1:]

        if self.cloud == 'aws':
            _download_file_s3(source_path, destination_path, self.bucket_name)
        if self.cloud == 'gcp':
            _download_file_gs(source_path, destination_path, self.bucket_name)
        
        print('File {} has been downloaded to {}'.format(
            source_path, destination_path), flush=True)

    def download_prefix(self, source_folder, destination_folder):
        result = urllib.parse.urlparse(source_folder)
        if result.scheme: source_folder = result.path[1:]

        if self.cloud == 'aws':
            raise NotImplementedError
        if self.cloud == 'gcp':
            for path, prefix in _list_folder_gs(source_folder, self.bucket_name):
                relpath = os.path.relpath(path, prefix)
                relpath = os.path.join(destination_folder, relpath)
                os.makedirs(os.path.dirname(relpath), exist_ok=True)
                _download_file_gs(path, relpath, self.bucket_name)
    

def _upload_file_s3(source_path, destination_path, bucket_name):
    import boto3 
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(Filename=source_path, Bucket=bucket_name, Key=destination_path)
    return f"s3://{bucket_name}/{destination_path}"


def _download_file_s3(source_path, destination_path, bucket_name):
    import boto3
    s3 = boto3.resource('s3')
    s3.Object(bucket_name, source_path).download_file(destination_path)


def _upload_file_gs(source_path, destination_path, bucket_name):
    from google.cloud import storage
    storage_client = storage.Client()
    storage_client.get_bucket(bucket_name).blob(destination_path).upload_from_filename(source_path)    
    return f"gs://{bucket_name}/{destination_path}"
    

def _download_file_gs(source_path, destination_path, bucket_name):
    from google.cloud import storage
    storage_client = storage.Client()
    storage_client.get_bucket(bucket_name).blob(source_path).download_to_filename(destination_path)

def _list_folder_gs(source_folder, bucket_name): 
    def _all_equal(items):
        return all(item == items[0] for item in items)

    from google.cloud import storage
    storage_client = storage.Client()

    filenames = []
    for blob in storage_client.get_bucket(bucket_name).list_blobs(prefix=source_folder):
        filenames.append(blob.name)

    common_prefix = ''.join(map(lambda x: x[0], itertools.takewhile(_all_equal, zip(*filenames))))
    for filename in filenames:
        yield filename, common_prefix 