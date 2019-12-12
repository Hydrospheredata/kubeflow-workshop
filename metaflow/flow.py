from metaflow import FlowSpec, Parameter, JSONType, S3, step, current, retry
from hydrosdk import sdk
import json, os, shutil, pprint, time, urllib.parse
import metaflow, numpy as np, requests
import _prepare             as prepare
import _train_classifier    as train_classifier
import _train_monitoring    as train_monitoring


class MNISTFlow(FlowSpec):

    mnist_url                   = Parameter('mnist_url', default='http://yann.lecun.com/exdb/mnist/')
    tensorflow_runtime          = Parameter('tensorflow_runtime', default="hydrosphere/serving-runtime-tensorflow-1.13.1:2.1.0")
    hydrosphere_url             = Parameter('hydrosphere_url', default="http://localhost/")
    bucket                      = Parameter('bucket', default='workshop-hydrosphere-mnist')
    prefix                      = Parameter('prefix', default='metaflow')
    classifier_learning_rate    = Parameter('classifier_learning_rate', default=0.01)
    classifier_batch_size       = Parameter('classifier_batch_size', default=256)
    classifier_epochs           = Parameter('classifier_epochs', default=10)
    classifier_model_name       = Parameter('classifier_model_name', default='metaflow_mnist_classifier')
    monitoring_learning_rate    = Parameter('monitoring_learning_rate', default=0.01)
    monitoring_batch_size       = Parameter('monitoring_batch_size', default=128)
    monitoring_steps            = Parameter('monitoring_steps', default=500)
    monitoring_model_name       = Parameter('monitoring_model_name', default='metaflow_mnist_monitoring')
    test_sample_size            = Parameter('test_sample_size', default=100)
    test_request_delay          = Parameter('test_request_delay', default=0.2)

    def get_run_metadata(self, **kwargs):
        metadata = {
            "flow_name"     : current.flow_name,
            "run_id"        : current.run_id,
            "origin_run_id" : current.origin_run_id,
            "namespace"     : current.namespace,
            "username"      : current.username,
        }
        kwargs_mapping = map(lambda item: (item[0], str(item[1])), kwargs.items())
        metadata.update(dict(list(kwargs_mapping)))
        return metadata

    def upload_model_s3(self, path):
        with S3(run=self, bucket=self.bucket, prefix=self.prefix) as s3:
            upload_tuples = []
            for root, dirs, files in os.walk(path):
                for file in files: 
                    full_path = os.path.join(root, file)
                    upload_tuples.append((full_path, full_path))
            s3.put_files(upload_tuples)

        return os.path.join(
            "s3://{}/metaflow/{}/{}/{}".format(
                self.bucket, current.flow_name, current.run_id, path)
            )
            
    @step
    def start(self):
        self.filenames = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz", 
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]
        self.next(self.prepare, foreach='filenames')

    @step
    def prepare(self):
        local_file, _   = prepare.download_file(self.mnist_url, self.input)
        local_file      = prepare.unpack_file(local_file)

        if local_file.endswith('-images-idx3-ubyte'):
            dataset     = local_file.replace('-images-idx3-ubyte', '')
            data        = prepare.process_images(local_file)
            type        = 'images'
        if local_file.endswith('-labels-idx1-ubyte'):
            dataset     = local_file.replace('-labels-idx1-ubyte', '')
            data        = prepare.process_labels(local_file)
            type        = 'labels'
        
        os.remove(local_file)
        self.data = (dataset, type, data)
        self.next(self.prepare_join)

    @step
    def prepare_join(self, inputs):
        for input in inputs:
            dataset, type, data = input.data
            if type == 'images' and dataset == 'train': 
                self.train_x = data
            if type == 'labels' and dataset == 'train': 
                self.train_y = data.astype(int)
            if type == 'images' and dataset == 't10k': 
                self.test_x = data
            if type == 'labels' and dataset == 't10k': 
                self.test_y = data.astype(int)
        self.next(self.train)

    @step
    def train(self):
        self.next(self.train_classifier, self.train_monitoring)

    @step
    def train_classifier(self):
        estimator = train_classifier.train(
            train_data      = (self.train_x, self.train_y),
            test_data       = (self.test_x, self.test_y),
            learning_rate   = self.classifier_learning_rate,
            batch_size      = self.classifier_batch_size,
            epochs          = self.classifier_epochs
        )
        
        self.classifier_path    = train_classifier.save(estimator)
        self.s3_classifier_path = self.upload_model_s3(self.classifier_path)
        self.next(self.train_join)
    
    @step
    def train_monitoring(self):
        self.monitoring_path = train_monitoring.train(
            train_data      = (self.train_x, self.train_y),
            test_data       = (self.test_x, self.test_y),
            learning_rate   = self.monitoring_learning_rate,
            batch_size      = self.monitoring_batch_size,
            steps           = self.monitoring_steps,
        )
        self.s3_monitoring_path = self.upload_model_s3(self.monitoring_path)
        self.next(self.train_join)
    
    @step
    def train_join(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.release_monitoring)

    @step
    def release_monitoring(self):
        payload_mapping = map(
            lambda x: os.path.join(self.monitoring_path, x), 
            os.listdir(self.monitoring_path))
        metadata = self.get_run_metadata(
            s3_model_path   = self.s3_monitoring_path, 
            learning_rate   = self.monitoring_learning_rate,
            batch_size      = self.monitoring_batch_size,
            training_steps  = self.monitoring_steps)
        model = sdk.Model() \
            .with_name(self.monitoring_model_name) \
            .with_payload(list(payload_mapping)) \
            .with_runtime(self.tensorflow_runtime) \
            .with_metadata(metadata)
        result = model.apply(self.hydrosphere_url)
        print(result)
        self.monitoring_model_version = result["modelVersion"]
        self.next(self.deploy_monitoring)
    
    @step
    def deploy_monitoring(self):
        model = sdk.Model.from_existing(
            name    = self.monitoring_model_name, 
            version = self.monitoring_model_version
        )
        application = sdk.Application.singular(
            name    = self.monitoring_model_name, 
            model   = model
        )
        result = application.apply(self.hydrosphere_url)
        print(result)
        self.next(self.release_classifier)
    
    @step
    def release_classifier(self):
        signature = sdk.Signature('predict') \
            .with_input('imgs', 'float32', [-1, 28, 28, 1], 'image') \
            .with_output('probabilities', 'float32', [-1, 10]) \
            .with_output('class_ids', 'int64', [-1, 1]) \
            .with_output('logits', 'float32', [-1, 10]) \
            .with_output('classes', 'string', [-1, 1])
        payload_mapping = map(
            lambda x: os.path.join(self.classifier_path, x), 
            os.listdir(self.classifier_path))
        monitoring = [
            sdk.Monitoring('Drift Detector') \
                .with_health(True) \
                .with_spec(
                    kind        = 'CustomModelMetricSpec', 
                    threshold   = 0.15, 
                    operator    = "<=",
                    application = self.monitoring_model_name), 
        ]
        metadata = self.get_run_metadata(
            s3_model_path   = self.s3_classifier_path, 
            learning_rate   = self.classifier_learning_rate,
            batch_size      = self.classifier_batch_size,
            epochs          = self.classifier_epochs,
        )
        model = sdk.Model() \
            .with_name(self.classifier_model_name) \
            .with_payload(list(payload_mapping)) \
            .with_runtime(self.tensorflow_runtime) \
            .with_monitoring(monitoring) \
            .with_metadata(metadata)
        result = model.apply(self.hydrosphere_url)
        print(result)
        self.classifier_model_version = result["modelVersion"]
        self.next(self.deploy_classifier_to_stage)

    @step
    def deploy_classifier_to_stage(self):
        model = sdk.Model.from_existing(
            name    = self.classifier_model_name, 
            version = self.classifier_model_version
        )
        application = sdk.Application.singular(
            name    = self.classifier_model_name + "_stage", 
            model   = model
        )
        result = application.apply(self.hydrosphere_url)
        print(result)
        self.next(self.test_endpoint)
    
    @retry
    @step
    def test_endpoint(self):
        predicted = []
        service_link = urllib.parse.urljoin(
            self.hydrosphere_url, f"/gateway/application/{self.classifier_model_name}_stage")
        for index, image in enumerate(self.test_x[:self.test_sample_size]):
            response = requests.post(service_link, json={'imgs': [image.reshape((1, 28, 28, 1)).tolist()]})
            print(f"{index} | {round(index / self.test_sample_size * 100)}% \n{response.text}")
            predicted.append(response.json()["class_ids"][0][0])
            time.sleep(self.test_request_delay)
        accuracy = np.sum(self.test_y[:self.test_sample_size] == np.array(predicted)) / self.test_sample_size
        print(f"Achieved accuracy of {accuracy}")
        self.next(self.deploy_classifier_to_prod)

    @step
    def deploy_classifier_to_prod(self):
        model = sdk.Model.from_existing(
            name    = self.classifier_model_name, 
            version = self.classifier_model_version
        )
        application = sdk.Application.singular(
            name    = self.classifier_model_name, 
            model   = model
        )
        result = application.apply(self.hydrosphere_url)
        print(result)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    MNISTFlow()