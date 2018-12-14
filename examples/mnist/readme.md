# Train, Deliver and Monitor ML Models to Production with a Single Command

Very often a workflow of training models and delivering them to the production environment contains huge gaps of manual work. Those could be either building a Docker image and deploying it to the Kubernetes cluster or packing the model to the Python package and installing it to your Python application. Or even changing your Java classes with the defined weights and re-compiling the whole project. Not to mention that all of this should be followed by testing your model's performance (which is not just a check of the ability to compile). Can this be interpreted as continuous delivery if you do it by hands? What if you could run the whole process of assembling/training/deploying/testing/running model via single command in your terminal? 

Let me show you how you can build the whole workflow of data gathering / model training / model deployment / model testing in a single file and how you can tune hyperparameters for all of this stages from one place. 

## Prerequisites

This tutorial will be done on a Kubernetes single node cluster created with Minikube. You will additionally need: 

- Docker
- Helm
- Kubectl 
- Ksonnet
- Argo

## Environment Preparation

Start a cluster and install Helm Tiller on it. 

```sh
$ minikube start 
$ helm init
```

After that create a working directory where will be stored model's files and initialize a ksonnet project inside that directory.

```sh
$ mkdir mnist; cd mnist
$ ks init demo; cd demo
```

Now you would need to deploy Argo and Tensorflow operators to the cluster. 

```sh
$ ks registry add kubeflow github.com/kubeflow/kubeflow/tree/v0.2.5/kubeflow
$ ks pkg install kubeflow/core@v0.2.5
$ ks pkg install kubeflow/argo 
$ ks generate core kubeflow-core --name=kubeflow-core
$ ks generate argo kubeflow-argo --name=kubeflow-argo
$ ks apply default -c kubeflow-core
$ ks apply default -c kubeflow-argo

$ cd ..  # cd to the parent `mnist` directory where we'll be working
```

Once you've done that, deploy a ML Lambda serving platform.

```sh
$ helm repo add hydro-serving https://hydrospheredata.github.io/hydro-serving-helm/
$ helm install --name serving hydro-serving/serving
```

We will additionally need a few resources on this cluster. PersistentVolumeClaims to store the data and trained models. 

```yaml
# pvc.yaml

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 20Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 20Gi
```

```sh 
$ kubectl apply -f pvc.yaml
```

Minikube v0.26+ uses RBAC by default. We will need to grant access for training ops. Let's define `service-account.yaml` for it. 

```yaml
# service-account.yaml

apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: tf-user
rules:
- apiGroups: [""]
  resources: ["pods", "pods/exec", "services"]
  verbs: ["create", "get", "list", "watch", "update", "patch"]
- apiGroups: [""]
  resources: ["configmaps", "serviceaccounts", "secrets"]
  verbs: ["get", "watch", "list"]
- apiGroups: [""]
  resources: ["persistentvolumeclaims"]
  verbs: ["create", "delete"]
- apiGroups: ["apps", "extensions", "batch"]
  resources: ["deployments", "jobs"]
  verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
- apiGroups: ["argoproj.io"]
  resources: ["workflows"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: ["kubeflow.org"]
  resources: ["tfjobs", "jobs"]
  verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: tf-user
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: tf-user
subjects:
- kind: ServiceAccount
  name: tf-user
  namespace: default
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: tf-user
```

```sh
$ kubectl apply -f service-account.yaml
```

Steps above define the whole environment that we need for our needs. Next we will create a simple MNIST classifier. Let's assemble the data. 

## Data gathering

Create directory `image`. This directory will hold all files of our model. (Think of it as of Docker image, not the actual digit picture).

```sh
$ mkdir image; cd image
```

MNIST dataset is located under the http://yann.lecun.com/exdb/mnist/ address in the binary format. We will download that data, process it into a numpy array and store it on the mounted path. 

```python
# domnload-mnist.py

from PIL import Image
import struct, numpy
import os, gzip, tarfile, shutil, glob
import urllib, urllib.parse, urllib.request


def download_files(base_url, base_dir, files):
    """ Download required data """

    downloaded = []
    os.makedirs(base_dir, exist_ok=True)

    for file in files:
        print(f"Started downloading {file}")
        download_url = urllib.parse.urljoin(base_url, file)
        download_path = os.path.join(base_dir, file)
        local_file, _ = urllib.request.urlretrieve(download_url, download_path)
        unpack_file(local_file, base_dir)
    
    return downloaded


def unpack_file(file, base_dir):
    """ Unpack the compressed file. """

    print(f"Unpacking {file}")
    with gzip.open(file, 'rb') as f_in, open(file[:-3],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file)


def preprocess_mnist_files(path, dataset):
    """ Preprocess downloaded MNIST datasets. """
    
    print(f"Preprocessing {os.path.join(path, dataset)}")
    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = numpy.fromfile(file, dtype=numpy.int8) #int8
        new_labels = numpy.zeros((num, 10))
        new_labels[numpy.arange(num), labels] = 1

    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = numpy.fromfile(file, dtype=numpy.uint8).reshape(num, rows, cols) #uint8
        imgs = imgs.astype(numpy.float32) / 255.0

    os.remove(label_file); os.remove(img_file)
    numpy.savez_compressed(os.path.join(path, dataset), imgs=imgs, labels=labels)


if __name__ == "__main__": 
    mnist_dir = os.environ.get("MNIST_DATA_DIR", "data/mnist")
    mnist_files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']
    
    downloaded = download_files(
        base_url="http://yann.lecun.com/exdb/mnist/", 
        base_dir=mnist_dir, 
        files=mnist_files)
    preprocess_mnist_files(mnist_dir, "train")
    preprocess_mnist_files(mnist_dir, "t10k")
```

As you can see files will be stored either in the directory defined by the `MNIST_DATA_DIR` environmnet variable, or locally under the `data/mnist` path if `MNIST_DATA_DIR` variable is unset. 

For the testing purposes we will also download a `notMNIST` dataset, which closely resembles MNIST - same 28x28px images, same 10 classes, but now those are letters and not digits.  

```python
# download-notmnist.py

from PIL import Image
import struct, numpy
import os, gzip, tarfile, shutil, glob
import urllib, urllib.parse, urllib.request


def download_files(base_url, base_dir, files):
    """ Download required data """

    downloaded = []
    os.makedirs(base_dir, exist_ok=True)

    for file in files:
        print(f"Started downloading {file}")
        download_url = urllib.parse.urljoin(base_url, file)
        download_path = os.path.join(base_dir, file)
        local_file, _ = urllib.request.urlretrieve(download_url, download_path)
        unpack_file(local_file, base_dir)
    
    return downloaded


def unpack_file(file, base_dir):
    """ Unpack the compressed file. """

    print(f"Unpacking {file}")
    with tarfile.open(file) as f_tar: 
        f_tar.extractall(base_dir)
    os.remove(file)


def preprocess_notmnist_files(path, dataset):
    """ Preprocess downloaded notMNIST datasets. """

    print(f"Preprocessing {os.path.join(path, dataset)}")
    files = glob.glob(os.path.join(path, dataset, "**", "*.png"))
    imgs = numpy.zeros((len(files), 28, 28))
    labels = numpy.zeros((len(files),))
    for index, file in enumerate(files): 
        try: 
            label = os.path.split(os.path.dirname(file))[-1]
            imgs[index,:,:] = numpy.array(Image.open(file))
            labels[index] = ord(label) - ord("A")
        except: pass

    shutil.rmtree(os.path.join(path, dataset), ignore_errors=True)
    numpy.savez_compressed(os.path.join(path, dataset), imgs=imgs, labels=labels)


if __name__ == "__main__": 
    notmnist_dir = os.environ.get("notMNIST_DATA_DIR", "data/notmnist")
    notmnist_files = [
        "notMNIST_small.tar.gz",]
        # "notMNIST_large.tar.gz"]

    download_files(
        base_url="http://yaroslavvb.com/upload/notMNIST/",
        base_dir=notmnist_dir,
        files=notmnist_files)
    preprocess_notmnist_files(notmnist_dir, "notMNIST_small")
    # preprocess_notmnist_files(notmnist_dir, "notMNIST_large")
```

## Building classification model

As the model backend we will use Tensorflow high-level Estimator API.

```python
# mnist-model.py

import os
import tensorflow as tf
import numpy as np


models_path = os.environ.get("MNIST_MODELS_DIR", "models/mnist")
models_path = os.path.join(models_path, "model")
base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
train_file = "train.npz"
test_file = "t10k.npz"

learning_rate = os.environ.get("LEARNING_RATE", 0.01)
num_steps = os.environ.get("LEARNING_STEPS", 10000)
batch_size = os.environ.get("BATCH_SIZE", 256)


def input_fn(file):
    with np.load(os.path.join(base_path, file)) as data:
        imgs = data["imgs"]
        labels = data["labels"].astype(int)
    return tf.estimator.inputs.numpy_input_fn(
        x = {"imgs": imgs}, y=labels, shuffle=True)


if __name__ == "__main__":
    imgs = tf.feature_column.numeric_column("imgs", shape=(28,28))
    train_fn, test_fn = input_fn(train_file), input_fn(test_file)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[256, 64],
        feature_columns=[imgs],
        n_classes=10,
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))

    tf.logging.set_verbosity(tf.logging.INFO)
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=num_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=test_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "imgs": tf.placeholder(tf.float32, shape=(None, 28, 28))})
    estimator.export_savedmodel(models_path, serving_input_receiver_fn)
```

Here we define a function `input_fn` that will produce images for our DNN Classifier. The network itself consists of 2 fully-connected hidden layers with 256 and 64 units respectively. As an activation function we use the default ReLU activation. As an optimizer we chose Adam with the learning rate that is configurable from the outside (via environment variable). After the training the model is stored in `saved_model` format under the specified path. We store both the graph and the weights. 

## Building concept model

To test that our model actually works on the data which is similar to the training set, we need to capture the essense of the training data. To do that we will additionally train an autoencoder to extract the most important features from the data. The difference between the reconstructed image and the original image will be our measure of "correctness" of the data. Higher L2-distance indicates that image is less likely to be from the training (or similiar) dataset. 

We chose to build this model with lower level Tensorflow API as it offers us more flexibility, but also produces more boilerplate code. 

```python
# mnist-concept.py

import os
import shutil
import numpy as np
import tensorflow as tf


learning_rate = os.environ.get("LEARNING_RATE", 0.01)
num_steps = os.environ.get("LEARNING_STEPS", 10000)
batch_size = os.environ.get("BATCH_SIZE", 246)
display_step = os.environ.get("DISPLAY_STEPS", 1000)

models_path = os.environ.get("MNIST_MODELS_DIR", "models/mnist")
models_path = os.path.join(models_path, "concept")
base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
train_file = "train.npz"

num_hidden_1 = 256 
num_hidden_2 = 128 
num_input = 784


# Import MNIST data
with np.load(os.path.join(base_path, train_file)) as data:
    imgs, labels = data["imgs"], data["labels"]

dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
dataset = dataset.batch(batch_size).repeat()
iterator = dataset.make_one_shot_iterator()
imgs, labels = iterator.get_next()


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

imgs_flattened = tf.layers.flatten(imgs)
encoder_op = encoder(imgs_flattened)
decoder_op = decoder(encoder_op)

y_pred, y_true = decoder_op, imgs_flattened
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2), axis=-1)
serving = tf.expand_dims(loss, 0)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    for i in range(1, num_steps+1):
        _, l = sess.run([optimizer, loss])
        if i % display_step == 0 or i == 1:
            print(f'Step {i}: Minibatch Loss: {np.mean(l)}')

    # Save model
    signature_map = {
        "infer": tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={"X": imgs}, 
            outputs={"reconstructed": serving})
    }

    shutil.rmtree(models_path, ignore_errors=True)
    builder = tf.saved_model.builder.SavedModelBuilder(models_path)
    builder.add_meta_graph_and_variables(
        sess=sess, 
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_map)
    builder.save()
```

Same as in the previous section, we train the image and export it in the `saved_model` format. 

## Packing image

Now, that we've done with our model, we need to build a Docker image from it. This will be our working image and it will contain all working files, that will be used during the training state. The image will be based on a raw Python Docker image, so we will need to additionally install some packages during the building stage. Create `requirements.txt` file. 

```
numpy==1.14.3
Pillow==5.2.0
tensorflow==1.9.0
hs==0.1.3
```

Now we can create a `Dockerfile`.

```Dockerfile
FROM python:3.6-slim
ADD ./requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt
ADD ./*.py /src/
ADD ./*.yaml /src/
WORKDIR /src/
```

Build an image and publish it in your public/private Docker registry. In the simpliest case it might be your personal [Docker Hub](https://hub.docker.com/) account. 

```sh
$ docker build -t {username}/mnist . 
$ docker push {username}/mnist:latest
```

Here we're naming the image with `mnist` name. By default it will be assigned with the `latest` tag. 

## Creating workflow 

As we've mentioned abobe, we will define the whole workflow using Argo's workflows. Create a `model-workflow.yaml` and add a basic structure to it.

```yaml 
# model-workflow.yaml

apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata: 
  generateName: hydro-workflow-
spec:
  entrypoint: mnist-workflow
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: data
    - name: models
      persistentVolumeClaim:
        claimName: models
  templates:
  - name: mnist-workflow
    steps:
    - - name: download-mnist
        template: nil
      - name: download-not-mnist
        template: nil
    - - name: train-mnist
        template: nil
      - name: train-mnist-concept
        template: nil
    - - name: upload
        template: nil
    - - name: deploy
        template: nil
    - - name: test
        template: nil
```

Here we define persistent volumes created above, all workflow steps and some other metadata. Downloading and Training will be done in parallel, while uploading/deploying/testing will be done consequently. Let's start with the downloading. We will do that with Kubernetes Jobs. 

```yaml
- name: download
  inputs: 
    parameters:
      - name: file
      value: download-file
  resource:
    action: apply
    successCondition: status.succeeded == 1
    failureCondition: status.failed > 3
    manifest: |
      apiVersion: batch/v1
      kind: Job
      metadata: 
        name: {{inputs.parameters.file}}
      spec: 
        template:
          spec:
            restartPolicy: Never
            containers:
            - name: main
              image: {username}/mnist
              command: ["python"]
              args: ["{{inputs.parameters.file}}.py"]
              volumeMounts:
              - name: data
                mountPath: /data
              env:
              - name: MNIST_DATA_DIR
                value: {{workflow.parameters.mnist-data-dir}}
              - name: notMNIST_DATA_DIR
                value: {{workflow.parameters.notmnist-data-dir}}
              volumes:
              - name: data
                persistentVolumeClaim:
                  claimName: data
```

As a base we use the container that we've built before. We have additionally provided some environment variables which we use in the downloading scripts. Environment variabels are specified via Argo parameters, which can be declared globally or locally. Global parameters are specified in one place and can be reached from everythere in the file, while local parameters are only specific to the declaration template. Using local parameters allows us to use a single template and specify which file we want to run. Let's put it all together. 

```yaml
# model-workflow.yaml

apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata: 
  generateName: hydro-workflow-
spec:
  arguments:
    parameters:
    - name: mnist-data-dir
      value: /data/mnist
    - name: notmnist-data-dir
      value: /data/notmnist
  entrypoint: mnist-workflow
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: data
    - name: models
      persistentVolumeClaim:
        claimName: models
  templates:
  - name: mnist-workflow
    steps:
    - - name: download-mnist
        template: download
        arguments:
          parameters:
            - name: file
              value: download-mnist
      - name: download-not-mnist
        template: download
        arguments:
          parameters:
            - name: file
              value: download-notmnist
    - - name: train-mnist
        template: nil
      - name: train-mnist-concept
        template: nil
    - - name: upload
        template: nil
    - - name: deploy
        template: nil
    - - name: test
        template: nil
  - name: download
    inputs: 
      parameters:
      - name: file
        value: download-file
    resource:
      action: apply
      successCondition: status.succeeded == 1
      failureCondition: status.failed > 3
      manifest: |
        apiVersion: batch/v1
        kind: Job
        metadata: 
          name: {{inputs.parameters.file}}
        spec: 
          template:
            spec:
              restartPolicy: Never
              containers:
              - name: main
                image: tidylobster/mnist
                command: ["python"]
                args: ["{{inputs.parameters.file}}.py"]
                volumeMounts:
                - name: data
                  mountPath: /data
                env:
                - name: MNIST_DATA_DIR
                  value: {{workflow.parameters.mnist-data-dir}}
                - name: notMNIST_DATA_DIR
                  value: {{workflow.parameters.notmnist-data-dir}}
              volumes:
              - name: data
                persistentVolumeClaim:
                  claimName: data
```

I will now breafly describe next templates' definitions and then we will join them altogether. The next template is training.

```yaml
- name: train
  inputs: 
    parameters:
    - name: file
        value: training-file
  resource: 
    action: apply
    successCondition: status.tfReplicaStatuses.Master.succeeded == 1
    failureCondition: status.tfReplicaStatuses.Master.failed > 3
    manifest: |
    apiVersion: kubeflow.org/v1alpha2
    kind: TFJob
    metadata:
        name: {{workflow.parameters.job-name}}-{{inputs.parameters.file}}
    spec:
        tfReplicaSpecs:
        Master:
            replicas: 1
            template:
            spec:
                containers:
                - name: tensorflow
                    image: tidylobster/mnist
                    command: ["python"]
                    args: ["{{inputs.parameters.file}}.py"]
                    volumeMounts:
                    - name: data
                        mountPath: /data
                    - name: models
                        mountPath: /models
                    env:
                    - name: MNIST_MODELS_DIR
                        value: {{workflow.parameters.mnist-models-dir}}
                    - name: MNIST_DATA_DIR
                        value: {{workflow.parameters.mnist-data-dir}}
                    - name: notMNIST_MODELS_DIR
                        value: {{workflow.parameters.notmnist-models-dir}}
                    - name: notMNIST_DATA_DIR
                        value: {{workflow.parameters.notmnist-data-dir}}
                volumes:
                - name: data
                    persistentVolumeClaim:
                    claimName: data
                - name: models
                    persistentVolumeClaim:
                    claimName: models
```