import importlib.util, psycopg2
import grpc, time, os, argparse
from tqdm import tqdm
import numpy as np
import hydro_serving_grpc as hs
import hashlib
from decouple import Config, RepositoryEnv

config = Config(RepositoryEnv("../config.env"))
POSTGRES_USER = config("POSTGRES__USER")
POSTGRES_PASS = config("POSTGRES__PASS")
POSTGRES_HOST = config("POSTGRES__HOST")
POSTGRES_PORT = config("POSTGRES__PORT")
POSTGRES_DB = config("POSTGRES__DB")


def generate_data(base_path, test_file, shuffle=False):
    # Read mages & labels, shuffle them and return
    with np.load(os.path.join(base_path, test_file)) as data:
        imgs, labels = data["imgs"], data["labels"]
        assert len(imgs) == len(labels)
    imgs = np.reshape(imgs, (len(imgs), 28, 28, 1))
    if shuffle:
        permute = np.random.permutation(len(imgs))
        return imgs[permute], labels[permute]
    return imgs, labels


def simulate_production_traffic(path="./", application_name="mnist_app", host=None, request_delay=1, request_amount=10000, file="test.npz", shuffle=False):
    conn = psycopg2.connect(f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS 
            requests (hex_uid varchar(256), ground_truth integer);
    ''')
    conn.commit()

    if not host:
        host = f"dev.k8s.hydrosphere.io"
    creds = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel(host, creds)
    # channel = grpc.insecure_channel(host)
    stub = hs.PredictionServiceStub(channel)

    # an application, that will be invoked
    model_spec = hs.ModelSpec(name=application_name)

    # basic shape for images
    tensor_shape = hs.TensorShapeProto(dim=[
        hs.TensorShapeProto.Dim(size=1),
        hs.TensorShapeProto.Dim(size=28),
        hs.TensorShapeProto.Dim(size=28),
        hs.TensorShapeProto.Dim(size=1),
    ])

    images, labels = generate_data(path, file, shuffle=shuffle)
    for index, (image, label) in tqdm(enumerate(zip(images, labels)), total=request_amount):
        if index == request_amount: break
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        # form a request and get a prediction
        tensor = hs.TensorProto(dtype=hs.DT_FLOAT, tensor_shape=tensor_shape, 
            float_val=image.flatten().tolist())
        request = hs.PredictRequest(model_spec=model_spec, inputs={"imgs": tensor})
        stub.Predict(request)

        # insert uid and ground_truth labels into database
        cur.execute("INSERT INTO requests VALUES (%s, %s)",
            (hashlib.sha1(image).hexdigest(), int(label)))
        conn.commit()
        time.sleep(request_delay)


if __name__ == "__main__":
    simulate_production_traffic(request_delay=0.6, shuffle=True)