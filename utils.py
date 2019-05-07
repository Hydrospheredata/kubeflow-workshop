import importlib.util, psycopg2
import grpc, time, os, argparse
from tqdm import tqdm
import numpy as np
import hydro_serving_grpc as hs


USER = "serving"
PASS = "hydro-serving"
PORT = "5432"
ADDRESS = "postgres"
DATABASE = "postgres"


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


def simulate_production_traffic(host=None, request_delay=0.5, request_amount=10000, file="combined.npz", shuffle=False):
    conn = psycopg2.connect(f"postgresql://{USER}:{PASS}@{ADDRESS}:{PORT}/{DATABASE}")
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS 
            requests (timestamp bigint, uid integer, ground_truth integer);
    ''')

    if not host:
        namespace = os.environ["NAMESPACE"]
        host = f"hydro-serving-ui-{namespace}:9091"
    channel = grpc.insecure_channel(host)
    stub = hs.PredictionServiceStub(channel)

    # an application, that will be invoked
    model_spec = hs.ModelSpec(name="mnist-app")

    # basic shape for images
    tensor_shape = hs.TensorShapeProto(dim=[
        hs.TensorShapeProto.Dim(size=1),
        hs.TensorShapeProto.Dim(size=28),
        hs.TensorShapeProto.Dim(size=28),
        hs.TensorShapeProto.Dim(size=1),
    ])

    images, labels = generate_data('/data', file, shuffle=shuffle)
    for index, (image, label) in tqdm(enumerate(zip(images, labels)), total=request_amount):
        if index == request_amount: break
        
        # form a request
        tensor = hs.TensorProto(dtype=hs.DT_FLOAT, tensor_shape=tensor_shape, 
            float_val=image.flatten().tolist())
        request = hs.PredictRequest(model_spec=model_spec, inputs={"imgs": tensor})
        
        # get prediction
        result = stub.Predict(request)
        
        # insert trace_id and ground_truth labels into database
        cur.execute("INSERT INTO requests VALUES (%s, %s, %s)",
            (result.trace_data.ts, result.trace_data.uid, int(label)))
        conn.commit()    
        time.sleep(request_delay)


if __name__ == "__main__":
    simulate_production_traffic()