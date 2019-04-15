import importlib.util, psycopg2
import grpc, time, os
import numpy as np
import hydro_serving_grpc as hs 

# Import download module from ./01-download
spec = importlib.util.spec_from_file_location("download", "./01-download/download.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

# Usage parameters 
requests_delay = 0.5
requests_count = 100 
target_file = "t10k.npz"
base_url="http://yann.lecun.com/exdb/mnist/"


def generate_data(mnist_base_path, test_file):
    # Read mages & labels, shuffle them and return back
    with np.load(os.path.join(mnist_base_path, test_file)) as data:
        imgs, labels = data["imgs"], data["labels"]
        assert len(imgs) == len(labels)
        permute = np.random.permutation(len(imgs))
        return imgs[permute], labels[permute]

conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/postgres")
cur = conn.cursor()

cur.execute('''
    CREATE TABLE IF NOT EXISTS 
        requests (timestamp bigint, uid integer, ground_truth integer)
''')

if __name__ == "__main__":

    # Download test images, if not exist
    if not os.path.exists(os.path.join(".", target_file)):
        download.download_files(
            base_url, '.', ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'])
        download.process_images('.', "t10k")

    creds = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("dev.k8s.hydrosphere.io:443", creds) 

    stub = hs.PredictionServiceStub(channel) 
    model_spec = hs.ModelSpec(name="mnist-app")
    tensor_shape = hs.TensorShapeProto(dim=[
        hs.TensorShapeProto.Dim(size=1),
        hs.TensorShapeProto.Dim(size=28),
        hs.TensorShapeProto.Dim(size=28)
    ])

    images, labels = generate_data('.', target_file)
    for index, (image, label) in enumerate(zip(images, labels)):
        if index == requests_count: break
        
        tensor = hs.TensorProto(dtype=hs.DT_FLOAT, tensor_shape=tensor_shape, 
            float_val=image.flatten().tolist())
        request = hs.PredictRequest(model_spec=model_spec, inputs={"imgs": tensor}) 
        result = stub.Predict(request)
        
        print(f"{index} | {int(index / requests_count * 100)}%", flush=True)
        cur.execute(
            "INSERT INTO requests VALUES (%s, %s, %s);",
            (result.trace_data.ts, result.trace_data.uid, int(label)))
        conn.commit()
        time.sleep(requests_delay)
    
conn.close()