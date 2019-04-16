import requests, psycopg2
import pickle, os, random, urllib.parse
import numpy as np
from tensorflow import make_ndarray
from hydro_serving_grpc.timemachine.reqstore_client import *


application_name = os.environ.get("APPLICATION_NAME", "mnist-app")
host_address = os.environ.get("CLUSTER_ADDRESS", "http://localhost")
mount_path = os.environ.get("MOUNT_PATH", "./")
data_path = os.path.join(mount_path, "data", "mnist")
reqstore_addr = urllib.parse.urljoin(host_address, "reqstore/")


def get_model_version_id(application_name):
    addr = urllib.parse.urljoin(host_address, f"api/v2/application/{application_name}")
    return requests.get(addr).json() \
        ["executionGraph"]["stages"][0]["modelVariants"][0]["modelVersion"]["id"]


if __name__ == "__main__":
    client = ReqstoreClient("https://dev.k8s.hydrosphere.io/reqstore/", 443)
    model_version_id = str(get_model_version_id(application_name))

    data = client.getRange(0, 1854897851804888100, model_version_id)
    for item in data:
        print(item)

    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/postgres")
    cur = conn.cursor()

    application_id = str(get_application_id(app_addr))
    binary_data = reqstore.APIHelper.subsample(reqstore_addr, application_id)
    records = reqstore.BinaryHelper.decode_records(binary_data)
    random.shuffle(records)
    
    imgs, labels = list(), list()
    for timestamp in records[:int(len(records) * 1)]:
        for entry in timestamp.entries:
            cur.execute("SELECT * FROM requests WHERE timestamp=? AND uid=?", (timestamp.ts, entry.uid))
            db_record = cur.fetchone()

            if not db_record: continue    
            request_image = make_ndarray(entry.request.inputs["imgs"]).reshape(28*28)
            imgs.append(request_image); labels.append(db_record[2])

    imgs, labels = np.array(imgs), np.array(labels)
    train_imgs, train_labels = imgs[:int(len(imgs) * 0.75)], labels[:int(len(labels) * 0.75)]
    test_imgs, test_labels = imgs[int(len(imgs) * 0.75):], labels[int(len(labels) * 0.75):]

    
    os.makedirs(data_path, exist_ok=True)
    print(f"New train subsample size: {len(train_imgs)}", flush=True)
    print(f"New test subsample size: {len(test_imgs)}", flush=True)
    
    np.savez_compressed(
        os.path.join(data_path, "subsample-train"), imgs=train_imgs, labels=train_labels)
    np.savez_compressed(
        os.path.join(data_path, "subsample-test"), imgs=test_imgs, labels=test_labels)