import requests, psycopg2
import pickle, os, random, urllib.parse
import numpy as np
from tensorflow import make_ndarray
from hydro_serving_grpc.timemachine.reqstore_client import *


application_name = os.environ.get("APPLICATION_NAME", "mnist-app")
host_address = os.environ.get("CLUSTER_ADDRESS", "http://localhost")
namespace = urllib.parse.urlparse(host_address).netloc.split(".")[0]
reqstore_address = urllib.parse.urljoin(host_address, "reqstore")
postgres_host = os.environ.get("KUBERNETES_SERVICE_HOST", "localhost")
postgres_host = postgres_host if postgres_host == "localhost" \
    else "postgres.{}.svc.cluster.local".format(namespace)
mount_path = os.environ.get("MOUNT_PATH", "./")
data_path = os.path.join(mount_path, "data", "mnist")


def get_model_version_id(application_name):
    addr = urllib.parse.urljoin(host_address, "api/v2/application/{}".format(application_name))
    return requests.get(addr).json() \
        ["executionGraph"]["stages"][0]["modelVariants"][0]["modelVersion"]["id"]


if __name__ == "__main__":
    client = ReqstoreHttpClient(reqstore_address)
    # client = ReqstoreClient(reqstore_address, False)
    model_version_id = str(get_model_version_id(application_name))

    conn = psycopg2.connect("postgresql://serving:hydro-serving@{}:5432/postgres".format(postgres_host))
    cur = conn.cursor()

    records = list(
        client.getRange(0, 1854897851804888100, model_version_id, limit=10000, reverse=False))
    random.shuffle(records)

    imgs, labels = list(), list()
    for timestamp in records:

        for entry in timestamp.entries:
            cur.execute("SELECT * FROM requests WHERE timestamp=%s AND uid=%s", (timestamp.ts, entry.uid))
            db_record = cur.fetchone()

            if not db_record: continue    
            request_image = make_ndarray(entry.request.inputs["imgs"]).reshape(28*28)
            imgs.append(request_image); labels.append(db_record[2])

    imgs, labels = np.array(imgs), np.array(labels)
    train_imgs, train_labels = imgs[:int(len(imgs) * 0.75)], labels[:int(len(labels) * 0.75)]
    test_imgs, test_labels = imgs[int(len(imgs) * 0.75):], labels[int(len(labels) * 0.75):]

    assert len(train_imgs) > 0, "Not enough training data"
    assert len(test_imgs) > 0, "Not enough testing data"

    os.makedirs(data_path, exist_ok=True)
    print("New train subsample size: {}".format(str(len(train_imgs))), flush=True)
    print("New test subsample size: {}".format(str(len(test_imgs))), flush=True)
    
    np.savez_compressed(
        os.path.join(data_path, "train"), imgs=train_imgs, labels=train_labels)
    np.savez_compressed(
        os.path.join(data_path, "t10k"), imgs=test_imgs, labels=test_labels)