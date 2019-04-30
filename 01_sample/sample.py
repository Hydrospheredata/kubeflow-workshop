import requests, psycopg2
import pickle, os, random, urllib.parse
import numpy as np
import datetime, argparse
from tensorflow import make_ndarray
from hydro_serving_grpc.timemachine.reqstore_client import *


def get_model_version_id(host_address, application_name):
    addr = urllib.parse.urljoin(host_address, "api/v2/application/{}".format(application_name))
    resp = requests.get(addr).json()
    assert resp.get("error") is None, resp.get("message")
    return resp["executionGraph"]["stages"][0]["modelVariants"][0]["modelVersion"]["id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mount-path',
        help='Path to PersistentVolumeClaim, deployed on the cluster',
        required=True)
    parser.add_argument(
        '--hydrosphere-address', required=True)
    parser.add_argument(
        '--model-name', required=True)
    
    args = parser.parse_args()
    arguments = args.__dict__

    application_name = "{}-app".format(arguments["model_name"])
    hydrosphere_address = arguments["hydrosphere_address"]
    # namespace = urllib.parse.urlparse(host_address).netloc.split(".")[0]
    reqstore_address = urllib.parse.urljoin(hydrosphere_address, "reqstore")

    postgres_host = "postgres"
    mount_path = arguments["mount_path"]
    timestamp = round(datetime.datetime.now().timestamp())
    data_path = os.path.join(mount_path, "data", "mnist", str(timestamp))
    pipeline_dataset_path = "/data_path.txt" if mount_path != "./" else "./data_path.txt"


    client = ReqstoreHttpClient(reqstore_address)
    # client = ReqstoreClient(reqstore_address, False)
    model_version_id = str(get_model_version_id(hydrosphere_address, application_name))

    conn = psycopg2.connect("postgresql://serving:hydro-serving@{}:5432/postgres".format(postgres_host))
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS 
            requests (timestamp bigint, uid integer, ground_truth integer);
    ''')

    print("Sample data from reqstore", flush=True)
    records = list(
        client.getRange(0, 1854897851804888100, model_version_id, limit=10000, reverse=False))
    random.shuffle(records)

    print("Prepare dataset", flush=True)
    imgs, labels = list(), list()
    for timestamp in records:

        for entry in timestamp.entries:
            cur.execute("SELECT * FROM requests WHERE timestamp=%s AND uid=%s", (timestamp.ts, entry.uid))
            db_record = cur.fetchone()

            if not db_record: continue    
            request_image = make_ndarray(entry.request.inputs["imgs"]).reshape(28*28)
            imgs.append(request_image); labels.append(db_record[2])

    if not imgs:
        imgs, labels = np.empty((0, 28, 28)), np.empty((0,))
    else: 
        imgs, labels = np.array(imgs), np.array(labels)
        
    train_imgs, train_labels = imgs[:int(len(imgs) * 0.75)], labels[:int(len(labels) * 0.75)]
    test_imgs, test_labels = imgs[int(len(imgs) * 0.75):], labels[int(len(labels) * 0.75):]

    assert len(train_imgs) > 0, "Not enough training data"
    assert len(test_imgs) > 0, "Not enough testing data"

    os.makedirs(data_path, exist_ok=True)

    # For workshop purpose only we add additional data for model re-training
    with np.load("notmnist.npz") as data:
        notmnist_imgs = data["imgs"]
        notmnist_labels = data["labels"] + 10
    
    with np.load("t10k.npz") as data:
        mnist_imgs = data["imgs"]
        mnist_labels = data["labels"]

    new_images = np.vstack([notmnist_imgs, mnist_imgs, train_imgs, test_imgs])
    new_labels = np.hstack([notmnist_labels, mnist_labels, train_labels, test_labels])
    
    # Shuffle training data
    permute = np.random.permutation(len(new_images))
    new_images = new_images[permute]
    new_labels = new_labels[permute]
    
    # Make train/test split
    train_imgs = new_images[:int(len(new_images) * 0.75)]
    train_labels = new_labels[:int(len(new_labels) * 0.75)]
    test_imgs = new_images[int(len(new_images) * 0.75):]
    test_labels = new_labels[int(len(new_labels) * 0.75):]

    os.makedirs(data_path, exist_ok=True)
    print("Train subsample size: {}".format(str(len(train_imgs))), flush=True)
    print("Test subsample size: {}".format(str(len(test_imgs))), flush=True)
    
    np.savez_compressed(
        os.path.join(data_path, "train"), imgs=train_imgs, labels=train_labels)
    np.savez_compressed(
        os.path.join(data_path, "test"), imgs=test_imgs, labels=test_labels)

    # Dump dataset location
    with open(pipeline_dataset_path, "w+") as file:
        file.write(data_path)