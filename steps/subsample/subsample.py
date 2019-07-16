import requests, psycopg2
import pickle, os, random, urllib.parse
import numpy as np
import datetime, argparse, hashlib
from hydro_serving_grpc.timemachine.reqstore_client import *
from decouple import Config, RepositoryEnv

# from storage import *
from orchestrator import *


config = Config(RepositoryEnv("config.env"))
HYDROSPHERE_LINK = config('HYDROSPHERE_LINK')
POSTGRES_HOST = config('POSTGRES__HOST')
POSTGRES_PASS = config('POSTGRES__PASS')
POSTGRES_USER = config('POSTGRES__USER')
POSTGRES_PORT = config('POSTGRES__PORT')
POSTGRES_DB = config('POSTGRES__DB')


def get_model_version_id(host_address, application_name):
    addr = urllib.parse.urljoin(host_address, f"api/v2/application/{application_name}")
    resp = requests.get(addr).json()
    assert resp.get("error") is None, resp.get("message")
    return resp["executionGraph"]["stages"][0]["modelVariants"][0]["modelVersion"]["id"]


def main(application_name, bucket_name, storage_path="/"):

    # Define helper classes     
    storage = Storage(bucket_name) 
    orchestrator = Orchestrator(storage_path=storage_path)

    # Define required variables
    data_path = os.path.join("data", str(round(datetime.datetime.now().timestamp())))
    reqstore_address = urllib.parse.urljoin(HYDROSPHERE_LINK, "reqstore")

    client = ReqstoreHttpClient(reqstore_address)
    model_version_id = str(get_model_version_id(HYDROSPHERE_LINK, application_name))

    # Initialize connection to Database 
    conn = psycopg2.connect(
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS 
            requests (hex_uid varchar(256), ground_truth integer);
    ''')
    conn.commit()

    print("Sample data from reqstore", flush=True)
    records = list(client.getRange(0, 1854897851804888100, model_version_id, limit=10000, reverse="false"))
    random.shuffle(records)

    print("Prepare dataset", flush=True)
    imgs, labels = list(), list()

    for timestamp in records:
        for entry in timestamp.entries:
            request_image = np.array(
                entry.request.inputs["imgs"].float_val, dtype=np.float32).reshape((28, 28))
            
            hex_uid = hashlib.sha1(request_image).hexdigest()
            cur.execute("SELECT * FROM requests WHERE hex_uid=%s", (hex_uid,))
            db_record = cur.fetchone()
            if not db_record: continue    
            
            imgs.append(request_image); labels.append(db_record[1])

    if not imgs:
        imgs, labels = np.empty((0, 28, 28)), np.empty((0,))
    else: 
        imgs, labels = np.array(imgs), np.array(labels)
        
    train_imgs, train_labels = imgs[:int(len(imgs) * 0.75)], labels[:int(len(labels) * 0.75)]
    test_imgs, test_labels = imgs[int(len(imgs) * 0.75):], labels[int(len(labels) * 0.75):]

    assert len(train_imgs) > 100, "Not enough training data"
    assert len(test_imgs) > 25, "Not enough testing data"

    print(f"Train subsample size: {len(train_imgs)}", flush=True)
    print(f"Test subsample size: {len(test_imgs)}", flush=True)

    np.savez_compressed(os.path.join(storage_path, "train.npz"), imgs=train_imgs, labels=train_labels)
    np.savez_compressed(os.path.join(storage_path, "test.npz"), imgs=test_imgs, labels=test_labels)

    storage.upload_file(os.path.join(storage_path, "train.npz"), os.path.join(data_path, "train.npz"))
    storage.upload_file(os.path.join(storage_path, "test.npz"), os.path.join(data_path, "test.npz"))
    
    orchestrator.export_meta("data_path", data_path, "txt")


def aws_lambda(event, context):
    return main(
        application_name=event["application_name"],
        bucket_name=event["bucket_name"],
        storage_path="/tmp/"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--application-name', required=True)
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        application_name=args.application_name,
        bucket_name=args.bucket_name,
    )
