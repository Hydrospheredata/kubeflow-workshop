import logging, sys

logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("subsample.log")])
logger = logging.getLogger(__name__)

import requests, psycopg2, tqdm
import pickle, os, random, urllib.parse
import numpy as np, wo
import datetime, argparse, hashlib
from hydro_serving_grpc.reqstore.reqstore_client import ReqstoreHttpClient


def get_model_versions(host_uri, application_name):
    """ 
    Return models, placed in the first stage of the given application. 
    
    Parameters
    ----------
    host_uri: str 
        URI of the Hydrosphere instance, where application is deployed.
    application_name: str
        Name of the application.
    
    Returns
    -------
    [(model_version_id: str, weight: int), ...]
    """
    logger.info("Retrieving model versions")
    addr = urllib.parse.urljoin(host_uri, f"api/v2/application/{application_name}")
    response = requests.get(addr)
    if response.ok:
        variants = response.json()["executionGraph"]["stages"][0]["modelVariants"]
        return map(lambda a: (str(a["modelVersion"]["id"]), a["weight"]), variants)
    else:
        raise ValueError(response.text)


def main(postgres_uri, reqstore_uri, hydrosphere_uri, application_name, limit, train_part, validation_part):

    logger.debug("Connecting to Reqstore")
    client = ReqstoreHttpClient(reqstore_uri)

    logger.debug("Connecting to PostgreSQL")
    conn = psycopg2.connect(postgres_uri)
    cur = conn.cursor()

    logger.debug("Creating `requests` table if not exists")
    cur.execute('''
        CREATE TABLE IF NOT EXISTS 
            requests (hex_uid varchar(256), ground_truth integer);
    ''')
    conn.commit()

    images, labels = [], []
    for model_version_id, weight in get_model_versions(hydrosphere_uri, application_name): 
        logger.info(f"Subsampling records from model_version={model_version_id}")
        records = client.getRange(
            from_ts=0, 
            to_ts=1854897851804888100, 
            folder=model_version_id, 
            limit=limit * weight / 100, 
            reverse="true",
        )
        logger.info(f"Sampled {len(records)} records")
        logger.info("Transforming records")
        for timestamp in tqdm.tqdm(records):
            for entry in timestamp.entries:
                flattened = np.array(entry.request.inputs["imgs"].float_val, dtype=np.float32)
                request_image = flattened.reshape((28, 28))
                
                cur.execute("""
                    SELECT * FROM requests WHERE hex_uid=%s
                """, (hashlib.sha1(request_image).hexdigest(), ))
                db_record = cur.fetchone()
                
                if not db_record: continue
                images.append(request_image) 
                labels.append(db_record[1])

    if not images:
        raise ValueError(f"Could not find any images for application={application_name}")

    images, labels = np.array(images), np.array(labels)
    train_imgs, train_labels = images[:int(len(images) * train_part)], labels[:int(len(labels) * train_part)]
    test_imgs, test_labels = images[int(len(images) * train_part):], labels[int(len(labels) * train_part):]

    assert len(train_imgs) > 100, "Not enough training data"
    assert len(test_imgs) > 25, "Not enough testing data"

    logger.info(f"Train subsample size: {len(train_imgs)}")
    logger.info(f"Test subsample size: {len(test_imgs)}")

    logger.debug("Writing training data to disk")
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/t10k", exist_ok=True)
    np.savez_compressed("data/train/imgs.npz", imgs=train_imgs)
    np.savez_compressed("data/train/labels.npz", labels=train_labels)
    np.savez_compressed("data/t10k/imgs.npz", imgs=test_imgs)
    np.savez_compressed("data/t10k/labels.npz", labels=test_labels)

    logger.debug("Calculating md5 hash")
    sample_version = wo.utils.io.md5_files([
        "data/train/imgs.npz", "data/train/labels.npz",
        "data/t10k/imgs.npz", "data/t10k/labels.npz",
    ])
    
    return {
        "sample_version": sample_version
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-path', required=True)
    parser.add_argument('--application-name', required=True)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--train-part', type=float, default=0.7)
    parser.add_argument('--validation-part', type=float, default=0.1)
    parser.add_argument('--dev', action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")
    kwargs = dict(vars(args))

    w = wo.Orchestrator(
        default_logs_path="mnist/logs",
        default_params={
            "postgres.host": "localhost",
            "postgres.port": "5432", 
            "postgres.user": "serving",
            "postgres.pass": "hydro-serving",
            "postgres.dbname": "serving",
            "uri.hydrosphere": "https://tm.k8s.hydrosphere.io",
            "uri.reqstore": "https://tm.k8s.hydrosphere.io/reqstore"
        },
        dev=args.dev,
    )
    config = w.get_config()
    
    try:

        # Download artifacts
        pass

        # Initialize runtime variables
        postgres_uri = f"postgresql://{config['postgres.user']}:{config['postgres.pass']}" \
            f"@{config['postgres.host']}:{config['postgres.port']}/{config['postgres.dbname']}"

        # Execute main script
        result = main(
            postgres_uri, 
            config['uri.reqstore'], 
            config["uri.hydrosphere"], 
            args.application_name,
            args.limit,
            args.train_part,
            args.validation_part,
        )

        # Prepare variables for logging
        output_data_path = os.path.join(
            args.output_data_path, f"sample-version={result['sample_version']}")

        # Upload artifacts 
        w.upload_prefix("data", output_data_path)
        
    except Exception as e:
        logger.exception("Main execution script failed")
    
    finally: 
        scheme, bucket, path = w.parse_uri(args.output_data_path)
        w.log_execution(
            outputs={"output_data_path": output_data_path},
            logs_bucket=f"{scheme}://{bucket}",
            logs_file="subsample.log",
        )
