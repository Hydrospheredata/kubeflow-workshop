import pickle, os
import numpy as np
import pandas as pd
import reqstore
from tensorflow import make_ndarray


addr = "http://localhost:7265"
name = "app1stage0"


if __name__ == "__main__":
    binary_data = reqstore.APIHelper.subsample(addr, name)
    records = reqstore.BinaryHelper.decode_records(binary_data)

    imgs, ids = list(), list()
    for entry in reqstore.splice_entries(records):
        id_ = entry.response.outputs["id"].int64_val[0]
        request_img = make_ndarray(entry.request.inputs["imgs"]).reshape(28*28)
        ids.append(id_); imgs.append(request_img.tolist())
    
    data = MNISTTable.raw(f"""
        SELECT id AS index, label FROM mnist_labels WHERE id IN ({",".join(ids)})
    """).dicts()

    df = pd.DataFrame(data)
    df.set_index("index", inplace=True)
    labels = df.loc[ids].values
    indexes = pd.isna(labels)
    imgs = np.array(imgs)[indexes,:]
    labels = labels[indexes]

    path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
    np.savez_compressed(os.path.join(path, "new_train"), imgs=np.array(imgs), labels=labels)