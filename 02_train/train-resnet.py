import keras.backend as K
import os, json, sys
import tensorflow as tf
import numpy as np
import argparse, datetime
from resnet import ResnetBuilder
from keras.optimizers import Adam


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data-path', 
        help='Path, where the current run\'s data was stored',
        required=True)
    parser.add_argument(
        '--mount-path',
        help='Path to PersistentVolumeClaim, deployed on the cluster',
        required=True)
    parser.add_argument(
        '--learning-rate', type=float, default=0.01)
    parser.add_argument(
        '--epochs', type=int, default=1)
    parser.add_argument(
        '--batch-size', type=int, default=256)
    parser.add_argument(
        '--dev', help="Flag for development purposes", type=bool, default=False)
    
    args = parser.parse_args()
    arguments = args.__dict__
    models_path = os.path.join(arguments["mount_path"], "models")

    # Prepare data inputs
    with np.load(os.path.join(arguments["data_path"], "train.npz")) as data:
        train_imgs = data["imgs"]
        train_labels = data["labels"].astype(int)
    
    with np.load(os.path.join(arguments["data_path"], "test.npz")) as data:
        test_imgs = data["imgs"]
        test_labels = data["labels"].astype(int)

    X_train, Y_train = np.reshape(train_imgs, (len(train_imgs), 28, 28, 1)), train_labels
    X_test, Y_test = np.reshape(test_imgs, (len(test_imgs), 28, 28, 1)), test_labels
    
    nb_classes = len(np.unique(np.hstack([train_labels, test_labels])))
    print("Number of classes: {}".format(nb_classes), flush=True)
    
    # Create the model
    sess = K.get_session()
    graph = sess.graph  # get Tensorflow graph

    model = ResnetBuilder.build_resnet_18((1, 28, 28), nb_classes)
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=arguments["learning_rate"]),
              metrics=['accuracy'])

    # Train the model 
    history = model.fit(
        X_train[:200], Y_train[:200], 
        epochs=arguments["epochs"], 
        batch_size=arguments["batch_size"], 
        validation_data=(X_test[:200], Y_test[:200]))
    accuracy = history.history["val_acc"][-1]
    
    imgs = graph.get_tensor_by_name("input_1:0")
    probabilities = graph.get_tensor_by_name("dense_1/Softmax:0")
    class_ids = tf.argmax(probabilities)

    # Export the model 
    signature_map = {
        "predict": tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={"imgs": imgs}, 
            outputs={
                "probabilities": probabilities, 
                "class_ids": class_ids
            }
        )
    }

    timestamp = round(datetime.datetime.now().timestamp())
    model_save_path = os.path.join(models_path, str(timestamp))
    print(model_save_path)  
    builder = tf.saved_model.builder.SavedModelBuilder(model_save_path)
    builder.add_meta_graph_and_variables(
        sess=sess,                                          # session, where the graph was initialized
        tags=[tf.saved_model.tag_constants.SERVING],        # tag your graph as servable using this constant
        signature_def_map=signature_map)
    builder.save()

    # Perform metrics calculations
    if arguments["dev"]: 
        accuracy_file = "./accuracy.txt"
        metrics_file = "./mlpipeline-metrics.json"
        model_path = "./model_path.txt"
    else: 
        accuracy_file = "/accuracy.txt"
        metrics_file = "/mlpipeline-metrics.json"
        model_path = "/model_path.txt"

    metrics = {
        'metrics': [
            {
                'name': 'accuracy-score',   # -- The name of the metric. Visualized as the column 
                                            # name in the runs table.
                'numberValue': accuracy,    # -- The value of the metric. Must be a numeric value.
                'format': "PERCENTAGE",     # -- The optional format of the metric. Supported values are 
                                            # "RAW" (displayed in raw format) and "PERCENTAGE" 
                                            # (displayed in percentage format).
            },
        ],
    }

    # Dump metrics
    with open(accuracy_file, "w+") as file:
        file.write(str(accuracy))
    
    with open(metrics_file, "w+") as file:
        json.dump(metrics, file)

    with open(model_path, "w+") as file:
        file.write(model_save_path)
    