from sklearn.metrics import accuracy_score
import os, time, requests
import numpy as np

host_address = os.environ.get("CLUSTER_ADDRESS", "https://dev.k8s.hydrosphere/io")
application_name = os.environ.get("APPLICATION_NAME", "mnist-app")
signature_name = os.environ.get("SIGNATURE_NAME", "predict")
warmup_images_count = int(os.environ.get("WARMUP_IMAGES_AMOUNT", 1000))
acceptable_accuracy = float(os.environ.get("ACCEPTABLE_ACCURACY", 0.90))

mnist_base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
test_file = "t10k.npz"

service_link = f"{host_address}/gateway/applications/{application_name}/{signature_name}"
print(f"Using URL :: {service_link}", flush=True)


def generate_data(mnist_base_path, test_file, warmup_images_count):
    with np.load(os.path.join(mnist_base_path, test_file)) as data:
        imgs, labels = data["imgs"], data["labels"]
        imgs, labels = imgs[:warmup_images_count], labels[:warmup_images_count]
        clean_images_count = len(imgs)

    noisy_imgs, noisy_labels = np.copy(imgs), np.copy(labels)
    noisy_imgs, noisy_labels = noisy_imgs[:warmup_images_count//2], noisy_labels[:warmup_images_count//2]
    noise = np.random.uniform(size=noisy_imgs.shape)
    return np.concatenate((imgs, noisy_imgs+noise)), np.concatenate((labels, noisy_labels)), clean_images_count


if __name__ == "__main__": 
    predicted = []

    data, labels, count = generate_data(mnist_base_path, test_file, warmup_images_count)
    for index, image in enumerate(data):
        response = requests.post(url=service_link, json={'imgs': [image.tolist()]})
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(0.4)
        
    assert accuracy_score(labels[:count], predicted[:count]) > acceptable_accuracy