import hydro_serving_grpc as hs
import numpy as np 
import grpc 

# connect to your ML Lamba instance
channel = grpc.insecure_channel("localhost:8080")
stub = hs.PredictionServiceStub(channel)

# 1. define a model, that you'll use
model_spec = hs.ModelSpec(name="seq2seq-mono", signature_name="serving_default")
# 2. define tensor_shape for Tensor instance
tensor_shape = hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=-1), hs.TensorShapeProto.Dim(size=1)])
# 3. define tensor with needed data
tensor = hs.TensorProto(dtype=hs.DT_STRING, tensor_shape=tensor_shape, string_val=np.array([b"hello , there , sugar"]))
# 4. create PredictRequest instance
request = hs.PredictRequest(model_spec=model_spec, inputs={"input_data": tensor})

# call Predict method
result = stub.Predict(request)
print(result)