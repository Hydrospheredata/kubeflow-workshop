import logging
import nltk
import numpy as np
import hydro_serving_grpc as hs
import os

# update searching path for nltk
nltk.data.path = ["/model/files/nltk_data"] + nltk.data.path


def tokenize(x):
	sentences = np.array(x.string_val)
	sentences = sentences.reshape([dim.size for dim in x.tensor_shape.dim])

	tokenized = np.copy(sentences)
	for index, sentence in enumerate(sentences):
		tokenized[index] = " ".join(nltk.word_tokenize(str(sentence[0], encoding="utf-8").lower()))
	
	tokenized = hs.TensorProto(
		dtype=hs.DT_STRING,
		string_val=tokenized.flatten(),
		tensor_shape=hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=-1), hs.TensorShapeProto.Dim(size=1)]))
	return hs.PredictResponse(outputs={"input_data": tokenized})
