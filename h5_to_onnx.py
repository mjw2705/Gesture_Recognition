from tensorflow.keras.models import load_model
import onnxruntime
import onnx
import keras2onnx


onnx_model_name = 'gesture_model.onnx'

model = load_model('gesture_model.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)