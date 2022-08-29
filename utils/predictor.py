import torch
import onnxruntime


class Detecton_yolov4_onnx_custom_ly(object):
    def __init__(self, onnx_dir, size=416):
        if torch.cuda.is_available():
            session = onnxruntime.InferenceSession(onnx_dir, None, providers=["CUDAExecutionProvider"]) #"TensorrtExecutionProvider"
        else:
            session = onnxruntime.InferenceSession(onnx_dir, None)
        session.get_modelmeta()
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.session = session
        self.size = size

    def get_prediction(self, in_image):
        prediction = self.session.run([self.output_name], {self.input_name: in_image})
        return prediction[0]