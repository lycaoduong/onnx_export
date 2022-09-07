from utils.convert_model import yolov4_darknet_2_onnx, yolov7_pt_2_onnx

def convert_yolov4_onnx(cfg, weight, save_dir='./export_model', save_name='v4_test'):
    in_s, out_s = yolov4_darknet_2_onnx(cfg, weight, 1, None, save_dir, onnx_file_name=save_name)
    print('Done. In: {}, Out: {}'.format(in_s, out_s))
    print('Check model on {}'.format(save_dir))

def convert_yolo7_onnx(weight, input_size=512, save_dir='./export_model', save_name='v7_test', dynamic=False):
    in_s, out_s = yolov7_pt_2_onnx(weight, input_size, save_dir, onnx_file_name=save_name, dynamic=dynamic)
    return 'Done. In: {}, Out: {}'.format(in_s, out_s)
    print('Check model on {}'.format(save_dir))

# For YoloV4
cfg_file = './pretrained/yolov4_PPE/samsung_PPE.cfg'
weight_file_v4 = './pretrained/yolov4_PPE/samsung_PPE.weights'

#For YoloV7
weight_file_v7 = './pretrained/yolov7_PPE_v1/best.pt'
input_size = 512 # Have to fix if you want to run on TensortRT, the value depending on training params
# Set dynamic is True if want to set Dynamic input and output (Support only CPU and CUDA)

if __name__ == '__main__':
    version = 4
    if version == 4:
        convert_yolov4_onnx(cfg_file, weight_file_v4)
    elif version == 7:
        convert_yolo7_onnx(weight_file_v7, input_size, dynamic=False)
