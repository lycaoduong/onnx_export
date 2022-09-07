from utils.convert_model import yolov4_darknet_2_onnx, yolov7_pt_2_onnx
import argparse


def convert_yolov4_onnx(cfg, weight, save_dir='./export_model', save_name='v4_test'):
    in_s, out_s = yolov4_darknet_2_onnx(cfg, weight, 1, None, save_dir, onnx_file_name=save_name)
    print('Done. In: {}, Out: {}'.format(in_s, out_s))
    print('Check model on {}'.format(save_dir))

def convert_yolo7_onnx(weight, input_size=512, save_dir='./export_model', save_name='v7_test', dynamic=False):
    in_s, out_s = yolov7_pt_2_onnx(weight, input_size, save_dir, onnx_file_name=save_name, dynamic=dynamic)
    print('Done. In: {}, Out: {}'.format(in_s, out_s))
    print('Check model on {}'.format(save_dir))

def get_args():
    parser = argparse.ArgumentParser('Convert YOLO to ONNX')
    parser.add_argument('-v', '--version', type=int, default=4, help='Choosing Yolo version: 4 or 7')
    parser.add_argument('-cfg', '--cfg_file', type=str, default='./pretrained/yolov4_PPE/samsung_PPE.cfg',
                        help='YoloV4 config file')
    parser.add_argument('-v4w', '--v4_weight', type=str, default='./pretrained/yolov4_PPE/samsung_PPE.weights',
                        help='YoloV4 weights file')
    parser.add_argument('-v7w', '--v7_weight', type=str, default='./pretrained/yolov7_PPE_v1/best.pt',
                        help='YoloV7 weights file')
    parser.add_argument('-in_size', '--input_size', type=int, default=512,
                        help='YoloV7 input image size')
    parser.add_argument('-dyn', '--dynamic', type=bool, default=False,
                        help='Set dynamic axes for V7 onnx')
    args = vars(parser.parse_args())
    return args

# For YoloV4
# cfg_file = './pretrained/yolov4_PPE/samsung_PPE.cfg'
# weight_file_v4 = './pretrained/yolov4_PPE/samsung_PPE.weights'
# For YoloV7
# weight_file_v7 = './pretrained/yolov7_PPE_v1/best.pt'
# input_size = 512 # Have to fix if you want to run on TensortRT, the value depending on training params
# Set dynamic is True if want to set Dynamic input and output (Support only CPU and CUDA)

if __name__ == '__main__':
    opt = get_args()
    version = opt.get('version')
    print('Convert YOLO V{} to ONNX ...'.format(version))
    if version == 4:
        cfg_file = opt.get('cfg_file')
        weight_file_v4 = opt.get('v4_weight')
        convert_yolov4_onnx(cfg_file, weight_file_v4)
    elif version == 7:
        weight_file_v7 = opt.get('v7_weight')
        input_size = opt.get('input_size')
        dynamic = opt.get('dynamic')
        convert_yolo7_onnx(weight_file_v7, input_size, dynamic=dynamic)
