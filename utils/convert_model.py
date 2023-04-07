import torch
import os
from networks.darknet.darknet_pytorch import Darknet_custom
import numpy as np
from models.yolo import Model


def yolov7_pt_2_onnx(cfg_file, weightfile, input_size=416, save_dir='./', onnx_file_name=None, dynamic=False):
    device = 'cpu'
    model = Model(cfg_file).to(device)
    weight = torch.load(weightfile, map_location=device)
    model.load_state_dict(weight, strict=False)
    model.eval()

    input_names = ["input"]
    output_names = ["output"]
    x = torch.randn((1, 3, input_size, input_size), requires_grad=True)
    if onnx_file_name is not None:
        onnx_file_name = "{}.onnx".format(onnx_file_name)
    else:
        onnx_file_name = "yolov7_pt.onnx"
    save_file_name = os.path.join(save_dir, onnx_file_name)
    if dynamic:
        dynamic_axes = {"input": {2: "img_w", 3: "img_h"}, "output": {0: "anchor_size"}}
    else:
        dynamic_axes = None
    torch.onnx.export(model,
                      x,
                      save_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes)

    if dynamic:
        input_shape = '[1 3 dynamic dynamic]'
        output_shape = '[dynamic 22]'
    else:
        num_output = int(3 * (np.power((input_size / 8), 2) + np.power((input_size / 16), 2) + np.power((input_size / 32), 2)))
        input_shape = '[{} 3 {} {}]'.format(1, input_size, input_size)
        output_shape = '[{} 6]'.format(num_output)
    return input_shape, output_shape


def yolov4_darknet_2_onnx(cfgfile, weightfile, batch_size=1, num_output=None, save_dir='./', onnx_file_name=None, quantization=False, batch=False):
    model = Darknet_custom(cfgfile, num_bb_filter=num_output, inference=True)
    model.print_network()
    model.load_weights(weightfile)
    model.eval()

    # if quantization:
    #     # Dynamic
    #     # model = torch.quantization.quantize_dynamic(
    #     #     model,  # the original model
    #     #     {torch.nn.Conv2d},  # a set of layers to dynamically quantize
    #     #     dtype=torch.qint8)  # the target dtype for quantized weights
    #     # Static
    #     model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    #     # model = torch.quantization.fuse_modules(model, [['conv', 'relu']])
    #     model = torch.quantization.prepare(model, inplace=False)
    #     model = torch.quantization.convert(model)

    print('Loading weights from %s... Done!' % (weightfile))

    if batch:
        batch_size = np.random.randint(2, 6)

    input_names = ["input"]
    output_names = ["output"]

    x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
    o = model(x).detach().cpu().numpy()
    # print(o.shape)

    if onnx_file_name is not None:
        onnx_file_name = "{}.onnx".format(onnx_file_name)
    else:
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, model.height, model.width)

    save_file_name = os.path.join(save_dir, onnx_file_name)
    # dynamic_axes = {"input": {2: "img_w", 3: "img_h"}, "output": {3: "anchor_size"}}
    if batch:
        dynamic_axes = {"input": {0: "num_batch"}, "output": {0: "num_batch"}}
        batch_size = 'Multi'
    else:
        dynamic_axes = None
    torch.onnx.export(model,
                      x,
                      save_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes)

    # test_model = Detecton_yolov4_onnx_custom_ly(save_file_name, size=model.height)

    # image = x.detach().numpy()
    # prediction = test_model.get_prediction(image)
    # num_output = prediction.shape[1]
    # num_output = int(3 * (np.power((model.height / 8), 2) + np.power((model.height / 16), 2) + np.power((model.height / 32), 2)))
    input_shape = '[{} 3 {} {}]'.format(batch_size, model.height, model.width)
    output_shape = '[{} {} 6]'.format(batch_size, o.shape[1])
    return input_shape, output_shape