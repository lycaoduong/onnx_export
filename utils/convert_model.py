import torch
import os
from networks.darknet.darknet_pytorch import Darknet_custom
import numpy as np
from networks.yolov7_pt.experimental import attempt_load


def yolov7_pt_2_onnx(weightfile, save_dir='./', onnx_file_name=None):
    device = 'cpu'
    model = attempt_load(weightfile, map_location=device)
    model.eval()

    input_names = ["input"]
    output_names = ["output"]
    x = torch.randn((1, 3, 512, 512), requires_grad=True)
    if onnx_file_name is not None:
        onnx_file_name = "{}.onnx".format(onnx_file_name)
    else:
        onnx_file_name = "yolov7_pt.onnx"
    save_file_name = os.path.join(save_dir, onnx_file_name)
    # dynamic_axes = {"input": {2: "img_w", 3: "img_h"}, "output": {0: "anchor_size"}}
    dynamic_axes = None
    torch.onnx.export(model,
                      x,
                      save_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes)

    input_shape = '[1 3 dynamic dynamic]'
    output_shape = '[dynamic 22]'
    return input_shape, output_shape


def yolov4_darknet_2_onnx(cfgfile, weightfile, batch_size=1, num_output=None, save_dir='./', onnx_file_name=None):
    model = Darknet_custom(cfgfile, num_bb_filter=num_output, inference=True)
    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    input_names = ["input"]
    output_names = ["output"]

    x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)

    if onnx_file_name is not None:
        onnx_file_name = "{}.onnx".format(onnx_file_name)
    else:
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, model.height, model.width)

    save_file_name = os.path.join(save_dir, onnx_file_name)
    # dynamic_axes = {"input": {2: "img_w", 3: "img_h"}, "output": {3: "anchor_size"}}
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
    num_output = int(3 * (np.power((model.height / 8), 2) + np.power((model.height / 16), 2) + np.power((model.height / 32), 2)))
    input_shape = '[{} 3 {} {}]'.format(batch_size, model.height, model.width)
    output_shape = '[{} 6]'.format(num_output)
    return input_shape, output_shape