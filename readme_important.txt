activate ui_onnx

pyinstaller onnx_export.py -p C:\Users\user\anaconda3\envs\ui_onnx\Lib\site-packages --hidden-import torchvision --hidden-import torch --hidden-import numpy -p C:\Users\user\anaconda3\envs\ui_onnx\Lib\site-packages\torch\lib --hidden-import models --hidden-import models.yolo --hidden-import models.common --hidden-import models.experimental

nvrtc64_112_0.dll
KERNEL32.dll
pyi-bindepend caffe2_nvrtc.dll -> api-ms-win-crt-runtime-l1-1-0.dll