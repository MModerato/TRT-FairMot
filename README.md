# TRT-FairMot
通过将FairMOT跟踪模型转换为TensorRT进行了推理阶段的加速，其中难点主要在于ONNX和TensorRT不支持DCNv2层，需要进行一些额外的操作。

基本步骤：

1. Pytorch模型转换为ONNX
2. 编译DCNv2插件层，将ONNX转换为TensorRT
3. 基于TensorRT进行模型推理

## Pytorch→ONNX
由于ONNX中不包含DCNv2算子，将FairMOT模型转换为ONNX模型之前需要对DCNv2层进行一些处理。这部分参考了https://github.com/CaoWGG/TensorRT-CenterNet/blob/master/readme/ctdet2onnx.md ，据作者说使用的是mmdetection中的DCNv2实现，原本的DCNv2实现是基于THC（转换过程会出错），而mmdetection中的DCNv2基于ATen实现（可以顺利转换）。
- 首先编译此版本DCNv2：
```
git clone https://github.com/CaoWGG/TensorRT-CenterNet.git
cd TensorRT-CenterNet/readme/dcn
python setup.py build_ext --inplace
```
- 为了能让ONNX识别出DCNv2层，需要在代码中添加symbolic函数（上述编译的dcn文件夹中已经对此进行了修改）：
```
class ModulatedDeformConvFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias,stride,padding,dilation,groups,deformable_groups):
        return g.op("DCNv2", input, offset, mask, weight, bias,
                    stride_i = stride,padding_i = padding,dilation_i = dilation,
                    groups_i = groups,deformable_group_i = deformable_groups)
```
- 在FairMOT源代码中先把上述编译后的dcn文件夹通过`sys.path.insert`的方式设置到PYTHONPATH中，然后把涉及`import DCNv2`的部分全都修改为`from dcn.modules.deform_conv import ModulatedDeformConvPack as DCN`。

- 对模型的forward函数还需进行部分修改才能顺利转换为onnx，创建并运行convert2onnx.py：
```
from lib.opts import opts
from lib.models.model import create_model, load_model
from types import MethodType
import torch.onnx as onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import OperatorExportTypes
from collections import OrderedDict

def pose_dla_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x)

    y = []
    for i in range(self.last_level - self.first_level):
        y.append(x[i].clone())
    self.ida_up(y, 0, len(y))

    ###############################################  修改的部分 ############################################################
    # 这里的修改是因为导出到onnx必须是返回数组，不能是dict（原版是dict）。
    z = {}
    for head in self.heads:
        z[head] = self.__getattr__(head)(y[-1])

    hm = z["hm"]
    wh = z["wh"]
    id_feature = z["id"]
    reg = z["reg"]
    
    return [hm, wh, reg, id_feature]
    #######################################################################################################################

opt = opts().init() 
opt.arch = 'dla_34'
opt.heads = OrderedDict([('hm', 1), ('reg', 2), ('wh', 4), ('id', 128)])
opt.head_conv = 256
print(opt)
model = create_model(opt.arch, opt.heads, opt.head_conv)
model.forward = MethodType(pose_dla_forward, model)
model = load_model(model, './fairmot_dla34.pth')
model.eval()
model.cuda()

input = torch.zeros([1, 3, 608, 1088]).cuda()
onnx.export(model, input, "./fairmot_dla34.onnx", verbose=True,
            operator_export_type=OperatorExportTypes.ONNX,
            output_names=["hm", "wh", "reg", "id_feature"])
```
**NOTICE**  在convert2onnx.py中的pose_dla_forward函数内也可增加max_pool、sigmoid等模型后处理涉及的部分计算，这样可以在TensorRT中完成尽可能多的计算。

## ONNX→TensorRT
由ONNX模型转换为TensorRT模型之前，需要先向TensorRT注册DCNv2层。
### 注册DCNv2
这里按照https://github.com/lesliejackson/TensorRT7-DCNv2-Plugin 的步骤：
- 安装cub-1.8.0
```
wget https://codeload.github.com/NVlabs/cub/zip/1.8.0 
unzip 1.8.0      # 解压得到cub-1.8.0/
rm 1.8.0
```
- 下载TensorRT release/7.0版本（事先记得安装好TensorRT-7.0.0.11，这里下载的是用于编译插件的源码）
```
git clone --branch release/7.0  https://github.com/NVIDIA/TensorRT
```
将TensorRT7-DCNv2-Plugin工程的DCNv2文件夹和CMakeLists.txt复制到TensorRT/plugin中，向TensorRT/plugin中的InferPlugin.cpp文件加入头文件`#include "DCNv2/DCNv2.hpp"`以及在initLibNvInferPlugins函数中加入`initializePlugin<nvinfer1::plugin::DCNv2PluginCreator>(logger, libNamespace);`，并修改DCNv2.cpp中155行和156行（因为_input_dims没有定义），然后：
```
cd TensorRT/
mkdir build  &&  cd build
cmake .. -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF -DTRT_LIB_DIR=/home/Setup/TensorRT-7.0.0.11_cuda10.2/lib -DTRT_BIN_DIR=`pwd`/out
-DBUILD_PLUGINS=ON -DCUB_ROOT_DIR=/home/cub-1.8.0/
make -j4
```
编译完成后在TensorRT/build/out/文件夹中会生成libnvinfer_plugin.so等链接库，**将这些链接库复制到TensorRT-7.0.0.11_cuda10.2/lib中**。
- 下载onnx-tensorrt 7.0分支：
```
git clone --recursive  --branch 7.0  https://github.com/onnx/onnx-tensorrt
```
- 将TensorRT7-DCNv2-Plugin工程中的builtin_op_importers.cpp复制到onnx-tensorrt，然后打开onnx_utils.hpp文件，增加头文件:
```
#include <stdexcept>
#include <limits>
```
- 安装protobuf-3.11.4 : `apt-get install protobuf-compiler libprotoc-dev`
- 下载onnx v1.6.0版本代码到onnx-tensorrt/third_party
- 编译onnx-tensorrt：
```
cd onnx-tensorrt/
mkdir build  &&  cd build
cmake .. -DTENSORRT_ROOT=/home/Setup/TensorRT-7.0.0.11_cuda10.2/ 
make -j4
```
编译完成后在onnx-tensorrt/build/文件夹中会生成libnvonnxparser.so等链接库以及可执行文件onnx2trt，**将这些链接库复制到TensorRT-7.0.0.11_cuda10.2/lib中**。


## 模型转换
到这一步，DCNv2层就已经注册到了TensorRT中，然后就可以使用官方提供的trtexec（位于文件夹TensorRT-7.0.0.11_cuda10.2/bin）来进行模型转换，具体使用方式通过`trtexec -h`了解。
eg:
```
./trtexec --onnx=./fairmot_dla34.onnx  \
          --int8   \
          --calib=./calib.txt   \ 
          --explicitBatch=1    \
          --workspace=512   \
          --verbose=True   \
          --saveEngine=./fairmot_dla34.engine
```
转换为int8量化模型的时候需要提供calib.txt，calib.txt中每一行是用来进行量化计算的图像路径，这些图像要经过resize（尺寸与网络输入相符）。

## 模型推理
### 创建输入/输出变量
考虑到FairMOT模型forward之后还有不少后处理是基于pytorch进行的计算，如果按照官方提供的allocate_buffers函数进行TensorRT变量创建，那么会产生许多GPU-CPU之间的数据切换增加耗时，因此这里利用pytorch接管了TensorRT变量创建：
```
    def allocate_buffers_v2(self):
        inputs = []
        outputs = []
        bindings = []
        with torch.no_grad():
            for binding in self.engine:
                # shape = (self.engine.max_batch_size, ) + tuple(self.engine.get_binding_shape(binding))
                shape = tuple(self.engine.get_binding_shape(binding))
                dtype = torch.float32
                device = torch.device('cuda')
                temp_tensor = torch.empty(size=shape, dtype=dtype, device=device)

                # Append the device buffer to device bindings.
                bindings.append(int(temp_tensor.data_ptr()))
                # Append to the appropriate list.
                if self.engine.binding_is_input(binding):
                    inputs.append(temp_tensor)
                else:
                    outputs.append(temp_tensor)
        return inputs, outputs, bindings
```
### 加载插件
虽然已经向TensorRT注册了DCNv2，但是在加载模型之前，仍然需要在代码中显式地进行加载：
```
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
```
### 推理部分
因为利用pytorch接管了TensorRT变量创建，所以在推理部分也相应地修改了代码：
```
    def do_inference_v2(self):
        # Run inference.
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        # Synchronize the stream
        self.stream.synchronize()
        return self.outputs

    def infer_v2(self, blob):
        with torch.no_grad():
            # Copy input image to host buffer
            blob = torch.from_numpy(blob)
            self.inputs[0].copy_(blob)

        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Run inference
        output = self.do_inference_v2()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        return output
```

## 速度对比
待续

## 参考
[1] FairMOT: https://github.com/ifzhang/FairMOT

[2] TensorRT-CenterNet: https://github.com/CaoWGG/TensorRT-CenterNet

[3] TensorRT7-DCNv2-Plugin: https://github.com/lesliejackson/TensorRT7-DCNv2-Plugin




