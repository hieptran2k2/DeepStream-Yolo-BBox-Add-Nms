# DeepStream-Yolo-BBox-Add-Nms
NVIDIA DeepStream SDK 7.1 application for YOLO-BBox models
--------------------------------------------------------------------------------------------------
### Improvements on this repository

* Custom ONNX model parser

### Getting started
#### Export model to onnx format with nms
```
from ultralytics import YOLO

# Load a model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(imgsz=640, format="onnx", conf=0.25, iou=0.5, agnostic_nms=False, dynamic=True, nms=True)
```
### Supported models
* [YOLOv11](https://docs.ultralytics.com/tasks/detect/#train)

### Basic usage

#### 1. Download the repo

```
git clone https://github.com/hieptran2k2/DeepStream-Yolo-BBox-Add-Nms.git
cd DeepStream-Yolo-BBox-Add-Nms
```
#### 2. Download the `cfg` and `weights` files from [Ultralytics](https://objects.githubusercontent.com/github-production-release-asset-2e65be/521807533/34b70ade-b6eb-4179-a60f-d6494307226b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250221T080530Z&X-Amz-Expires=300&X-Amz-Signature=797b6690d26f075652bc9d62aa47579b394eacec2ac5276de204f21d9c1ac9b4&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dyolo11n.pt&response-content-type=application%2Foctet-stream) repo to the DeepStream-Yolo folder

#### 3. Compile the lib

3.1. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 7.1 = 12.6
  ```

* Jetson platform

  ```
  DeepStream 7.1 = 12.6
  ```

3.2. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo-BBox clean && make -C nvdsinfer_custom_impl_Yolo-BBox
```

#### 4. Edit the `config_infer_primary.txt` file according to your model (example for YOLOv4)

```
[property]
...
onnx-file=path/to/model.onnx
model-engine-file=path/to/model.engine
labelfile-path=/path/to/label.txt
...
```
#### 5. Run
```
python main.py -i <uri1> [uri2] -o /path/to/output/file -c /path/to/config/file
```
* Note

|       Flag          |                                   Describe                             |                             Example                          |
| :-----------------: | :--------------------------------------------------------------------: | :----------------------------------------------------------: |
| -i or --input       |      Path to input streams                                             | file:///path/to/file (h264, mp4, ...)  or rtsp://host/video1 |
| -o or --output      |      Path to output file                                               |                          /output/out.mp4                     |
| -c or  --configfile |      Choose the config-file to be used with specified pgie             |                      /model/pgie/config.txt                  |
| --file-loop         |      Loop the input file sources after EOS if input is file           |                                                               |

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: With DeepStream 7.1, the docker containers do not package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode. This change could affect processing certain video streams/files like mp4 that include audio track. Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features:

```
/opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

### Reference
- DeepStream-Yolo: https://github.com/marcoslucianops/DeepStream-Yolo
- DeepStream SDK Python: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
