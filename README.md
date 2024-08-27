# FaceRecognition

Face recognition in PyTorch.

## Installation

Use Python 3.10:

```bash
conda create -n FaceRecognition python=3.10
```

- For CPU:

  ```bash
  pip install torch torchvison
  ```

- For AMD GPU:

  ```bash
  pip install torch==2.3.1 torchvision
  pip install torch-directml
  pip install numpy<2
  ```

- For NVIDIA GPU:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
  ```

Then:

```bash
pip install opencv-python rich
git clone https://github.com/MarkIvory2973/ProxyTest.git
```

⚠ ***If you don't use AMD GPU:***

```python
# src/ResNet18/__init__.py
#import torch, torch_directml, os, cv2, random
import torch, os, cv2, random
```

## Usage

```bash
cd FaceRecognition
python train -d E:/Datasets -c ./checkpoints -e 40 -r 0.02 -s 16 -g 0.94
python register -i 0 -c ./checkpoints -n Guest01
python verify -i 0 -d E:/Datasets -c ./checkpoints -n Guest01
```

## Parameters

Train mode:

|Parameter|Required|Default|Description|
|:-|:-:|:-|:-|
|--datasets-root,-d|✓|-|Datasets root|
|--checkpoints-root,-c|-|./checkpoints|Checkpoints root|
|--total-epoch,-e|-|50|Total epoch|
|--learning-rate,-r|-|0.01|Learning rate|
|--batch-size,-s|-|8|Batch size|
|--gamma,-g|-|0.95|The gamma of ExponentialLR|

Register mode:
|Parameter|Required|Default|Description|
|:-|:-:|:-|:-|
|--camera-id,-i|-|0|Camera ID|
|--checkpoints-path,-c|-|./checkpoints|Checkpoints path|
|--username,-n|✓|-|Username|

Verify mode:
|Parameter|Required|Default|Description|
|:-|:-:|:-|:-|
|--camera-id,-i|-|0|Camera ID|
|--datasets-root,-d|✓|-|Datasets root|
|--checkpoints-path,-c|-|./checkpoints|Checkpoints path|
|--username,-n|✓|-|Username|

## The structure of datasets folder

```bash
Datasets_Root/
├──CelebA_1/
│  │──000001.jpg
│  │──000002.jpg
│  │──000003.jpg
│  │  ...
│  └──202599.jpg
│──CelebA_2/
│  │──1/
│  │  │──000023.jpg
│  │  │──004506.jpg
│  │  │──006439.jpg
│  │  │  ...
│  │  └──157602.jpg
│  │──2/
│  │  │──016188.jpg
│  │  │──051523.jpg
│  │  │──068490.jpg
│  │  │  ...
│  │  └──155885.jpg
│  │──3/
│  │  │──003029.jpg
│  │  │──003206.jpg
│  │  │──008838.jpg
│  │  │  ...
│  │  └──161022.jpg
│  │  ...
│  └──10177/
│     │──010230.jpg
│     │──013391.jpg
│     │──053390.jpg
│     │  ...
│     └──158603.jpg
└──LFW_2/
   │──Aaron_Eckhart/
   │  └──Aaron_Eckhart_0001.jpg
   │──Aaron_Guiel/
   │  └──Aaron_Guiel_0001.jpg
   │──Aaron_Patterson/
   │  └──Aaron_Patterson_0001.jpg
   │  ...
   └──Zydrunas_Ilgauskas/
      └──Zydrunas_Ilgauskas_0001.jpg
```

## Result

![The result of training]()
