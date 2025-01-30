# FaceRecognition

Face recognition in PyTorch.

## Installation

Create an environment (Python **3.10**):

```bash
conda create -n FaceRecognition python=3.10
```

Install dependencies:

- For CPU:

  ```bash
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

- For AMD GPU (for **Windows** users):

  ```bash
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip3 install torch-directml
  ```

- For NVIDIA GPU:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
  ```

```bash
pip install opencv-python rich
```

Clone this repository:

```bash
git clone https://github.com/MarkIvory2973/ProxyTest.git
```

## Usage

```bash
cd FaceRecognition
python src/main.py train -d E:/Datasets -c ./checkpoints -e 40 -r 0.02 -s 16 -g 0.94
python src/main.py register -i 0 -c ./checkpoints -n Guest01
python src/main.py verify -i 0 -d E:/Datasets -c ./checkpoints -n Guest01
```

## Parameters

Train mode:

|Parameter|Required|Default|Description|
|:-|:-:|:-|:-|
|--datasets-root,-d|✓|-|Datasets root|
|--checkpoints-root,-c|-|./checkpoints|Checkpoints root|
|--total-epoch,-e|-|100|Total epoch|
|--learning-rate,-r|-|0.02|Learning rate|
|--batch-size,-s|-|16|Batch size|
|--gamma,-g|-|0.98|The gamma of ExponentialLR|

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

## Result

⚠ The metrics in `checkpoint.N.pth` are based on the output of **each 16 batch**, so their `Accuracy` may be not correct enough (**lower than actual**).

I tested again with `batch_size=1`. Here is the result of `checkpoint.best.pth` (`checkpoint.100.pth`):

||Accuracy (%)|
|:-:|-:|
|LFW|94.03|

## References

1. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

2. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
