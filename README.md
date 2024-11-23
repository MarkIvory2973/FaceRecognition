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

## Training

I trained 50 epochs using ResNet18 for backbone and repeated the training of **the error set**(epoch & loss ≥ 10) on top of training the CelebA dataset.

## Results

11/23/2024:

![The result of training](https://raw.githubusercontent.com/MarkIvory2973/FaceRecognition/refs/heads/main/imgs/result%20(11-23-2024).png)

8/27/2024:

![The result of training](https://raw.githubusercontent.com/MarkIvory2973/FaceRecognition/refs/heads/main/imgs/result%20(8-27-2024).png)

## References

1. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

2. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
