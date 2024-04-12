# Face-Liveness

This repository contains the implementation of the paper titled "Deep Pixel-wise Binary Supervision for Face Anti-Spoofing" in the TensorFlow framework. The paper introduces a novel technique for face anti-spoofing using deep pixel-wise binary supervision.

## Overview

Face anti-spoofing is a crucial task in computer vision, especially in security-sensitive applications such as face recognition systems. Spoofing attacks involve presenting a fake face or image to a face recognition system to gain unauthorized access. The proposed technique in this paper leverages deep pixel-wise binary supervision to enhance the robustness of face anti-spoofing systems against such attacks.

## Paper

The paper detailing the approach implemented in this repository can be found on [arXiv](https://arxiv.org/pdf/1907.04047v1.pdf).

## Implementation

The implementation is provided in TensorFlow, a popular deep learning framework. The codebase includes the necessary scripts to train, evaluate, and test the face anti-spoofing model using the deep pixel-wise binary supervision technique.

## Requirements

- Python 3.x
- TensorFlow
- Other dependencies specified in `requirements.txt`

Install the required dependencies using:

```
pip install -r requirements.txt
```

## Usage

1. **Training**: Train the face anti-spoofing model using the provided training script. Customize the training parameters as needed.

```bash
python Train.py
```

2. **Testing**: Test the trained model on unseen data using the testing script. This script provides predictions and performance metrics on a real-time frames from your camera.

```bash
python Test.py
```

## Results

The results obtained from the experiments conducted with the implemented technique are summarized in the paper. Additional details and analysis can be found in the paper and supplementary materials.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project inspired from [Face-Anti-Spoofing-using-DeePixBiS](https://github.com/Saiyam26/Face-Anti-Spoofing-using-DeePixBiS)
