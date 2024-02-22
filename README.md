# DeepLabV3-CelebHQ
![Logotype](./misc/logo.png)

Trained Torch version of the DeepLab on the CelebHQ. Also dataset used in the model training are [here](https://github.com/switchablenorms/CelebAMask-HQ).

# Requirements
This project requires version of Python of at least 3.10.* Other requirements are listed in the [requirements.txt](./requirements.txt) file. To install all requirements:
```
python3 -m pip install -r requirements.txt
```

# Usage
## Training
TBD
## Evaluation
TBD

# TODO
This list will be updated throughout the time. Contributions are hugely appreciated!

Task name | Progress |
----------|----------|
Implement Tensorboard logging|:white_square_button:|
Implement callback for precision|:white_square_button:|
Implement callback for precision|:white_square_button:|
Implement script for the evaluation|:white_square_button:| 
Train weights and make them public|:white_square_button:|

# Citations
```
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```