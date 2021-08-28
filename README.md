# my_CNNs_classification

## Reference
[1] [MobileNet V1 Model](https://arxiv.org/pdf/1704.04861.pdf)\
[2] [MobileNet V2 Model](https://arxiv.org/pdf/1801.04381.pdf)\
[3] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)\
[4] [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

## Project Structure
```
    my_CNNs_classification
                    |
                    ├── config
                    |	    ├── cifar.yaml       
                    |	    └── hymenoptera.yaml
                    |
                    ├── models
                    |	    ├── efficientnet.py  # definition of EfficientNet model
                    |	    ├── mobile_net.py    # definition of MobileNet v1 model
                    |     ├── mobilenet_v2.py  # definition of MobileNet V2 model
                    |     └── se_resnet.py     # definition of ResNet model with Squeeze and Excitation class                    
                    |
                    ├── my_predict
                    |	    ├── inference.py
                    |     ├── test.py
                    |     ├── yaml_file.yaml
                    |     └── hymenoptera.yaml
                    |
                    ├── dataset
                    |	    ├── cifar.py
                    |     ├── dataset.py
                    |     └── dogcat_dataset.py     
                    |
                    ├── training.py
                    ├── test.py
                    └── utils.py
```

## TODO
 - [X] Applying models which is proposed in paper [1, 2, 3, 4].
 - [X] Upload CIFAR10 and Hymenoptera dataset.
 - [X] Use early stoping to stop training when model is not optimizer in some(setting with 10) epochs.
 - [X] Use lr_scheduler to tune hypeparameters in optimizer function. 
 - [X] Inference with test dataset after training to check accuracy.
 - [X] Applying transfer learning (TL) to train.
 
 
 ## Results
| Model | CIFAR 10 | Hymenoptera |
| ---         |     ---      |          --- |
| MobileNet V1 |   acc: > 87%    |  acc: > 95%   |
| MobileNet V2 |   acc: > 89%   |  acc: > 95%   |
| ResNet |   no result (my colab account can't response memory)   |  acc: > 95%   |
| TL with mobilenetV2(pretrained=True) |   acc: > 95%   |  acc: > 97%   |
| TL with mobilenetV2(pretrained=False) |   acc: > 89%   |  acc: > 97%   |



## Contributor
*Xuan-Phong Pham*
