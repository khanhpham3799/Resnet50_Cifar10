# Welcom to Duy Khanh channel
This is the modified of Resnet50 model for CIFAR10 dataset.
## Model summary (Resnet_Unet_model.py)
Resnet50 + 1 block Unet 

![model]()

## Data preparation

Cifar10-dataset transform 

* Crop, padding

* Rotate

* Flipping (vertical, horizontal)

* Normalization

## Training

* Loss: Cross Entropy

* Optimizer: SGD (lr = 0.001)

* Scheduler: ReduceLROnPlateau(maximize test accuracy)

* If train_acc >99 for 30 epochs continously -> Repeat transform with different ratio

## Checkpoint file link

## Result
![Loss and accuracy]()

![Confusion matrix]()
