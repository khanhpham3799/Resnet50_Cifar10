# Welcom to Duy Khanh channel
This is the modified of Resnet50 model for CIFAR10 dataset.
## Model summary (Resnet_Unet_model.py)
Resnet50 + 1 block Unet 

![Resnetmodel_summary1](https://user-images.githubusercontent.com/64471569/132592352-b19e6447-251e-4a32-9cca-80792f321df9.png)

![Resnetmodel_summary2](https://user-images.githubusercontent.com/64471569/132592404-790f6457-4fb7-4b03-b32e-a41cef133227.png)

![Resnetmodel_summary3](https://user-images.githubusercontent.com/64471569/132592433-1e28609d-e0ab-4520-8576-d8c450a4ed9d.png)

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
Loss and accuracy
![Loss and accuracy](https://user-images.githubusercontent.com/64471569/132589044-605fe954-9ed8-4795-86fd-919afb76433d.png)
Confusion matrix
![Confusion matrix](https://user-images.githubusercontent.com/64471569/132592485-0f7e4227-012b-4198-b751-64bffa7f45ea.png)
