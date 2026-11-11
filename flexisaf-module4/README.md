# FlexiSAF Module 4: Clothing Classifier (Computer Vision)

Project chosen from DataCamp's [Computer Vision Projects](https://www.datacamp.com/blog/computer-vision-projects) list: **Clothing Classifier** (beginner).

A small convolutional neural network (CNN), built with PyTorch, learns to recognise 10 types of clothing from 28x28 grayscale images.

## Dataset
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research: 60,000 training images and 10,000 test images across 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). The script downloads it automatically into `data/` on first run.

## Steps
1. **Load data**: download Fashion-MNIST and scale pixel values from 0-255 to 0-1.
2. **Train**: fit a small CNN (two convolution + pooling layers, then two dense layers with dropout) for 3 epochs using the Adam optimiser.
3. **Evaluate**: test on the 10,000 unseen images, report overall and per-class accuracy, and save a confusion matrix and sample predictions.

## Results
- Test accuracy: **89.1%** on unseen images
- Easiest classes: Sandal (99.0%), Bag (98.0%), Trouser (96.3%)
- Hardest class: Shirt (68.5%), which the model often confuses with T-shirt/top, Pullover and Coat because they look similar at 28x28 pixels

Accuracy can vary slightly between runs and machines.

## How to run
```
pip install -r requirements.txt
python clothing_classifier/train.py
```
Training takes about 1-3 minutes on a normal laptop CPU.

## Output
- `evidence/training_output.txt`: training progress, test accuracy, per-class accuracy
- `evidence/confusion_matrix.png`: which classes get mixed up
- `evidence/sample_predictions.png`: 18 test images with predicted and actual labels
- `clothing_classifier/model.pt`: the trained model weights
