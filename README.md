# Minimal code and simple experiments to play with Denoising Diffusion Probabilistic Models (DDPMs)

All experiments have tensorboard visualizations for samples / train curves etc.

1. To run the toy data experiments:
```
python scripts/train_toy.py --dataset swissroll --save_path logs/swissroll
```

2. To run the discrete mode collapse experiment:
```
python scripts/train_mnist.py --save_path logs/mnist_3 --n_stack 3
```

This requires the pretrained mnist classifier:
```
python scripts/train/mnist_classifier.py
```

3. To run the CIFAR image generation experiment:
```
python scripts/train_cifar.py --save_path logs/cifar
```

4. To run the CelebA image generation experiments:
```
python scripts/train_celeba.py --save_path logs/celeba
```
