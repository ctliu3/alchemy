FCN
===


Paper: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

Year: 2015

Keywords: fully convolutional network (FCN), upsamples, deconv, segmentation

#### The method
- Network structure: see [this](http://dgschwend.github.io/netscope/#/preset/fcn-16s), which is a two-steram network
- Use FCN.
 	1. Then the output is `#class x width x height` (instead of `#class x 1 x 1`). 
 	2. The output has the same width and height with the input.
 	3. The width and height will be smaller then the input image size (pooling layer, downsampling) .
 	4. How to do the pixel-level classification? -> upsampling (concretely, deconv).
- Dense prediction. Combine the shallow and deep layer to calculate the predicted value. FCN-32s (conv7 32x upsampled), FCN-16s (2x conv7 + pool4 -> 16x upsampled), FCN-8s (4x conv7 + 2x pool4 + pool3 -> 8x upsampled) means three concat strategies.
- Sampling (vs patch-wise sampling).
- Class balancing.