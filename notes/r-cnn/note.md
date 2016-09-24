R-CNN
===
Paper: [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

Year: 2013

#### Pipeline

- Selective search -> CNN features -> class-specific linear SVM.
- In test stage, non-maximum suppression is used.

#### Keynotes

- **Basic strategies**. Selective search, non-maximum suppression, linear SVM.
- **Use pre-trained model is really useful**. It's obvious.
- **Samples selection in fine-tuning CNN stage and SVM**. It's different.
  - In fine-tuning CNN phase, augmentation (avoid overfitting) + samples selection (IoU >= 0.5 is positive, others are negative samples for **ALL** classes).
  - For training SVMs, the ground-true boxes are positive and the samples, which IoU < 0.3, are negative.
  - *Notes*: the samples for SVM are more class-specific. Better positive samples and class-specific negative samples.
- **Bounding-box regression**. Big contribution of this paper.
  - BB regression is class-specific.
  - One of the input is the last convolution feature. x,y are scale-invariant translation and w,h are log-space translation.
  - Only take effect when the predicted box is close to the ground-true.
- **The composition of each mini-batch**. 128 = 32 (positive) + 96 (negative).
- **Hard negative mining method in SVM stage**.
- **Region to CNN input**. Warp method (anisotropically scales each object proposal to the CNN input size) performs best.

#### Numbers

- One image has around 2000 region candidates.
- 10ms+ in GPU and around 1 minute in CPU.
