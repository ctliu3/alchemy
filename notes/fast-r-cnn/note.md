Fast R-CNN
===
Paper: [Fast R-CNN](http://arxiv.org/abs/1504.08083)

Year: 2015

#### Keynotes

- Fast when comparing with R-CNN and SPPnet.
- **Fast R-CNN architecture**
    - The input of the network are images and RoIs. The RoIs can be obtained by selective search or DPM.
    - The output of RoI pooling layers is feed to a sequence of fully connected (fc) layers and finally branch into two sibling output layers: classification branch and regression branch.
- **RoI pooling layer** : Use max pooling in the region of interest, the output is a fixed H x W feature map, where H and W are hyper-parameters.  The pooled region size is determined by the input size and H, W parameters.
- **Sampling**: Use two images and their RoIs in one mini-batch. This sampling strategy can improve the training efficiency. And this sampling seems do not cause slow training convergence issue.
- Others
    - Multi-task learning helps.
    - SVD reduces the number of parameter and only costs little accuracy decrease.

#### Questions

- Not quite understand BP of RoI pooling layer.
