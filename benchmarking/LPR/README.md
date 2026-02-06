# LPR for WildCross
WildCross is an extension to the original [Wild-Places](https://csiro-robotics.github.io/Wild-Places/) dataset designed for LPR.
To ensure consistency between this and our previous work, we keep all benchmarking code related to our WildCross LPR experiments within the Wild-Places repository. 

To train and evaluate on the WildCross data splits and replicate the results of our paper, simply clone the [Wild-Places repository](https://github.com/csiro-robotics/Wild-Places) and check out the `WildCross_splits` branch using the following command:

```
git clone git@github.com:csiro-robotics/Wild-Places.git
cd Wild-Places
git checkout WildCross_splits
```
Within this branch you can find the training and evaluation scripts to replicate our LPR experiments using LoGG3D-Net, MinkLoc3Dv2, and HOTFormerLoc, as well as links to our pre-trained weights for said models.