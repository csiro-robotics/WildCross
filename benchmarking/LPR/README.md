# LPR for WildCross

## Training 
To train LPR on WildCross, we leverage the tools in the Wild-Places LPR repository.  To train the network, first clone the Wild-Places repository and check out the `WildCross_splits` branch using the following command:

```
git clone git@github.com:csiro-robotics/Wild-Places.git
cd Wild-Places
git checkout WildCross_splits
```
Then, follow the instructions on this branch on how to generate the training splits for each method and perform LPR training.

## Evaluation
In the `eval` folder, we provide the template for a generic evaluation script for inter and intra-sequence place recognition performance evaluation.  To use this script, replace the `model_factory` and `get_latent_vector` placeholder functions with ones which load the model to be evaluated and extract place descriptors in order to get the LPR performance for a given pre-trained model and checkpoint.