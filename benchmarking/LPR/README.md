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
In the `eval` folder, we provide the template for a generic evaluation script for inter and intra-sequence place recognition performance evaluation.  To use this script, replace the `model_factory` and `get_latent_vector` placeholder functions with ones which load the model to be evaluated and extract place descriptors in order to get the LPR performance for a given pre-trained model and checkpoint.  Then, first generate the pickle files using the following command:

```
python eval/make_testing_pickles.py --dataset_root /path/to/wildcross --save_dir /pickle/save/dir
```

And then use the following commands to either run inter-sequence evaluation:
```
python inter-sequence-wildcross.py \
    --test_pickle_files /pickle/save/dir/venman_testing_info.pickle /pickle/save/dir/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir /path/to/results/save/dir \
    --split_idx CROSSFOLD_SPLIT_IDX \
    --ckpt /path/to/pretrained/ckpt.pth \
    --dataset_root /path/to/wildcross/root
```

Or intra-sequence evaluation:
```
python intra-sequence-wildcross.py \
    --test_pickle_files /pickle/save/dir/venman_testing_info.pickle /pickle/save/dir/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir /path/to/results/save/dir \
    --split_idx CROSSFOLD_SPLIT_IDX \
    --ckpt /path/to/pretrained/ckpt.pth \
    --dataset_root /path/to/wildcross/root
```

**Note**: As in the paper, here we use 1-indexing such that `split_idx 1` means that V-01 and K-01 are held out for evaluation and the rest of the data is used for training.