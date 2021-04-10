# GeoLayout3D

The repo is the 3D part implementation of [GeoLayout(ECCV 2020)](https://arxiv.org/abs/2008.06286)

### References

[1] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. International Conference on 3D Vision (3DV), 2017.



### Requirements

- CUDA 10.1
- pytorch 1.6.0
- torchvision 0.7.0
- numpy 1.19.1
- h5py 2.10.0
- pandas 1.1.0
- scipy 1.5.2
- Pillow 8.2.0
- scikit_learn 0.24.1
- opencv_python 4.5.1.48



### Usages

First, you should download the dataset **Matterport3D-Layout** following the instruction of https://github.com/vsislab/Matterport3D-Layout

Then, you should build the environment following the steps, anaconda is recommended

```
conda env create -f environment.yaml
conda activate geolayout
```

Then, you should set the parameters in train_utils.py, such as **data_dir**(the place to store the dataset) and **save_dir**(the place to store your train-valid info).

Then, to train the network, you should type

```
python main.py
```

After training, you can type this to post process the data and get the results

```
python process.py --bs=1
```

