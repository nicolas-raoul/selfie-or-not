# Goal
This project's goal is to detect selfies among random pictures.

# Usage
Install the necessary packages, for instance on Ubuntu 2018.04:
```
sudo apt-get update && apt-get install -y python-numpy python-virtualenv
virtualenv tensorflow_env
pip install tensorflow keras
```
Then just run:
```
python run-preprocessed.py
```

# Data samples
All images are from Wikimedia Commons under the same name. You can also use your own images and generate their thumbnails with the included `preprocess-images.sh`.
