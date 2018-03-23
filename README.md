# README #

This README would normally document whatever steps are necessary to get your application up and running.

### Setup

```
git clone git@bitbucket.org:toancauxanh/yolo-aitl.git
cd yolo-aitl
python3 -m venv --system-site-packages venv
source venv/bin/activate
(venv) pip install -r requirements.txt
(venv) ./init.py
```

For simpler, the commands below will be presented without the prefix "(venv)".

Depend on your system, let's choose `tensorflow` or `tensorflow-gpu` should be installed:

```
pip install tensorflow

# or

pip install tensorflow-gpu
```

Then:

- download [yolo.weights](http://49.156.52.21:7777/YOLO/yolo.weights) into `temp/weights/`
- download [initial-model.h5](http://49.156.52.21:7777/checkpoints/initial-model.h5) into `temp/checkpoints/`
- download [person-small.tar.gz](http://49.156.52.21:7777/dataset/person/person-small.tar.gz) (10MB) or [person-full.tar.gz](http://49.156.52.21:7777/dataset/person/person-full.tar.gz) (310MB)
- extract compressed `.tar.gz` file into `temp/` folder
- run `seed` to generate training set & evaluation set

For example:

```
cd yolo-aitl

wget http://49.156.52.21:7777/YOLO/yolo.weights -P temp/weights/
wget http://49.156.52.21:7777/checkpoints/initial-model.h5 -P temp/checkpoints/
wget http://49.156.52.21:7777/dataset/person/person-small.tar.gz -P temp/

tar -xzvf temp/person-small.tar.gz -C temp
./seed.py -d temp/person-small/
```

`seed.py` would automatically separate dataset to training set and evaluation set for you.


Lastly, if everything is ok, you can start training now:


```
./train.py
```

If the server 49.156.52.21 does not work, here is the alternative:

https://mega.nz/#F!fXBVgYrZ!sP1cSnhJVKME15XGWImTxw


### Prediction


Using `predict.py`, you can:


1, Detect person from specified image

```
./predict.py -f tests/images/01.jpg

```


2, Detect all .jpg image from specified folder

```
./predict.py -d path_to_input_folder -o path_to_output_folder
```


3, Randomly detect an image stored in `test/images`

```
./predict.py
```


4, Detect person from AVI or MP4 video file

```
./predict.py -f path_to_your_video.avi
```


### TensorBoard

```
tensorboar --logdir=temp/logs
```


### Test

```
./test.sh
```

