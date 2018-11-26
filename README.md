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

- download [yolo.weights](https://mega.nz/#!HTAXUKzS!Rjp2fda8wDtZ3svORzp0NN5iHJFJJ_9Nin-1H22KH54) into `temp/weights/`
- download [initial-model.h5](https://mega.nz/#!GfZXwRZK!usMEKy7jzSTu8xIQzQudomewd3CY477XvjFt5Zws_ss) into `temp/checkpoints/`
- download [person-small.tar.gz](https://mega.nz/#!GOhlEa7A!OJAOHo5icQAxB_dirrSs3CtOtDp9-KCjWkAQfJbht8M) (10MB) or [person-full.tar.gz](https://mega.nz/#!HX5ABaia!1ROcMJwRo3NUM8e9Vz6dUUTEdOfzHLDTY2-b3AZAxwQ) (310MB)
- extract compressed `.tar.gz` file into `temp/` folder
- run `seed` to generate training set & evaluation set

For example:

```
cd yolo-aitl
tar -xzvf temp/person-small.tar.gz -C temp
./seed.py -d temp/person-small/
```

`seed.py` would automatically separate dataset to training set and evaluation set for you.


Lastly, if everything is ok, you can start training now:


```
./train.py
```


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

5, Detect persom from camera

```
./predict -c N
```

With N is camera index.


### TensorBoard

```
tensorboar --logdir=temp/logs
```


### Test

```
./test.sh
```

