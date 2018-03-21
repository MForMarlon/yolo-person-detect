# README #

This README would normally document whatever steps are necessary to get your application up and running.

### Setup

```
cd worker-recognition
python3 -m venv venv
source venv/bin/activate
(venv) pip install -r requirements.txt
(venv) ./init.py
```

Before running train.py, you need:

- download [yolo.weights](https://pjreddie.com/media/files/yolo.weights) into `temp/weights/`
- download [pretrained.h5](https://mega.nz/#!GfZXwRZK!usMEKy7jzSTu8xIQzQudomewd3CY477XvjFt5Zws_ss) model file into `temp/`
- download [person-small.tar.gz](https://mega.nz/#!GOhlEa7A!OJAOHo5icQAxB_dirrSs3CtOtDp9-KCjWkAQfJbht8M) (10MB) or [person-full.tar.gz](https://mega.nz/#!HX5ABaia!1ROcMJwRo3NUM8e9Vz6dUUTEdOfzHLDTY2-b3AZAxwQ) (310MB)
- extract `person-*.tar.gz` to get `person-small` or `person-large` folder
- run `seed.py` to generate training set & evaluation set


```
tar -xzvf person-full.tar.gz -C temp
./seed.py -d temp/person-full/
```


If everything is ok, try:

```
(venv) ./train.py
```


### Prediction

```
(venv) ./predict.py -f tests/images/01.jpg

# or randomly predict

(venv) ./predict.py

# AVI and MP4 video file also work

(venv) ./predict.py -f path_to_your_video.avi
```


### Test

```
(venv) ./test.sh
```

