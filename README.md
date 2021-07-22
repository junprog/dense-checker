Dense-Checker
===

## Requirements

* torch
* torchvision
* numpy
* python-opencv
* flask

## Demo (current status)

Checking the dense situation

Run this script.
```bash
$ python app.py
```

And open `localhost:8000` on web browser. 

## Docker (not support yet)

```bash
sudo docker build -t dense-check .
sudo docker run -it --name test --runtime nvidia --network host -t dense-check
```