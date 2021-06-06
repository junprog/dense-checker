Dense-Checker
===

## Application development with COVID-19

This project will help to avoid the close or dense situation between people with DEEP LEARNING and so on...

## Document

* Ideas and solutions : [Google Docs](https://docs.google.com/document/d/1SBujBopHUcR5McblwqASUc4pZSVqQubFaBeCCH0w1pM/edit?usp=sharing)

* Progress Management : [Google Spread Sheets](https://docs.google.com/spreadsheets/d/1siRg7qTlEc6rQirJbn1OY48E0qZ5ZRyTXaP1s8RAq74/edit?usp=sharing)

## Demo (Current status)

### Requirements

* torch
* torchvision
* numpy
* python-opencv
* flask

### Run this script.
```bash
$ python app.py
```

And open `localhost:8000` on web browser. 

## Docker (Only Jetson Support)

### Run this script.
```bash
sudo docker build -t dense-check .
sudo docker run --device /dev/video0:/dev/video0:mwr -it --name test --runtime nvidia --network host -p 8000:8000 -t dense-check
```

And open `localhost:8000` on web browser. 