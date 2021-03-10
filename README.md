Dense-Checker
===

## Application development with COVID-19

This project will help to avoid the close or dense situation between people with DEEP LEARNING and so on...

## Document

* Ideas and solutions : [Google Docs](https://docs.google.com/document/d/1SBujBopHUcR5McblwqASUc4pZSVqQubFaBeCCH0w1pM/edit?usp=sharing)

* Progress Management : [Google Spread Sheets](https://docs.google.com/spreadsheets/d/1siRg7qTlEc6rQirJbn1OY48E0qZ5ZRyTXaP1s8RAq74/edit?usp=sharing)

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

## Docker

```bash
sudo docker build -t dense-check .
sudo docker run -it --name test --runtime nvidia --network host -t dense-check
```