FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

RUN apt-get -y update
RUN apt-get install -y python3-opencv
RUN pip3 install Flask

# file copy
COPY . /app
WORKDIR /app

EXPOSE 8000

CMD ["python3", "app.py"]