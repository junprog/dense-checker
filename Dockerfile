FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

RUN apt-get -y update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    python3-opencv
    
RUN pip3 install Flask

# file copy
COPY . /app
WORKDIR /app

CMD ["python3", "app.py"]