FROM python:3.11

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install numpy 
RUN pip install matplotlib 
RUN pip install torch
RUN pip install torchvision
RUN pip install Pillow 
RUN pip install streamlit
