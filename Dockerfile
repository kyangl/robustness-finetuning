FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y python3 git

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
    accelerate>=0.20.1 \
    transformers[torch] \
    torch \
    torchvision \
    datasets \
    adapters \
    evaluate \
    scikit-learn \
    argparse 

# set the working directory in the container
WORKDIR /workspace

COPY . /workspace 