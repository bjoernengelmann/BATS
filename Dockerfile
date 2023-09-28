FROM python:3.8

RUN apt-get update && apt-get install -y \
	git \
	openjdk-17-jdk \
	openjdk-17-jre 
	
WORKDIR /workspace
ADD . /workspace

RUN pip install --upgrade pip
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements
RUN python -m spacy download en_core_web_sm
RUN pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements_lg.txt
RUN pip install --no-cache-dir -U pyunpack
