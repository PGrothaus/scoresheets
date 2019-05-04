FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y vim

RUN apt-get install -y protobuf-compiler
RUN apt-get install -y python-pil
RUN apt-get install -y python-lxml
RUN apt-get install -y python-tk

RUN pip install Cython==0.25.2
RUN pip install contextlib2==0.5.5
RUN pip install scikit-image==0.13.0
RUN pip install PyYAML==3.12
RUN pip install progressbar2==3.34.3

RUN pip install python-chess==0.23.10

WORKDIR /opt
RUN git clone -b 3.4 https://github.com/opencv/opencv_contrib
WORKDIR /opt
RUN git clone -b 3.4 https://github.com/opencv/opencv
WORKDIR /opt/opencv
RUN mkdir build
WORKDIR /opt/opencv/build

RUN apt-get update && apt-get install -y build-essential cmake
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D BUILD_NEW_PYTHON_SUPPORT=ON \
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 -D INSTALL_C_EXAMPLES=OFF \
 -D INSTALL_PYTHON_EXAMPLES=OFF \
 -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
 -D PYTHON_EXECUTABLE=/usr/bin/python2.7 \
 -D BUILD_EXAMPLES=OFF /opt/opencv
RUN make -j $(nproc)
RUN make install
RUN ldconfig

### CHANGE NOTHING ABOVE THIS LINE

RUN pip install editdistance
RUN pip install numpy
RUN apt-get install python3-tk
RUN pip install shapely

WORKDIR /src
COPY . /src
WORKDIR /src

RUN chmod +x eval.sh
ENV PYTHONPATH "$PYTHONPATH:/opt/opencv/build/lib"
