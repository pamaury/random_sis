FROM sagemath/sagemath:latest

RUN sudo apt update
RUN sudo apt install -y git autoconf libtool make
RUN sage --pip install tabulate
RUN sage --pip install pandas
RUN git clone https://github.com/malb/lattice-estimator.git
RUN git clone https://github.com/fplll/g6k.git
RUN cd g6k && sage --pip install -r requirements.txt && sage --python setup.py build_ext --inplace && sage --python setup.py install
VOLUME /home/sage/mnt

ENTRYPOINT /bin/bash
WORKDIR /home/sage/mnt/
