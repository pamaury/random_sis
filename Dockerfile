FROM sagemath/sagemath:latest

RUN sudo apt update
RUN sudo apt install -y git autoconf libtool make
RUN sage --pip install tabulate
RUN sage --pip install pandas
RUN git clone https://github.com/malb/lattice-estimator.git
VOLUME /home/sage/mnt

ENTRYPOINT /bin/bash
WORKDIR /home/sage/mnt/
