FROM centos:7

USER root

RUN yum install -y gcc \
                   gcc-c++ \
                   make \
                   git \
                   python3 \
                   python3-devel \
                   python3-pip