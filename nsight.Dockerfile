# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg \
        libasound2t64 \
        libfreetype6 \
        libglib2.0-0 \
        libnss3 \
        libsqlite3-0 \
        libx11-xcb1 \
        libxcb-glx0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxml2 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        libgl1 \
        libglx-mesa0 \
        libxkbfile-dev \
        openssh-client \
        wget \
        xcb \
        openjdk-21-jdk \
        dnsutils \
        telnet \
        iproute2 iputils-ping \
        xkb-data && \
    apt-get clean

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        qt6-base-dev && \
    apt-get clean

# CMake e Ninja per la compilazione
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        cmake \
        ninja-build && \
    apt-get clean

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        zsh && \
    apt-get clean

CMD ["zsh"]

RUN cd /tmp && \
    wget  https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/nsight-systems-2025.3.1_2025.3.1.90-1_amd64.deb && \
    apt-get install -y ./nsight-systems-2025.3.1_2025.3.1.90-1_amd64.deb && \
    rm -rf /tmp/*
