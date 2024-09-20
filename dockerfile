# Use the official Python image as the base image
FROM python:3.8.15-slim

# Set environment variables
ENV LANG=C.UTF-8

# Install necessary packages and configure them
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        netbase \
        wget \
        gnupg \
        dirmngr \
        git \
        mercurial \
        openssh-client \
        subversion \
        procps \
        libbluetooth-dev \
        tk-dev \
        uuid-dev \
        autoconf \
        automake \
        bzip2 \
        dpkg-dev \
        file \
        g++ \
        gcc \
        imagemagick \
        libbz2-dev \
        libc6-dev \
        libcurl4-openssl-dev \
        libdb-dev \
        libevent-dev \
        libffi-dev \
        libgdbm-dev \
        libglib2.0-dev \
        libgmp-dev \
        libjpeg-dev \
        libkrb5-dev \
        liblzma-dev \
        libmagickcore-dev \
        libmagickwand-dev \
        libmaxminddb-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libpng-dev \
        libpq-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libtool \
        libwebp-dev \
        libxml2-dev \
        libxslt-dev \
        libyaml-dev \
        make \
        patch \
        unzip \
        xz-utils \
        zlib1g-dev \
        tmux; \
    if apt-cache show 'default-libmysqlclient-dev' 2>/dev/null | grep -q '^Version:'; then \
        apt-get install -y --no-install-recommends default-libmysqlclient-dev; \
    else \
        apt-get install -y --no-install-recommends libmysqlclient-dev; \
    fi; \
    rm -rf /var/lib/apt/lists/*; \
    pip install --no-cache-dir jupyterlab pandas pydicom nibabel; \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113; \
    apt-get clean;

# Add the Jupyter Lab start command to the bash history
RUN echo "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root" >> /root/.bash_history

# Set the command to run on container start
CMD ["bash"]