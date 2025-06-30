FROM python:3

# Install python development requirements
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Install maven and clean up
RUN apt-get update \
    && apt-get install -y maven \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set the working directory
WORKDIR /workspace

# Add normal user. USER_UID, and USER_GID are passed from local environment by devcontainer.json.
# Otherwise all files are created with root privileges.
ARG USER_UID
ARG USER_GID

RUN groupadd --gid $USER_GID devuser \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash devuser

USER devuser