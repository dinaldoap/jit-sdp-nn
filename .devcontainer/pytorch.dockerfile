FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

# Install Linux tools
RUN apt-get update -q && \
    apt-get install -yq openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Config locale
ENV LC_ALL C.UTF-8

# Config standard user
ARG USERNAME=pytorch

# Or your actual UID, GID on Linux if not the default 1000
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    #
    # [Optional] Add sudo support
    apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    rm -rf /var/lib/apt/lists/*

# Technically optional
ENV HOME /home/$USERNAME

# Config VS Code server directory
RUN mkdir "${HOME}/.vscode-server" && \
  chown -R ${USERNAME}:${USERNAME} "${HOME}/.vscode-server"

    
# Set the default user
USER $USERNAME

# Set the default shell to bash instead of sh
ENV SHELL /bin/bash

# Install python dependencies
ADD --chown=pytorch:pytorch conda.yml /workspace/
ADD --chown=pytorch:pytorch requirements.txt /workspace/
ADD --chown=pytorch:pytorch setup.py /workspace/
RUN cd /workspace && \
    conda env create --name pytorch --file conda.yml && \
    rm conda.yml && \
    rm requirements.txt && \
    rm setup.py

CMD ["/bin/bash"]