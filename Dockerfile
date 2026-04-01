FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# install SSH server and dependencies
RUN apt-get update && \
	apt-get install -y openssh-server git awscli tmux && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# configure SSH
RUN mkdir -p /var/run/sshd /root/.ssh && \
	chmod 700 /root/.ssh && \
	sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
	sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
	ssh-keygen -A

RUN pip install --no-cache-dir numpy tqdm pyarrow

WORKDIR /workspace