FROM slimerl/slime:latest

USER root

# Install ARC runner binary
ARG RUNNER_VERSION=2.323.0
RUN mkdir -p /home/runner && \
    curl -fsSL https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    | tar xz -C /home/runner && \
    /home/runner/bin/installdependencies.sh && \
    chown -R root:root /home/runner

# Runner expects to be called as root or with the right perms
WORKDIR /home/runner
