# One Hachi-Co party per container, for the emulated wide-area runs of scripts/wan_docker.py. The image builds
# hachi-co-party from the working tree for the machine's architecture (arm64 on Apple Silicon), with the Rust release
# of the local toolchain, and adds what scripts/cluster.py uses: GNU time for the peak memory, tc (netem) for the
# emulated network, and ping and iperf3 to measure it.
#
#   python3 scripts/wan_docker.py build
ARG RUST_VERSION=1.99.1
FROM rust:${RUST_VERSION}-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p hachi-co --bin hachi-co-party
ARG COMMIT=unknown
ARG DIRTY=false
RUN printf '{"commit": "%s", "dirty": %s, "rustc": "%s", "built_in": "docker %s"}\n' \
      "$COMMIT" "$DIRTY" "$(rustc -V)" "$(uname -m)" > target/release/hachi-co-party.build.json

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends time iproute2 iputils-ping iperf3 procps ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/hachi-co-party /src/target/release/hachi-co-party.build.json /usr/local/bin/
WORKDIR /run/hco
CMD ["sleep", "infinity"]
