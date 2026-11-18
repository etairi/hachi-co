# Hachi-Co in a container: the release build of hachi-co-party and the scripts, to run the quick start or any preset
# without installing Rust or Python. All provers of a run share the container and talk over its loopback.
#
#   docker build -f docker/quickstart.Dockerfile -t hachi-co .
#   docker run --rm hachi-co                     # two provers at bench-l16-n2, then the verifier
#   docker run --rm hachi-co python3 scripts/run_parties.py --preset secure100-l20-n2-a8 --n 2 --out /tmp/run
#
# On x86-64 the build targets the CPU of the machine that builds the image (.cargo/config.toml), so build the image
# on the machine that runs it. --build-arg JOBS=<k> limits the build to k parallel jobs (default: every core).
# docker/Dockerfile is a different image: one prover per container, for scripts/wan_docker.py.
ARG RUST_VERSION=1.99.1
FROM rust:${RUST_VERSION}-bookworm AS build
ARG JOBS
WORKDIR /hachi-co
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY crates crates
RUN cargo build --release --locked ${JOBS:+-j $JOBS} -p hachi-co --bin hachi-co-party

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 time procps \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /hachi-co
COPY --from=build /hachi-co/target/release/hachi-co-party target/release/hachi-co-party
COPY scripts scripts
CMD ["python3", "scripts/run_parties.py", "--preset", "bench-l16-n2", "--n", "2", "--out", "/tmp/run"]
