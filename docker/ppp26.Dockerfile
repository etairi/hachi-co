# PPP26's implementation (Pappas, Papadopoulos, Papamanthou, ePrint 2026/729; the paper's reference [46],
# github.com/ChristodoulosPappas/Code-Based-Scalable-coSNARKs) with scripts/ppp26/timing.patch, for x86-64: it needs
# AVX2 and MPI, and its CMake flags are -march=native -mavx -msha. On Apple Silicon the image runs under Rosetta, which
# has AVX2 but no SHA extensions, so it serves to test scripts/ppp26.py, not to time anything.
#
#   python3 scripts/ppp26.py fetch && python3 scripts/ppp26.py docker-build
FROM --platform=linux/amd64 ubuntu:24.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake libopenmpi-dev openmpi-bin openssh-client time procps \
 && rm -rf /var/lib/apt/lists/*
COPY . /ppp26
WORKDIR /ppp26
# The repository carries a CMake build directory of the authors' machine: build afresh.
RUN rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)" \
 && mpicc -shared -fPIC -O2 -o libhco_mpi_count.so ../hco/mpi_count.c
ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
CMD ["sleep", "infinity"]
