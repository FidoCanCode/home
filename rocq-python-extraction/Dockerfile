FROM ocaml/opam:debian-12-ocaml-5.3

# Install system build tools and Python 3 (needed for .py syntax validation)
RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        make \
        build-essential \
        pkg-config \
        libgmp-dev \
        linux-libc-dev \
        python3 \
    && sudo rm -rf /var/lib/apt/lists/*

# Add the rocq-prover opam repository, which carries rocq-stdlib.9.1.0 (not
# yet in the default opam repository).  Then install Rocq 9.1, rocq-stdlib
# 9.1.0, and dune, and clean up the opam cache.  rocq-stdlib.9.1.0 provides
# PrimInt63, PrimFloat, and PrimString (needed for MLuint/MLfloat/MLstring
# test coverage).
RUN opam repo add rocq-released https://rocq-prover.org/opam/released \
    && opam install -y rocq-core.9.1.0 rocq-stdlib.9.1.0 dune \
    && opam clean --all

WORKDIR /workspace
USER opam
