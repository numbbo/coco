# Coco

[![Crates.io](https://img.shields.io/crates/v/coco-rs)](https://crates.io/crates/coco-rs)
[![docs.rs](https://img.shields.io/docsrs/coco?color=blue)](https://docs.rs/coco-rs/latest/coco_rs/)

Rust bindings for the COCO Numerical Black-Box Optimization Benchmarking Framework.

See https://github.com/numbbo/coco and https://numbbo.github.io/coco/.

## Building coco-sys

This is only necessary when updating COCO. A regular build only requires compilers for Rust and C as well as a call to `cargo build`.

### Requirements

- `git`
- `gcc` (or any other C compiler)
- `bindgen` (`cargo install bindgen`)
- `bash` (for `generate.sh`)

### Build Steps

```sh
$ git submodule update --init --recursive
$ cd coco-sys
$ ./generate.sh
$ cargo build
```