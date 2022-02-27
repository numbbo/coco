# Coco

[![Crates.io](https://img.shields.io/crates/v/coco-rs)](https://crates.io/crates/coco-rs)
[![docs.rs](https://img.shields.io/docsrs/coco?color=blue)](https://docs.rs/coco-rs/latest/coco_rs/)

Rust bindings for the COCO Numerical Black-Box Optimization Benchmarking Framework.

See https://github.com/numbbo/coco and https://numbbo.github.io/coco/.

## Building and packaging

### Requirements

- `git`
- `gcc` (or any other C compiler)
- `bindgen` (`cargo install bindgen`)
    - and `libclang` (install `libclang-dev` on Ubuntu)
- `bash` (for `generate.sh`)

### coco-sys

```sh
$ python do.py build-rust
$ cd code-experiments/build/rust/coco-sys
$ cargo build
$ # and when publishing
$ cargo package --allow-dirty
$ cargo publish --allow-dirty
```

### coco-rs

```sh
$ cd code-experiments/build/rust
$ cargo build
$ # and when publishing
$ cargo package
$ cargo publish
```