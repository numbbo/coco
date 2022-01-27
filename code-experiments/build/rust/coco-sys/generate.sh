#!/usr/bin/env bash

root=$(pwd)

mkdir -p "$root/vendor/coco-prebuilt"

cd "$root/vendor/coco"
python3 do.py build-c

cd "$root/vendor/coco-prebuilt"
cp "$root/vendor/coco/code-experiments/build/c/coco.c" "./"
cp "$root/vendor/coco/code-experiments/build/c/coco.h" "./"
cp "$root/vendor/coco/code-experiments/src/coco_internal.h" "./"

cd "$root"
bindgen wrapper.h \
    -o vendor/coco-prebuilt/coco.rs \
    --blocklist-item FP_NORMAL \
    --blocklist-item FP_SUBNORMAL \
    --blocklist-item FP_ZERO \
    --blocklist-item FP_INFINITE \
    --blocklist-item FP_NAN

cd "$root/vendor/coco"
git clean -fd
