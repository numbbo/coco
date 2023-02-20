extern crate bindgen;

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");

    let coco_out = Path::new("vendor");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .size_t_is_usize(true)
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_NAN")
        .generate()
        .expect("Unable to generate bindings");

    bindings.write_to_file(coco_out.join("coco.rs"))
        .expect("Couldn't write bindings to 'vendor/coco.rs'!");

    cc::Build::new()
        .file(coco_out.join("coco.c"))
        .file("wrapper.c")
        .warnings(false)
        .compile("coco");
}
