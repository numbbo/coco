extern crate bindgen;
use std::env;
use std::path::Path;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");

    let src_path = Path::new("vendor");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

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

    bindings
        .write_to_file(out_path.join("coco_sys.rs"))
        .expect("Couldn't write bindings to 'OUT_DIR/coco_sys.rs'!");

    cc::Build::new()
        .file(src_path.join("coco.c"))
        .file("wrapper.c")
        .warnings(false)
        .compile("coco");
}
