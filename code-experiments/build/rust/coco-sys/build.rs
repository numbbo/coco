use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    generate_source();
    generate_bindings();
    compile_and_link();
}

fn generate_source() {
    let src_path = Path::new("vendor");

    let required_files = ["coco.c", "coco.h", "coco_internal.h"];
    let files_exist = || {
        required_files
            .iter()
            .all(|file| src_path.join(file).exists())
    };

    if !files_exist() {
        let build_script_path = Path::new("../../../../do.py");
        assert!(build_script_path.exists(), "build script does not exist");

        Command::new("python")
            .arg(build_script_path)
            .arg("prepare-build-rust")
            .output()
            .expect("failed to run prepare-build-rust command");
    }

    assert!(files_exist(), "failed to generate code files");
}

fn generate_bindings() {
    println!("cargo:rerun-if-changed=wrapper.h");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
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
}

fn compile_and_link() {
    let src_path = Path::new("vendor");

    cc::Build::new()
        .file(src_path.join("coco.c"))
        .file("wrapper.c")
        .warnings(false)
        .compile("coco");
}
