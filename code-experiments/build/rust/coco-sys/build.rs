use std::path::Path;

fn main() {
    compile_and_link_coco();
}

fn compile_and_link_coco() {
    let coco_out = Path::new("vendor/coco-prebuilt");

    cc::Build::new()
        .file(coco_out.join("coco.c"))
        .file("wrapper.c")
        .warnings(false)
        .compile("coco");
}
