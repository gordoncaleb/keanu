//! The protobuf types in `src/generated` are generated from Keanu's schema in `proto/keanu/graph.proto` and committed, so users of the crate need neither `protoc` nor a build script. This
//! test regenerates them with the pure Rust `protox` and fails if the committed files are stale. Run with `KEANU_REGENERATE=1` to update them.

use std::path::Path;

#[test]
fn generated_code_is_up_to_date() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let proto = root.join("proto");
    let files = ["keanu/graph.proto"];
    let descriptors = protox::compile(files, [&proto]).expect("the schema compiles");
    let descriptor_bytes = prost::Message::encode_to_vec(&descriptors);

    let out = tempfile_dir();
    // BTreeMaps, so that saved metadata is written in a deterministic order
    prost_build::Config::new()
        .btree_map(["."])
        .out_dir(&out)
        .compile_fds(descriptors)
        .expect("prost generates the types");
    pbjson_build::Builder::new()
        .register_descriptors(&descriptor_bytes)
        .expect("descriptors register")
        .btree_map(["."])
        .out_dir(&out)
        .build(&[".keanu"])
        .expect("pbjson generates the serde impls");

    let regenerate = std::env::var_os("KEANU_REGENERATE").is_some();
    let mut stale = Vec::new();
    for name in ["keanu.rs", "keanu.serde.rs"] {
        let generated = std::fs::read_to_string(out.join(name)).unwrap();
        let committed_path = root.join("src/generated").join(name);
        let committed = std::fs::read_to_string(&committed_path).unwrap_or_default();
        if generated != committed {
            if regenerate {
                std::fs::write(&committed_path, generated).unwrap();
            } else {
                stale.push(name);
            }
        }
    }
    assert!(
        stale.is_empty(),
        "{stale:?} are stale: run KEANU_REGENERATE=1 cargo test -p keanu-io --test generated_code"
    );
}

fn tempfile_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("keanu-io-generated-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
