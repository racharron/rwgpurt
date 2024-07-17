use std::ffi::OsString;



#[derive(clap::Parser)]
#[command(version, about, author, long_about)]
/// A raytracer that runs on the GPU, powered by wgpu.
pub struct Args {
    #[arg(short, long)]
    /// The file with some settings.  Overrides the default, is overriden in turn by additional flags.
    pub config: Option<OsString>,
    #[command(flatten)]
    pub settings: Settings,
}

#[derive(Clone, Copy, Debug, clap::Args, serde::Deserialize)]
pub struct Settings {
    /// The number of pixel samples.
    #[arg(short, long, default_value = "8")]
    #[serde(default = "samples_default")]
    pub samples: usize,
    /// The maximum number of vertices in a leaf.
    #[arg(short, long, default_value = "8")]
    #[serde(default = "vertices_default")]
    pub vertices: usize,
    /// The maximum number of indices (number of triangles + 2) in a leaf.
    #[arg(short, long, default_value = "8")]
    #[serde(default = "indices_default")]
    pub indices: usize,
}

fn samples_default() -> usize {
    8
}
fn vertices_default() -> usize {
    8
}
fn indices_default() -> usize {
    8
}