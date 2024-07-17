use serde::{Deserialize, Serialize};
use crate::camera::Camera;

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub pixel_samples: usize,
    pub max_vertex_count: usize,
    pub max_index_count: usize,
    pub camera: Camera,
}

