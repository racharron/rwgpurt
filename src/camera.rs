use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraPushConstants {
    focus: Vec3,
    base: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}
