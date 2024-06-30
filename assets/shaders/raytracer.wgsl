/*

struct CameraPushConstants {
    focus: vec3<f32>,
    base: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
}

var<push_constant> camera: CameraPushConstants;
*/

/*
override rt_wgs_x: u32;
override rt_wgs_y: u32;
*/

@group(0) @binding(0)
var frame: texture_storage_2d<bgra8unorm, write>;

const rt_wgs_x: u32 = 16;
const rt_wgs_y: u32 = 16;

@compute @workgroup_size(rt_wgs_x, rt_wgs_y)
//@compute @workgroup_size(16, 16)
fn raytrace(@builtin(global_invocation_id) id: vec3<u32>) {
    let blue = f32(id.x % rt_wgs_x) / f32(rt_wgs_x - 1);
    let green = f32(id.y % rt_wgs_y) / f32(rt_wgs_y - 1);
    let red = select(1., 0.5, ((id.x / 512) & 1) == 0);
    textureStore(frame, id.xy, vec4<f32>(blue, green, red, 1.));
}
