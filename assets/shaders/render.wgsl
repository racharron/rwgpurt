var<push_constant> width: u32;

@group(0) @binding(0)
var<storage, read_write> frame: array<vec4<f32>>;

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    switch (i) {
        case 0u: { return vec4(-1., -1., 0., 1.); }
        case 1u: { return vec4(-1., 3., 0., 1.); }
        case 2u: { return vec4(3., -1., 0., 1.); }
        default: { return vec4<f32>(); }
    }
}

@fragment
fn fragment(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(p.xy);
    return frame[pixel.y * width + pixel.x];
}