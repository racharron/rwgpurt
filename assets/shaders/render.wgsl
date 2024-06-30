@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var xs = array<f32, 4>(-1., 1., -1., 1.);
    var ys = array<f32, 4>(1., 1., -1., -1.);
    return vec4<f32>(xs[idx], ys[idx], 0., 1.);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return abs(abs(vec4<f32>(position.x / 100., position.y / 100., 0., 1.)));
}
