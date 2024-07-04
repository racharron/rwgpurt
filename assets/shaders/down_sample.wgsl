var<push_constant> y_offset: u32;

@group(1) @binding(0)
var<storage, read> sections: array<array<array<array<vec4<f32>, SG_SIZE>, SG_SIZE_X>, SG_SIZE_Y>>;

@group(2) @binding(0)
var bar: texture_storage_2d<rgba32float, write>;


@compute @workgroup_size(SG_SIZE_X, SG_SIZE_Y)
fn down_sample(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var sum = vec4<f32>();
    var c = vec4<f32>();
    for (var i = 0u; i < SG_SIZE; i += 1u) {
        let y = sections[wid.x][lid.x][lid.y][i] - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    textureStore(bar, vec2(0, y_offset) + wid.xy, sum);
}
