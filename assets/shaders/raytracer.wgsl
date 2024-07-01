const EPSILON: f32 = pow(2., -23.);

struct PushConstants {
    origin_max: vec4<f32>,
    base_uframe: vec4<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
}

var<push_constant> camera: PushConstants;

/*
override rt_wgs_x: u32;
override rt_wgs_y: u32;
*/

@group(0) @binding(0)
var<storage, read> indices: array<u32>;

@group(0) @binding(1)
var<storage, read> vertex_positions: array<vec3<f32>>;

@group(0) @binding(2)
var<storage, read> vertex_diffuse: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read> vertex_specular: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> vertex_emmisivity: array<vec3<f32>>;

@group(1) @binding(0)
var frame: texture_storage_2d<bgra8unorm, write>;

const rt_wgs_x: u32 = 8;
const rt_wgs_y: u32 = 8;

@compute @workgroup_size(rt_wgs_x, rt_wgs_y)
//@compute @workgroup_size(16, 16)
fn raytrace(@builtin(global_invocation_id) id: vec3<u32>) {
    if all(id.xy < textureDimensions(frame)) {
        let origin = camera.origin_max.xyz;
        let base = camera.base_uframe.xyz;
        let frame_count = bitcast<u32>(camera.base_uframe);
        let direction = normalize(base + f32(id.x) * camera.horizontal + f32(id.y) * camera.vertical - origin);
        var depth = camera.origin_max.w;
        var color = vec4(1.);
        for (var i = 0u; i+2 < arrayLength(&indices); i++) {
            let a = indices[i];
            let b = indices[i+1];
            let c = indices[i+2];
            let intersection = intersection(
                origin, direction,
                vertex_positions[a], vertex_positions[b], vertex_positions[c],
            );
            if intersection.intersects && intersection.position_distance.w < depth {
                depth = intersection.position_distance.w;
                color = (vertex_diffuse[a] + vertex_diffuse[b] + vertex_diffuse[c]) / 3.;
            }
        }
        textureStore(frame, id.xy, color);
    }
}

struct MaybeIntersection {
    intersects: bool,
    position_distance: vec4<f32>,
}

fn intersection(origin: vec3<f32>, direction: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> MaybeIntersection {
    let e1 = b - a;
    let e2 = c - a;

    let ray_cross_e2 = cross(direction, e2);
    let det = dot(e1, ray_cross_e2);

    if det > -EPSILON && det < EPSILON {
        // This ray is parallel to this triangle.
        return MaybeIntersection(false, vec4<f32>());
    }

    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = inv_det * dot(s, ray_cross_e2);
    if u < 0.0 || u > 1.0 {
        return MaybeIntersection(false, vec4<f32>());
    }

    let s_cross_e1 = cross(s, e1);
    let v = inv_det * dot(direction, s_cross_e1);
    if v < 0.0 || u + v > 1.0 {
        return MaybeIntersection(false, vec4<f32>());
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = inv_det * dot(e2, s_cross_e1);

    if t > EPSILON { // ray intersection
        return MaybeIntersection(true, vec4(origin + direction * t, t));
    } else { // This means that there is a line intersection but not a ray intersection.
        return MaybeIntersection(false, vec4<f32>());
    }
}

/// Jarzynski and Olano prng
fn jo_prng(input: u32) -> u32 {
    let state = input * 747796405 + 2891336453;
    let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22u) ^ word;
}

fn pcg3d(input: vec3<u32>) -> vec3<u32> {
    var v = input;
    v = v * 1664525 + 1013904223;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    v ^= v >> vec3(16);
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    return v;
}

fn pcg4d(input: vec4<u32>) -> vec4<u32> {
    var v = input;
    v = v * 1664525 + 1013904223;
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    v ^= v >> vec4(16);
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    return v;
}
