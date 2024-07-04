const EPSILON: f32 = pow(2., -23.);

struct PushConstants {
    origin_max: vec4<f32>,
    base_uframe: vec4<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
}

struct Jitter {
    offset: vec2<f32>,
    rand: vec2<u32>,
}

struct MaterialParameters {
    metallicity: f32,
    roughness: f32,
}

var<push_constant> push_constants: PushConstants;

/*
override rt_wgs_x: u32;
override rt_wgs_y: u32;
*/

@group(0) @binding(0)
var<storage, read> indices: array<u32>;

@group(0) @binding(10)
var<storage, read> positions: array<vec3<f32>>;

//  Normals go here.

@group(0) @binding(12)
var<storage, read> diffuse: array<vec4<f32>>;

@group(0) @binding(13)
var<storage, read> specular: array<vec4<f32>>;

@group(0) @binding(14)
var<storage, read> emmisivity: array<vec3<f32>>;

@group(0) @binding(15)
var<storage, read> parameters: array<MaterialParameters>;

@group(1) @binding(0)
var frame: texture_storage_2d<bgra8unorm, write>;

@group(1) @binding(1)
var<uniform> jitters: array<Jitter, SAMPLE_COUNT>;

const rt_wgs_x: u32 = 8;
const rt_wgs_y: u32 = 8;

@compute @workgroup_size(rt_wgs_x, rt_wgs_y)
fn raytrace(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(frame);
    if all(id.xy < dimensions) {
        let render_distance = push_constants.origin_max.w;
        let index_count = arrayLength(&indices);
        let origin = push_constants.origin_max.xyz;
        let base = push_constants.base_uframe.xyz;
        let frame_count = bitcast<u32>(push_constants.base_uframe);
        let x = push_constants.horizontal;
        let y = push_constants.vertical;
        var pixel = vec3<f32>();
        var pixel_error = vec3<f32>();
        for (var s = 0u; s < SAMPLE_COUNT; s += 1u) {
            var depth = render_distance;
            var jitter = jitters[s].offset;
            var direction = normalize(
                base
                + (f32(id.x) + jitter.x) * x
                + (f32(id.y) + jitter.y) * y
                - origin
            );
            var color = miss(direction);
            for (var tri = 0u; tri + 2 < index_count; tri += 1u) {
                let a = indices[tri];
                let b = indices[tri+1];
                let c = indices[tri+2];
                let intersection = intersection(
                    origin, direction,
                    positions[a], positions[b], positions[c],
                );
                if intersection.intersects && intersection.position_distance.w < depth {
                    depth = intersection.position_distance.w;
                    color = (diffuse[a].xyz + diffuse[b].xyz + diffuse[c].xyz) / 3.;
                }
            }
            let y = color - pixel_error;
            let t = pixel + y;
            pixel_error = (t - pixel) - y;
            pixel = t;
        }
        textureStore(frame, id.xy, vec4(pixel / f32(SAMPLE_COUNT), 1.));
    }
}

fn miss(direction: vec3<f32>) -> vec3<f32> {
    return vec3(1.);
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

//  Generates a random number in the range (-0.5, 0.5). (Half Centered around Zero).
fn gen_f32_hcz(rand: u32) -> f32 {
    let exponent = 126u << 23;
    let sign = 0x80000000 & rand;
    let base = bitcast<f32>(exponent | (0x7FFFFF & rand));
    return bitcast<f32>(sign | bitcast<u32>(base - 0.5));
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
