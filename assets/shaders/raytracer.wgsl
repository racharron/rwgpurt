const EPSILON: f32 = pow(2., -23.);
const TAU: f32 = 6.283185307179586;
//  Plastic number, or the second most irrational number
//  See "The Unreasonable Effectiveness of Quasirandom Sequences"
const PHI_2: f32 = 1.32471795724474602596;
const A_1: f32 = 1. / PHI_2;
const A_2: f32 = 1. / (PHI_2*PHI_2);

struct PushConstants {
    origin_max: vec4<f32>,
    base_uframe: vec4<f32>,
    horizontal_width: vec4<f32>,
    vertical_height: vec4<f32>,
}

struct MeshletData {
    triangle_count: u32,
    vertex_count: u32,
}

var<push_constant> push_constants: PushConstants;

/*
override rt_wgs_x: u32;
override rt_wgs_y: u32;
*/

@group(0) @binding(0)
var<uniform> meshlet_data: MeshletData;

@group(0) @binding(1)
var<uniform> indices: array<vec4<u32>, MAX_INDEX_COUNT>;

@group(0) @binding(10)
var<uniform> positions: array<vec3<f32>, MAX_VERTEX_COUNT>;

//  Normals go here.

@group(0) @binding(12)
var<uniform> diffusive_transparency: array<vec4<f32>, MAX_VERTEX_COUNT>;

@group(0) @binding(13)
var<uniform> specular_metallicity: array<vec4<f32>, MAX_VERTEX_COUNT>;

@group(0) @binding(14)
var<uniform> emissivity_roughness: array<vec4<f32>, MAX_VERTEX_COUNT>;

@group(1) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;

const rt_wgs_x: u32 = 8;
const rt_wgs_y: u32 = 8;

@compute @workgroup_size(rt_wgs_x, rt_wgs_y)
fn raytrace(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = bitcast<u32>(push_constants.horizontal_width.w);
    let height = bitcast<u32>(push_constants.vertical_height.w);
    if all(id.xy < vec2(width, height)) {
        let render_distance = push_constants.origin_max.w;
        let base = push_constants.base_uframe.xyz;
        let frame_count = bitcast<u32>(push_constants.base_uframe.w);
        let x = push_constants.horizontal_width.xyz;
        let y = push_constants.vertical_height.xyz;
        var pixel = vec3<f32>();
        var pixel_error = vec3<f32>();
        for (var s = 0u; s < SAMPLE_COUNT; s += 1u) {
            var origin = push_constants.origin_max.xyz;
//            var jitter = quasirandom((pcg3d(vec3(id.xy, frame_count)).z & 0xFF) * SAMPLE_COUNT + s) - 0.5;
            let urand1 = pcg4d(vec4(id.xy, frame_count, s));
            let jitter = vec2(f32((urand1.w & 0xFFFF) + 1), f32((urand1.w >> 16) + 1)) / f32(0x10001) - 0.5;
            var direction = normalize(
                base
                + (f32(id.x) + jitter.x) * x
                + (f32(id.y) + jitter.y) * y
                - origin
            );
            var sum = vec3(0.);
            var prod = vec3(1.);
            var skip = bitcast<u32>(-1);
            for (var b = 0u; b < 4; b += 1u) {
                var depth = render_distance;
                var diffuse = vec3(0.);
                var emissive = vec3(0.);
                var normal = vec3(0.);
                var contact = vec3(0.);
                var flip = false;
                var missed = true;
                for (var tri = 0u; tri < meshlet_data.triangle_count; tri += 1u) {
                    if tri == skip {
                        continue;
                    }
                    let a = indices[tri].x;
                    let b = indices[select(tri+1, tri+2, flip)].x;
                    let c = indices[select(tri+2, tri+1, flip)].x;
                    let p_a = positions[a];
                    let p_b = positions[b];
                    let p_c = positions[c];
                    let ab = p_b - p_a;
                    let bc = p_c - p_b;
                    let scaled_normal = cross(bc, ab);
                    let intersection = intersection(origin, direction, p_a, p_b, p_c);
                    if intersection.intersects && intersection.position_distance.w < depth && dot(scaled_normal, direction) < 0. {
                        missed = false;
                        normal = normalize(scaled_normal);
                        contact = intersection.position_distance.xyz;
                        depth = intersection.position_distance.w;
                        diffuse = (diffusive_transparency[a].xyz + diffusive_transparency[b].xyz + diffusive_transparency[c].xyz) / 3.;
                        emissive = (emissivity_roughness[a].xyz + emissivity_roughness[b].xyz + emissivity_roughness[c].xyz);
                        skip = tri;
                    }
                    flip = !flip;
                }
                if missed {
                    break;
                } else {
                    sum += prod*emissive;
                    prod *= diffuse;
                    origin = contact;
                    let urand = pcg4d(vec4(id.xy, frame_count, 4 * s + b));
                    let nrand = vec4(box_muller(urand.z), box_muller(urand.w));
                    let rand_norm = normalize(nrand.xyz);
                    direction = normal + select(rand_norm, -rand_norm, dot(rand_norm, normal) < 0.);
                }
            }
            let y = (sum + prod * miss(direction, frame_count)) - pixel_error;
            let t = pixel + y;
            pixel_error = (t - pixel) - y;
            pixel = t;
        }
        output[id.y * width + id.x] = vec4(pixel / f32(SAMPLE_COUNT), 1.);
    }
}

fn miss(direction: vec3<f32>, frame_count: u32) -> vec3<f32> {
    let angle = TAU * f32(frame_count % 256) / 256.;
    return vec3(0.5 + 0.5 * dot(direction.xz, vec2(cos(angle), sin(angle))));
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

fn box_muller(seed: u32) -> vec2<f32> {
    let u1 = f32((seed & 0xFFFFu) + 1u) / f32(0x10001);
    let u2 = f32((seed >> 16u) + 1u) / f32(0x10001);
    let z0 = sqrt(-2. * log(u1)) * cos(TAU * u2);
    let z1 = sqrt(-2. * log(u1)) * sin(TAU * u2);
    return vec2(z0, z1);
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

fn quasirandom(n: u32) -> vec2<f32> {
    return (vec2(A_1 * f32(n), A_2 * f32(n)) + 0.5) % 1.;
}
