const EPSILON: f32 = pow(2., -23.);

struct Vertex {
    position_reflectivity: vec4<f32>,
    color: vec4<f32>,
}

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
var<storage, read> vertices: array<Vertex>;

@group(0) @binding(1)
var<storage, read> indices: array<u32>;

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
        var color = vec4<f32>(1.);
        for (var i = 0u; i+2 < arrayLength(&indices); i++) {
            let a = vertices[indices[i]];
            let b = vertices[indices[i+1]];
            let c = vertices[indices[i+2]];
            let intersection = intersection(
                origin, direction,
                 a.position_reflectivity.xyz, b.position_reflectivity.xyz, c.position_reflectivity.xyz
            );
            if intersection.intersects && intersection.position_distance.w < depth {
                depth = intersection.position_distance.w;
                color = (a.color + b.color + c.color) / 3.;
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
        return MaybeIntersection(true, vec4<f32>(origin + direction * t, t));
    } else { // This means that there is a line intersection but not a ray intersection.
        return MaybeIntersection(false, vec4<f32>());
    }
}
