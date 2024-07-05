struct paramsStruct {
    zoom: f32, // not needed for this shader but included because it's easier
    gravity_constant: f32,
    max_impulse: f32,
    expansion_factor: f32,
    n_particles: u32
}

@group(0)
@binding(0)
var<uniform> params: paramsStruct;

@group(0)
@binding(1)
var<storage, read_write> position_buffer: array<vec2<f32>>;

@group(0)
@binding(2)
var<storage, read_write> velocity_buffer: array<vec2<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > params.n_particles) { return; }
    let position: vec2<f32> = position_buffer[global_id.x];
    
    for (var i: u32 = 0; i < params.n_particles; i++)
    {
        if (i == global_id.x) { continue; }
        let otherPosition: vec2<f32> = position_buffer[i];

        // https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation
        let directionVec: vec2<f32> = otherPosition - position;
        let distSqr: f32 = dot(directionVec, directionVec);
        let directionUnitVec: vec2<f32> = normalize(directionVec);
        if (distSqr == 0) {
            continue;
        }

        let impulse: f32 = min(params.max_impulse, params.gravity_constant / distSqr);

        velocity_buffer[global_id.x] += impulse * directionUnitVec;
    }
    velocity_buffer[global_id.x] += position * params.expansion_factor;
}

@compute @workgroup_size(1)
fn update_positions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    position_buffer[global_id.x] += velocity_buffer[global_id.x];
}
