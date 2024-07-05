struct paramsStruct {
    draw_blobs: u32,
    optimum_distance: f32,
    dampening: f32,
    falloff: f32,
    attraction_force: f32,
    repulsion_force: f32,
    heat_loss: f32,
    heat_gain: f32,
    heat_zone: f32,
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

@group(0)
@binding(3)
var<storage, read_write> heat_buffer: array<f32>;

fn force_function(distance: f32) -> f32 {
    let adjustedDist = distance - params.optimum_distance;
    if (adjustedDist < params.optimum_distance) {
        return params.repulsion_force * -exp(-params.falloff * adjustedDist);
    }
    else {
        return params.attraction_force * exp(-params.falloff * adjustedDist);
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > params.n_particles) { return; }
    let position: vec2<f32> = position_buffer[global_id.x];
    
    var newVelocity: vec2<f32> = velocity_buffer[global_id.x];
    for (var i: u32 = 0; i < params.n_particles; i++)
    {
        if (i == global_id.x) { continue; }
        let otherPosition: vec2<f32> = position_buffer[i];

        let directionVec: vec2<f32> = otherPosition - position;
        let directionUnitVec: vec2<f32> = normalize(directionVec);
        let dist: f32 = distance(position, otherPosition);
        let impulse: f32 = force_function(dist);

        newVelocity += impulse * directionUnitVec;
        // May (will) cause race conditions... hopefully not too bad?
        // Makes weird streaks bounce around everywhere. Leaving it here because it's cool
        //velocity_buffer[global_id.x] += exp(1100 * -dist) * velocity_buffer[i] / 2;
    }

    newVelocity /= params.dampening;

    newVelocity.y += heat_buffer[global_id.x];

    // Reflect velocity when hitting walls
    if ((position.x >= 1 && newVelocity.x > 0) || (position.x <= -1 && newVelocity.x < 0)) {
        newVelocity.x = -newVelocity.x;
    }
    else if ((position.y >= 1 && newVelocity.y > 0) || (position.y <= -1 && newVelocity.y < 0)) {
        newVelocity.y = -newVelocity.y;
    }
    velocity_buffer[global_id.x] = newVelocity;
}

@compute @workgroup_size(1)
fn update_positions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    position_buffer[global_id.x] += velocity_buffer[global_id.x];

    // Also update heat
    if (position_buffer[global_id.x].y <= params.heat_zone) {
        heat_buffer[global_id.x] += params.heat_gain;
    }

    heat_buffer[global_id.x] -= params.heat_loss;
}
