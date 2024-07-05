struct paramsStruct {
    optimum_distance: f32,
    dampening: f32,
    falloff: f32,
    attraction_force: f32,
    repulsion_force: f32,
    n_particles: u32
}

@group(0)
@binding(0)
var<uniform> params: paramsStruct;

@group(0)
@binding(1)
var<storage, read_write> position_buffer: array<vec2<f32>>;


// vertex shader based off of https://github.com/gfx-rs/wgpu/blob/trunk/examples/src/uniform_values/shader.wgsl
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coord: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32,) -> VertexOutput {
    var vertices = array<vec2<f32>, 3>(
        vec2<f32>(-1., 1.),
        vec2<f32>(3.0, 1.),
        vec2<f32>(-1., -3.0),
    );

    var out: VertexOutput;
    out.coord = vertices[vertex_index];
    out.position = vec4<f32>(out.coord, 0.0, 1.0);

    return out;
}

const color: vec4<f32> = vec4(1, 0.5, 0, 1);

@fragment
fn fs_main(@location(0) coord: vec2<f32>) -> @location(0) vec4<f32> {
    var count: u32 = 0;
    for (var i: u32 = 0; i < params.n_particles; i++) {
        if (distance(position_buffer[i], coord) < 0.005) {
            if (count >= 0) {
                return color;
            }
            count++;
        }
    }
    return vec4<f32>(0,0,0,1);
}
