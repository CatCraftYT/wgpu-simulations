struct paramsStruct {
    zoom: f32,
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

const col1: vec3<f32> = vec3(0, 0, 0);
const col2: vec3<f32> = vec3(0.056, 0.056, 0.224);
const col3: vec3<f32> = vec3(1, 0.239, 0.239);
const col4: vec3<f32> = vec3(1, 1, 1);

// https://stackoverflow.com/q/47285778
fn get_color(count: f32) -> vec4<f32> {
    var color: vec3<f32>;

    color = mix(col1, col2, smoothstep(0.0, 0.33, count));
	color = mix(color, col3, smoothstep(0.33, 0.66, count));
	color = mix(color, col4, smoothstep(0.66, 1.0, count));

	return vec4(color, 1.0);
}

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


@fragment
fn fs_main(@location(0) coord: vec2<f32>) -> @location(0) vec4<f32> {
    let adjustedCoord: vec2<f32> = coord * params.zoom;

    var count: f32 = 0;
    for (var i: u32 = 0; i < params.n_particles; i++) {
        if (distance(position_buffer[i], adjustedCoord) < params.zoom / 400) {
            count = count + 1.0;
        }
    }
    return get_color(count / 6.0);
}
