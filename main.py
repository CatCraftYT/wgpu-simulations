import sys
import wgpu
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run

# Get required arguments for sim
###----------###
arg_names = [
    "zoom",
    "density",
    "gravity_constant",
    "max_impulse",
    "expansion_factor",
    "n_particles"
]

args = sys.argv[1:]
if len(args) != len(arg_names):
    print(f"Incorrect number of arguments provided. Required arguments are: {', '.join(arg_names)}")
    sys.exit(1)
###----------###

# Create parameters
sim_values = np.zeros((), dtype=[
    ("zoom", "float32"),
    ("gravity_constant", "float32"),
    ("max_impulse", "float32"),
    ("expansion_factor", "float32"),
    ("n_particles", "uint32")
])

for n, arg in enumerate(arg_names):
    if arg in sim_values.dtype.names: # type: ignore
        sim_values[arg] = args[n]

# Create canvas (window)
canvas = WgpuCanvas(title="Gravity simulation")

# Get GPU device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
device = adapter.request_device()

# Get canvas context to render to
context = canvas.get_context()
render_texture_format = context.get_preferred_format(device.adapter)
context.configure(device=device, format=render_texture_format)

# Load shaders from files
with open("rendering_shader.wgsl", "r") as shader_file:
    rendering_shader = device.create_shader_module(code=shader_file.read())

with open("compute_shader.wgsl", "r") as shader_file:
    compute_shader = device.create_shader_module(code=shader_file.read())

# Create buffer containing our sim parameters
sim_values_buffer = device.create_buffer_with_data(
    data=sim_values, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

rng = np.random.default_rng()
# Create position data
position_data = np.zeros(sim_values["n_particles"], dtype=[
    ("position", "float32", (2))
])
# Create velocity data
velocity_data = np.zeros(sim_values["n_particles"], dtype=[
    ("velocity", "float32", (2))
])
# Init position data
for i in range(0, sim_values["n_particles"]):
    position_data[i][0] = (rng.random((2), dtype=np.float32) * 2 - 1) * float(args[1])

# Init velocity data
for i in range(0, sim_values["n_particles"]):
    velocity_data[i][0] = np.delete(np.cross(np.append(position_data[i][0], 0), [0, 0, -1]), 2)
    velocity_data[i][0] = velocity_data[i][0] / np.linalg.norm(velocity_data[i][0]) * 0.75

# Create GPU buffer for position data
# size is sizeof(float) * (2 for vec2) * n_particles
position_buffer = device.create_buffer_with_data(
    data=position_data, usage=wgpu.BufferUsage.STORAGE
)
# Create GPU buffer for velocity data
velocity_buffer = device.create_buffer_with_data(
    data=velocity_data, usage=wgpu.BufferUsage.STORAGE
)

# Create memory binding layouts
binding_layouts = [
    # Sim parameters
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.FRAGMENT,
        "buffer": {
            "type": wgpu.BufferBindingType.uniform,
        },
    },
    # Position buffer
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.FRAGMENT,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    # Velocity buffer
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]
bindings = [
    {
        "binding": 0,
        "resource": {"buffer": sim_values_buffer, "offset": 0, "size": sim_values_buffer.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": position_buffer, "offset": 0, "size": position_buffer.size},
    },
    {
        "binding": 2,
        "resource": {"buffer": velocity_buffer, "offset": 0, "size": velocity_buffer.size},
    },
]

# Put everything together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

# Create compute pipeline
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={
        "module": compute_shader,
        "entry_point": "main"
    },
)

# Create compute pipeline for updating positions
compute_pipeline_positions = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={
        "module": compute_shader,
        "entry_point": "update_positions"
    },
)

# Create render pipeline
render_pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={
        "module": rendering_shader,
        "entry_point": "vs_main",
        "buffers": []
    },
    depth_stencil=None,
    multisample=None,
    fragment={
        "module": rendering_shader,
        "entry_point": "fs_main",
        "targets": [
            {
                "format": render_texture_format,
            },
        ],
    },
)

frame_number = 0

# Run every frame
def draw_frame():
    global frame_number
    render_texture = context.get_current_texture()
    command_encoder = device.create_command_encoder()

    # Setup compute pipeline for velocity
    if frame_number % 5 == 0:
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(sim_values["n_particles"], 1, 1)
        compute_pass.end()

    # Setup compute pipeline for position updating
    compute_pass_positions = command_encoder.begin_compute_pass()
    compute_pass_positions.set_pipeline(compute_pipeline_positions)
    compute_pass_positions.set_bind_group(0, bind_group)
    compute_pass_positions.dispatch_workgroups(sim_values["n_particles"], 1, 1)
    compute_pass_positions.end()

    # Setup render pipeline
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": render_texture.create_view(),
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_bind_group(0, bind_group)
    render_pass.draw(3, 1, 0, 0)

    render_pass.end()
    device.queue.submit([command_encoder.finish()])
    frame_number += 1

def render_loop():
    draw_frame()
    canvas.request_draw(render_loop)

# Start rendering/processing loop
canvas.request_draw(render_loop)
run()
