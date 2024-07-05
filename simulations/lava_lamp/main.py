from sys import path as module_path
from os import getcwd
from os.path import dirname
import wgpu
import numpy as np

module_path.append(getcwd())
import sim_helper

# Get required arguments for sim
args = sim_helper.get_args([
    "draw_blobs",
    "optimal_distance",
    "dampening",
    "falloff",
    "attraction_force",
    "repulsion_force",
    "heat_loss",
    "heat_gain",
    "heat_zone",
    "n_particles",
])

# Create parameters
sim_values = sim_helper.create_parameters(args, [
    "uint32", # can't be a bool because it's not shareable with the shader
    "float32",
    "float32",
    "float32",
    "float32",
    "float32",
    "float32",
    "float32",
    "float32",
    "uint32",
])

simulation = sim_helper.Simulation("Lava Lamp Simulation", inaccuracy = 1, file_path = dirname(__file__))

# Create buffer containing our sim parameters
simulation.create_buffer(
    usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    buffer_type = wgpu.BufferBindingType.uniform,
    visibility = wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.FRAGMENT,
    data = sim_values
)

rng = np.random.default_rng()
# Function to init position data between -1 and 1 x/y
def position_data_function(data):
    for i in range(0, sim_values["n_particles"]):
        data[i][0] = (rng.random((2), dtype=np.float32) * 2 - 1)

# Create position data
position_data = sim_helper.create_data_array(
    n_elements = sim_values["n_particles"],
    data_function=position_data_function,
    dtype=[
        ("position", "float32", (2))
    ]
)
# Create velocity data
velocity_data = sim_helper.create_data_array(
    n_elements = sim_values["n_particles"],
    data_function=None,
    dtype=[
        ("velocity", "float32", (2))
    ]
)
# Create heat data
heat_data = sim_helper.create_data_array(
    n_elements = sim_values["n_particles"],
    data_function=None,
    dtype=[
        ("heat", "float32")
    ]
)

# Create GPU buffer for position data
simulation.create_buffer(
    usage = wgpu.BufferUsage.STORAGE,
    buffer_type = wgpu.BufferBindingType.storage,
    visibility = wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.FRAGMENT,
    data = position_data
)
# Create GPU buffer for velocity data
simulation.create_buffer(
    usage = wgpu.BufferUsage.STORAGE,
    buffer_type = wgpu.BufferBindingType.storage,
    visibility = wgpu.ShaderStage.COMPUTE,
    data = velocity_data
)
# Create GPU buffer for heat data
simulation.create_buffer(
    usage = wgpu.BufferUsage.STORAGE,
    buffer_type = wgpu.BufferBindingType.storage,
    visibility = wgpu.ShaderStage.COMPUTE,
    data = heat_data
)

simulation.finalize_buffers()

# Create compute pipelines
simulation.create_compute_pipeline(
    entry_point = "main",
    n_workgroups = int(sim_values["n_particles"]),
    inaccurate = True
)
simulation.create_compute_pipeline(
    entry_point = "update_positions",
    n_workgroups = int(sim_values["n_particles"]),
    inaccurate = False
)

# Create render pipeline
simulation.create_render_pipeline()

simulation.run()
