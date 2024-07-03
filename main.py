import sys
import wgpu
from wgpu.gui.auto import WgpuCanvas, run

# Get required arguments for sim
###----------###
argNames = [
    "particle_count",
    "gravity_constant",
    "max_impulse",
    "expansion_factor"
]

args = sys.argv[1:]
if len(args) != len(argNames):
    print(f"Not enough arguments provided. Required arguments are: {', '.join(argNames)}")
    sys.exit(1)
###----------###

# Create canvas (window)
canvas = WgpuCanvas(title="Gravity simulation")

# Get GPU device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
device = adapter.request_device()

# Get canvas context to render to
present_context = canvas.get_context()
render_texture_format = present_context.get_preferred_format(device.adapter)
present_context.configure(device=device, format=render_texture_format)

# Load shaders from files
with open("rendering_shader.wgsl", "r") as shader_file:
    rendering_module = device.create_shader_module(code=shader_file.read())

with open("compute_shader.wgsl", "r") as shader_file:
    compute_module = device.create_shader_module(code=shader_file.read())

# Run every frame
def draw_frame():
    pass

# Start rendering/processing loop
canvas.request_draw(draw_frame)
run()
