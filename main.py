import sys
import wgpu
from PyQt6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas, run

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
app = QtWidgets.QApplication([])
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

# Create empty pipeline memory layout (for now)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

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

# Run every frame
def draw_frame():
    render_texture = context.get_current_texture()
    command_encoder = device.create_command_encoder()

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
    render_pass.draw(3, 1, 0, 0)

    render_pass.end()
    device.queue.submit([command_encoder.finish()])

# Start rendering/processing loop
canvas.request_draw(draw_frame)
run()
