from typing import Any
import wgpu
from os.path import join as path_join
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np


class Simulation():
    bindings: list[dict]
    binding_layouts: list[dict]
    bind_group: wgpu.GPUBindGroup

    pipeline_layout: wgpu.GPUPipelineLayout
    render_pipeline: wgpu.GPURenderPipeline
    compute_pipelines: list[tuple[bool, int, wgpu.GPUComputePipeline]]

    device: wgpu.GPUDevice
    canvas: wgpu.gui.WgpuCanvasBase
    render_context: wgpu.GPUCanvasContext
    render_shader: wgpu.GPUShaderModule
    compute_shader: wgpu.GPUShaderModule

    def __init__(self, name="Super Awesome Simulation", inaccuracy = 1, file_path = "./"):
        self.inaccuracy = inaccuracy
        self.compute_pipelines = []
        self.binding_layouts = []
        self.bindings = []

        # Create canvas (window)
        self.canvas = WgpuCanvas(title=name)

        # Get GPU device
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        self.device = adapter.request_device()

        # Get canvas context to render to
        self.render_context = self.canvas.get_context()
        render_texture_format = self.render_context.get_preferred_format(self.device.adapter)
        self.render_context.configure(device=self.device, format=render_texture_format)

        # Load shaders from files
        with open(path_join(file_path, "rendering_shader.wgsl"), "r") as shader_file:
            self.render_shader = self.device.create_shader_module(code=shader_file.read())

        with open(path_join(file_path, "compute_shader.wgsl"), "r") as shader_file:
            self.compute_shader = self.device.create_shader_module(code=shader_file.read())
    
    def create_buffer(self, usage, buffer_type, visibility, data):
        buffer = self.device.create_buffer_with_data(
            data=data,
            usage=usage
        )

        self.binding_layouts.append({
                "binding": len(self.bindings),
                "visibility": visibility,
                "buffer": {
                    "type": buffer_type,
                },
            }
        )
        self.bindings.append({
            "binding": len(self.bindings),
            "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
            }
        )

        return buffer
    
    def finalize_buffers(self):
        bind_group_layout = self.device.create_bind_group_layout(entries=self.binding_layouts)
        self.bind_group = self.device.create_bind_group(layout=bind_group_layout, entries=self.bindings)
        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    
    def create_compute_pipeline(self, entry_point: str, n_workgroups: int, inaccurate: bool):
        compute_pipeline = self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={
                "module": self.compute_shader,
                "entry_point": entry_point
            },
        )
        self.compute_pipelines.append((inaccurate, n_workgroups, compute_pipeline))

    # Only one render pipeline is needed for these simulations
    def create_render_pipeline(self):
        render_pipeline = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex={
                "module": self.render_shader,
                "entry_point": "vs_main",
                "buffers": []
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": self.render_shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.render_context.get_preferred_format(self.device.adapter),
                    },
                ],
            },
        )
        self.render_pipeline = render_pipeline
    
    def run(self):
        self.frame_number = 0

        def draw_frame():
            render_texture = self.render_context.get_current_texture()
            command_encoder = self.device.create_command_encoder()

            for inaccurate, n_workgroups, compute_pipeline in self.compute_pipelines:
                if inaccurate and self.frame_number % self.inaccuracy != 0:
                    continue
                    
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(compute_pipeline)
                compute_pass.set_bind_group(0, self.bind_group)
                compute_pass.dispatch_workgroups(n_workgroups, 1, 1)
                compute_pass.end()

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

            render_pass.set_pipeline(self.render_pipeline)
            render_pass.set_bind_group(0, self.bind_group)
            render_pass.draw(3, 1, 0, 0)

            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self.frame_number += 1
        
        def render_loop():
            draw_frame()
            self.canvas.request_draw(render_loop)
        
        self.canvas.request_draw(render_loop)
        run()