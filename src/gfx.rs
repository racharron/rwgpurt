use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState, ColorTargetState, ColorWrites, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, Features, FragmentState, FrontFace, include_wgsl, Instance, InstanceDescriptor, Limits, Maintain, MaintainBase, PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, PushConstantRange, Queue, RenderPipeline, RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, Surface, SurfaceConfiguration, TextureAspect, TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState};
use winit::event_loop::ActiveEventLoop;
use std::future::Future;
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct Graphics {
    instance: Instance,
    window: Arc<Window>,
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    overrides: Overrides,
    compute_pipeline: ComputePipeline,
    // render_pipeline: RenderPipeline,
}

struct Overrides {
    rt_wgs_x: u32,
    rt_wgs_y: u32,
}

impl Graphics {
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }
    pub fn draw(&mut self) {

        let frame = self.surface.get_current_texture().unwrap();
        let texture = &frame.texture;
        let x = texture.width().div_ceil(16);
        let y = texture.height().div_ceil(16);
        let view = texture.create_view(&TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.compute_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(
                0,
                &bind_group,
                &[]
            );
            cpass.dispatch_workgroups(x, y, 1);
        }

        let command_buffer = encoder.finish();
        self.device.poll(Maintain::WaitForSubmissionIndex(self.queue.submit([command_buffer])));
        frame.present();
    }
}

pub fn create_graphics(event_loop: &ActiveEventLoop) -> impl Future<Output =Graphics> + 'static {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::VULKAN,
        ..Default::default()
    });

    let window_attrs = Window::default_attributes();

    let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
    let surface = instance
        .create_surface(window.clone())
        .unwrap();

    async move {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        assert!(surface.get_capabilities(&adapter).usages.contains(TextureUsages::STORAGE_BINDING));

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: Features::PUSH_CONSTANTS | Features::BGRA8UNORM_STORAGE,
                    required_limits: Limits {
                        max_push_constant_size: 128,
                        ..Default::default()
                    },
                },
                None,
            )
            .await
            .unwrap();
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        config.usage = TextureUsages::STORAGE_BINDING;
        //  TODO: check if Bgra8UnormSrgb can be used
        config.format = TextureFormat::Bgra8Unorm;

        surface.configure(&device, &config);

        let raytrace_shader = device.create_shader_module(include_wgsl!("../assets/shaders/raytracer.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: config.format,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let overrides = Overrides {
            rt_wgs_x: 16,
            rt_wgs_y: 16,
        };
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &bind_group_layout
                ],
                push_constant_ranges: &[
                    // PushConstantRange { stages: ShaderStages::COMPUTE, range: 0..1 }
                ],
            })),
            module: &raytrace_shader,
            entry_point: "raytrace",
            compilation_options: PipelineCompilationOptions {
                constants: &overrides.get_map(),
                zero_initialize_workgroup_memory: false,
            },
        });
        /*
        let render_shader = device.create_shader_module(include_wgsl!("../assets/shaders/render.wgsl"));
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            })),
            vertex: VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[
                    Some(ColorTargetState {
                        format: config.format,
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
                    })
                ],
            }),
            multiview: None,
        });*/

        Graphics {
            instance,
            window,
            surface,
            config,
            adapter,
            device,
            queue,
            overrides,
            compute_pipeline,
            // render_pipeline,
        }
    }
}

impl Overrides {
    fn get_map(&self) -> HashMap<String, f64> {
        let mut map = HashMap::default();
        map.insert("rt_wgs_x".into(), self.rt_wgs_x as f64);
        map.insert("rt_wgs_y".into(), self.rt_wgs_y as f64);
        map
    }
}

pub mod wgpu_surface {
    use std::sync::Arc;
    use wgpu::Surface;
    use winit::window::Window;

    self_cell::self_cell!(
        pub struct WgpuSurface {
            owner: Arc<Window>,
            #[covariant]
            dependent: Surface,
        }
    );
}
