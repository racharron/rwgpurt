use crate::camera::Camera;
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use glam::{Vec3, Vec4};
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    include_wgsl, Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferUsages, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, Features, Instance, InstanceDescriptor, Limits, Maintain,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PushConstantRange, Queue, ShaderStages,
    StorageTextureAccess, Surface, SurfaceConfiguration, TextureFormat, TextureUsages,
    TextureViewDescriptor, TextureViewDimension,
};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
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
    vertices: Buffer,
    indices: Buffer,
    world_bind_group: BindGroup,
    compute_pipeline: ComputePipeline,
}

struct Overrides {
    rt_wgs_x: u32,
    rt_wgs_y: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    position: Vec3,
    reflectivity: f32,
    color: Vec4,
}

impl Graphics {
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }
    pub fn draw(&mut self, camera: &Camera, current_frame: u32) {
        let frame = self.surface.get_current_texture().unwrap();
        let texture = &frame.texture;
        let width = texture.width();
        let height = texture.height();
        let x = width.div_ceil(self.overrides.rt_wgs_x);
        let y = height.div_ceil(self.overrides.rt_wgs_y);
        let view = texture.create_view(&TextureViewDescriptor::default());
        let texture_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.compute_pipeline.get_bind_group_layout(1),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_push_constants(
                0,
                bytes_of(&PushConstants::new(camera, width, height, current_frame)),
            );
            cpass.set_bind_group(0, &self.world_bind_group, &[]);
            cpass.set_bind_group(1, &texture_bind_group, &[]);
            cpass.dispatch_workgroups(x, y, 1);
        }

        let command_buffer = encoder.finish();
        self.device.poll(Maintain::WaitForSubmissionIndex(
            self.queue.submit([command_buffer]),
        ));
        frame.present();
    }
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}

pub fn create_graphics(event_loop: &ActiveEventLoop) -> impl Future<Output = Graphics> + 'static {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::VULKAN,
        ..Default::default()
    });

    let window_attrs = Window::default_attributes();

    let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
    let surface = instance.create_surface(window.clone()).unwrap();

    async move {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        assert!(surface
            .get_capabilities(&adapter)
            .usages
            .contains(TextureUsages::STORAGE_BINDING));

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

        let raytrace_shader =
            device.create_shader_module(include_wgsl!("../assets/shaders/raytracer.wgsl"));

        let overrides = Overrides {
            rt_wgs_x: 8,
            rt_wgs_y: 8,
        };

        let world_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let vertices = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&[
                //  0
                Vertex {
                    position: Vec3::new(-1.5, 1., -0.5),
                    reflectivity: 0.,
                    color: Vec4::new(1., 0., 0., 1.),
                },
                //  1
                Vertex {
                    position: Vec3::new(-1., -1., 0.),
                    reflectivity: 0.,
                    color: Vec4::new(1., 0., 0., 1.),
                },
                //  2
                Vertex {
                    position: Vec3::new(-0.5, 1., 0.5),
                    reflectivity: 0.,
                    color: Vec4::new(1., 1., 0., 1.),
                },
                //  3
                Vertex {
                    position: Vec3::new(0., -1., 1.),
                    reflectivity: 0.,
                    color: Vec4::new(0., 1., 0., 1.),
                },
                //  4
                Vertex {
                    position: Vec3::new(0.5, 1., 1.5),
                    reflectivity: 0.,
                    color: Vec4::new(0., 1., 1., 1.),
                },
                //  5
                Vertex {
                    position: Vec3::new(1., -1., 2.),
                    reflectivity: 0.,
                    color: Vec4::new(0., 0., 1., 1.),
                },
                //  6
                Vertex {
                    position: Vec3::new(1.5, 1., 2.5),
                    reflectivity: 0.,
                    color: Vec4::new(0., 0., 1., 1.),
                },
            ]),
            usage: BufferUsages::STORAGE,
        });
        let indices = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&[0, 1, 2, 3, 4, 5, 6]),
            usage: BufferUsages::STORAGE,
        });

        let world_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &world_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: vertices.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: indices.as_entire_binding(),
                },
            ],
        });
        let frame_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&world_bind_group_layout, &frame_bind_group_layout],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<PushConstants>() as u32,
                }],
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
            vertices,
            indices,
            world_bind_group,
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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstants {
    focus: Vec3,
    view_distance: f32,
    base: Vec3,
    frame: u32,
    horizontal: Vec3,
    _padding2: u32,
    vertical: Vec3,
    _padding3: u32,
}

impl PushConstants {
    pub fn new(camera: &Camera, screen_width: u32, screen_height: u32, frame: u32) -> Self {
        let hw = (0.5 * camera.view_angles.x).tan();
        let hh = (0.5 * camera.view_angles.y).tan();
        let base = Vec3::new(
            -hw + hw / (screen_width as f32),
            hh - hh / (screen_height as f32),
            1.,
        );
        Self {
            focus: camera.position,
            view_distance: camera.view_distance,
            base: camera.position + camera.orientation * base,
            frame,
            horizontal: camera.orientation * (2.0 * Vec3::X * hw / screen_width as f32),
            _padding2: 0,
            vertical: camera.orientation * (2.0 * Vec3::NEG_Y * hh / screen_height as f32),
            _padding3: 0,
        }
    }
}
