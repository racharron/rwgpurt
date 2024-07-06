use crate::camera::Camera;
use bytemuck::{bytes_of, cast_slice, Pod, try_cast_slice, Zeroable};
use glam::{Vec3, Vec4};
use std::borrow::Cow;
use std::collections::HashMap;
use std::future::Future;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferAsyncError, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, Extent3d, Features, FragmentState, FrontFace, Instance, InstanceDescriptor, InstanceFlags, Limits, LoadOp, Maintain, MapMode, MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PresentMode, PrimitiveState, PrimitiveTopology, PushConstantRange, QuerySet, QuerySetDescriptor, QueryType, Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, StoreOp, Surface, SurfaceConfiguration, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

const SAMPLES: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MaterialParameters {
    metallicity: f32,
    roughness: f32,
}

struct WorldBuffers {
    indices: Buffer,
    position: Buffer,
    diffuse: Buffer,
    specular: Buffer,
    emissivity: Buffer,
    material_parameters: Buffer,
}

struct TimestampResources {
    query_set: QuerySet,
    query_buffer: Buffer,
    transfer_buffer: Buffer,
}

pub struct Raytracer {
    pipeline: ComputePipeline,
    shader: ShaderModule,
    world_bind_group: BindGroup,
    output_bind_group: BindGroup,
}

pub struct Renderer {
    window: Arc<Window>,
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    shader: ShaderModule,
    input_bind_group: BindGroup,
}

pub struct Graphics {
    renderer: Renderer,
    device: Device,
    queue: Queue,
    timestamp_resources: TimestampResources,
    overrides: Overrides,
    raytracer: Raytracer,
}

struct Overrides {
    rt_wgs_x: u32,
    rt_wgs_y: u32,
}

const INDICES: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];

const VERTEX_POSITIONS: [Vec3; 7] = [
    //  0
    Vec3::new(-2., 1., 1.),
    //  1
    Vec3::new(-2., -1., 2.),
    //  2
    Vec3::new(-1., 1., 3.),
    //  3
    Vec3::new(0., -1., 3.),
    //  4
    Vec3::new(1., 1., 3.),
    //  5
    Vec3::new(2., -1., 2.),
    //  6
    Vec3::new(2., 1., 1.),
];

const VERTEX_DIFFUSE: [Vec4; 7] = [
    Vec4::new(-1.5, 1., -0.5, 0.25),
    Vec4::new(-1., -1., 0., 0.5),
    Vec4::new(-0.5, 1., 0.5, 0.75),
    Vec4::new(0., -1., 1., 1.),
    Vec4::new(0.5, 1., 1.5, 1.),
    Vec4::new(1., -1., 2., 1.),
    Vec4::new(1.5, 1., 2.5, 1.),
];

const VERTEX_SPECULAR: [Vec4; 7] = [
    Vec4::new(-1.5, 1., -0.5, 1.),
    Vec4::new(-1., -1., 0., 1.),
    Vec4::new(-0.5, 1., 0.5, 1.),
    Vec4::new(0., -1., 1., 1.),
    Vec4::new(0.5, 1., 1.5, 0.75),
    Vec4::new(1., -1., 2., 0.5),
    Vec4::new(1.5, 1., 2.5, 0.25),
];

const VERTEX_EMISSIVITY: [Vec3; 7] = [
    Vec3::splat(0.5),
    Vec3::ZERO,
    Vec3::ZERO,
    Vec3::ONE,
    Vec3::ZERO,
    Vec3::ZERO,
    Vec3::splat(0.5),
];

const VERTEX_MATERIAL_PARAMETERS: [MaterialParameters; 7] = [
    MaterialParameters {
        metallicity: 0.,
        roughness: 0.,
    },
    MaterialParameters {
        metallicity: 1.,
        roughness: 0.,
    },
    MaterialParameters {
        metallicity: 0.,
        roughness: 1.,
    },
    MaterialParameters {
        metallicity: 0.5,
        roughness: 0.5,
    },
    MaterialParameters {
        metallicity: 1.,
        roughness: 0.,
    },
    MaterialParameters {
        metallicity: 0.,
        roughness: 1.,
    },
    MaterialParameters {
        metallicity: 1.,
        roughness: 1.,
    },
];

impl Graphics {
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.renderer.config.width = size.width;
        self.renderer.config.height = size.height;
        self.renderer
            .surface
            .configure(&self.device, &self.renderer.config);
        let interface = new_interface_texture(&self.device, size);
        let view = interface.create_view(&TextureViewDescriptor::default());
        let entries = &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&view),
        }];
        self.renderer.input_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.renderer.pipeline.get_bind_group_layout(0),
            entries,
        });
        self.renderer.pipeline = Self::new_render_pipeline(
            &self.device,
            self.renderer.config.format,
            &self.renderer.shader,
            &self.renderer.pipeline.get_bind_group_layout(0),
        );
        self.raytracer.output_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.raytracer.pipeline.get_bind_group_layout(1),
            entries,
        });
        self.raytracer.pipeline = new_raytracer_pipeline(
            &self.device,
            &self.raytracer.shader,
            &self.overrides,
            &self.raytracer.pipeline.get_bind_group_layout(0),
            &self.raytracer.pipeline.get_bind_group_layout(1),
        );
    }

    pub fn draw(&mut self, camera: &Camera, current_frame: u32, rand: u32) -> Duration {
        let frame = self.renderer.surface.get_current_texture().unwrap();
        assert!(!frame.suboptimal);
        let texture = &frame.texture;
        let width = texture.width();
        let height = texture.height();
        let x = width.div_ceil(self.overrides.rt_wgs_x);
        let y = height.div_ceil(self.overrides.rt_wgs_y);
        let mut compute_encoder = self.device.create_command_encoder(&Default::default());
        compute_encoder.write_timestamp(&self.timestamp_resources.query_set, 0);
        {
            let mut cpass = compute_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.raytracer.pipeline);
            cpass.set_push_constants(
                0,
                bytes_of(&PushConstants::new(
                    camera,
                    width,
                    height,
                    current_frame,
                    rand,
                )),
            );
            cpass.set_bind_group(0, &self.raytracer.world_bind_group, &[]);
            cpass.set_bind_group(1, &self.raytracer.output_bind_group, &[]);

            cpass.dispatch_workgroups(x, y, 1);
        }
        compute_encoder.write_timestamp(&self.timestamp_resources.query_set, 1);
        self.timestamp_resources.do_read(&mut compute_encoder);
        self.queue.submit([compute_encoder.finish()]);
        let slice = self.timestamp_resources.transfer_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slice.map_async(MapMode::Read, move |res| sender.send(res).unwrap());
        let mut render_encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let view = texture.create_view(&TextureViewDescriptor::default());
            let mut render_pass = render_encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Default::default()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.renderer.pipeline);
            render_pass.set_bind_group(0, &self.renderer.input_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        } /*
          self.device.poll(Maintain::WaitForSubmissionIndex(
              self.queue.submit([compute_encoder.finish(), render_encoder.finish()]),
          ));*/
        self.device.poll(Maintain::WaitForSubmissionIndex(
            self.queue.submit([render_encoder.finish()]),
        ));
        frame.present();
        receiver.recv().unwrap().unwrap();
        let time = {
            let timestamps = slice.get_mapped_range();
            let timestamps = try_cast_slice::<u8, u64>(&timestamps[..]).unwrap();
            timestamps[1].wrapping_sub(timestamps[0]) as f32 * self.queue.get_timestamp_period()
        };
        self.timestamp_resources.transfer_buffer.unmap();
        Duration::from_nanos(time.round() as u64)
    }
    pub fn request_redraw(&self) {
        self.renderer.window.request_redraw();
    }

    fn new_render_pipeline(
        device: &Device,
        format: TextureFormat,
        module: &ShaderModule,
        bind_group_layout: &BindGroupLayout,
    ) -> RenderPipeline {
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: VertexState {
                module,
                entry_point: "vertex",
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module,
                entry_point: "fragment",
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        })
    }
}

pub fn create_graphics(event_loop: &ActiveEventLoop) -> impl Future<Output = Graphics> + 'static {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::VULKAN,
        flags: if cfg!(debug_assertions) {
            InstanceFlags::advanced_debugging()
        } else {
            InstanceFlags::empty()
        },
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
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: Features::PUSH_CONSTANTS
                        | Features::BGRA8UNORM_STORAGE
                        | Features::TIMESTAMP_QUERY
                        | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
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
        //  TODO: check if Bgra8UnormSrgb can be used
        config.present_mode = PresentMode::AutoVsync;

        surface.configure(&device, &config);

        let interface = new_interface_texture(&device, size);

        let raytrace_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(
                &std::fs::read_to_string("assets/shaders/raytracer.wgsl")
                    .unwrap()
                    .replace("SAMPLE_COUNT", &SAMPLES.to_string()),
            )),
        });
        let render_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(
                &std::fs::read_to_string("assets/shaders/render.wgsl").unwrap(),
            )),
        });

        let overrides = Overrides {
            rt_wgs_x: 8,
            rt_wgs_y: 8,
        };

        let read_buffer_entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let world_buffers = WorldBuffers::new(&device);

        let world_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    ..read_buffer_entry
                },
                BindGroupLayoutEntry {
                    binding: 10,
                    ..read_buffer_entry
                },
                BindGroupLayoutEntry {
                    binding: 12,
                    ..read_buffer_entry
                },
                BindGroupLayoutEntry {
                    binding: 13,
                    ..read_buffer_entry
                },
                BindGroupLayoutEntry {
                    binding: 14,
                    ..read_buffer_entry
                },
                BindGroupLayoutEntry {
                    binding: 15,
                    ..read_buffer_entry
                },
            ],
        });
        let world_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &world_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: world_buffers.indices.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: world_buffers.position.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: world_buffers.diffuse.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 13,
                    resource: world_buffers.specular.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 14,
                    resource: world_buffers.emissivity.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 15,
                    resource: world_buffers.material_parameters.as_entire_binding(),
                },
            ],
        });

        let output_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: interface.format(),
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });
        let output_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &output_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(
                    &interface.create_view(&TextureViewDescriptor::default()),
                ),
            }],
        });

        let compute_pipeline = new_raytracer_pipeline(
            &device,
            &raytrace_shader,
            &overrides,
            &world_bind_group_layout,
            &output_bind_group_layout,
        );
        let timestamp_resources = TimestampResources::new(&device);

        let input_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let input_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &input_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(
                    &interface.create_view(&TextureViewDescriptor::default()),
                ),
            }],
        });

        let render_pipeline = Graphics::new_render_pipeline(
            &device,
            config.format,
            &render_shader,
            &input_bind_group_layout,
        );

        let renderer = Renderer {
            window,
            surface,
            config,
            pipeline: render_pipeline,
            shader: render_shader,
            input_bind_group,
        };

        let raytracer = Raytracer {
            world_bind_group,
            pipeline: compute_pipeline,
            output_bind_group,
            shader: raytrace_shader,
        };

        Graphics {
            renderer,
            device,
            queue,
            raytracer,
            timestamp_resources,
            overrides,
        }
    }
}

fn new_raytracer_pipeline(
    device: &Device,
    raytrace_shader: &ShaderModule,
    overrides: &Overrides,
    world_bind_group_layout: &BindGroupLayout,
    output_bind_group_layout: &BindGroupLayout,
) -> ComputePipeline {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&world_bind_group_layout, &output_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..size_of::<PushConstants>() as u32,
            }],
        })),
        module: &raytrace_shader,
        entry_point: "raytrace",
        compilation_options: PipelineCompilationOptions {
            constants: &overrides.get_map(),
            zero_initialize_workgroup_memory: false,
        },
    })
}

fn new_interface_texture(device: &Device, size: PhysicalSize<u32>) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[TextureFormat::Rgba32Float],
    })
}

impl WorldBuffers {
    pub fn new(device: &Device) -> Self {
        let indices = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&INDICES),
            usage: BufferUsages::STORAGE,
        });
        let position = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&VERTEX_POSITIONS.map(|p| p.extend(f32::NAN))),
            usage: BufferUsages::STORAGE,
        });
        let diffuse = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&VERTEX_DIFFUSE),
            usage: BufferUsages::STORAGE,
        });
        let specular = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&VERTEX_SPECULAR),
            usage: BufferUsages::STORAGE,
        });
        let emissivity = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&VERTEX_EMISSIVITY.map(|e| e.extend(f32::NAN))),
            usage: BufferUsages::STORAGE,
        });
        let material_parameters = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(&VERTEX_MATERIAL_PARAMETERS),
            usage: BufferUsages::STORAGE,
        });
        WorldBuffers {
            indices,
            position,
            diffuse,
            specular,
            emissivity,
            material_parameters,
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

impl TimestampResources {
    pub fn new(device: &Device) -> Self {
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: None,
            ty: QueryType::Timestamp,
            count: 2,
        });
        let query_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: size_of::<[u64; 2]>() as _,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let transfer_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: size_of::<[u64; 2]>() as _,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            query_buffer,
            transfer_buffer,
        }
    }
    pub fn do_read(&self, encoder: &mut CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.query_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.query_buffer,
            0,
            &self.transfer_buffer,
            0,
            size_of::<[u64; 2]>() as _
        );
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstants {
    focus: Vec3,
    view_distance: f32,
    base: Vec3,
    frame: u32,
    horizontal: Vec3,
    rand: u32,
    vertical: Vec3,
    _padding3: u32,
}

impl PushConstants {
    pub fn new(
        camera: &Camera,
        screen_width: u32,
        screen_height: u32,
        frame: u32,
        rand: u32,
    ) -> Self {
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
            rand,
            vertical: camera.orientation * (2.0 * Vec3::NEG_Y * hh / screen_height as f32),
            _padding3: 0,
        }
    }
}
