use std::borrow::Cow;
use crate::camera::Camera;
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::future::Future;
use std::mem::size_of;
use std::sync::Arc;
use rand::{Rng, SeedableRng};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{include_wgsl, Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferUsages, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, Features, Instance, InstanceDescriptor, Limits, Maintain, PipelineCompilationOptions, PipelineLayoutDescriptor, PushConstantRange, Queue, ShaderStages, StorageTextureAccess, Surface, SurfaceConfiguration, TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension, BufferDescriptor, ShaderModuleDescriptor, ShaderSource};
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Jitter {
    offset: Vec2,
    rand: u64,
}

struct WorldBuffers {
    indices: Buffer,
    position: Buffer,
    diffuse: Buffer,
    specular: Buffer,
    emissivity: Buffer,
    material_parameters: Buffer,
}

struct JitterBuffers<R: Rng> {
    samples: usize,
    current: Buffer,
    next: Buffer,
    transfer: Buffer,
    rng: R,
}

pub struct Graphics {
    instance: Instance,
    window: Arc<Window>,
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    world_buffers: WorldBuffers,
    jitter_buffers: JitterBuffers<rand::rngs::StdRng>,
    overrides: Overrides,
    world_bind_group: BindGroup,
    compute_pipeline: ComputePipeline,
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
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(self.jitter_buffers.get().as_entire_buffer_binding())
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
        assert_eq!(adapter.limits().min_subgroup_size, adapter.limits().max_subgroup_size);
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
            device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(Cow::Borrowed(&std::fs::read_to_string("assets/shaders/raytracer.wgsl")
                    .unwrap()
                    .replace("SAMPLE_COUNT", &SAMPLES.to_string())
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
        let world_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &world_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: indices.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: position.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: diffuse.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 13,
                    resource: specular.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 14,
                    resource: emissivity.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 15,
                    resource: material_parameters.as_entire_binding(),
                },
            ],
        });
        let frame_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: config.format,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&world_bind_group_layout, &frame_bind_group_layout],
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
        });

        let world_buffers = WorldBuffers {
            indices,
            position,
            diffuse,
            specular,
            emissivity,
            material_parameters,
        };

        let jitter_buffers = JitterBuffers::new(&device, rand::rngs::StdRng::seed_from_u64(123), SAMPLES);
        Graphics {
            instance,
            window,
            surface,
            config,
            adapter,
            device,
            queue,
            world_buffers,
            jitter_buffers,
            overrides,
            world_bind_group,
            compute_pipeline,
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

impl<R: Rng> JitterBuffers<R> {
    pub fn new(device: &Device, mut rng: R, samples: usize) -> Self {
        let jitters = Self::new_jitters(&mut rng, samples);
        let current = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(jitters.as_slice()),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let size = (jitters.len() * size_of::<Jitter>()) as _;
        let next = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transfer = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });
        JitterBuffers {
            samples,
            current,
            next,
            transfer,
            rng,
        }
    }

    fn new_jitters(mut rng: &mut R, samples: usize) -> Vec<Jitter> {
        use rand::distributions::*;
        let dist = Uniform::new(-0.5, 0.5);
        std::iter::repeat_with(|| {
            let x = dist.sample(&mut rng);
            let y = dist.sample(&mut rng);
            let rand = rng.gen();
            Jitter {
                offset: Vec2::new(x, y),
                rand,
            }
        }).take(samples).collect::<Vec<_>>()
    }
    pub fn get(&mut self) -> &Buffer {
        &self.current
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
