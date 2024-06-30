use crate::camera::Camera;
use crate::gfx::*;
use crate::input::KeyboardState;
use crate::metrics::Metrics;
use glam::{Quat, Vec2, Vec3};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowId;

pub enum App {
    Empty,
    Running {
        graphics: Graphics,
        keyboard: KeyboardState,
        camera: Camera,
        metrics: Metrics,
        last_move: Instant,
    },
}

impl App {
    pub fn take(&mut self) -> Self {
        std::mem::replace(self, App::Empty)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let App::Empty = self else { return };
        let last_move = Instant::now();
        let graphics = pollster::block_on(create_graphics(event_loop));
        let now = Instant::now();
        *self = App::Running {
            graphics,
            keyboard: KeyboardState::new(),
            camera: Camera {
                position: Vec3::ZERO,
                view_distance: 1000.,
                orientation: Quat::IDENTITY,
                view_angles: Vec2::new(2., 1.5),
            },
            metrics: Metrics::new(),
            last_move,
        };
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: ()) {
        let App::Running { graphics, .. } = self else {
            return;
        };
        graphics.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match (self, event) {
            (_, WindowEvent::CloseRequested) => event_loop.exit(),
            (App::Empty, _) => {}
            (App::Running { graphics, .. }, WindowEvent::Resized(size)) => {
                graphics.resize(size);
            }
            (
                App::Running {
                    graphics,
                    camera,
                    keyboard,
                    metrics,
                    last_move,
                },
                WindowEvent::RedrawRequested,
            ) => {
                let now = Instant::now();
                let last_frame_time = now - *last_move;
                *last_move = now;
                camera.fly_around(keyboard.view(), last_frame_time);
                graphics.draw(camera, metrics.current_frame());
                metrics.advance_frame();
            }
            (App::Running { keyboard, .. }, WindowEvent::KeyboardInput { event, .. }) => {
                keyboard.add_event(event)
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let App::Running { graphics, .. } = self else {
            return;
        };
        graphics.request_redraw()
    }
}
