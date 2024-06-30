use crate::gfx::Vertex;
use gfx::PushConstants;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod camera;
mod gfx;
mod input;
mod metrics;

fn main() {
    env_logger::init();
    assert_eq!(std::mem::size_of::<glam::Vec3>(), 4 * 3);
    assert_eq!(std::mem::size_of::<Vertex>(), 32);
    assert_eq!(std::mem::size_of::<PushConstants>(), 64);
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app::App::Empty).unwrap()
}
