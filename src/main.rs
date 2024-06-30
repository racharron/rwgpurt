use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod gfx;
mod camera;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::<gfx::Graphics>::with_user_event()
        .build()
        .unwrap();
    event_loop.set_control_flow(ControlFlow::/*Poll*/Wait);
    event_loop
        .run_app(&mut app::App::Empty)
        .unwrap()
}
