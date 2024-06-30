use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowId;
use crate::gfx::*;

pub enum App {
    Empty,
    Running {
        graphics: Graphics,
    }
}

impl App {
    pub fn take(&mut self) -> Self {
        std::mem::replace(self, App::Empty)
    }
}

impl winit::application::ApplicationHandler<Graphics> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let App::Empty = self else { return };
        let graphics = pollster::block_on(create_graphics(event_loop));
        *self = App::Running {
            graphics
        };
    }

    fn user_event(&mut self, _: &ActiveEventLoop, event: Graphics) {
        *self = App::Running {
            graphics: event
        };
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        match (self, event) {
            (_, WindowEvent::CloseRequested)    =>  event_loop.exit(),
            (App::Empty, _) => {},
            (App::Running { graphics }, WindowEvent::Resized(size)) =>   {
                graphics.resize(size);
            },
            (App::Running { graphics }, WindowEvent::RedrawRequested) => {
                graphics.draw();
            },
            _   =>  {}
        }
    }
}
