use crate::input::KeyboardView;
use glam::{Quat, Vec2, Vec3};
use std::time::Duration;
use serde::{Deserialize, Serialize};
use winit::keyboard::KeyCode;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Camera {
    pub position: Vec3,
    pub view_distance: f32,
    pub orientation: Quat,
    pub view_angles: Vec2,
}

impl Camera {
    pub fn fly_around(&mut self, keyboard: KeyboardView, step: Duration) {
        let mut delta = Vec3::ZERO;
        if keyboard.is_held(KeyCode::KeyW) {
            delta += Vec3::Z;
        }
        if keyboard.is_held(KeyCode::KeyS) {
            delta += Vec3::NEG_Z;
        }
        if keyboard.is_held(KeyCode::KeyA) {
            delta += Vec3::NEG_X;
        }
        if keyboard.is_held(KeyCode::KeyD) {
            delta += Vec3::X;
        }
        if keyboard.is_held(KeyCode::Space) {
            delta += Vec3::Y;
        }
        if keyboard.is_held(KeyCode::KeyC) {
            delta += Vec3::NEG_Y;
        }
        delta = self.orientation * delta;
        let mut axis = Vec3::ZERO;
        if keyboard.is_held(KeyCode::ArrowUp) {
            axis += Vec3::NEG_X;
        }
        if keyboard.is_held(KeyCode::ArrowDown) {
            axis += Vec3::X;
        }
        if keyboard.is_held(KeyCode::ArrowLeft) {
            axis += Vec3::NEG_Y;
        }
        if keyboard.is_held(KeyCode::ArrowRight) {
            axis += Vec3::Y;
        }
        if keyboard.is_held(KeyCode::KeyE) {
            axis += Vec3::NEG_Z;
        }
        if keyboard.is_held(KeyCode::KeyQ) {
            axis += Vec3::Z;
        }
        if keyboard.is_held(KeyCode::ShiftLeft) {
            delta *= 2.;
            axis *= 2.;
        }
        if keyboard.is_held(KeyCode::ControlLeft) {
            delta *= 0.5;
            axis *= 0.5;
        }
        let spin = Quat::from_scaled_axis(self.orientation * axis * step.as_secs_f32());
        self.position += delta * step.as_secs_f32();
        self.orientation = spin * self.orientation;
    }
}
