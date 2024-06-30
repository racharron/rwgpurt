use hashbrown::{HashMap, HashSet};
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Debug)]
pub struct KeyboardState {
    held: HashSet<KeyCode>,
    changed: HashMap<KeyCode, ElementState>,
}

#[derive(Debug)]
pub struct KeyboardView<'a> {
    held: &'a HashSet<KeyCode>,
    changed: HashMap<KeyCode, ElementState>,
}

impl KeyboardState {
    pub fn new() -> Self {
        Self {
            held: HashSet::new(),
            changed: HashMap::new(),
        }
    }
    pub fn add_event(&mut self, input: KeyEvent) {
        let PhysicalKey::Code(key_code) = input.physical_key else {
            return;
        };
        self.changed.insert(key_code, input.state);
        match input.state {
            ElementState::Pressed => self.held.insert(key_code),
            ElementState::Released => self.held.remove(&key_code),
        };
    }
    pub fn view(&mut self) -> KeyboardView {
        KeyboardView {
            held: &self.held,
            changed: std::mem::replace(&mut self.changed, HashMap::new()),
        }
    }
}

impl KeyboardView<'_> {
    pub fn is_held(&self, key_code: KeyCode) -> bool {
        self.held.contains(&key_code)
    }
    pub fn just_changed(&self, key_code: KeyCode) -> bool {
        self.changed.contains_key(&key_code)
    }
    pub fn just_changed_into(&self, key_code: KeyCode, state: ElementState) -> bool {
        self.changed.get(&key_code).cloned() == Some(state)
    }
    pub fn just_pressed(&self, key_code: KeyCode) -> bool {
        self.just_changed_into(key_code, ElementState::Pressed)
    }
    pub fn just_released(&self, key_code: KeyCode) -> bool {
        self.just_changed_into(key_code, ElementState::Released)
    }
}
