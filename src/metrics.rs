use std::time::Instant;

#[derive(Clone)]
pub struct Metrics {
    start: Instant,
    last_debug_time: Instant,
    total_frame_count: u32,
    last_debug_frame: u32,
}

impl Metrics {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_debug_time: now,
            total_frame_count: 0,
            last_debug_frame: 0,
        }
    }
    pub fn current_frame(&self) -> u32 {
        self.total_frame_count
    }
    pub fn advance_frame(&mut self) {
        let now = Instant::now();
        let time = now - self.last_debug_time;
        if time.as_secs() != 0 {
            let total = now - self.start;
            let fps = (self.total_frame_count - self.last_debug_frame) as f32 / time.as_secs_f32();
            let avg = self.total_frame_count as f32 / total.as_secs_f32();
            println!("[{:?}] fps: {fps:.1}, avg: {avg:.1}", total);
            self.last_debug_frame = self.total_frame_count;
            self.last_debug_time = now;
        }
        self.total_frame_count += 1;
    }
}
