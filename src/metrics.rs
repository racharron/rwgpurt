use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Metrics {
    start: Instant,
    last_debug_time: Instant,
    total_frame_count: u32,
    last_debug_frame: u32,
    total_raytracing_time: Duration,
    frame_raytracing_time: Duration,
}

impl Metrics {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_debug_time: now,
            total_frame_count: 0,
            last_debug_frame: 0,
            total_raytracing_time: Duration::ZERO,
            frame_raytracing_time: Duration::ZERO,
        }
    }
    pub fn current_frame(&self) -> u32 {
        self.total_frame_count
    }
    pub fn advance_frame(&mut self, raytracing_time: Duration) {
        let now = Instant::now();
        let time = now - self.last_debug_time;
        self.frame_raytracing_time += raytracing_time;
        if time.as_secs() != 0 {
            let total = now - self.start;
            let frame_count = self.total_frame_count - self.last_debug_frame;
            let fps = frame_count as f32 / time.as_secs_f32();
            let frame_avg = self.total_frame_count as f32 / total.as_secs_f32();
            let frtt = 1e3 * (self.frame_raytracing_time / frame_count).as_secs_f64();
            self.total_raytracing_time += self.frame_raytracing_time;
            self.frame_raytracing_time = Duration::ZERO;
            let artt = 1e3 * (self.total_raytracing_time / self.total_frame_count).as_secs_f64();
            println!("{:?}", total);
            println!("\tfps:\t{fps:.1},\t\tavg:\t{frame_avg:.1}");
            println!("\trt:\t{frtt:.1}ms,\t\tavg:\t{artt:.1}ms");
            println!();
            self.last_debug_frame = self.total_frame_count;
            self.last_debug_time = now;
        }
        self.total_frame_count += 1;
    }
}
