#![allow(dead_code)]

use clap::Parser;
use enum_iterator::{next_cycle, previous_cycle};
use rayon::prelude::*;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::rect::Rect;
use sdl2::video::WindowBuilder;
use std::io::{Error, ErrorKind};
use std::thread::{self, JoinHandle};
use std::{io::Write, path::PathBuf};

mod complex;
mod vec3;
mod vec4;

use complex::*;
use vec3::*;
use vec4::*;

type RGB8 = (u8, u8, u8);

fn screen_coords_to_complex_coords(
    px: f32,
    py: f32,
    args: &ProgramArgs,
    dxmin: f32,
    dxmax: f32,
    dymin: f32,
    dymax: f32,
) -> Complex {
    let x = (px / args.screen_width as f32) * (dxmax - dxmin) + dxmin;
    let y = (py / args.screen_height as f32) * (dymax - dymin) + dymin;

    Complex { re: x, im: y }
}

fn write_image(file: PathBuf, width: i32, height: i32, pixels: &[(u8, u8, u8)]) {
    let mut writer =
        std::io::BufWriter::new(std::fs::File::create(file).expect("Failed to open output file"));
    writeln!(&mut writer, "P3\n{} {}\n255", width, height).unwrap();
    for (r, g, b) in pixels {
        writeln!(&mut writer, "{} {} {}", r, g, b).unwrap();
    }
}

fn hsv_to_rgb(hsv: (f32, f32, f32)) -> (u8, u8, u8) {
    let (h, s, v) = hsv;
    let h = h.clamp(0f32, 360f32);

    if h.abs() < 1.0E-5f32 {
        let v = (v * 255f32) as u8;
        return (v, v, v);
    }

    //
    // Make hue to be in the [0, 6) range.
    let hue = if h == 360f32 { 0f32 } else { h / 60f32 };

    //
    // Get integer and fractional part of hue.
    let int_part = hue.floor() as i32;
    let frac_part = hue - int_part as f32;

    let p = v * (1f32 - s);

    let q = v * (1f32 - (s * frac_part));

    let t = v * (1f32 - (s * (1f32 - frac_part)));

    let color_table: [f32; 6 * 3] = [
        //
        // Case 0
        v, t, p, //
        // Case 1
        q, v, p, //
        // Case 2
        p, v, t, //
        // Case 3
        p, q, v, //
        // Case 4
        t, p, v, //
        // Case 5
        v, p, q,
    ];

    let r = (color_table[(int_part * 3 + 0) as usize] * 255f32) as u8;
    let g = (color_table[(int_part * 3 + 1) as usize] * 255f32) as u8;
    let b = (color_table[(int_part * 3 + 2) as usize] * 255f32) as u8;

    (r, g, b)
}

fn color_simple(n: f32, max_iterations: i32, _z: Complex) -> RGB8 {
    let px = 255u8 - ((n / max_iterations as f32) * 255f32) as u8;
    (px, px, px)
}

fn color_smooth(n: f32, max_iterations: i32, _z: Complex) -> RGB8 {
    let t = n as f32 / max_iterations as f32;
    let u = 1f32 - t;

    (
        ((9f32 * u * t * t * t) * 255f32) as u8,
        ((15f32 * u * u * t * t) * 255f32) as u8,
        ((8.5f32 * u * u * u * t) * 255f32) as u8,
    )
}

fn color_log(n: f32, _max_iterations: i32, z: Complex) -> RGB8 {
    let n = n + 1f32 - (z.abs().ln() / 2f32.ln()).ln();

    (
        ((-(n * 0.025_f32).cos() + 1_f32) * 127f32) as u8,
        ((-(n * 0.080_f32).cos() + 1_f32) * 127f32) as u8,
        ((-(n * 0.120_f32).cos() + 1_f32) * 127f32) as u8,
    )
}

fn color_hsv(iterations: f32, max_iterations: i32, z: Complex) -> (u8, u8, u8) {
    let iterations = if iterations == max_iterations as f32 {
        max_iterations as f32
    } else {
        iterations as f32 + 1f32 - z.abs().log2().log10()
    };

    let hsv = (
        (iterations / max_iterations as f32) * 360f32,
        1f32,
        if iterations < max_iterations as f32 {
            1f32
        } else {
            0f32
        },
    );

    hsv_to_rgb(hsv)
}

fn color_orbit_trap(e: &WorkResult, _args: &ProgramArgs) -> (u8, u8, u8) {
    let f = 1f32 + e.trap.log2() / 16f32;

    f32_color_to_u8_color((f, f * f, f * f * f))
}

fn f32_color_to_u8_color((r, g, b): (f32, f32, f32)) -> RGB8 {
    ((r * 255f32) as u8, (g * 255f32) as u8, (b * 255f32) as u8)
}

fn color_orbit_trap2(e: &WorkResult, args: &ProgramArgs) -> (u8, u8, u8) {
    let col = Vec3::same(e.dmin.w);
    let col = mix(
        col,
        Vec3::new(1.0f32, 0.80f32, 0.60f32),
        1f32.min((e.dmin.x * 0.25f32).powf(0.20f32)),
    );

    let col = mix(
        col,
        Vec3::new(0.72f32, 0.70f32, 0.60f32),
        1f32.min((e.dmin.y * 0.50f32).powf(0.50f32)),
    );

    let col = mix(
        col,
        Vec3::same(1f32),
        1f32 - 1f32.min((e.dmin.z * 1.00).powf(0.15f32)),
    );

    let col = 1.25f32 * col * col;
    let col = col * col * (Vec3::same(3f32) - 2f32 * col);

    let (px, py) = (
        e.pixel.0 as f32 / args.screen_width as f32,
        e.pixel.1 as f32 / args.screen_height as f32,
    );

    let u = 0.5f32 * (16f32 * px * (1f32 - px) * py * (1f32 - py)).powf(0.15f32);
    let col = col * (Vec3::same(0.5f32) + Vec3::same(u));

    f32_color_to_u8_color((col.x, col.y, col.z))
}

fn linear_interpolate(a: f32, b: f32, t: f32) -> f32 {
    a * (1f32 - t) + b * t
}

#[derive(Copy, Clone, Debug)]
struct WorkPackage {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, clap::ValueEnum, enum_iterator::Sequence)]
enum Coloring {
    BlackWhite,
    Hsv,
    Smooth,
    Log,
    Histogram,
    OrbitTrap,
    OrbitTrap2,
}

#[derive(clap::Parser, Copy, Clone, Debug)]
#[command(author, version, about, long_about = None)]
struct ProgramArgs {
    #[arg(long, default_value_t = 1024)]
    screen_width: i32,
    #[arg(long, default_value_t = 1024)]
    screen_height: i32,
    #[arg(short = 't', long, default_value_t = 8, value_parser = clap::value_parser!(i32).range(4..512))]
    tile_size: i32,
    #[arg(short = 'i', long, default_value_t = 16, value_parser = clap::value_parser!(i32).range(4..2048))]
    iterations: i32,
    #[arg(short = 'w', long, default_value_t = 8, value_parser = clap::value_parser!(i32).range(4..512))]
    workers: i32,
    #[arg(short = 'c', value_enum, default_value_t = Coloring::BlackWhite)]
    coloring: Coloring,
    #[arg(short = 'z', default_value_t = 1f32)]
    zoom: f32,
    #[arg(default_value_t = 0f32)]
    ox: f32,
    #[arg(default_value_t = 0f32)]
    oy: f32,
}

impl std::default::Default for ProgramArgs {
    fn default() -> Self {
        Self {
            screen_width: 1024,
            screen_height: 1024,
            tile_size: 8,
            iterations: 16,
            workers: 8,
            coloring: Coloring::BlackWhite,
            zoom: 1f32,
            ox: 0f32,
            oy: 0f32,
        }
    }
}

impl ProgramArgs {
    pub const MIN_TERATIONS: i32 = 4;
    pub const MAX_ITERATIONS: i32 = 2048;
    pub const ZOOM_IN_FACTOR: f32 = 0.85f32;
    pub const ZOOM_OUT_FACTOR: f32 = 2f32;
    const FRACTAL_XMIN: f32 = -2f32;
    const FRACTAL_XMAX: f32 = 2f32;
    const FRACTAL_YMIN: f32 = -1f32;
    const FRACTAL_YMAX: f32 = 1f32;

    const FRACTAL_HALF_WIDTH: f32 =
        (ProgramArgs::FRACTAL_XMAX - ProgramArgs::FRACTAL_XMIN) * 0.5f32;
    const FRACTAL_HALF_HEIGHT: f32 =
        (ProgramArgs::FRACTAL_YMAX - ProgramArgs::FRACTAL_YMIN) * 0.5f32;
}

#[derive(Copy, Clone, Debug)]
struct WorkResult {
    pixel: (i32, i32),
    z: Complex,
    trap: f32,
    dmin: Vec4,
    n: f32,
}

struct HistogramColoringState {
    histogram: Vec<i32>,
    hues: Vec<f32>,
    total: i32,
}

impl HistogramColoringState {
    fn new(escapes: &[WorkResult], args: &ProgramArgs) -> HistogramColoringState {
        let mut histogram = vec![0i32; args.iterations as usize];

        for y in 0..args.screen_height {
            for x in 0..args.screen_width {
                let e = escapes[(y * args.screen_width + x) as usize];

                if e.n < args.iterations as f32 {
                    histogram[e.n.floor() as usize] += 1;
                }
            }
        }

        let total: i32 = histogram.iter().sum();
        let mut hues = Vec::<f32>::new();
        let mut h: f32 = 0f32;

        for i in 0..args.iterations {
            h += histogram[i as usize] as f32 / total as f32;
            hues.push(h);
        }
        hues.push(h);

        HistogramColoringState {
            histogram,
            hues,
            total,
        }
    }

    fn colorize(&self, iterations: f32, args: &ProgramArgs, z: Complex) -> RGB8 {
        let iterations = if iterations >= args.iterations as f32 {
            args.iterations as f32
        } else {
            iterations as f32 + 1f32 - z.abs().log2().log10()
        };

        let hsv = (
            360f32
                - linear_interpolate(
                    self.hues[iterations.floor() as usize],
                    self.hues[iterations.ceil() as usize],
                    iterations.fract(),
                ) * 360f32,
            1f32,
            if iterations < args.iterations as f32 {
                1f32
            } else {
                0f32
            },
        );

        hsv_to_rgb(hsv)
    }
}

fn pick_main_display(vid_sys: &sdl2::VideoSubsystem) -> std::io::Result<Rect> {
    vid_sys
        .num_video_displays()
        .ok()
        .map(|display_count| {
            (0..display_count).find_map(|dpy_idx| {
                vid_sys
                    .display_bounds(dpy_idx)
                    .ok()
                    .filter(|bounds| bounds.x == 0 && bounds.y == 0)
            })
        })
        .flatten()
        .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to get main display"))
}

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
enum ModKeys {
    LeftControl,
    RightControl,
}

fn main() -> std::io::Result<()> {
    let sdl_ctx = sdl2::init().expect(&format!("Failed to init SDL2: {}", sdl2::get_error()));
    let video_subsys = sdl_ctx.video().expect(&format!(
        "Failed to get video subsystem: {}",
        sdl2::get_error()
    ));

    let screen_bounds = pick_main_display(&video_subsys)?;
    println!("Main display bounds {:?}", screen_bounds);

    let mut args = {
        let args = ProgramArgs::parse();
        ProgramArgs {
            screen_width: screen_bounds.width() as i32,
            screen_height: screen_bounds.height() as i32,
            ..args
        }
    };

    let mut escapes = vec![
        WorkResult {
            pixel: (0, 0),
            z: Complex::new(0f32, 0f32),
            trap: 1e20f32,
            dmin: Vec4::same(1000f32),
            n: 0f32
        };
        (args.screen_width * args.screen_height) as usize
    ];

    let packages_x = args.screen_width / args.tile_size;
    let packages_y = args.screen_height / args.tile_size;

    let mut work_packages = Vec::<WorkPackage>::new();
    for y in 0..args.tile_size {
        for x in 0..args.tile_size {
            work_packages.push(WorkPackage {
                x0: (x * packages_x).min(args.screen_width),
                y0: (y * packages_y).min(args.screen_height),
                x1: ((x + 1) * packages_x).min(args.screen_width),
                y1: ((y + 1) * packages_y).min(args.screen_width),
            })
        }
    }

    let main_window = WindowBuilder::new(
        &video_subsys,
        "Mandelbrot Explorer",
        screen_bounds.width(),
        screen_bounds.height(),
    )
    .borderless()
    .fullscreen_desktop()
    .maximized()
    .build()
    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;

    sdl_ctx.mouse().warp_mouse_in_window(
        &main_window,
        (screen_bounds.width() / 2) as i32,
        (screen_bounds.height() / 2) as i32,
    );

    let mut canvas = main_window
        .into_canvas()
        .build()
        .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;

    let mut event_pump = sdl_ctx
        .event_pump()
        .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;

    let texture_creator = canvas.texture_creator();

    let mut fractal_texture = texture_creator
        .create_texture(
            canvas.default_pixel_format(),
            sdl2::render::TextureAccess::Streaming,
            screen_bounds.width(),
            screen_bounds.height(),
        )
        .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;

    println!("Texture {:?}", fractal_texture.query());

    let mut pixels = vec![255u8; (screen_bounds.width() * screen_bounds.height() * 4) as usize];

    fractal_texture
        .update(None, &pixels, (screen_bounds.width() * 4) as usize)
        .expect("Failed to update texture");

    let mut thread_pool = ThreadPool::new();
    thread_pool.recompute_fractal(&args, &work_packages);

    let mut key_state: [bool; 2] = [false, false];

    'main_loop: loop {
        let mut rebuild_texture = false;
        thread_pool.main_loop(&args, &mut escapes, &mut rebuild_texture);

        if rebuild_texture {
            let histogram_coloring = if args.coloring == Coloring::Histogram {
                Some(HistogramColoringState::new(&escapes, &args))
            } else {
                None
            };

            for y in 0..args.screen_height {
                for x in 0..args.screen_width {
                    let e = escapes[(y * args.screen_width + x) as usize];

                    let (r, g, b) = match args.coloring {
                        Coloring::BlackWhite => color_simple(e.n, args.iterations, e.z),
                        Coloring::Hsv => color_hsv(e.n, args.iterations, e.z),
                        Coloring::Log => color_log(e.n, args.iterations, e.z),
                        Coloring::Smooth => color_smooth(e.n, args.iterations, e.z),
                        Coloring::Histogram => histogram_coloring
                            .as_ref()
                            .map(|histogram| histogram.colorize(e.n, &args, e.z))
                            .unwrap(),

                        Coloring::OrbitTrap => color_orbit_trap(&e, &args),
                        Coloring::OrbitTrap2 => color_orbit_trap2(&e, &args),
                    };

                    pixels[(y * args.screen_width * 4 + x * 4 + 2) as usize] = r;
                    pixels[(y * args.screen_width * 4 + x * 4 + 1) as usize] = g;
                    pixels[(y * args.screen_width * 4 + x * 4 + 0) as usize] = b;
                }
            }

            fractal_texture
                .update(None, &pixels, (screen_bounds.width() * 4) as usize)
                .expect("Failed to update texture");
        }

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'main_loop,

                Event::KeyUp { keycode, .. } => {
                    if let Some(keycode) = keycode {
                        match keycode {
                            Keycode::LCtrl => {
                                key_state[ModKeys::LeftControl as usize] = false;
                            }
                            _ => {}
                        }
                    }
                }

                Event::KeyDown { keycode, .. } => {
                    if let Some(keycode) = keycode {
                        match keycode {
                            Keycode::LCtrl => {
                                key_state[ModKeys::LeftControl as usize] = true;
                            }

                            Keycode::Backspace => {
                                args = ProgramArgs {
                                    screen_width: screen_bounds.width() as i32,
                                    screen_height: screen_bounds.height() as i32,
                                    ..ProgramArgs::default()
                                };

                                thread_pool.recompute_fractal(&args, &work_packages);
                            }
                            Keycode::PageDown => {
                                args.coloring = next_cycle(&args.coloring).unwrap();
                                thread_pool.recompute_fractal(&args, &work_packages);
                            }
                            Keycode::PageUp => {
                                args.coloring = previous_cycle(&args.coloring).unwrap();
                                thread_pool.recompute_fractal(&args, &work_packages);
                            }
                            Keycode::KpPlus => {
                                args.iterations = (args.iterations * 2)
                                    .clamp(ProgramArgs::MIN_TERATIONS, ProgramArgs::MAX_ITERATIONS);
                                thread_pool.recompute_fractal(&args, &work_packages);
                            }
                            Keycode::KpMinus => {
                                args.iterations = (args.iterations / 2)
                                    .clamp(ProgramArgs::MIN_TERATIONS, ProgramArgs::MAX_ITERATIONS);
                                thread_pool.recompute_fractal(&args, &work_packages);
                            }

                            _ => {}
                        }
                    }
                }

                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Left,
                    x,
                    y,
                    ..
                } => {
                    let Complex { re: cx, im: cy } = screen_coords_to_complex_coords(
                        x as f32,
                        y as f32,
                        &args,
                        ProgramArgs::FRACTAL_XMIN,
                        ProgramArgs::FRACTAL_XMAX,
                        ProgramArgs::FRACTAL_YMIN,
                        ProgramArgs::FRACTAL_YMAX,
                    );

                    //
                    // also zoom when centering if CTRL is down
                    if key_state[ModKeys::LeftControl as usize] {
                        args.zoom *= ProgramArgs::ZOOM_IN_FACTOR;
                    }

                    args.ox += cx * args.zoom;
                    args.oy += cy * args.zoom;

                    thread_pool.recompute_fractal(&args, &work_packages);
                }

                Event::MouseWheel { y, .. } => {
                    let zoom = if y > 0 {
                        //
                        // zoom in
                        args.zoom * ProgramArgs::ZOOM_IN_FACTOR
                    } else {
                        //
                        // zoom out
                        (args.zoom * ProgramArgs::ZOOM_OUT_FACTOR).min(1f32)
                    };

                    if zoom != args.zoom {
                        args.zoom = zoom;
                        thread_pool.recompute_fractal(&args, &work_packages);
                    }
                }
                _ => {}
            }
        }

        canvas
            .copy(&fractal_texture, None, None)
            .expect("Failed to copy texture data");
        canvas.present();
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    Ok(())
}

struct PoolWorker {
    handle: JoinHandle<()>,
    tx: crossbeam::channel::Sender<ThreadPoolMessage>,
}

#[derive(Clone, Debug)]
enum ThreadPoolMessage {
    Quit,
    Work {
        pkgs: Vec<WorkPackage>,
        args: ProgramArgs,
    },
    WorkPackageResult(Vec<WorkResult>),
    ComputationDone(i32),
}

struct ThreadPool {
    worker: Option<JoinHandle<()>>,
    work_in_progress: bool,
    rx: crossbeam::channel::Receiver<ThreadPoolMessage>,
    tx: crossbeam::channel::Sender<ThreadPoolMessage>,
}

fn worker_thread(
    rx: crossbeam::channel::Receiver<ThreadPoolMessage>,
    tx: crossbeam::channel::Sender<ThreadPoolMessage>,
    id: i32,
) {
    'main_loop: loop {
        let msg = rx.recv().expect("Worker recv() error");

        match msg {
            ThreadPoolMessage::Quit => {
                println!("Worker {} quitting ...", id);
                break 'main_loop;
            }
            ThreadPoolMessage::Work { pkgs, args } => {
                println!(
                    "Worker {} starting work, packages {}, args {:?}",
                    id,
                    pkgs.len(),
                    args
                );

                let fxmin = args.ox - ProgramArgs::FRACTAL_HALF_WIDTH * args.zoom;
                let fxmax = args.ox + ProgramArgs::FRACTAL_HALF_WIDTH * args.zoom;
                let fymin = args.oy - ProgramArgs::FRACTAL_HALF_HEIGHT * args.zoom;
                let fymax = args.oy + ProgramArgs::FRACTAL_HALF_HEIGHT * args.zoom;

                pkgs.par_iter()
                    .map(|pkg| {
                        let pixels_x = pkg.x1 - pkg.x0;
                        let pixels_y = pkg.y1 - pkg.y0;

                        assert!(pixels_x > 0);
                        assert!(pixels_y > 0);

                        let mut escapes: Vec<WorkResult> =
                            Vec::with_capacity((pixels_x * pixels_y) as usize);

                        for y in pkg.y0..pkg.y1 {
                            for x in pkg.x0..pkg.x1 {
                                let c = screen_coords_to_complex_coords(
                                    x as f32, y as f32, &args, fxmin, fxmax, fymin, fymax,
                                );

                                let mut z = Complex::new(0f32, 0f32);
                                let mut iterations = 0;
                                let mut trap = 1e20f32;
                                let mut dmin = Vec4::same(1000f32);

                                while z.abs_squared() <= 4f32 && iterations < args.iterations {
                                    z = z * z + c;
                                    iterations += 1;

                                    let dzz = dot(z, z);

                                    trap = trap.min(dzz);

                                    dmin = min(
                                        dmin,
                                        Vec4::new(
                                            (z.im + 0.5f32 * z.re.sin()).abs(),
                                            (1f32 + z.re + 0.5f32 * z.im.sin()).abs(),
                                            dzz,
                                            (fract(z) - Complex::new(0.5f32, 0.5f32)).abs(),
                                        ),
                                    );
                                }

                                escapes.push(WorkResult {
                                    pixel: (x, y),
                                    z,
                                    trap,
                                    dmin,
                                    n: iterations as f32,
                                });
                            }
                        }

                        escapes
                    })
                    .for_each(|escapes| {
                        tx.send(ThreadPoolMessage::WorkPackageResult(escapes))
                            .expect("Failed to send work package result to main");
                    });

                tx.send(ThreadPoolMessage::ComputationDone(id))
                    .expect("Worker failed to send result");
            }
            _ => {}
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum LoopAction {
    Nothing,
    RebuildFractalTexture,
}

impl ThreadPool {
    fn new() -> ThreadPool {
        //
        // sender for pool, receiver for worker
        let (send_to_worker, receiver_worker) =
            crossbeam::channel::unbounded::<ThreadPoolMessage>();
        //
        // sender for worker, receiver for pool
        let (send_to_main, receiver_main) = crossbeam::channel::unbounded::<ThreadPoolMessage>();

        let worker = Some(thread::spawn(move || {
            worker_thread(receiver_worker, send_to_main, 0)
        }));

        ThreadPool {
            work_in_progress: false,
            worker,
            tx: send_to_worker,
            rx: receiver_main,
        }
    }

    fn main_loop(
        &mut self,
        args: &ProgramArgs,
        escapes: &mut Vec<WorkResult>,
        rebuild_texture: &mut bool,
    ) {
        if self.work_in_progress {
            while let Ok(worker_msg) = self.rx.try_recv() {
                match worker_msg {
                    ThreadPoolMessage::ComputationDone(id) => {
                        println!("worker {} finished computation", id);
                        self.work_in_progress = false;
                        //
                        // rebuild fractal texture
                        println!("Rebuilding fractal texture!");
                        *rebuild_texture = true;
                    }
                    ThreadPoolMessage::WorkPackageResult(res) => {
                        assert!(self.work_in_progress);
                        for r in res {
                            let (x, y) = r.pixel;
                            escapes[(y * args.screen_width + x) as usize] = r;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn recompute_fractal(&mut self, args: &ProgramArgs, work_pkgs: &Vec<WorkPackage>) {
        if self.work_in_progress {
            println!("Cant recompute, previous job is unfinished");
            return;
        }

        println!("Recomputing fractal!");

        self.work_in_progress = true;
        self.tx
            .send(ThreadPoolMessage::Work {
                pkgs: work_pkgs.clone(),
                args: *args,
            })
            .expect("Failed to send message to workers");
    }
}

impl std::ops::Drop for ThreadPool {
    fn drop(&mut self) {
        self.tx
            .send(ThreadPoolMessage::Quit)
            .expect("Failed to send quit to workers!");

        self.worker
            .take()
            .map(|handle| handle.join().expect("Failed to join worker thread"));
    }
}
