#![allow(dead_code)]

use clap::Parser;
use std::sync::{Arc, Mutex};
use std::thread;
use std::{io::Write, path::PathBuf};

mod complex;

use complex::Complex;

type RGB8 = (u8, u8, u8);

fn screen_coords_to_complex_coords(
    px: f32,
    py: f32,
    sxmin: f32,
    sxmax: f32,
    symin: f32,
    symax: f32,
    dxmin: f32,
    dxmax: f32,
    dymin: f32,
    dymax: f32,
) -> Complex {
    let x = (px / (sxmax - sxmin)) * (dxmax - dxmin) + dxmin;
    let y = (py / (symax - symin)) * (dymax - dymin) + dymin;

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
    let px = ((n / max_iterations as f32) * 255f32) as u8;
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
        ((-(n * 0.025_f32).cos() + 1_f32) * 255f32) as u8,
        ((-(n * 0.080_f32).cos() + 1_f32) * 255f32) as u8,
        ((-(n * 0.120_f32).cos() + 1_f32) * 255f32) as u8,
    )
}

fn color_hsv(iterations: f32, max_iterations: i32, _z: Complex) -> (u8, u8, u8) {
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

fn color_hsv_smooth(iterations: f32, max_iterations: i32, z: Complex) -> (u8, u8, u8) {
    let m = if iterations >= max_iterations as f32 {
        max_iterations as f32
    } else {
        iterations + 1f32 - z.abs().log2().log10()
    };

    color_hsv(m, max_iterations, z)
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

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum Coloring {
    BlackWhite,
    Hsv,
    Smooth,
    SmoothHSV,
    Log,
    Histogram,
}

#[derive(clap::Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct ProgramArgs {
    #[arg(long, default_value_t = 1024)]
    screen_width: i32,
    #[arg(long, default_value_t = 1024)]
    screen_height: i32,
    #[arg(short = 't', long, default_value_t = 8, value_parser = clap::value_parser!(i32).range(4..512))]
    tile_size: i32,
    #[arg(short = 'i', long, default_value_t = 16)]
    iterations: i32,
    #[arg(short = 'w', long, default_value_t = 8, value_parser = clap::value_parser!(i32).range(1..512))]
    workers: i32,
    #[arg(short = 'c', value_enum, default_value_t = Coloring::BlackWhite)]
    coloring: Coloring,
}

#[derive(Copy, Clone, Debug)]
struct WorkResult {
    pixel: (i32, i32),
    z: Complex,
    n: f32,
}

fn main() {
    let args = ProgramArgs::parse();
    println!("{:?}", args);

    let mut pixels = vec![(0u8, 0u8, 0u8); (args.screen_width * args.screen_height) as usize];
    let mut escapes = vec![
        WorkResult {
            pixel: (0, 0),
            z: Complex::new(0f32, 0f32),
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

    let (sender, receiver) = std::sync::mpsc::channel::<WorkResult>();

    let work_packages = Arc::new(Mutex::new(work_packages));
    let workers = (0..args.workers)
        .map(|_| {
            let work_queue = Arc::clone(&work_packages);
            let chan_sender = sender.clone();

            thread::spawn(move || 'main_loop: loop {
                let maybe_work = work_queue
                    .lock()
                    .ok()
                    .map(|mut wkqueue| wkqueue.pop())
                    .flatten();

                if let Some(work_pkg) = maybe_work {
                    for py in work_pkg.y0..work_pkg.y1 {
                        for px in work_pkg.x0..work_pkg.x1 {
                            let c = screen_coords_to_complex_coords(
                                px as f32,
                                py as f32,
                                0f32,
                                args.screen_width as f32,
                                0f32,
                                args.screen_height as f32,
                                -2.75f32,
                                1f32,
                                -1f32,
                                1f32,
                            );

                            let mut z = Complex::new(0f32, 0f32);
                            let mut iterations = 0;

                            while z.abs_squared() <= 4f32 && iterations < args.iterations {
                                z = z * z + c;
                                iterations += 1;
                            }

                            let m = if iterations == args.iterations {
                                args.iterations as f32
                            } else {
                                iterations as f32 + 1f32 - z.abs().log2().log10()
                            };

                            chan_sender
                                .send(WorkResult {
                                    pixel: (px, py),
                                    z,
                                    n: m,
                                })
                                .unwrap();
                        }
                    }
                } else {
                    break 'main_loop;
                }
            })
        })
        .collect::<Vec<_>>();

    drop(sender);

    for res in receiver.iter() {
        let (px, py) = res.pixel;
        escapes[(py * args.screen_width + px) as usize] = res;
    }

    for w in workers {
        w.join().expect("Failed to join worker thread!");
    }

    for y in 0..args.screen_height {
        for x in 0..args.screen_width {
            let e = escapes[(y * args.screen_width + x) as usize];

            let pixel_color = match args.coloring {
                Coloring::BlackWhite => color_simple(e.n, args.iterations, e.z),
                Coloring::Hsv => color_hsv(e.n, args.iterations, e.z),
                Coloring::Log => color_log(e.n, args.iterations, e.z),
                Coloring::Smooth => color_smooth(e.n, args.iterations, e.z),
                Coloring::SmoothHSV => color_hsv_smooth(e.n, args.iterations, e.z),
                Coloring::Histogram => {
                    let mut histogram = vec![0i32; args.iterations as usize];
                    // let mut values = vec![0f32; (args.screen_width * args.screen_height) as usize];

                    if e.n < args.iterations as f32 {
                        histogram[e.n.floor() as usize] += 1;
                    }

                    let total: i32 = histogram.iter().sum();
                    let mut hues = Vec::<f32>::new();
                    let mut h: f32 = 0f32;

                    for i in 0..args.iterations {
                        h += histogram[i as usize] as f32 / total as f32;
                        hues.push(h);
                    }
                    hues.push(h);

                    todo!("Not implemented yet");
                }
            };

            pixels[(y * args.screen_width + x) as usize] = pixel_color;
        }
    }

    //
    // colorize fractal

    // for y in 0..args.screen_height {
    //     for x in 0..args.screen_width {
    //         let m = values[(y * args.screen_width + x) as usize];

    //         let hsv = (
    //             360f32
    //                 - linear_interpolate(
    //                     hues[m.floor() as usize],
    //                     hues[m.ceil() as usize],
    //                     m.fract(),
    //                 ) * 360f32,
    //             1f32,
    //             if m < args.iterations as f32 {
    //                 1f32
    //             } else {
    //                 0f32
    //             },
    //         );

    //         pixels[(y * args.screen_width + x) as usize] = hsv_to_rgb(hsv);
    //     }
    // }

    write_image(
        "mandel.pbm".into(),
        args.screen_width,
        args.screen_height,
        &pixels,
    );
}
