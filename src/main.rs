extern crate png;
extern crate rgb;

mod complex;
mod window;

use complex::Complex;
use rgb::RGB8;
use window::Window;

fn screen_domain_to_complex_domain(scr: &Window<u32>, cmpl: &Window<f32>, p: Complex) -> Complex {
    Complex::new(
        p.re / scr.width() as f32 * cmpl.width() + cmpl.xmin,
        p.im / scr.height() as f32 * cmpl.height() + cmpl.ymin,
    )
}

fn escape<F>(c: Complex, max_iterations: u32, f: F) -> u32
where
    F: Fn(Complex, Complex) -> Complex,
{
    let mut iter_count = 0u32;
    let mut z = Complex::new(0f32, 0f32);

    while z.abs_squared() < 4f32 && iter_count < max_iterations {
        z = f(z, c);
        iter_count += 1;
    }

    iter_count
}

fn color_simple(n: u32, max_iterations: u32) -> RGB8 {
    let px = ((n as f32 / max_iterations as f32) * 255f32) as u8;
    RGB8::new(px, px, px)
}

fn color_smooth(n: u32, max_iterations: u32) -> RGB8 {
    let t = n as f32 / max_iterations as f32;
    let u = 1f32 - t;

    RGB8::new(
        ((9f32 * u * t * t * t) * 255f32) as u8,
        ((15f32 * u * u * t * t) * 255f32) as u8,
        ((8.5f32 * u * u * u * t) * 255f32) as u8,
    )
}

fn mandlebrot<F, C>(
    screen: &Window<u32>,
    fractal: &Window<f32>,
    max_iterations: u32,
    f: F,
    color_fun: C,
) -> Vec<RGB8>
where
    F: Fn(Complex, Complex) -> Complex + Copy,
    C: Fn(u32, u32) -> RGB8 + Copy,
{
    let mut pixels: Vec<RGB8> = Vec::new();
    pixels.reserve(screen.size() as usize);

    for y in screen.ymin..screen.ymax {
        for x in screen.xmin..screen.xmax {
            let c =
                screen_domain_to_complex_domain(screen, fractal, Complex::new(x as f32, y as f32));
            let n = escape(c, max_iterations, f);

            pixels.push(color_fun(n, max_iterations));
        }
    }

    pixels
}

fn write_image(filename: &str, width: u32, height: u32, pixels: &[RGB8]) -> std::io::Result<()> {
    use png::HasParameters;
    use std::fs::File;
    use std::io::BufWriter;

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set(png::ColorType::RGB).set(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;

    let img_data =
        unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 3) };

    png_writer.write_image_data(&img_data)?;

    Ok(())
}

fn main() {
    let screen = window::Window::new(0, 1024, 0, 1024);
    let fractal = window::Window::new(-1.5, 1.5, -1.5, 1.5);
    let pixels = mandlebrot(
        &screen,
        &fractal,
        128,
        |z: Complex, c: Complex| z * z + c,
        color_smooth,
    );
    write_image(
        "mandelbrot.png",
        screen.width() as u32,
        screen.height() as u32,
        &pixels,
    )
    .expect("Failed to write fractal image!");
}
