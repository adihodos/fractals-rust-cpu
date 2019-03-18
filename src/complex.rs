use std::ops::{Add, Mul, Sub};

#[derive(Copy, Clone, Debug)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Complex {
        Complex { re, im }
    }

    pub fn abs_squared(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub fn abs(&self) -> f32 {
        self.abs_squared().sqrt()
    }
}

impl Add for Complex {
    type Output = Complex;

    fn add(self, rhs: Complex) -> Complex {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, rhs: Complex) -> Complex {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Mul<f32> for Complex {
    type Output = Complex;

    fn mul(self, k: f32) -> Complex {
        Complex {
            re: self.re * k,
            im: self.im * k,
        }
    }
}

impl Mul<Complex> for f32 {
    type Output = Complex;

    fn mul(self, c: Complex) -> Complex {
        c * self
    }
}
