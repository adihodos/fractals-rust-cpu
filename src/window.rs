use std::ops::{Add, Mul, Sub};

#[derive(Copy, Clone, Debug)]
pub struct Window<T> {
    pub xmin: T,
    pub xmax: T,
    pub ymin: T,
    pub ymax: T,
}

impl<T> Window<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    pub fn new(xmin: T, xmax: T, ymin: T, ymax: T) -> Window<T> {
        Window {
            xmin,
            xmax,
            ymin,
            ymax,
        }
    }

    pub fn width(&self) -> T {
        self.xmax - self.xmin
    }

    pub fn height(&self) -> T {
        self.ymax - self.ymin
    }

    pub fn size(&self) -> T {
        self.width() * self.height()
    }
}
