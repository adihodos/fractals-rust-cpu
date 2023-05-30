#[derive(Copy, Clone, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn same(v: f32) -> Self {
        Self::new(v, v, v)
    }
}

impl std::ops::Add<Vec3> for Vec3 {
    type Output = Self;

    fn add(self, rhs: Vec3) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub<Vec3> for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, k: f32) -> Self::Output {
        Self {
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
        }
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

pub fn sqrt(v: Vec3) -> Vec3 {
    Vec3::new(v.x.sqrt(), v.y.sqrt(), v.z.sqrt())
}

pub fn sin(v: Vec3) -> Vec3 {
    Vec3::new(v.x.sin(), v.y.sin(), v.z.sin())
}

pub fn mix(x: Vec3, y: Vec3, a: f32) -> Vec3 {
    //  x×(1−a)+y×a.
    Vec3::new(
        x.x * (1f32 - a) + y.x * a,
        x.y * (1f32 - a) + y.y * a,
        x.z * (1f32 - a) + y.z * a,
    )
}
