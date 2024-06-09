#[cfg(not(feature = "f32"))]
use std::f64::consts::E;
#[cfg(feature = "f32")]
use std::f32::consts::E;
use super::Float;


pub struct Activation<'a> {
    pub function: &'a dyn Fn(Float) -> Float,
    pub derivative: &'a dyn Fn(Float) -> Float
}

pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + E.powf(-x)),
    derivative: &|x| x * (1.0 - x)
};
