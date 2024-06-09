#[cfg(not(feature = "f32"))]
use core::f64::consts::E;
#[cfg(feature = "f32")]
use core::f32::consts::E;
#[cfg(not(feature = "f32"))]
use libm::pow;
#[cfg(feature = "f32")]
use libm::powf;
use super::Float;


/// Helper container type holding the closures for the activation function and the derivative. 
/// 
/// Used for forward and backwards propagation in the neural network. 
pub struct Activation<'a> {
    pub function: &'a dyn Fn(Float) -> Float,
    pub derivative: &'a dyn Fn(Float) -> Float
}

/// Sigmoid activation function, used a lot in the examples and tests. 
#[cfg(not(feature = "f32"))]
pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + pow(E, -x)),
    derivative: &|x| x * (1.0 - x)
};

/// Sigmoid activation function, used a lot in the examples and tests. 
#[cfg(feature = "f32")]
pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + powf(E, -x)),
    derivative: &|x| x * (1.0 - x)
};
