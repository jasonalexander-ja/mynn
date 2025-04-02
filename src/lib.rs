//! # mynn 
//! 
//! [![crates.io](https://img.shields.io/crates/v/mynn)](https://crates.io/crates/mynn)
//! [![Released API docs](https://docs.rs/mynn/badge.svg)](https://docs.rs/mynn)
//! [![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENCE)
//! 
//! An hobbyist no-std neural network library. 
//! 
//! ## Explanation 
//! 
//! This is a small library (currently ~200 lines minus doc comments and helper macros) I initially created during my lunch break when I had attempted to represent the shape of a neural network in Rust's type system, the result was I was able to make all the vectors into fixed sized arrays and allow the neural network to be no-std and in theory usable on microcontroller and embedded platforms.  
//! 
//! See this [example](https://github.com/jasonalexander-ja/mynn-attiny-example) of a pre-trained model approximating an XOR running on an ATtiny85. 
//! 
//! ## Installation 
//! 
//! Command line: 
//! ```text 
//! cargo add mynn 
//! ```
//! 
//! Cargo.toml: 
//! ```text 
//! mynn = "0.1.2" 
//! ``` 
//! 
//! To use `f32` in all operations, supply the `f32` flag:
//! 
//! ```text
//! mynn = { version = "0.1.2", features = ["f32"] }
//! ```
//! 
//! To remove recursion, use `recurse-opt`:
//! ```text
//! mynn = { version = "0.1.1", features = ["recurse-opt"] }
//! ```
//! 
//! This will cause all the recursive method calls on each layer to be inlined, on larger models this may increase the size of the generated code, tradeoffs need to be considered.
//! 
//! ## Example  
//! 
//! Short example approximates the output of a XOR gate. 
//! 
//! ```rust
//! use mynn::make_network;
//! use mynn::activations::SIGMOID;
//! 
//! fn main() {
//!     let inputs = [[0.0, 0.0],  [0.0, 1.0], [1.0, 0.0],  [1.0, 1.0]];
//!     let targets = [[0.0], [1.0], [1.0], [0.0]];
//! 
//! 
//!     let mut network = make_network!(2, 3, 1);
//!     network.train(0.5, inputs, targets, 10_000, &SIGMOID);
//! 
//! 
//!     println!("0 and 0: {:?}", network.predict([0.0, 0.0], &SIGMOID));
//!     println!("1 and 0: {:?}", network.predict([1.0, 0.0], &SIGMOID));
//!     println!("0 and 1: {:?}", network.predict([0.0, 1.0], &SIGMOID));
//!     println!("1 and 1: {:?}", network.predict([1.0, 1.0], &SIGMOID));
//! }
//! ```
#![no_std]

/// Contains types for and an example activation function. 
pub mod activations;
/// Contains the types and functionality for processing matrices. 
pub mod matrix;
/// Contains the types and functionality for the neural network. 
pub mod network;

/// Centralized type for floating point operations that can be easily changed to [f32] or [f64] (default is [f64], use `f32` feature for [f32]).  
#[cfg(not(feature = "f32"))]
pub type Float = f64;
/// Centralized type for floating point operations that can be easily changed to [f32] or [f64] (default is [f64], use `f32` feature for [f32]).  
#[cfg(feature = "f32")]
pub type Float = f32;



/// Helper macro, finds and evaluates to the final value from a token tree. 
/// 
/// # Example
/// ```
/// use mynn::last_arg;
/// 
/// let foo = last_arg!(0, 1, 2, 3, 4);
/// assert_eq!(foo, 4);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! last_arg {
    ($x:expr) => {$x};
    ($x:expr, $($xs:expr),+) => {$crate::last_arg!($($xs),+)};
}

/// Helper macro, instantiates the inner recursive elements to a neural network without the type. 
/// 
/// When used in combination with [make_net_type] within [make_network] this can instantiate a neural network. 
/// 
/// # Example
/// ```
/// use mynn::network::{ProcessLayer, EndLayer};
/// use mynn::instantiate_net;
/// 
/// let network: ProcessLayer::<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>> = ProcessLayer::new(instantiate_net!(2, 3, 1));
/// let network2: ProcessLayer::<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>> = ProcessLayer::new(ProcessLayer::new(EndLayer()));
/// 
/// assert_eq!(std::any::type_name_of_val(&network), std::any::type_name_of_val(&network2));
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! instantiate_net {
    ($a:expr, $b:expr) => {
        ($crate::network::EndLayer())
    };
    ($a:expr, $($b:tt),*) => {
        $crate::network::ProcessLayer::new($crate::instantiate_net!($($b),*))
    }
}

/// Helper macro, generates a type definition for the recursive types in a neural network. 
/// 
/// When used with [instantiate_net] in [make_network] it can make instantiating large neural networks less verbose. 
/// 
/// # Example 
/// ```
/// use mynn::network::{ProcessLayer, EndLayer};
/// use mynn::make_net_type;
/// 
/// let network: make_net_type!(2, 3, 1) = ProcessLayer::new(ProcessLayer::new(EndLayer()));
/// ```
#[macro_export]
#[doc(hidden)]
macro_rules! make_net_type {
    ($neurons:expr) => {
        $crate::network::EndLayer::<$neurons>
    };
    ($neurons:expr, $next:expr) => {
        $crate::network::ProcessLayer::<$next, $neurons, $next, $crate::make_net_type!($next)>
    };
    ($neurons:expr, $next:expr, $($c:tt),*) => {
        $crate::network::ProcessLayer::<$next, $neurons, {$crate::last_arg!($($c),*)}, $crate::make_net_type!($next, $($c),*)>
    };
}

/// Helper macro used to initialize a neural network, simply pass a comma separated list the number of neurons for each layer, works for any sized neural network. 
/// 
/// # Example 
/// ```
/// use mynn::network::{ProcessLayer, EndLayer};
/// use mynn::make_network;
/// 
/// let network = make_network!(2, 3, 1);
/// let network2 = ProcessLayer::<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>>::new(ProcessLayer::new(EndLayer()));
/// 
/// assert_eq!(std::any::type_name_of_val(&network), std::any::type_name_of_val(&network2));
/// ```
#[macro_export]
macro_rules! make_network {
    ($neurons:expr) => {
        $crate::network::EndLayer::<$neurons>()
    };
    ($neurons:expr, $next:expr) => {
        $crate::network::ProcessLayer::<
            $next, 
            $neurons, 
            $next, 
            $crate::make_net_type!($next)
        >::new($crate::instantiate_net!($neurons, $next))
    };
    ($neurons:expr, $next:expr, $($c:tt),*) => {
        $crate::network::ProcessLayer::<
            $next, 
            $neurons, 
            {$crate::last_arg!($($c),*)}, 
            $crate::make_net_type!($next, $($c),*)
        >::new($crate::instantiate_net!($neurons, $next, $($c),*))
    };
}




