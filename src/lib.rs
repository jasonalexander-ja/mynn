#![no_std]

pub mod activations;
pub mod matrix;
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




