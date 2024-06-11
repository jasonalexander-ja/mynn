# mynn 

[![crates.io](https://img.shields.io/crates/v/mynn)](https://crates.io/crates/mynn)
[![Released API docs](https://docs.rs/mynn/badge.svg)](https://docs.rs/mynn)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENCE)

An hobbyist no-std neural network library. 

## Explanation 

This is a small library (currently ~200 lines minus doc comments and helper macros) I initially created during my lunch break when I had attempted to represent the shape of a neural network in the rust type system, the result was I was able to make all the vectors into fixed sized arrays and allow the neural network to be no-std and usable microcontroller and embedded platforms.  

## Installation 

Command line: 
```text 
cargo add mynn 
```

Cargo.toml: 
```text 
mynn = "0.1.0" 
``` 

To use `f32` in all operations, supply the `f32` flag:

```text
mynn = { version = "0.1.0", features = ["f32"] }
```

## Example  

Simple estimating the output of a XOR gate. 

```rust
use mynn::make_network;
use mynn::activations::SIGMOID;

fn main() {
    let inputs: [[f64; 2]; 4] = [[0.0, 0.0],  [0.0, 1.0], [1.0, 0.0],  [1.0, 1.0]];
    let targets: [[f64; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];


    let mut network = make_network!(2, 3, 1);
    network.train(0.5, inputs, targets, 10_000, &SIGMOID);


    println!("0 and 0: {:?}", network.predict([0.0, 0.0], &SIGMOID));
    println!("1 and 0: {:?}", network.predict([1.0, 0.0], &SIGMOID));
    println!("0 and 1: {:?}", network.predict([0.0, 1.0], &SIGMOID));
    println!("1 and 1: {:?}", network.predict([1.0, 1.0], &SIGMOID));
}
```
