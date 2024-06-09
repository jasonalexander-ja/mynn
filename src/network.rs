use super::{activations::Activation, matrix::Matrix};
use super::Float;

/// Generic type for all layers in a neural network defining standard const parameter and behaviour. 
/// 
/// # Type Parameters
/// * `NEURONS` The number of neurons in that layer. 
/// * `END_S` The number of neurons in the final layer, used when passing back an array of predictions. 
pub trait Layer<const NEURONS: usize, const END_S: usize> {

    /// Feeds forward data and returns (I.E. predicts) an array of data based on it's current learned state. 
    /// 
    /// # Parameters 
    /// * `feed` The data to be predicted upon, a matrix with 1 column and number of rows equal to the number of neurons. 
    /// * `act` The Activation function to be used. 
    fn feed_forward<'a>(&mut self, feed: Matrix<NEURONS, 1>, act: &Activation<'a>) -> [Float; END_S];

    /// Back propagates (I.E. makes corrections or "learns") based on the previous outputs and the expected outputs. 
    /// 
    /// # Parameters 
    /// * `l_rate` The learning rate, is multiplied with the calculated difference gradient to allow for smaller/greater changes per learning revision. 
    /// * `outputs` The outputs from the previous prediction. 
    /// * `targets` The actual targeted value for the previous prediction. 
    /// * `act` The activation function. 
    fn back_propagate<'a>(&mut self, l_rate: Float, outputs: [Float; END_S], targets: [Float; END_S], act: &Activation<'a>) -> BackProps<NEURONS>;
}


/// Type for an active (I.E. containing neurons) layer. 
/// 
/// Has type bounds to ensure the next layer must have equal number of neurons as there are rows in the weights and biases matrices. 
/// 
/// # Type Parameters
/// * `ROWS` The number of rows in the weights, biases, and number of neurons that must be in the next layer. 
/// * `NEURONS` The number of neurons (number of columbs in the weights matrix) in this layer. 
/// * `END_S` The number of neurons in the final layer, used when passing back an array of predictions. 
/// * `T` The type of the next layer, must implement [Layer]. 
pub struct ProcessLayer<const ROWS: usize, const NEURONS: usize, const END_S: usize, T: Layer<ROWS, END_S>> {
    /// The next layer. 
    pub next: T,
    pub weights: Matrix<ROWS, NEURONS>,
    pub biases: Matrix<ROWS, 1>,
    /// The data that was last passed in during a feed forward, used to make corrections during back propagation. 
    pub data: Matrix<NEURONS, 1>
}

impl <const ROWS: usize, const NEURONS: usize, const END_S: usize, T: Layer<ROWS, END_S>> ProcessLayer<ROWS, NEURONS, END_S, T> {

    /// Instantiates a new layer, accepts the next layer in the linked list as a parameter. 
    /// 
    /// # Example 
    /// ```
    /// use mynn::network::{ProcessLayer, EndLayer};
    /// 
    /// let network: ProcessLayer::<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>> = ProcessLayer::new(ProcessLayer::new(EndLayer()));
    /// ```
    pub fn new(next: T) -> ProcessLayer<ROWS, NEURONS, END_S, T> {
        ProcessLayer {
            next,
            weights: Matrix::zeros(),
            biases: Matrix::zeros(),
            data: Matrix::zeros(),
        }
    }

    /// Accepts an array of data, feeding it forward down each layer, returning the predicted result based on the current learning state. 
    /// 
    /// # Parameters 
    /// * `data` The data for the prediction to be made upon, must have equal number of values as neurons in the first layer. 
    /// * `act` The activation function to be used. 
    /// 
    /// # Example 
    /// ```
    /// use mynn::{make_network, activations::SIGMOID};
    /// 
    /// let inputs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    /// let targets = [[0.0], [0.0], [0.0], [1.0]]
    /// let network = make_network!(2, 3, 1);
    /// 
    /// network.train(0.5, inputs, targets, 10_000, &SIGMOID);
    /// 
    /// println!("1 and 1: {:?}", network.predict([1.0, 1.0], &SIGMOID));
    /// ```
    pub fn predict<'a>(&mut self, data: [Float; NEURONS], act: &Activation<'a>) -> [Float; END_S] {
        self.feed_forward(Matrix::from([data]).transpose(), act)
    }

    /// Trains a neural network list, accepts 2 arrays of equal length with the data and expected results. 
    /// 
    /// # Parameters 
    /// * `l_rate` The learning rate, is multiplied with the calculated difference gradient to allow for smaller/greater changes per learning revision. 
    /// * `inputs` Array of possible inputs, each index in this array must correspond with the same index in the `targets`. 
    /// * `targets` Array of targets, each index in this array must correspond with the same index in the `inputs`. 
    /// * `epochs` Number of epochs (feeding forward/preedicting and then back propagating/learning).
    /// * `act` The activation function. 
    pub fn train<'a, const DATA_S: usize>(&mut self, l_rate: Float, inputs: [[Float; NEURONS]; DATA_S], targets: [[Float; END_S]; DATA_S], epochs: usize, act: &Activation<'a>) {
        for _ in 1..=epochs {
            for i in 0..DATA_S {
                let outputs = self.feed_forward(Matrix::from([inputs[i]]).transpose(), act);
                self.back_propagate(l_rate, outputs, targets[i].clone(), act);
            }
        }
    }

}

impl <const ROWS: usize, const NEURONS: usize, const END_S: usize, T: Layer<ROWS, END_S>> Layer<NEURONS, END_S> for ProcessLayer<ROWS, NEURONS, END_S, T> {
    fn feed_forward<'a>(&mut self, feed: Matrix<NEURONS, 1>, act: &Activation<'a>) -> [Float; END_S] {
        self.data = feed;
        let result = self.weights.multiply(&self.data)
            .add(&self.biases)
            .map(act.function);
        self.next.feed_forward(result, act)
    }

    fn back_propagate<'a>(&mut self, l_rate: Float, outputs: [Float; END_S], targets: [Float; END_S], act: &Activation<'a>) -> BackProps<NEURONS> {
        let BackProps(errors, gradients) = self.next.back_propagate(l_rate, outputs, targets, act);
        let gradients = gradients.dot_multiply(&errors).map(&|x| x * l_rate);

        self.weights = self.weights.add(&gradients.multiply(&self.data.transpose()));
        self.biases = self.biases.add(&gradients);

        let errors = self.weights.transpose().multiply(&errors);
        let gradients = self.data.map(&act.derivative);

        BackProps(errors, gradients)
    }
}


/// The end layer, this terminates the neural network linked list, just accepts the number of neurons in the final layer. 
/// 
/// # Type Parameters
/// * `END_S` Number of neurons in the end layer. 
pub struct EndLayer<const END_S: usize>();

impl <const END_S: usize> Layer<END_S, END_S> for EndLayer<END_S> {
    fn feed_forward<'a>(&mut self, feed: Matrix<END_S, 1>, _act: &Activation<'a>) -> [Float; END_S] {
        feed.transpose().data[0]
    }

    fn back_propagate<'a>(&mut self, _l_rate: Float, outputs: [Float; END_S], targets: [Float; END_S], act: &Activation<'a>) -> BackProps<END_S> {
        let parsed = Matrix::from([outputs]).transpose();
        let errors = Matrix::from([targets]).transpose().subtract(&parsed);
        let gradients = parsed.map(&act.derivative);
        BackProps(errors, gradients)
    }
}

/// Helper type for passing parameters back through the the neural network during back propagation. 
/// `(errors, gradients)`
pub struct BackProps<const COLS: usize>(Matrix<COLS, 1>, Matrix<COLS, 1>);



