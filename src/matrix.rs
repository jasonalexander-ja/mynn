use rand::{thread_rng, Rng};
use super::Float;

/// Type that represents a matrix, uses fixed size arrays based on the `ROWS` and `COLS` const parameters. 
#[derive(Clone)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
	pub data: [[Float; COLS]; ROWS],
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
	/// Initializes a matrix with all zeros. 
	pub fn zeros() -> Matrix<ROWS, COLS> {
		Matrix {
			data: [[0.0; COLS]; ROWS]
		}
	}

	pub fn random() -> Matrix<ROWS, COLS> {
		let mut rng = thread_rng();
		let mut data = [[0.0; COLS]; ROWS];

		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = rng.gen::<Float>() * 2.0 - 1.0;
			}
		}

		Matrix {
			data
		}
	}

	/// Will multiply with another matrix with number of rows equal to the number of rows as to this matrix's cols. 
	pub fn multiply<const OTHER_COLS: usize>(&self, other: &Matrix<COLS, OTHER_COLS>) -> Matrix<ROWS, OTHER_COLS> {

		let mut res = Matrix::<ROWS, OTHER_COLS>::zeros();

		for i in 0..ROWS {
			for j in 0..OTHER_COLS {
				let mut sum = 0.0;
				for k in 0..COLS {
					sum += self.data[i][k] * other.data[k][j];
				}

				res.data[i][j] = sum;
			}
		}

		res
	}

	/// Will add all the values to an equally sized matrix. 
	pub fn add(&self, other: &Matrix<ROWS, COLS>) -> Matrix<ROWS, COLS> {

		let mut data = [[0.0; COLS]; ROWS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = self.data[row][col] + other.data[row][col];
			}
		}

		Matrix {
			data
		}
	}

	/// Will multiply all the values to an equally sized matrix. 
	pub fn dot_multiply(&self, other: &Matrix<ROWS, COLS>) -> Matrix<ROWS, COLS> {

		let mut data = [[0.0; COLS]; ROWS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = self.data[row][col] * other.data[row][col];
			}
		}

		Matrix {
			data
		}
	}

	/// Will subtract all the values to an equally sized matrix. 
	pub fn subtract(&self, other: &Matrix<ROWS, COLS>) -> Matrix<ROWS, COLS> {

		let mut data = [[0.0; COLS]; ROWS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = self.data[row][col] - other.data[row][col];
			}
		}

		Matrix {
			data
		}
	}

	/// Maps all the internal values with a given closure. 
	pub fn map(&self, function: &dyn Fn(Float) -> Float) -> Matrix<ROWS, COLS> {

		let mut data = [[0.0; COLS]; ROWS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = function(self.data[row][col]);
			}
		}

		Matrix {
			data
		}
	}

	/// Creates a new matrix from a given 2-dimensional array. 
	pub fn from(data: [[Float; COLS]; ROWS]) -> Matrix<ROWS, COLS> {
		Matrix {
			data
		}
	}

	/// Swaps the rows and the columns. 
	pub fn transpose(&self) -> Matrix<COLS, ROWS> {
		let mut data = [[0.0; ROWS]; COLS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[col][row] = self.data[row][col];
			}
		}
		Matrix {
			data
		}
	}
}
