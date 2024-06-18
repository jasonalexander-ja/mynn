use fastrand::Rng;
use super::Float;
use core::fmt;


const SEED: u64 = 6_447_991_239_222_745_267;

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

	#[cfg(not(feature = "f32"))]
	pub fn random() -> Matrix<ROWS, COLS> {
		let mut rng = Rng::with_seed(SEED);
		let mut data = [[0.0; COLS]; ROWS];

		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = rng.f64() * 2.0 - 1.0;
			}
		}

		Matrix {
			data
		}
	}

	#[cfg(feature = "f32")]
	pub fn random() -> Matrix<ROWS, COLS> {
		let mut rng = Rng::with_seed(SEED);
		let mut data = [[0.0; COLS]; ROWS];

		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = rng.f32() * 2.0 - 1.0;
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

impl<const ROWS: usize, const COLS: usize> fmt::Debug for Matrix<ROWS, COLS> {
	fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt.debug_list().entries(self.data.iter()).finish()
	}
}


/// Type that represents a matrix, uses fixed size arrays based on the `ROWS` and `COLS` const parameters. 
#[derive(Clone)]
pub struct NormalizedMatrix<const ROWS: usize, const COLS: usize> {
	pub data: [[Float; COLS]; ROWS],
	pub cols: usize,
	pub rows: usize,
}

impl<const ROWS: usize, const COLS: usize> NormalizedMatrix<ROWS, COLS> {
	/// Initializes a matrix with all zeros. 
	pub fn zeros(cols: usize, rows: usize) -> Self {
		NormalizedMatrix {
			data: [[0.0; COLS]; ROWS],
			rows,
			cols
		}
	}

	/// Will multiply with another matrix with number of rows equal to the number of rows as to this matrix's cols. 
	pub fn multiply<const OTHER_COLS: usize>(&mut self, other: &NormalizedMatrix<COLS, OTHER_COLS>) {

		for i in 0..self.rows {
			for j in 0..other.cols {
				let mut sum = 0.0;
				for k in 0..self.cols {
					sum += self.data[i][k] * other.data[k][j];
				}
				self.data[i][j] = sum;
			}
		}
		self.cols = other.cols
	}

	/// Will add all the values to an equally sized matrix. 
	pub fn add(&mut self, other: &NormalizedMatrix<ROWS, COLS>) {
		for row in 0..self.rows {
			for col in 0..self.cols {
				self.data[row][col] = self.data[row][col] + other.data[row][col];
			}
		}
	}

	/// Will multiply all the values to an equally sized matrix. 
	pub fn dot_multiply(&mut self, other: &NormalizedMatrix<ROWS, COLS>) {
		for row in 0..self.rows {
			for col in 0..self.cols {
				self.data[row][col] = self.data[row][col] * other.data[row][col];
			}
		}
	}

	/// Will subtract all the values to an equally sized matrix. 
	pub fn subtract(&mut self, other: &NormalizedMatrix<ROWS, COLS>){
		for row in 0..self.rows {
			for col in 0..self.cols {
				self.data[row][col] = self.data[row][col] - other.data[row][col];
			}
		}
	}

	/// Maps all the internal values with a given closure. 
	pub fn map(&mut self, function: &dyn Fn(Float) -> Float) {
		for row in 0..ROWS {
			for col in 0..COLS {
				self.data[row][col] = function(self.data[row][col]);
			}
		}
	}

	/// Swaps the rows and the columns. 
	pub fn transpose(&self) -> NormalizedMatrix<COLS, ROWS> {
		let mut data = [[0.0; ROWS]; COLS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[col][row] = self.data[row][col];
			}
		}
		NormalizedMatrix {
			data,
			rows: self.cols,
			cols: self.rows
		}
	}
}
