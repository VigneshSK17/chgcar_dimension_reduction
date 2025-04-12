# CHGCAR Dimension Reduction

Provides the ability to reduce the dimensions of CHGCARS from (X,Y,Z) -> (X/A, Y/A, Z/A, W) by utilizing different basis functions.

The current available basis function options are: Chebyshev, Lagrange, and Gaussian (untuned).

# Installation

1. Clone the repository: `git clone https://github.com/VigneshSK17/chgcar_dimension_reduction.git`
2. Install python dependencies using `pip install -r requirements.txt`
3. Run `python main.py --mode lagrange chgcars/` to start compression

# Help

```
Usage: main.py [OPTIONS] PATH

  Reduce dimensions of chgcars located in PATH by FACTOR with DEGREE

Options:
  --factor INTEGER                Divisor by which you wish to reduce
                                  dimensions of chgcar
  --degree INTEGER                The degree of the 4th dimension of data
                                  created during dimension reduction
  --max-workers INTEGER           Maximum number of worker processes (defaults
                                  to CPU count)
  --mode [chebyshev|lagrange|gaussian]
                                  [required]
  --output TEXT                   Path to save results in CSV format
  --help                          Show this message and exit.
```


