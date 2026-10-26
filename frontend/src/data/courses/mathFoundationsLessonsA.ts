import type { CourseLesson } from "./types";

const floatingPointTutorial = { title: "Python tutorial: floating-point arithmetic, issues and limitations", url: "https://docs.python.org/3/tutorial/floatingpoint.html" };
const mathModule = { title: "Python documentation: math — mathematical functions", url: "https://docs.python.org/3/library/math.html" };
const numericTypes = { title: "Python documentation: numeric types — int, float, complex", url: "https://docs.python.org/3/builtins/stdtypes.html#numeric-types-int-float-complex" };
const emulatingNumericTypes = { title: "Python reference: emulating numeric types with special methods", url: "https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types" };
const nestedListComprehensions = { title: "Python tutorial: nested list comprehensions", url: "https://docs.python.org/3/tutorial/datastructures.html#nested-list-comprehensions" };
const zipFunction = { title: "Python documentation: zip and its strict argument", url: "https://docs.python.org/3/builtins/functions.html#zip" };
const statisticsModule = { title: "Python documentation: statistics — mathematical statistics functions", url: "https://docs.python.org/3/library/statistics.html" };
const randomReproducibility = { title: "Python documentation: random — notes on reproducibility", url: "https://docs.python.org/3/library/random.html#notes-on-reproducibility" };
const fractionsModule = { title: "Python documentation: fractions — rational numbers", url: "https://docs.python.org/3/library/fractions.html" };
const mmlBook = { title: "Mathematics for Machine Learning, by Deisenroth, Faisal and Ong (book website)", url: "https://mml-book.github.io/" };

/** Lessons 1 to 3 of the Mathematics for Machine Learning from First Principles course. */
export const mathFoundationsLessonsA: CourseLesson[] = [
  {
    id: "python-math-foundations", title: "Turn formulas into Python and handle rounding", minutes: 45,
    intro: "Every formula in machine learning eventually runs as code. This lesson maps Python's numbers, expressions, functions, loops, lists and classes onto mathematical notation, and shows where computer arithmetic differs from arithmetic on paper: floats are rounded, rounding errors can accumulate, and subtracting nearly equal numbers can destroy accuracy.",
    sections: [
      { title: "Know which numbers are exact", body: "Python integers are exact at any size, so 2¹⁰⁰ and 30! are stored digit for digit. A float is a binary number with a 53-bit significand: it carries about 16 significant decimal digits, and the gap between 1 and the next float, the machine epsilon, is 2⁻⁵² ≈ 2.2 × 10⁻¹⁶. Most decimals, including 0.1, have no exact binary form and are rounded when stored. In expressions, / always returns a float, // rounds down towards −∞, and exponentiation binds more tightly than a leading minus sign, so minus three squared gives −9 unless you add brackets." },
      { title: "Translate notation into code", body: "A Python function plays the role of f(x), with every multiplication written out: 3x² − 2x + 1 becomes 3 * x * x - 2 * x + 1. A sum Σ or product Π becomes a loop with an accumulator that starts at 0 or 1, which is why the empty sum is 0 and 0! = 1. Recursion mirrors a recursive definition but needs a base case. A matrix is a list of equally long rows, so aᵢⱼ is A[i-1][j-1]. Special methods such as __add__ let a class use the + sign." },
      { title: "Expect rounding and control it", body: "Because 0.1 + 0.2 is stored as 0.30000000000000004, never compare floats with ==. Use math.isclose instead, and give it an abs_tol when one side may be zero. Adding many floats one at a time lets rounding errors accumulate, whereas math.fsum rounds only once. Subtracting nearly equal numbers, called catastrophic cancellation, turns tiny earlier errors into large relative ones: the shortcut variance formula, mean of squares minus squared mean, can even return a negative number. Rearranging the formula, as the two-pass variance does, avoids the problem." },
    ],
    example: { title: "Cancellation in a variance, worked by hand", body: "Take 4, 7, 13 and 16. The mean is 10, the deviations are −6, −3, 3 and 6, their squares add up to 36 + 9 + 9 + 36 = 90, and the population variance is 90 / 4 = 22.5. Adding 10⁹ to every value leaves the variance unchanged. The shortcut formula then subtracts (10⁹ + 10)² = 1,000,000,020,000,000,100 from a mean of squares that is only 22.5 larger, but neighbouring floats near 10¹⁸ are 128 apart, so the 22.5 is lost: a straightforward loop returns −128.0. The two-pass formula subtracts the mean first and still gives 22.5." },
    code: {
      title: "Watch rounding accumulate",
      source: "import math\n\ntotal = 0.0\nfor _ in range(10):\n    total += 0.1\nprint(total)  # 0.9999999999999999\nprint(math.fsum([0.1] * 10))  # 1.0\nprint(total == 1.0)  # False\nprint(math.isclose(total, 1.0))  # True\nprint(0.1 + 0.2)  # 0.30000000000000004",
      explanation: "The loop rounds after each of its ten additions, and the errors leave the total just below 1. math.fsum tracks the exact running sum and rounds once. Python's built-in sum has compensated for this error since version 3.12, which is why the snippet uses an explicit loop: loops you write yourself, and most other languages, behave like this one.",
    },
    notebook: { path: "math-foundations/01_python_math_foundations.ipynb", title: "Python foundations: numbers, functions, lists and floating point" },
    practice: [
      "Run the notebook from the top and read the output of Step 6. Explain why the shortcut variance returns −128.0 for the shifted data while statistics.pvariance still gives 22.5.",
      "Complete Exercise 1 (polynomial and n_choose_k) and Exercise 2 (mean, variance and mean_squared_error), rerunning each check cell until it reports that all tests passed.",
      "Complete Exercise 3 (reshape, flatten and diagonal_matrix) and Exercise 4 (relative_error, approx_equal and float_predictions), writing down your six float predictions before you run any of the expressions.",
    ],
    references: [floatingPointTutorial, mathModule, numericTypes, emulatingNumericTypes],
    quiz: {
      question: "In exact arithmetic a computed float should equal 0.3. Which test is the safest way to check it?",
      options: ["computed == 0.3", "round(computed) == round(0.3)", "math.isclose(computed, 0.3)"],
      correctIndex: 2,
      explanation: "Rounding error can make an exact comparison fail even when the mathematics says the values are equal, as with 0.1 + 0.2. math.isclose accepts a small relative difference. Rounding both sides to whole numbers is far too coarse: 0.3 and 0.4 both round to 0.",
    },
  },
  {
    id: "linear-algebra-from-scratch", title: "Build vectors, matrices and transformations from scratch", minutes: 50,
    intro: "Datasets are matrices, model parameters are vectors and matrices, and most of the arithmetic inside machine learning is linear algebra. This lesson builds that arithmetic from Python lists: vector operations and the dot product, matrix multiplication with careful shape checks, norms and distances, and 2-D transformations that compose through matrix multiplication.",
    sections: [
      { title: "Represent and validate vectors and matrices", body: "A vector is a list of numbers and a matrix is a list of rows of equal length. A dataset with m examples and n features is an m × n matrix whose rows are feature vectors. Check the shape before computing anything: an empty list, an empty row or ragged rows of different lengths do not form a matrix, and the error should say which rule failed. Vectors combine entry by entry, and the dot product x · y = Σ xᵢyᵢ measures how far two vectors point the same way; 0 means they are perpendicular." },
      { title: "Multiply matrices by matching the inner size", body: "The product of an m × n matrix A and an n × p matrix B is the m × p matrix with entries (AB)ᵢⱼ = Σₖ aᵢₖbₖⱼ: the dot product of row i of A with column j of B. Three nested loops compute it. The inner sizes must agree, so a 2 × 3 matrix times another 2 × 3 matrix is undefined. Multiplication is associative but not commutative, the identity matrix leaves every matrix unchanged, and the transpose reverses the order of a product: (AB)ᵀ = BᵀAᵀ." },
      { title: "Measure vectors and transform the plane", body: "Norms measure size: the L1 norm adds absolute values, the L2 norm is the Euclidean length and the L∞ norm takes the largest absolute entry. A distance is the norm of a difference, as used by nearest-neighbour methods. A 2 × 2 matrix transforms points of the plane: scaling stretches the axes, a rotation by θ uses cos θ and sin θ, and a shear slides points sideways. Applying B and then A equals applying the single matrix AB, so matrix multiplication composes transformations, rightmost first." },
    ],
    example: { title: "A rectangular product by hand", body: "Multiply the 2 × 3 matrix A = [[1, 2, 3], [4, 5, 6]] by the 3 × 2 matrix B = [[7, 8], [9, 10], [11, 12]]. The inner sizes match, so AB is 2 × 2. Its top-left entry is row 1 of A dotted with column 1 of B: 1·7 + 2·9 + 3·11 = 7 + 18 + 33 = 58, and the other entries are 64, 139 and 154. Reversing the order gives BA, a 3 × 3 matrix whose top-left entry is 7·1 + 8·4 = 39, so AB ≠ BA." },
    code: {
      title: "The classic triple loop",
      source: "def matrix_multiply(A, B):\n    m, n, p = len(A), len(A[0]), len(B[0])\n    if len(B) != n:\n        raise ValueError(f\"cannot multiply {m}x{n} by {len(B)}x{p}\")\n    C = [[0] * p for _ in range(m)]\n    for i in range(m):\n        for j in range(p):\n            C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))\n    return C\n\n\nA = [[1, 2, 3], [4, 5, 6]]\nB = [[7, 8], [9, 10], [11, 12]]\nprint(matrix_multiply(A, B))  # [[58, 64], [139, 154]]\nprint(matrix_multiply(B, A)[0])  # [39, 54, 69]",
      explanation: "This short version checks only that the inner sizes agree. The notebook's version also rejects empty matrices and ragged rows before it computes anything, and its error message names both shapes.",
    },
    notebook: { path: "math-foundations/02_linear_algebra_from_scratch.ipynb", title: "Linear algebra from scratch: vectors, matrices and transformations" },
    practice: [
      "Run the notebook from the top, check the Step 3 products against your own hand calculation of AB and BA, and read the ValueError message produced by multiplying A by itself.",
      "Complete Exercise 1 (cosine_similarity and normalise) and Exercise 2 (outer_product and trace), rerunning each check cell until it reports that all tests passed.",
      "Complete Exercise 3 (linear_layer and batch_linear_layer) and Exercise 4 (compose_transformations and determinant_2x2), then explain why scaling and then rotating gives a different matrix from rotating and then scaling.",
    ],
    references: [nestedListComprehensions, zipFunction, mathModule, mmlBook],
    quiz: {
      question: "A has shape 2 × 3 and B has shape 3 × 4. Which statement is true?",
      options: ["AB is 2 × 4 and BA is not defined", "AB is 3 × 3, because the outer sizes cancel", "AB and BA are both defined, with shape 2 × 4"],
      correctIndex: 0,
      explanation: "The inner sizes of A (2 × 3) and B (3 × 4) are both 3, so AB exists and takes the outer sizes, 2 × 4. For BA the inner sizes are 4 and 2, which do not match, so that product is undefined.",
    },
  },
  {
    id: "probability-statistics-from-scratch", title: "Compute probabilities and estimate parameters from scratch", minutes: 55,
    intro: "Data are noisy and models are uncertain. This lesson implements the core of probability and statistics with the standard library: descriptive statistics, the Bernoulli, binomial and normal distributions, reproducible sampling, conditional probability and Bayes' theorem, covariance and correlation, and maximum likelihood estimation, the principle behind the most common training losses.",
    sections: [
      { title: "Summarise data and choose the right variance", body: "The mean is pulled by outliers and the median is not. The population variance divides the sum of squared deviations by n; the sample variance divides by n − 1, which corrects for measuring deviations from the sample mean and makes the estimate unbiased. Python's statistics module offers both, as pvariance and variance, while NumPy's var divides by n unless you pass ddof=1. Covariance shows whether two variables move together, and the correlation r rescales it to lie between −1 and 1, measuring only straight-line association." },
      { title: "Describe uncertainty with distributions and seeded samples", body: "A Bernoulli variable is 1 with probability p. A binomial count of successes in n trials has P(K = k) = C(n, k) pᵏ(1 − p)ⁿ⁻ᵏ, where math.comb computes C(n, k). The normal density is a density, not a probability: probabilities are areas under it, so it can exceed 1 when σ is small. Create a generator with random.Random(seed) and pass it into your functions, so that every experiment is reproducible. Sample averages then settle towards their expected values as the sample grows." },
      { title: "Update beliefs and fit parameters", body: "Conditional probability, P(A | B) = P(A and B) / P(B), restricts attention to the outcomes in which B happened. Bayes' theorem reverses a condition, P(H | E) = P(E | H) P(H) / P(E), and shows why a rare condition produces many false positives. Maximum likelihood chooses the parameters that make the observed data most probable. Working with log-likelihoods avoids underflow and gives p̂ = k/n for coin flips, and μ̂ = x̄ and σ̂² = Σ(xᵢ − x̄)²/n for Gaussian data. Because that variance divides by n, it is slightly biased." },
    ],
    example: { title: "Bayes' theorem for a screening test", body: "A condition affects 1% of people. A test detects it 95% of the time and wrongly flags 5% of healthy people. Out of 10,000 people, 100 have the condition and 95 of them test positive; 9,900 do not, and 495 of them also test positive. Only 95 of the 590 positive results are true, so P(condition | positive) = 0.95 × 0.01 / (0.95 × 0.01 + 0.05 × 0.99) = 0.0095 / 0.059 ≈ 0.161. A second, independent positive result raises the probability to about 0.785." },
    code: {
      title: "Two variances and a posterior",
      source: "import statistics\n\ndata = [2, 4, 4, 4, 5, 5, 7, 9]\nprint(statistics.pvariance(data))  # 4, dividing by n\nprint(statistics.variance(data))  # 4.571428571428571, dividing by n - 1\n\nprior, sensitivity, false_positive_rate = 0.01, 0.95, 0.05\nevidence = sensitivity * prior + false_positive_rate * (1 - prior)\nprint(round(sensitivity * prior / evidence, 3))  # 0.161",
      explanation: "The same data give two different variances because the functions answer different questions: pvariance describes these eight values, while variance estimates the spread of the population they were drawn from. The last line applies Bayes' theorem to the screening example.",
    },
    notebook: { path: "math-foundations/03_probability_statistics_from_scratch.ipynb", title: "Probability and statistics from scratch: distributions, Bayes and likelihood" },
    practice: [
      "Run the notebook from the top. In Step 6, compare the average maximum likelihood variance with the average sample variance over 20,000 samples of size 5, and explain why their ratio is about 0.8.",
      "Complete Exercise 1 (standardise and covariance_matrix) and Exercise 2 (binomial_cdf and estimate_dice_probability), drawing every random number from the generator that the check passes in.",
      "Complete Exercise 3 (posterior and repeated_test_posterior) and Exercise 4 (exponential_log_likelihood and exponential_mle), deriving the exponential rate estimate on paper before you code it.",
    ],
    references: [statisticsModule, randomReproducibility, fractionsModule, mmlBook],
    quiz: {
      question: "A condition affects 1% of people. A test detects 95% of cases and wrongly flags 5% of healthy people. Roughly what share of positive results are true positives?",
      options: ["About 95%", "About 16%", "About 50%"],
      correctIndex: 1,
      explanation: "Out of 10,000 people, 95 of the 100 with the condition test positive, but so do 495 of the 9,900 without it. Only 95 of the 590 positives, about 16%, have the condition: the low base rate matters as much as the accuracy of the test.",
    },
  },
];
