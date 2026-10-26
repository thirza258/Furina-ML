import type { CourseLesson } from "./types";

const mathModule = { title: "Python documentation: math — mathematical functions", url: "https://docs.python.org/3/library/math.html" };
const mathIsFinite = { title: "Python documentation: math.isfinite", url: "https://docs.python.org/3/library/math.html#math.isfinite" };
const floatingPoint = { title: "Python tutorial: floating-point arithmetic, issues and limitations", url: "https://docs.python.org/3/tutorial/floatingpoint.html" };
const floatInfo = { title: "Python documentation: sys.float_info", url: "https://docs.python.org/3/library/sys.html#sys.float_info" };
const linearRegression = { title: "Python documentation: statistics.linear_regression", url: "https://docs.python.org/3/library/statistics.html#statistics.linear_regression" };
const randomModule = { title: "Python documentation: random — generate pseudo-random numbers", url: "https://docs.python.org/3/library/random.html" };

/** Lessons 4 and 5 of the Mathematics for Machine Learning from First Principles course. */
export const mathFoundationsLessonsB: CourseLesson[] = [
  {
    id: "calculus-from-scratch", title: "Compute derivatives and gradients from scratch", minutes: 55,
    intro: "Training a model rests on one question: if a parameter changes slightly, how does the loss change? This lesson answers it in plain Python. You estimate derivatives from function values, measure how accurate the estimates are, and use them to check gradients that you derive by hand.",
    sections: [
      { title: "Estimate a derivative from function values", body: "The derivative f′(x) is the limit of the slope (f(x + h) − f(x)) / h as h shrinks to zero. A computer cannot take that limit, so it picks a small h. This forward difference has a truncation error proportional to h. The central difference, (f(x + h) − f(x − h)) / 2h, samples both sides of x, so the first-order errors cancel, leaving an error proportional to h². For the same two evaluations of f it is usually far more accurate. A partial derivative varies one input at a time and holds the rest fixed." },
      { title: "Choose the step size with evidence", body: "Smaller is not always better. A Python float holds about 16 significant digits, so each computed value of f carries a rounding error of roughly machine epsilon, 2.2 × 10⁻¹⁶, times its size, and the difference quotient divides that error by h. As h shrinks, truncation error falls but rounding error grows, so the total error is smallest in between: near h = 10⁻⁸ for forward differences and near 10⁻⁵ for central differences. The notebook sweeps h from 10⁻¹ to 10⁻¹⁵ and shows both sides of this trade-off." },
      { title: "Apply the chain rule and check every gradient", body: "A loss is a chain of simple operations. The chain rule multiplies their local derivatives along a path and adds the contributions of different paths. A computational graph organises this as a forward pass followed by a backward pass, which is how backpropagation works. Finite differences are too slow for training but ideal for checking: compare an analytic gradient a with a numerical one b through the relative error ‖a − b‖ / (‖a‖ + ‖b‖) at several random points. Correct gradients usually give 10⁻⁸ or less, so a tolerance of 10⁻⁶ exposes genuine bugs." },
    ],
    example: { title: "Two difference quotients and a chain rule, by hand", body: "For f(x) = x² at x = 3 with h = 0.1, the forward difference is (3.1² − 3²) / 0.1 = (9.61 − 9) / 0.1 = 6.1, an error of exactly h. The central difference is (3.1² − 2.9²) / 0.2 = (9.61 − 8.41) / 0.2 = 6.0, which is exact for any quadratic. For a composite function, y = sin(x²) has dy/dx = cos(x²) · 2x, which at x = 1 is 2 cos 1 ≈ 1.0806046117. A central difference with h = 10⁻⁵ gives 1.0806046115, within about 2 × 10⁻¹⁰." },
    code: { title: "Forward and central differences side by side", source: "import math\n\n\ndef forward(f, x, h):\n    return (f(x + h) - f(x)) / h\n\n\ndef central(f, x, h):\n    return (f(x + h) - f(x - h)) / (2 * h)\n\n\nexact = math.cos(1.0)  # the derivative of sin at x = 1\nfor h in (1e-2, 1e-5, 1e-8, 1e-11):\n    forward_error = abs(forward(math.sin, 1.0, h) - exact)\n    central_error = abs(central(math.sin, 1.0, h) - exact)\n    print(f\"h = {h:.0e}: forward error {forward_error:.1e}, central error {central_error:.1e}\")", explanation: "At h = 10⁻⁵ the central error is about 10⁻¹¹, more than a hundred thousand times smaller than the forward error. At h = 10⁻¹¹ both estimates are worse than at larger steps, because rounding error now dominates. The notebook extends this to a full sweep of h." },
    notebook: { path: "math-foundations/04_calculus_from_scratch.ipynb", title: "Calculus from scratch: derivatives, gradients and the chain rule" },
    practice: [
      "Run the notebook from the top. In the Step 4 table, find the step size with the smallest error for each method and explain why the error grows again below it.",
      "Complete Exercise 1 (second_derivative, a central difference for curvature) and Exercise 2 (grad_g, a gradient you derive by hand), and rerun each check cell until it reports correct.",
      "Complete Exercise 3 (gradient_check) so that it rejects every broken gradient in its check, then attempt the stretch Exercise 4 (directional_derivative) and confirm that −∇f gives the steepest descent.",
    ],
    references: [mathModule, floatingPoint, floatInfo],
    quiz: {
      question: "A forward-difference estimate improves as h shrinks from 10⁻² to 10⁻⁸, then becomes worse at h = 10⁻¹². What causes the deterioration?",
      options: ["Truncation error grows as h shrinks", "Python cannot evaluate f at such close points", "Rounding error is divided by a tiny h"],
      correctIndex: 2,
      explanation: "Truncation error shrinks with h. Each computed value of f carries a rounding error of about machine epsilon times its size, and the difference quotient divides the difference of two such values by h. Below about 10⁻⁸ that magnified rounding error outweighs the truncation error. Python evaluates f correctly at nearby points; the precision is lost in the subtraction and division.",
    },
  },
  {
    id: "optimization-from-scratch", title: "Minimise a loss with gradient descent from scratch", minutes: 60,
    intro: "Training a model is a search for the parameters that make a loss small. This lesson builds that search, gradient descent, in plain Python, and shows how the learning rate, the stopping rule and the shape of the loss decide whether it succeeds.",
    sections: [
      { title: "Step against the gradient", body: "Gradient descent repeats θ ← θ − η∇L(θ), stepping against the gradient, which points in the direction of steepest increase. The learning rate η sets the step size. On L(x) = (x − 3)² + 1 each step multiplies the distance to the minimum by 1 − 2η: η = 0.001 crawls, η = 0.1 converges quickly, η = 0.9 overshoots but still converges, and η = 1.5 doubles the distance at every step until the loss overflows to infinity. In general, a direction whose curvature is a is stable only if η < 2/a." },
      { title: "Decide when to stop", body: "Gradient descent rarely lands exactly on a minimum, so it needs stopping rules. Stop when the gradient norm falls below a tolerance, when the change in loss becomes negligible, or when an iteration cap is reached, and stop at once if math.isfinite reports that the loss has become inf or nan. The rules measure different things: a tiny change in loss can mean that the steps are tiny rather than that the minimum is close. Check the gradient norm before trusting convergence, and record which rule fired." },
      { title: "Expect valleys and overflow", body: "A non-convex loss can have several valleys, and gradient descent ends in whichever one its starting point drains into, so try several seeded starting points. A convex loss, such as the mean squared error of a linear model, has no false valleys: every local minimum is global. Floating-point limits matter as well. math.exp(1000) raises OverflowError, and log(1 + eˣ) must be rewritten as max(x, 0) + log(1 + e^(−|x|)) to stay accurate for large and very negative x." },
    ],
    example: { title: "Fit a line by formula and by descent", body: "For the points (1, 3.1), (2, 4.9), (3, 7.2), (4, 8.8), (5, 11.1) and (6, 12.9), the means are x̄ = 3.5 and ȳ = 8. The sums Σ(x − x̄)(y − ȳ) = 34.6 and Σ(x − x̄)² = 17.5 give the least-squares slope w = 34.6 / 17.5 ≈ 1.977143 and intercept b = 8 − 3.5w = 1.08. Gradient descent on the mean squared error, starting from w = b = 0 with η = 0.05, reaches the same line within 3,000 steps." },
    code: { title: "Gradient descent against the closed form", source: "xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]\nys = [3.1, 4.9, 7.2, 8.8, 11.1, 12.9]\nn = len(xs)\n\nw = b = 0.0\nfor step in range(3000):\n    residuals = [w * x + b - y for x, y in zip(xs, ys)]\n    grad_w = 2 / n * sum(r * x for r, x in zip(residuals, xs))\n    grad_b = 2 / n * sum(residuals)\n    w, b = w - 0.05 * grad_w, b - 0.05 * grad_b\n\nx_mean, y_mean = sum(xs) / n, sum(ys) / n\nsxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))\nsxx = sum((x - x_mean) * (x - x_mean) for x in xs)\nprint(f\"gradient descent: w = {w:.6f}, b = {b:.6f}\")\nprint(f\"closed form:      w = {sxy / sxx:.6f}, b = {y_mean - sxy / sxx * x_mean:.6f}\")", explanation: "Both lines print w = 1.977143 and b = 1.080000. Change both learning rates from 0.05 to 0.07 and the same loop diverges to astronomically large values, because 0.07 is above this loss's stability limit of about 0.063." },
    notebook: { path: "math-foundations/05_optimization_from_scratch.ipynb", title: "Optimisation from scratch: gradient descent and its pitfalls" },
    practice: [
      "Run the notebook from the top. Before reading the Step 4 verdicts, predict each learning rate's outcome from the factor 1 − 2η, then compare.",
      "Complete Exercise 1 (gradient_descent_with_stopping) and Exercise 2 (best_learning_rate), and explain why a learning rate of 10⁻¹³ triggers false convergence.",
      "Complete Exercise 3 (fit_through_origin) and compare it with Σxy / Σx², then attempt the stretch Exercise 4 (momentum_descent) and compare its progress on the bowl with plain gradient descent.",
    ],
    references: [mathIsFinite, linearRegression, randomModule],
    quiz: {
      question: "Gradient descent on L(x) = (x − 3)² multiplies the distance to the minimum by 1 − 2η at every step. What happens with η = 1.5?",
      options: ["Each step doubles the distance, so it diverges", "It converges, but more slowly than with η = 0.1", "It lands exactly on the minimum after one step"],
      correctIndex: 0,
      explanation: "With η = 1.5 the factor is 1 − 3 = −2: each step jumps across the minimum and doubles the distance, so the loss grows until it overflows to inf, which math.isfinite detects. The rate that lands on the minimum in a single step is 0.5.",
    },
  },
];
