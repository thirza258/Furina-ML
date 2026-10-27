import type { Course } from "./types";

const survey = { title: "Baydin et al. (2018): Automatic differentiation in machine learning, a survey (JMLR)", url: "https://jmlr.org/papers/v18/17-468.html" };
const rumelhart = { title: "Rumelhart, Hinton and Williams (1986): Learning representations by back-propagating errors", url: "https://www.nature.com/articles/323533a0" };
const autogradMechanics = { title: "PyTorch documentation: autograd mechanics", url: "https://docs.pytorch.org/docs/stable/notes/autograd.html" };
const autogradTutorial = { title: "PyTorch tutorial: automatic differentiation with torch.autograd", url: "https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html" };
const tensorBackward = { title: "PyTorch documentation: torch.Tensor.backward", url: "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html" };
const optimisation = { title: "PyTorch tutorial: optimizing model parameters", url: "https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html" };
const numericTypes = { title: "Python reference: emulating numeric types (__add__, __radd__ and friends)", url: "https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types" };
const namingAndBinding = { title: "Python reference: naming and binding (how closures see outer variables)", url: "https://docs.python.org/3/reference/executionmodel.html#naming-and-binding" };
const graphlib = { title: "Python documentation: graphlib, topological sorting of graph-like structures", url: "https://docs.python.org/3/library/graphlib.html" };
const linearRegression = { title: "Python documentation: statistics.linear_regression", url: "https://docs.python.org/3/library/statistics.html#statistics.linear_regression" };
const randomModule = { title: "Python documentation: random, seeded pseudo-random numbers", url: "https://docs.python.org/3/library/random.html" };

export const autogradCourse: Course = {
  id: "autograd",
  title: "Build Your Own Autograd",
  level: "Advanced",
  track: "Deep learning",
  description: "Build a tiny automatic differentiation engine yourself, one typed cell at a time, with nothing but the Python standard library: computational graphs, a Value class whose operations record their own backward rules, a topological sort and backward(), and finally a small neural network trained by your engine. You will see what frameworks such as PyTorch do inside loss.backward(). The engine is an educational model and is not production-equivalent: it handles one number at a time in pure Python.",
  prerequisites: [
    "Derivatives and the chain rule, at the level of Mathematics for Machine Learning from First Principles",
    "Python classes and methods; operator methods and closures are explained as you go",
    "The forward pass, loss and backpropagation of a small network, as in Neural Networks from Scratch",
  ],
  prerequisiteCourses: ["math-foundations", "neural-networks"],
  practiceMode: "notebook",
  dependencyTier: 0,
  nextCourseId: "base-models",
  outcomes: [
    "Represent an expression as a computational graph and back-propagate through it by hand",
    "Implement a scalar Value class whose operations record their backward rules as closures",
    "Write a topological sort and backward(), and explain why gradients accumulate and must be reset",
    "Verify any gradient against central differences and explain why reverse mode is cheap",
    "Train a small neural network with your own engine and map each piece onto PyTorch",
  ],
  capstone: {
    title: "Extend your engine: a new operation and a momentum optimiser",
    brief: "Add one new operation to your Value engine, such as sigmoid, a square root or a power with a Value exponent, together with its backward rule. Then write a tensor-free momentum optimiser that keeps a velocity for every parameter, and compare it with plain gradient descent on the two-moons network from notebook 04, using the same seeds, data and number of steps.",
    deliverables: [
      "The new operation with its backward rule, a practice-style check against a reference and a gradient check below 10⁻⁶",
      "A Momentum class with step() and zero_grad() whose velocities persist between steps and are never reset by zero_grad",
      "Loss curves and training accuracy for gradient descent and for momentum with β = 0.9, over at least three seeds",
      "A short note on what would have to change for the same network to run on tensors, as PyTorch does",
    ],
    rubric: [
      "Every new rule multiplies by out.grad and accumulates with +=, and its gradient check passes, including on a value used twice",
      "The update follows v ← βv + ∇L and p ← p − ηv, and β = 0 reproduces plain gradient descent exactly",
      "Comparisons use the same seeds, data and step counts, and report the spread across seeds rather than one run",
      "Claims stay within what was measured on the generated data, and the engine is not presented as production-ready",
    ],
  },
  references: [survey, rumelhart, autogradMechanics, autogradTutorial],
  lessons: [
    {
      id: "computational-graphs", title: "Computational graphs and the chain rule, by hand", minutes: 50,
      intro: "Every expression a model computes is a graph of simple operations. This lesson evaluates such a graph forwards, works out each operation's local derivative, and applies the chain rule as a single backwards walk that gives the derivative of the output with respect to every input, checked against central differences.",
      sections: [
        { title: "An expression is a graph", body: "Write L = (tanh(wx + b) − t)² one operation at a time and name every intermediate result: p = wx, z = p + b, y = tanh z, e = y − t and L = e · e. Each result is a node, and an edge joins each argument to the node that uses it. The inputs are the leaves and L is the output. The graph has no cycles, so its nodes can be listed with each one after its arguments. Evaluating them in that order is the forward pass, which keeps every intermediate value." },
        { title: "Local derivatives and the backwards walk", body: "Each operation has a simple local derivative: addition gives 1 to both arguments, subtraction 1 and −1, multiplication gives each argument the other one, and tanh gives 1 − out², using its own output. To differentiate the output, set its gradient to 1 and visit the nodes in reverse order. Each node adds its gradient times the local derivative into the gradient of each argument. Because the walk runs backwards, a node passes its gradient on only after every node that uses it has finished." },
        { title: "Why contributions add, and how to check them", body: "A value used in several places affects the output along several paths, and the multivariable chain rule adds the contributions. That is why the walk uses +=: in L = e · e, each use of e contributes e, so the gradient is 2e. To check a gradient without any derivative rules, nudge one input by ±h and take the central difference (L(x + h) − L(x − h)) / 2h. With h near 10⁻⁶ it agrees to about ten digits, but it costs two forward passes per input." },
      ],
      example: { title: "A backwards walk with numbers", body: "With w = 0.5, x = 2, b = −0.3 and t = 1: p = 1, z = 0.7, y = tanh 0.7 ≈ 0.604368, e ≈ −0.395632 and L ≈ 0.156525. Walking back: ∂L/∂e = 2e ≈ −0.791264; ∂L/∂z = ∂L/∂e · (1 − y²) ≈ −0.791264 × 0.634740 ≈ −0.502247; then ∂L/∂w = ∂L/∂z · x ≈ −1.004494 and ∂L/∂x = ∂L/∂z · w ≈ −0.251123. A central difference with h = 10⁻⁶ reproduces ∂L/∂w to about ten decimal places." },
      code: { title: "The chain rule against a central difference", source: "import math\n\n\ndef loss(w, x=2.0, b=-0.3, t=1.0):\n    return (math.tanh(w * x + b) - t) ** 2\n\n\nw, h = 0.5, 1e-6\ny = math.tanh(w * 2.0 - 0.3)\nchain_rule = 2 * (y - 1.0) * (1 - y * y) * 2.0   # dL/dy · dy/dz · dz/dw\ncentral = (loss(w + h) - loss(w - h)) / (2 * h)\nprint(f\"{chain_rule:.10f} {central:.10f}\")       # both -1.0044937397", explanation: "The product of local derivatives and the central difference agree to ten decimal places. The notebook builds the general version: a forward pass, a backwards walk and a numerical check for any graph stored as a dictionary." },
      notebook: { path: "autograd/01_computational_graphs.ipynb", title: "Computational graphs: values forwards, gradients backwards" },
      practice: [
        "Type forward_values and local_derivatives, run each check cell until it shows ✅, and compare the drawn graph with the hand calculation.",
        "Type backward_gradients and numerical_gradient, and confirm that every gap in the comparison table is below 10⁻⁹.",
        "Run the 🧪 cells with w = 3 and t = −1, and with step sizes from 10⁻² to 10⁻¹²; explain the vanishing gradient and the best step size, then try Exercises 1 and 2.",
      ],
      references: [survey, rumelhart],
      quiz: {
        question: "In L = e · e the value e is used twice. What is ∂L/∂e?",
        options: ["e, the local derivative of one use", "2e, the sum of both contributions", "e², the value of L itself"],
        correctIndex: 1,
        explanation: "Each use of e contributes its local derivative, the other factor e, times ∂L/∂L = 1. The chain rule adds the two contributions, giving 2e, the derivative of e².",
      },
    },
    {
      id: "scalar-autodiff", title: "A scalar autodiff engine: the Value class", minutes: 60,
      intro: "An autodiff engine records the graph while ordinary code runs. Here you build Value, a number that remembers the values it came from and carries a backward rule for the operation that made it, and you write and check the rule of every operation the engine supports.",
      sections: [
        { title: "A number that remembers its history", body: "A Value stores data, the number itself; grad, the derivative of the final output with respect to it; _children, the values it was computed from; _op, the operation's name; label, for printing; and _backward, a function that passes its gradient on to its children. Python's operator methods turn a + b into a.__add__(b) and 2 + a into a.__radd__(2), so ordinary arithmetic on Values builds the graph as a side effect, without a separate description of the graph." },
        { title: "Backward rules as closures", body: "Every operation follows one pattern: compute the forward value, create the output with its children, then define an inner function that adds local derivative × out.grad into each child's grad, and attach it as out._backward. The inner function is a closure: it still sees its inputs and output after the operation has returned. It multiplies by out.grad because of the chain rule, and it adds with += because a child can receive gradient from several uses." },
        { title: "The rules, and how to trust them", body: "Addition passes out.grad to both children. Multiplication passes b · out.grad to a and a · out.grad to b. A constant power aⁿ passes n · aⁿ⁻¹, exp passes its own output, log passes 1/a and tanh passes 1 − out², each multiplied by out.grad. Subtraction, negation and division need no rules of their own, because they are built from these. Each rule is checked twice: against a reference on test cases, and against a central difference of the forward pass." },
      ],
      example: { title: "One rule, step by step", body: "Take out = a · b with a = 2 and b = −3, so out.data = −6, and suppose the backward pass delivers out.grad = 0.5. The closure adds b · out.grad = −1.5 into a.grad and a · out.grad = 1.0 into b.grad. For a · a with a = 1.5 and out.grad = 1, the single node receives 1.5 twice, 3.0 in total, which is the derivative of a² at 1.5. With = instead of +=, it would keep only 1.5." },
      code: { title: "A closure that remembers its inputs", source: "class Node:\n    def __init__(self, data, children=()):\n        self.data, self.grad, self._children = data, 0.0, children\n        self._backward = None\n\n\ndef mul(a, b):\n    out = Node(a.data * b.data, (a, b))\n\n    def _backward():\n        a.grad += b.data * out.grad\n        b.grad += a.data * out.grad\n\n    out._backward = _backward\n    return out\n\n\nx = Node(1.5)\nsquare = mul(x, x)\nsquare.grad = 1.0\nsquare._backward()\nprint(x.grad)   # 3.0 = d(x²)/dx at 1.5: both uses add their share", explanation: "The closure still sees a, b and out after mul has returned. Because x is both arguments, it receives 1.5 twice. The notebook builds the full Value class, with every operator and a check for each rule." },
      notebook: { path: "autograd/02_scalar_autodiff.ipynb", title: "A scalar autodiff engine: the Value class" },
      practice: [
        "Read the Value class and the worked value_relu, then type value_add and value_mul; each check should show ✅ and a central-difference gap near 10⁻¹⁰.",
        "Type value_pow, value_exp, value_log and value_tanh, then use the 🧪 cells to watch tanh saturate and log fail outside its domain.",
        "In Put it together, back-propagate the neuron by calling each _backward in reverse order; then swap two nodes in that order and explain which gradients break (Exercise 3).",
      ],
      references: [numericTypes, namingAndBinding, autogradMechanics],
      quiz: {
        question: "What does value_tanh's backward closure add into a.grad?",
        options: ["1 − a.data², computed from the input", "out.grad alone, with no local factor", "(1 − out.data²) × out.grad"],
        correctIndex: 2,
        explanation: "The derivative of tanh a is 1 − tanh² a, which is 1 − out², and the chain rule multiplies it by the gradient arriving from above, out.grad. Using the input, 1 − a², is a common bug.",
      },
    },
    {
      id: "reverse-mode-autodiff", title: "Reverse-mode autodiff: topological order and backward()", minutes: 55,
      intro: "One call to backward() should fill in every gradient of any expression. This lesson finds the order in which to run the backward rules, explains why gradients must accumulate and be reset between steps, verifies whole expressions numerically and measures why reverse mode is so cheap.",
      sections: [
        { title: "Topological order", body: "A node may pass its gradient on only when that gradient is complete, after every node that uses it has added its share. A topological order lists each node after all of its children, and running the backward rules in the reverse of that order guarantees completeness. A depth-first search produces one: visit the children, then append the node, keeping a set of visited nodes so that a value reached along two paths is listed once. backward() sets the root's gradient to 1, then runs every rule in reverse order." },
        { title: "Accumulate, then reset", body: "Gradients accumulate. Inside one graph that is essential: a value used twice must receive the sum of both contributions, and a rule written with = silently keeps only the last one. Across training steps it is a hazard. Every step builds a new graph from the same parameters, so their grad fields still hold the previous step's gradients. A training loop therefore resets the parameters' gradients before each forward pass, which is what optimizer.zero_grad() does in PyTorch." },
        { title: "Verification and cost", body: "A gradient check compares backward() with central differences for a whole expression and reports the largest gap: about 10⁻⁹ for correct code, about the size of the gradient for a broken rule. Reverse mode obtains all n partial derivatives from one forward and one backward pass, while central differences need 2n forward passes and forward mode one pass per input. For a loss with millions of parameters that difference decides everything; the price is keeping the whole graph in memory until backward() runs." },
      ],
      example: { title: "Fitting a line with the engine", body: "Six points with x from −2 to 3, starting from w = c = 0, give a mean squared error of 15.785. Each step resets both gradients, rebuilds the loss, calls backward() and moves w and c by 0.1 times their gradients. After 150 steps the loss is 0.018, with w = 2.00286 and c = 1.01524, matching the least-squares answer from statistics.linear_regression to five decimal places." },
      code: { title: "A topological order from the standard library", source: "from graphlib import TopologicalSorter\n\n# each node -> the nodes it was computed from (its children)\ngraph = {\"u\": [\"a\", \"b\"], \"v\": [\"u\", \"a\"], \"out\": [\"u\", \"v\"]}\norder = list(TopologicalSorter(graph).static_order())\nprint(order)                   # ['a', 'b', 'u', 'v', 'out']: children first\nprint(list(reversed(order)))   # the order in which backward() runs the rules", explanation: "Python's graphlib finds a topological order for a graph given as a dictionary of predecessors. The notebook writes the same search by hand for Value objects and runs every _backward in the reversed order." },
      notebook: { path: "autograd/03_reverse_mode_autodiff.ipynb", title: "Reverse-mode autodiff: backward() for any expression" },
      practice: [
        "Type topological_order and backward_pass, and check that loss.backward() reproduces the neuron gradients from notebook 02.",
        "Run the = versus += experiment and the zeroing demonstration, then type zero_grad and gradient_check, whose check includes a broken case that must fail.",
        "Use the graph-size 🧪 cell to compare reverse mode with central differences as the number of inputs grows, then finish Put it together's line fit and Exercise 1, Kahn's algorithm.",
      ],
      references: [graphlib, linearRegression, tensorBackward, survey],
      quiz: {
        question: "A training loop calls backward() on each step's new loss but never resets the gradients. What does a parameter's grad hold?",
        options: ["The sum of every step's gradient so far", "Only the latest gradient, because each graph is new", "The latest gradient, doubled on every step"],
        correctIndex: 0,
        explanation: "backward() adds into grad. The intermediate nodes are new on every step, but the parameters are the same Value objects, so their gradients pile up unless zero_grad resets them before each forward pass.",
      },
    },
    {
      id: "differentiable-neural-network", title: "A neural network trained by your own engine", minutes: 60,
      intro: "With a working engine, a neural network takes a few lines: neurons whose weights are Values, layers of neurons and a multilayer perceptron. This lesson trains one on two interleaved half-moons with a loss built from Values, then maps every piece onto PyTorch and states what a production framework adds.",
      sections: [
        { title: "Networks built from Values", body: "A neuron holds n weights and a bias as Values and computes tanh(b + Σ wᵢxᵢ), or the plain sum for a linear output unit. A layer is a list of neurons that read the same inputs, and a multilayer perceptron chains layers: MLP(2, [8, 1]) has 33 parameters. Weights start from a normal distribution with standard deviation 1/√n, which keeps tanh units responsive, and random starting values make the neurons differ, so that they can learn different features." },
        { title: "A loss and a training step", body: "The mean squared error of the scores against targets of ±1 is itself a Value, so loss.backward() reaches every parameter. A training step always runs in the same order: reset the parameters' gradients, run the forward pass to build a new graph, compute the loss, call backward(), then move each parameter's data a small step against its gradient. In the notebook, 200 full-batch steps with a learning rate of 0.2 take the loss from about 2.19 to 0.11 and the training accuracy to about 97%." },
        { title: "From your engine to PyTorch", body: "PyTorch uses the same ideas. Parameters are tensors with requires_grad=True, loss.backward() runs the autograd engine, gradients appear in .grad, and optimizer.zero_grad() and optimizer.step() replace the hand-written reset and update. The educational engine is not production-equivalent. Frameworks put a whole array in one graph node, run vectorised kernels on GPUs, provide hundreds of tested operations and manage memory, while your engine handles one Python float at a time." },
      ],
      example: { title: "Counting parameters and graph nodes", body: "In MLP(2, [8, 1]) the hidden layer has 8 × (2 + 1) = 24 parameters and the output layer 1 × (8 + 1) = 9, so 33 in all. One training step on 60 points builds a graph of about 4,700 Values, and backward() runs each node's rule once. A tensor framework would describe the same step with a handful of array operations." },
      code: { title: "Count the parameters of an MLP", source: "def count_parameters(n_in, sizes):\n    total = 0\n    for width in sizes:\n        total += width * (n_in + 1)   # one weight per input, plus a bias, for each neuron\n        n_in = width\n    return total\n\n\nprint(count_parameters(2, [8, 1]))       # 33\nprint(count_parameters(2, [16, 16, 1]))  # 337", explanation: "Each neuron has one weight per input plus a bias. The engine also creates a Value for every intermediate result, so the graph of one training step is much larger than the number of parameters." },
      notebook: { path: "autograd/04_differentiable_neural_network.ipynb", title: "A differentiable neural network, trained by your engine" },
      practice: [
        "Type neuron_forward, layer_forward and mlp_forward, and check that each shows ✅, with gradients that the engine computed for you.",
        "Type mse_loss and train_step, then run Put it together and confirm that the loss falls below 30% of its starting value and the accuracy reaches at least 85%.",
        "Use the 🧪 cells to compare learning rates of 0.02, 0.2 and 0.6, and a straight line ([1]) with a hidden layer; then write hinge_loss in Exercise 1.",
      ],
      references: [autogradTutorial, optimisation, randomModule],
      quiz: {
        question: "Why does the update use param.data -= learning_rate * param.grad instead of param = param - learning_rate * param.grad?",
        options: ["Both forms update the model in exactly the same way", "The second form is faster but less precise", "The second only rebinds a local name"],
        correctIndex: 2,
        explanation: "param = param - … builds a new Value node and binds the loop variable to it, leaving the parameter stored in the model unchanged. Changing param.data moves the parameter itself, outside the graph.",
      },
    },
  ],
};
