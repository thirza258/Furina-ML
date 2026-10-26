import type { Course } from "./types";

const numpyQuickstart = { title: "NumPy: the absolute basics for beginners", url: "https://numpy.org/doc/stable/user/absolute_beginners.html" };
const numpyBroadcasting = { title: "NumPy: broadcasting", url: "https://numpy.org/doc/stable/user/basics.broadcasting.html" };
const rumelhart = { title: "Rumelhart, Hinton and Williams (1986): Learning representations by back-propagating errors", url: "https://www.nature.com/articles/323533a0" };
const heInit = { title: "He et al. (2015): Delving deep into rectifiers (He initialisation)", url: "https://arxiv.org/abs/1502.01852" };
const deepLearningBook = { title: "Goodfellow, Bengio and Courville: Deep Learning (free online book)", url: "https://www.deeplearningbook.org/" };

export const neuralNetworksCourse: Course = {
  id: "neural-networks", title: "Neural Networks from Scratch", level: "Intermediate", track: "Deep learning",
  description: "Build a neural network yourself, one typed cell at a time: neurons, layers, the forward pass, loss functions, gradients, backpropagation and gradient descent, all in NumPy. By the end your own network learns, is evaluated honestly, and is saved and reloaded, and you know what every line does.",
  prerequisites: ["Python functions, loops and lists", "Vectors, matrices and derivatives at the level of Mathematics for Machine Learning from First Principles"],
  prerequisiteCourses: ["math-foundations"], practiceMode: "notebook", dependencyTier: 1, nextCourseId: "base-models",
  outcomes: [
    "Implement a neuron, dense layers and a full forward pass in NumPy",
    "Derive and implement backpropagation, and verify it with a gradient check",
    "Train a network with mini-batch gradient descent and read its learning curves",
    "Run an honest train/validation/test protocol and save and reload your own model",
  ],
  capstone: {
    title: "Your own neural network, end to end",
    brief: "In notebook 12, package the functions you wrote into a NeuralNetwork class and run a complete pipeline: split, standardise, train with the epoch chosen on validation data, test once, save, reload and explain the model's limits.",
    deliverables: ["The completed final-project notebook, run top to bottom", "Training and validation loss curves with the kept epoch marked", "Test accuracy beside the majority-class baseline, with the number of test examples", "A short write-up: protocol, three inspected mistakes and the model's limitations"],
    rubric: ["No scaling statistic or model choice used the test set", "The validation set chose the epoch, and the test set was evaluated once", "The reloaded model reproduces the predictions exactly", "Claims are limited to the data and seed actually used"],
  },
  references: [deepLearningBook, rumelhart],
  lessons: [
    {
      id: "python-numpy-foundations", title: "Python and NumPy for neural networks", minutes: 50,
      intro: "Every neural network is arrays and arithmetic. Before building one, get fluent with the tools: shapes, vectorised operations, broadcasting, matrix multiplication and reproducible random numbers, all of which you will type yourself.",
      sections: [
        { title: "Shapes are the contract", body: "A batch of inputs is a matrix with one row per example and one column per feature. Every operation in a network either keeps the batch axis and changes the feature axis, or reduces over an axis. Printing shapes after each step catches more bugs than any other habit, because a wrong shape often still produces numbers." },
        { title: "Vectorise instead of looping", body: "A Python loop over every multiplication is slow and verbose. NumPy runs the same arithmetic in compiled code: X @ W multiplies whole matrices, and broadcasting stretches a bias vector across every row without copying it. You will write matrix multiplication with three loops first, then see that @ gives the same numbers far faster." },
        { title: "Reproducible randomness and scaled features", body: "Networks start from random weights, so experiments must fix their random seed to be repeatable. Inputs on very different scales slow learning, so features are standardised to mean zero and unit spread, using statistics from the training data only. The notebook ends by generating the spiral dataset that the later notebooks learn." },
      ],
      example: { title: "One matrix product, many dot products", body: "For X of shape (4, 3) and W of shape (3, 2), X @ W has shape (4, 2). Entry (i, j) is the dot product of row i of X with column j of W: 3 multiplications and 2 additions, so 24 multiplications for the whole result." },
      notebook: { path: "neural-networks/01_python_numpy_foundations.ipynb", title: "Python and NumPy foundations for neural networks" },
      practice: ["Type dot_loop and matmul_loops, then compare your loops with @ on the timing cell.", "Type standardize and check that every training column ends with mean 0 and standard deviation 1.", "Type make_spirals and plot the three arms; you will train on them in notebook 08."],
      references: [numpyQuickstart, numpyBroadcasting],
      quiz: { question: "X has shape (32, 10) and W has shape (10, 4). What is the shape of X @ W?", options: ["(10, 4)", "(32, 4)", "(32, 10)"], correctIndex: 1, explanation: "The inner dimensions (10) match and disappear; the batch axis (32) and the output axis (4) remain." },
    },
    {
      id: "build-a-neuron", title: "Build a neuron", minutes: 40,
      intro: "A neuron computes a weighted sum of its inputs, adds a bias and passes the result through an activation function. You will type each step, set weights by hand to compute logic gates, and discover the one simple problem a single neuron cannot solve.",
      sections: [
        { title: "Weighted sum and bias", body: "The neuron's pre-activation is z = w · x + b. Each weight scales one input and its sign decides whether that input pushes the output up or down. The bias shifts the result, acting as a threshold: with weights (10, 10), a bias of −15 makes the neuron fire only when both inputs are 1, while −5 makes either input enough." },
        { title: "Activation functions", body: "The activation bends the straight-line output of the weighted sum. The sigmoid squashes any number into (0, 1), which reads as a probability; it is written with logaddexp so that very negative inputs do not overflow. ReLU keeps positive values and zeroes negative ones; it is cheap and works well in deep networks." },
        { title: "One neuron draws one line", body: "A neuron answers 1 on one side of the boundary w · x + b = 0, which is a straight line for two inputs. That is enough for AND and OR, but XOR puts its 1s on opposite corners of the square, so no single line separates them. Searching thousands of weight settings never gets past three correct answers out of four." },
      ],
      example: { title: "A neuron by hand", body: "With x = (1, 2), w = (0.5, −1) and b = 0.1: z = 0.5 × 1 + (−1) × 2 + 0.1 = −1.4. The sigmoid gives 1 / (1 + e^1.4) ≈ 0.198, and ReLU gives 0." },
      notebook: { path: "neural-networks/02_build_a_neuron.ipynb", title: "Build a neuron" },
      practice: ["Type weighted_sum, sigmoid and relu, and run each practice check until it prints ✅.", "Type neuron and neuron_batch; confirm the batch version matches a loop of single calls.", "In the 🧪 cell, change the bias and find the range that turns AND into OR."],
      quiz: { question: "Why can a single neuron not compute XOR?", options: ["Its decision boundary is a single straight line", "The sigmoid cannot output exactly 0 or 1", "XOR needs more than two inputs"], correctIndex: 0, explanation: "XOR's 1s and 0s sit on opposite diagonals of the square, and no straight line separates them. Hidden layers solve this by bending the boundary." },
    },
    {
      id: "build-a-neural-network", title: "Build a neural network", minutes: 45,
      intro: "Stack neurons into layers and layers into a network. You will write the dense layer as one matrix product, initialise the weights so signals keep a sensible scale, count parameters, and wire a two-hidden-unit network by hand that solves XOR.",
      sections: [
        { title: "A layer is a matrix", body: "A layer of m neurons reading n inputs keeps its weights in a matrix W of shape (n, m), one column per neuron, and its biases in a vector of length m. For a batch X, the whole layer is Z = XW + b. A network is a list of such layers with nonlinear activations between them." },
        { title: "Initialisation matters", body: "If all weights start equal, every neuron in a layer computes the same thing and receives the same update forever. Random weights break the symmetry, and their scale must suit the layer width. He initialisation, designed for ReLU, draws weights with standard deviation √(2 / n_in) and starts biases at zero." },
        { title: "Hidden layers bend the boundary", body: "Two ReLU hidden units, h₁ = max(0, x₁ + x₂) and h₂ = max(0, x₁ + x₂ − 1), combined as h₁ − 2h₂, compute XOR exactly. Remove the ReLU and the network collapses into one straight line again. On the spiral data a random network is still only guessing, which is why it has to be trained." },
      ],
      example: { title: "Counting parameters", body: "A [2, 16, 3] network has 2 × 16 + 16 = 48 parameters in its first layer and 16 × 3 + 3 = 51 in its second: 99 in total. A [2, 64, 3] network has 2 × 64 + 64 + 64 × 3 + 3 = 387." },
      notebook: { path: "neural-networks/03_build_a_neural_network.ipynb", title: "Build a neural network" },
      practice: ["Type dense_forward, init_network and count_parameters, and run each check until it prints ✅.", "Run the hand-built XOR network on your dense_forward, then remove the ReLU and explain what breaks.", "In the 🧪 cell, change the seed and hidden width of the random spiral network and note that accuracy stays near chance."],
      references: [heInit],
      quiz: { question: "What goes wrong if every weight in a layer starts at the same value?", options: ["The layer cannot be multiplied by the input", "The network overflows immediately", "Every neuron in the layer stays identical during training"], correctIndex: 2, explanation: "Identical neurons receive identical gradients, so they never become different. Random initialisation breaks this symmetry." },
    },
    {
      id: "forward-propagation", title: "Forward propagation", minutes: 40,
      intro: "The forward pass carries a batch through every layer to class probabilities. You will type a numerically stable softmax and a forward function that keeps a cache of each layer's input and output, which backpropagation will reuse.",
      sections: [
        { title: "From scores to probabilities", body: "The last layer outputs one score, or logit, per class. Softmax exponentiates each score and divides by the row's total, giving positive probabilities that sum to 1. Subtracting the row's maximum before exponentiating changes nothing mathematically but stops large scores from overflowing to infinity." },
        { title: "Keep a cache", body: "For each layer, forward stores the layer's input A_prev and its pre-activation Z. The backward pass needs exactly these arrays: A_prev to compute weight gradients and Z to know where each ReLU was active. Storing them costs memory proportional to the batch size, which is why large batches need more memory." },
        { title: "Batches are independent", body: "Every row of a batch flows through the network on its own, so the batched forward pass gives the same probabilities as running one example at a time. The batch version is much faster because it turns many small vector operations into a few large matrix products." },
      ],
      example: { title: "Softmax of (1, 2, 3)", body: "e¹, e² and e³ are about 2.718, 7.389 and 20.086, totalling 30.193. Dividing gives probabilities of about 0.090, 0.245 and 0.665, which sum to 1." },
      notebook: { path: "neural-networks/04_forward_propagation.ipynb", title: "Forward propagation" },
      practice: ["Type softmax and check it on logits around 1000 without overflow.", "Type forward with its cache, then predict and accuracy.", "Run the batched-versus-looped comparison and the 🧪 timing cell for batch sizes 1, 32 and 600."],
      quiz: { question: "Why does softmax subtract each row's maximum before exponentiating?", options: ["To make the probabilities sum to 1", "To avoid overflow without changing the result", "To make the largest probability exactly 1"], correctIndex: 1, explanation: "The factor e^(−max) cancels between numerator and denominator, so the probabilities are unchanged, but every exponent becomes at most 0." },
    },
    {
      id: "loss-functions", title: "Loss functions", minutes: 45,
      intro: "Training needs one number that says how wrong the network is. You will type mean squared error, binary cross-entropy and multiclass cross-entropy, see how confident mistakes are punished, and fix the numerical problems of computing logarithms of tiny probabilities.",
      sections: [
        { title: "Match the loss to the task", body: "Mean squared error suits predicting numbers. Binary cross-entropy suits yes-or-no probabilities, and multiclass cross-entropy suits one-of-several classes: it is the negative log of the probability given to the true class, averaged over the batch. A perfect, confident prediction costs nothing." },
        { title: "Confidence is punished when it is wrong", body: "The penalty −log p grows without limit as the probability of the true class approaches zero. One confidently wrong prediction can outweigh many correct ones, which pushes the network towards calibrated, honest probabilities rather than overconfident guesses." },
        { title: "Compute it stably", body: "Probabilities can round to exactly zero, and log(0) is minus infinity. Clipping probabilities avoids that, and computing the loss directly from logits with the log-sum-exp trick avoids it more accurately. Frameworks combine softmax and cross-entropy into one function for this reason." },
      ],
      example: { title: "Cross-entropy for one example", body: "If the true class receives probability 0.7, the loss is −ln 0.7 ≈ 0.357. At 0.1 it is about 2.303, and at 0.001 about 6.908: a hundred times less probability costs about three times as much loss." },
      notebook: { path: "neural-networks/05_loss_functions.ipynb", title: "Loss functions: measuring how wrong a prediction is" },
      practice: ["Type mse, binary_cross_entropy and cross_entropy, and check each with practice.", "Type cross_entropy_from_logits and push the logits to extreme values in the 🧪 cell.", "Compute the loss of the untrained network in the Put it together section and compare it with ln(3)."],
      quiz: { question: "A three-class network outputs equal probabilities for every example. What is its cross-entropy?", options: ["0", "1/3", "ln 3, about 1.10"], correctIndex: 2, explanation: "Each true class receives probability 1/3, and −ln(1/3) = ln 3 ≈ 1.10. That is the loss an untrained network should start near." },
    },
    {
      id: "gradients-and-derivatives", title: "Gradients and derivatives", minutes: 45,
      intro: "A gradient tells every parameter which way is downhill for the loss. You will compute derivatives numerically, assemble gradients of whole arrays, and build the gradient check that later proves your backpropagation is right.",
      sections: [
        { title: "Derivatives as slopes", body: "A derivative measures how much a function's output changes for a tiny change in its input. A central difference, (f(x + h) − f(x − h)) / 2h, estimates it accurately. Too large a step blurs the slope; too small a step drowns it in floating-point rounding, so a step around 1e-5 works well for float64." },
        { title: "Partial derivatives and the chain rule", body: "With many parameters, a partial derivative measures one parameter's effect while holding the others fixed, and the gradient collects them all. The chain rule multiplies local slopes along a computation: for a neuron's loss, dL/dw = dL/da · da/dz · dz/dw. Backpropagation is this rule applied systematically." },
        { title: "Gradient checking", body: "A numerical gradient is slow, because it evaluates the loss twice per parameter, but it is simple and hard to get wrong. Comparing it with an analytic gradient through a relative error catches mistakes in backpropagation: values around 1e-7 mean agreement, while values near 1 mean a bug." },
      ],
      example: { title: "A central difference by hand", body: "For f(x) = x² at x = 3 with h = 0.01: f(3.01) = 9.0601 and f(2.99) = 8.9401, so the estimate is 0.12 / 0.02 = 6.0, matching the exact derivative 2x = 6." },
      notebook: { path: "neural-networks/06_gradients_and_derivatives.ipynb", title: "Gradients and derivatives: which way is downhill?" },
      practice: ["Type numerical_derivative and numerical_gradient, and check them with practice.", "Type relative_error, then use it to compare a sigmoid neuron's analytic gradient with the numerical one.", "In the 🧪 cell, sweep the step size of the gradient check and explain the U-shaped error."],
      quiz: { question: "Why is a central difference preferred to a forward difference?", options: ["Its error shrinks with h² rather than h", "It needs only one function evaluation", "It gives exact derivatives for every function"], correctIndex: 0, explanation: "The first-order error terms cancel in the central difference, so for small h its truncation error is proportional to h² instead of h." },
    },
    {
      id: "backpropagation", title: "Backpropagation", minutes: 60,
      intro: "Backpropagation computes every gradient of the network in one backward pass, reusing the forward pass's cache. You will derive and type each rule, assemble the full backward pass and prove it right with a gradient check of every parameter.",
      sections: [
        { title: "Start at the loss", body: "For softmax with mean cross-entropy, the gradient of the loss with respect to the last layer's logits is remarkably simple: (P − Y) / n, the predicted probabilities minus the one-hot labels, divided by the batch size. Backpropagation starts here and walks backwards through the layers." },
        { title: "Three rules for a dense layer", body: "Given dZ, the gradient flowing into a layer's output, the weight gradient is A_prevᵀ dZ, the bias gradient is dZ summed over the batch, and the gradient passed to the previous layer is dZ Wᵀ. Through a ReLU, the gradient is passed where the unit was active and zeroed where it was not." },
        { title: "Trust, but verify", body: "A backward pass with a subtle bug often still produces numbers, and training then fails mysteriously. Comparing every parameter's analytic gradient with a numerical one on a tiny network, with relative errors below 1e-6, proves the implementation correct before it is used for training." },
      ],
      example: { title: "The output gradient for one example", body: "With probabilities (0.2, 0.7, 0.1) and true class 1 in a batch of one, dL/dZ = (0.2, 0.7 − 1, 0.1) = (0.2, −0.3, 0.1). Raising the true class's logit lowers the loss, and raising the others raises it." },
      notebook: { path: "neural-networks/07_backpropagation.ipynb", title: "Backpropagation: every gradient in one backward pass" },
      practice: ["Type softmax_cross_entropy_grad, dense_backward and relu_backward, and check each with practice.", "Type the full backward function and gradient-check every parameter of a small network.", "In the 🧪 cell, break one gradient on purpose and watch the check catch it."],
      references: [rumelhart],
      quiz: { question: "In a dense layer Z = A W + b, what is the gradient of the loss with respect to W?", options: ["dZ Wᵀ", "Aᵀ dZ", "dZ summed over the batch"], correctIndex: 1, explanation: "Aᵀ dZ gives the weight gradient. dZ Wᵀ is the gradient passed back to the layer's input, and the batch sum of dZ is the bias gradient." },
    },
    {
      id: "train-with-gradient-descent", title: "Train with gradient descent", minutes: 50,
      intro: "Put everything together and train. You will type the update rule, shuffled mini-batches and one training epoch, then watch your own network, built entirely from your own functions, learn the three-class spirals.",
      sections: [
        { title: "The update rule", body: "Gradient descent moves every parameter a small step against its gradient: W becomes W − η ∂L/∂W and b becomes b − η ∂L/∂b. The learning rate η sets the step size. Too small and training barely moves; too large and the loss oscillates or explodes to infinity." },
        { title: "Epochs, batches and updates", body: "An epoch is one pass over the training data. Mini-batch descent shuffles the data each epoch and takes one step per batch, so with 480 examples and batches of 32 an epoch makes 15 updates. Full-batch descent makes one accurate update per epoch, and stochastic descent one noisy update per example." },
        { title: "Watch validation, not just training", body: "Training loss tells you whether the network can fit the data; validation loss, on examples it never trains on, tells you whether it generalises. When training loss keeps falling but validation loss rises, the network has started to memorise." },
      ],
      example: { title: "Counting updates", body: "With 480 training examples, a batch size of 32 and 120 epochs, training makes 15 × 120 = 1,800 updates. With a batch size of 480 it would make only 120, and with a batch size of 1 it would make 57,600." },
      notebook: { path: "neural-networks/08_train_with_gradient_descent.ipynb", title: "Train with gradient descent" },
      practice: ["Type sgd_step, minibatch_indices and train_epoch, and check each with practice.", "Train the [2, 64, 3] network and read the loss and accuracy curves and the decision regions.", "In the 🧪 learning-rate cell, compare 0.02, 1.0 and 50.0 and describe what each does."],
      references: [deepLearningBook],
      quiz: { question: "Training loss keeps falling while validation loss starts to rise. What is most likely happening?", options: ["The learning rate is too small", "The gradients are computed incorrectly", "The network is starting to memorise the training data"], correctIndex: 2, explanation: "Diverging training and validation losses are the signature of overfitting. A wrong gradient or a tiny learning rate would stop the training loss falling too." },
    },
    {
      id: "final-project", title: "Final project: your own network, end to end", minutes: 90,
      intro: "Package your functions into your own NeuralNetwork class and run a complete, honest pipeline: split the data before looking at it, standardise with training statistics, keep the epoch with the best validation loss, test once, save, reload and explain what the model can and cannot do.",
      sections: [
        { title: "Three sets, three jobs", body: "The training set fits the weights, the validation set makes choices such as which epoch to keep, and the test set measures the chosen model once. Using the test set to make a choice turns it into another validation set, and the final number stops being an honest estimate." },
        { title: "Early stopping by checkpoint", body: "Instead of guessing how many epochs to train, the class keeps a copy of the parameters from the epoch with the lowest validation loss and restores them at the end. This is the simplest defence against overfitting, and it needs no extra hyperparameters." },
        { title: "Save, reload, explain", body: "A model is only useful if it can be reloaded exactly. The class saves its weights with np.savez and checks that the reloaded model gives identical predictions. The write-up then states the baseline, the sample size, inspected mistakes and the limits of what was shown." },
      ],
      example: { title: "Reading a test result", body: "With 150 test examples, one more mistake moves accuracy by 1/150 ≈ 0.7 percentage points. A result of 0.96 against a majority-class baseline of 0.33 is a large gain, but it describes this dataset and seed, not other data." },
      notebook: { path: "neural-networks/12_final_project.ipynb", title: "Final project: your own neural network, end to end" },
      practice: ["Type train_val_test_split and fit_standardizer, and check them with practice.", "Train the NeuralNetwork class, mark the kept epoch on the loss curves, then evaluate the test set once.", "Save and reload the model, then complete the write-up cell and the self-review rubric."],
      quiz: { question: "Why are the standardisation statistics computed from the training set only?", options: ["To keep information about the validation and test sets out of training", "Because the test set is too small to have a mean", "So that every feature ends between 0 and 1"], correctIndex: 0, explanation: "Statistics from the whole dataset would let held-out data influence the model, making validation and test results optimistic." },
    },
  ],
};
