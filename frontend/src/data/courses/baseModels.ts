import type { Course } from "./types";

const notebook1 = { path: "base-models/01_tensors_and_autograd.ipynb", title: "Tensors and autograd: build the engine" };
const notebook2 = { path: "base-models/02_datasets_and_losses.ipynb", title: "Datasets, batches and losses" };
const notebook3 = { path: "base-models/03_training_loop_and_checkpoints.ipynb", title: "The training loop: modules, optimisers, diagnostics and checkpoints" };
const notebook4 = { path: "base-models/04_evaluate_reload_capstone.ipynb", title: "Evaluate, reload and the capstone" };

const autodiffSurvey = { title: "Baydin et al. (2018): Automatic differentiation in machine learning, a survey (JMLR)", url: "https://www.jmlr.org/papers/v18/17-468.html" };
const deepLearningBook = { title: "Goodfellow, Bengio and Courville: Deep Learning (free online book)", url: "https://www.deeplearningbook.org/" };
const numpyBroadcasting = { title: "NumPy: broadcasting", url: "https://numpy.org/doc/stable/user/basics.broadcasting.html" };
const numpyLoad = { title: "NumPy: numpy.load and the allow_pickle safeguard", url: "https://numpy.org/doc/stable/reference/generated/numpy.load.html" };

export const baseModelsCourse: Course = {
  id: "base-models", title: "Base Model Training", level: "Intermediate", track: "Deep learning",
  description: "Build the machinery that trains a neural network, then use it. In NumPy notebooks that you type yourself, you write a tensor engine with automatic differentiation, a leak-free data pipeline, modules, the Adam optimiser and a diagnosed training loop, then save checkpoints that resume exactly and evaluate a reloadable base model honestly.",
  prerequisites: ["Python functions, classes, loops and lists", "NumPy arrays, matrix multiplication and the chain rule, at the level of Neural Networks from Scratch", "Training, validation and test splits and basic metrics, as in Model Evaluation, or equivalent knowledge"],
  prerequisiteCourses: ["neural-networks", "evaluation"], practiceMode: "notebook", dependencyTier: 1, nextCourseId: "intermediate-models",
  outcomes: [
    "Build a reverse-mode automatic differentiation engine in NumPy and gradient-check every operation",
    "Split data by source, fit preprocessing on training rows only and build mini-batch pipelines",
    "Write modules, SGD and Adam and a correct training step, and diagnose a failing loop",
    "Save checkpoints that resume training exactly, and reload a verified model for prediction",
    "Select on validation data, test once, and report results with baselines and seed variation",
  ],
  lab: {
    filename: "base_model.py", title: "Optional: the same experiment in PyTorch",
    description: "This script runs locally with PyTorch on a CPU and is not needed for the course: the notebooks cover every step. It trains a small multilayer perceptron on generated data, selects a checkpoint on validation loss, evaluates untouched test rows and verifies a checkpoint reload, so you can compare a framework's version of each step with the one you built. No model or dataset download is needed.",
    exercises: ["Run the default experiment and record the validation loss and the test accuracy.", "Find the lines that do the work of your train_step, your Adam optimiser, your checkpoint selection and your reload check.", "Add a second random seed and explain the variation without choosing the best test score."],
  },
  capstone: {
    title: "Deliver a reproducible base model",
    brief: "In notebook 4, run a complete experiment on the new generated rings task, or on a documented classification dataset of your own. Write the protocol before any result, compare one controlled change across seeds on validation data, test the chosen model once, and reload it to verify its predictions.",
    deliverables: ["The protocol, written before the comparison: split unit, baseline, validation metric, budget, selection rule and the one controlled change", "The completed capstone cells, re-run from a restarted kernel, with the printed report and the versions it records", "The test confusion matrix beside the majority baseline and the number of test rows, and a passing reload check", "Three inspected mistakes and a short statement of limitations"],
    rubric: ["No fitted transform or model choice used test data, and the test set was opened once", "Training and evaluation modes are handled correctly, and the best checkpoint was copied rather than referenced", "The reloaded model reproduces the selected model's logits within tolerance", "Conclusions compare the change with the spread across seeds and stay within the generated data"],
  },
  references: [
    autodiffSurvey,
    deepLearningBook,
    { title: "PyTorch: learn the basics (for the optional PyTorch script)", url: "https://docs.pytorch.org/tutorials/beginner/basics/intro.html" },
  ],
  lessons: [
    {
      id: "what-is-a-base-model", title: "Define the model and its training objective", minutes: 30,
      intro: "Here, a base model starts from random weights and learns an initial task. It may be a small supervised network or a language model trained on next-token prediction. “Base” describes a training stage, not a guarantee of capability. In this course you build every part of that training yourself.",
      sections: [
        { title: "Separate architecture, weights and objective", body: "The architecture specifies the operations and connections. Weights are the numeric values learned within that structure. The objective is the scalar quantity that training tries to reduce. Two networks with the same architecture can behave very differently because they learned from different data or losses. Write down inputs, outputs and the intended prediction before choosing layer sizes. In notebook 1 the architecture is a 2 → 16 → 2 network, its 82 weights start at random values, and the objective is the average cross-entropy." },
        { title: "Distinguish pretraining from adaptation", body: "Training from scratch initialises weights without importing a learned checkpoint. Pretraining often learns reusable patterns from a broad dataset. Fine-tuning starts from learned weights and adapts them to a new objective or distribution. A pretrained language model is not automatically an instruction-following assistant; later courses examine the extra data and objectives involved. Whichever stage you are in, the machinery is the same: a forward pass, a loss, gradients and an update, which is exactly what this course builds." },
        { title: "Make the first experiment small enough to inspect", body: "Begin with a task whose correct answer and baseline you understand. Limit data and model size so that you can inspect a batch, run a few updates and repeat the experiment cheaply. Frameworks such as PyTorch hide the gradient machinery; here you write it in NumPy, so nothing is hidden. The notebooks train on generated data, so their accuracies describe only that teaching distribution. Their purpose is to make every training step visible and testable." },
      ],
      example: { title: "A concrete model contract", body: "Input: two measured coordinates. Output: two class logits. Target: which side of a nonlinear boundary contains the point. Baseline: always choose the most common training class. Acceptance: improve validation performance, then measure the chosen checkpoint once on the held-out test partition." },
      notebook: notebook1,
      practice: ["Read section 1 of notebook 1, then write an input/output contract with a baseline for a task you understand.", "Type count_parameters and check that the 2 → 16 → 2 network has 82 parameters.", "In the 🧪 cell, predict the parameter count for hidden widths 32 and 64, then feed three input features and read the shape error."],
      references: [
        { title: "Bommasani et al. (2021): On the opportunities and risks of foundation models", url: "https://arxiv.org/abs/2108.07258" },
        deepLearningBook,
      ],
      quiz: { question: "Which change makes an experiment fine-tuning rather than training from scratch?", options: ["Increasing the batch size and the number of epochs", "Starting from previously learned weights", "Saving a checkpoint at the end of training"], correctIndex: 1, explanation: "Fine-tuning reuses a learned starting point. Batch size, epochs and checkpoint saving can be part of either workflow." },
    },
    {
      id: "tensors-and-shapes", title: "Tensors, shapes and automatic differentiation", minutes: 90,
      intro: "A tensor is an array with a shape and a numeric type. In this course it also remembers how it was computed, so that gradients can flow back through it. Keeping track of shapes is one of the fastest ways to understand a neural network and to catch errors before they become misleading metrics.",
      sections: [
        { title: "Follow one batch through the network", body: "For tabular inputs, [batch, features] becomes [batch, hidden] after a linear layer and [batch, classes] after a classification head. In this course a linear layer computes xW + b, with W of shape [inputs, outputs]; frameworks such as PyTorch store the transpose and compute xWᵀ + b. An activation such as ReLU introduces nonlinearity; stacking only linear layers still produces a linear mapping. Broadcasting reuses one bias for every example, so the bias gradient sums over the batch." },
        { title: "Record the forward pass, then run it backwards", body: "Reverse-mode automatic differentiation records each operation's inputs and a local rule during the forward pass. The backward pass visits the operations in reverse topological order: each receives the gradient of the loss with respect to its output and sends one gradient back to each input, with that input's shape. A value used twice receives the sum of both contributions. Every rule you write in notebook 1 is checked against central differences, which are too slow for training but ideal for testing." },
        { title: "Choose numeric types deliberately", body: "Neural layers receive floating-point inputs; class targets are integer indices, one per example; a boolean mask means something different from a floating additive mask. Inspect dtype, shape and a few values together, because a correctly shaped tensor can still hold the wrong thing. Numerical range matters too: a softmax that skips subtracting the row maximum overflows on large logits. Frameworks such as PyTorch also place each tensor on a device, and one operation's tensors must share it; NumPy arrays always live in main memory." },
      ],
      example: { title: "Count the trainable parameters", body: "A 2 → 16 → 2 MLP has 2×16 + 16 parameters in its first layer and 16×2 + 2 in its output layer: 82 in total. The batch dimension changes how many examples are processed, not the number of learned parameters. For a batch of 32, the first bias of shape [16] receives an upstream gradient of shape [32, 16], and its own gradient is that array summed over the 32 rows." },
      code: { title: "A forward and backward pass with your engine", source: "rng = np.random.default_rng(0)\nW1, b1 = parameter(2, 16, rng=rng), zeros_parameter(16)\nW2, b2 = parameter(16, 2, rng=rng), zeros_parameter(2)\nx = Tensor(rng.normal(size=(8, 2)))\nlogits = (x @ W1 + b1).relu() @ W2 + b2\nassert logits.shape == (8, 2)\nlogits.sum().backward()\nprint(sum(p.data.size for p in (W1, b1, W2, b2)), b1.grad.shape)  # 82 (16,)", explanation: "Run it after building the engine in notebook 1. The forward pass records a graph, and backward() fills in a gradient for every parameter, each with its parameter's shape. It does not train the model or turn logits into probabilities." },
      notebook: notebook1,
      practice: ["Type unbroadcast, topological_order and backward_pass, and pass their checks.", "Type tensor_mul, matmul_backward, relu_backward, sum_backward and tensor_mean, and read each gradient check.", "Type log_softmax_forward and log_softmax_backward, then run the 🧪 experiments and the final training run on your own engine."],
      references: [
        autodiffSurvey,
        numpyBroadcasting,
        { title: "He et al. (2015): Delving deep into rectifiers (He initialisation)", url: "https://arxiv.org/abs/1502.01852" },
      ],
      quiz: { question: "A bias of shape (16,) is added to a batch of shape (32, 16). What shape must the bias gradient have?", options: ["(32, 16), exactly the shape of the upstream gradient", "(32,), with one number per example in the batch", "(16,), summed over the batch"], correctIndex: 2, explanation: "The bias was reused for every example, so its gradient adds the 32 contributions and keeps the bias's own shape. That is the job of unbroadcast in your backward pass." },
    },
    {
      id: "datasets-and-batches", title: "Construct datasets and batches without leakage", minutes: 40,
      intro: "The data pipeline decides what the optimiser sees and what evaluation means. Split examples before fitting transforms, and keep enough metadata to prove where each example came from.",
      sections: [
        { title: "Assign three separate roles", body: "Training examples update weights. Validation examples guide checkpoint selection and experimental decisions. Test examples support the final evaluation after those decisions. Split by person, document, recording or time when examples share an underlying source. Randomly separating fragments from the same source can make the test unrealistically easy: in notebook 2, a memorising nearest-neighbour classifier scores 88% with a row split but 61% with a split by source, on identical data." },
        { title: "Fit transforms on training data", body: "Estimate feature means, scales and learned vocabularies using the training partition only. Apply those fixed values to validation and test inputs, and save them with the model, because the deployed model needs the same transform. Training augmentation can introduce variation, but validation preprocessing should be stable enough that checkpoint comparisons reflect model changes rather than random crops or noise. A constant column needs a floor on its scale, so that standardising it cannot divide by zero." },
        { title: "Understand what batching changes", body: "A batch averages or sums contributions from multiple examples before an update. Shuffling training examples every epoch helps avoid learning from an arbitrary file order. Keep validation deterministic and include its final partial batch, so that every validation row counts exactly once. When batch sizes vary, average the loss by the number of examples or valid tokens, not by giving every batch equal weight: a final batch of two rows would otherwise count each of its rows twice as much." },
      ],
      example: { title: "A split by original source", body: "If one recording yields 100 short clips, put all 100 into the same partition. Otherwise the model may recognise recording conditions rather than learn the task. Record the original recording ID in a split manifest even when the training tensors no longer contain it." },
      notebook: notebook2,
      practice: ["Type group_split and confirm that no source appears in two partitions.", "Type fit_standardiser and shuffled_batches, then check that every training row appears once per epoch.", "Run 🧪 Experiment 1 and explain why the row split scores higher than the split by source."],
      references: [
        { title: "scikit-learn user guide: cross-validation iterators for grouped data", url: "https://scikit-learn.org/stable/modules/cross_validation.html" },
        { title: "scikit-learn user guide: common pitfalls, including data leakage", url: "https://scikit-learn.org/stable/common_pitfalls.html" },
      ],
      quiz: { question: "Where should examples used to select the best epoch come from?", options: ["A validation partition", "The final test partition", "Only the examples with the highest training loss"], correctIndex: 0, explanation: "Checkpoint selection is a model-development decision. It uses validation data so the test partition remains independent of that choice." },
    },
    {
      id: "loss-and-gradients", title: "Connect predictions, loss and gradients", minutes: 40,
      intro: "Training needs a differentiable objective that responds to prediction errors. Automatic differentiation traces the forward computation and computes how changing each trainable parameter would change that objective.",
      sections: [
        { title: "Match the loss to the output", body: "For mutually exclusive classes, cross-entropy consumes unnormalised logits and integer class indices. Do not apply a separate softmax first: a loss computed on probabilities is flatter and cannot reach zero. For multilabel classification, independent binary outputs need a different target representation and loss. For regression, squared error penalises large misses strongly; an absolute-error objective behaves differently near outliers. In notebook 2 you write cross_entropy from your engine's log-softmax and indexing." },
        { title: "Interpret a gradient as a local signal", body: "A gradient describes a local slope, not a promise that a large update will improve the whole problem. Backpropagation applies the chain rule through the computation graph. Calling backward() fills in parameter gradients; it does not update weights. An optimiser performs the update using those gradients and any state it maintains. For cross-entropy, the gradient with respect to the logits is (p − onehot(y)) / n, which your engine derives without being told the formula." },
        { title: "Debug broken computation graphs", body: "Detaching predictions, converting the loss to a Python number before backward, or computing the forward pass in a mode that records nothing removes the path needed for learning. Logging the loss as a number is fine after backward. Check that intended trainable parameters have finite gradients and that at least some weights change after an update. In notebook 2, detached logits fail silently: backward runs, no gradient arrives, and every weight stays exactly where it started." },
      ],
      example: { title: "One squared-error derivative", body: "For prediction ŷ = wx, target y and loss (wx − y)², the gradient with respect to w is 2(wx − y)x. With x = 2, w = 1 and y = 3, the gradient is −4. A gradient-descent step with learning rate 0.1 moves w to 1.4, making the prediction 2.8." },
      code: { title: "The worked example with your engine", source: "w = Tensor(1.0, requires_grad=True)\nloss = (w * 2.0 - 3.0) ** 2      # prediction wx with x = 2, target y = 3\nloss.backward()\nprint(w.grad)                    # -4.0\nw.data -= 0.1 * w.grad           # one gradient-descent step\nprint(w.data, 2 * w.data)        # 1.4 2.8", explanation: "Run it after the recap cell of notebook 2. The engine reproduces the hand calculation; backward() only computes the gradient, and the update is a separate line." },
      notebook: notebook2,
      practice: ["Type cross_entropy and confirm that its gradient equals (p − onehot(y)) / n.", "Type squared_error_grad and verify the worked example by hand, with your engine and with central differences.", "Run 🧪 Experiment 2: detach the logits, convert the loss to a float, and explain what each does to training."],
      references: [
        deepLearningBook,
        { title: "Rumelhart, Hinton and Williams (1986): Learning representations by back-propagating errors", url: "https://www.nature.com/articles/323533a0" },
      ],
      quiz: { question: "What does loss.backward() do?", options: ["Loads the next batch of training examples", "Computes gradients through the recorded graph", "Updates every weight with the optimiser and saves a checkpoint"], correctIndex: 1, explanation: "Backward computes gradients. The optimiser's step() applies an update, while data loading and checkpoint saving are separate operations." },
    },
    {
      id: "training-loop", title: "Write and diagnose the training loop", minutes: 50,
      intro: "A training loop is a sequence of explicit state changes. Make those changes visible before adding frameworks, accelerators or distributed execution.",
      sections: [
        { title: "Perform one optimiser step correctly", body: "Switch the network into training mode, clear gradients, compute predictions and loss, check that the loss is finite, backpropagate, then let the optimiser update the weights. Gradients accumulate across backward calls, in your engine as in PyTorch, so forgetting to clear them changes the algorithm. Intentional gradient accumulation is useful, but its normalisation and update schedule must be designed explicitly. Adam adds per-parameter step sizes from running averages, with a bias correction for their zero start." },
        { title: "Separate evaluation mode from gradient recording", body: "model.eval() changes the behaviour of layers such as dropout and batch normalisation. It does not disable gradient computation; frameworks such as PyTorch provide a separate no-gradient context for validation. Your engine records the operations but never calls backward on a validation loss, so no weight changes. Return to model.train() before the next update. This prevents validation from using training-time randomness or updating running statistics, and keeps scores comparable between checkpoints." },
        { title: "Use a tiny-batch diagnostic", body: "Try overfitting a handful of correctly labelled examples. Failure can expose detached gradients, a missing zero_grad, an unsuitable learning rate or insufficient capacity. Success is a wiring check, not evidence of generalisation: in notebook 3 the network memorises shuffled labels just as easily. Also log the loss, the global gradient norm, weight changes, class balance and predictions that collapse to one label, because a loss can stay finite while the run explodes." },
      ],
      code: { title: "The core update", source: "model.train()\noptimiser.zero_grad()\nloss = cross_entropy(model(Tensor(X_batch), rng), y_batch)\nassert np.isfinite(loss.data), \"non-finite loss: check the learning rate and the data\"\nloss.backward()\noptimiser.step()\n\nmodel.eval()\nvalidation_logits = model(Tensor(X_val)).data   # no backward(), no step()", explanation: "This is the body of train_step from notebook 3, followed by the validation recipe. The model, optimiser and data come from the notebook; the validation forward pass never calls backward() or step()." },
      example: { title: "Recognise an accumulated-gradient bug", body: "If you intended one update per batch but never clear gradients, the second update includes gradients from both the first and second batches. That is different from averaging two micro-batches and taking one deliberately normalised update. In notebook 3, this bug leaves the tiny-batch loss stuck near 0.43 instead of falling below 0.001." },
      notebook: notebook3,
      practice: ["Type Linear and dropout, then check that evaluation mode gives identical outputs on repeated calls.", "Type adam_update and train_step, and pass the tiny-batch test on 16 examples.", "Type grad_norm, then run 🧪 Experiments 1 and 2: the deliberate bugs and the exploding learning rate."],
      references: [
        { title: "Kingma and Ba (2014): Adam, a method for stochastic optimisation", url: "https://arxiv.org/abs/1412.6980" },
        { title: "Srivastava et al. (2014): Dropout, a simple way to prevent neural networks from overfitting (JMLR)", url: "https://jmlr.org/papers/v15/srivastava14a.html" },
        { title: "Bengio (2012): Practical recommendations for gradient-based training of deep architectures", url: "https://arxiv.org/abs/1206.5533" },
      ],
      quiz: { question: "Which validation setup is correct?", options: ["model.eval(), a forward pass, and no backward or update", "model.train() followed by backward on the validation loss", "model.eval() alone, trusted to switch off all gradient recording"], correctIndex: 0, explanation: "Evaluation mode controls layers such as dropout; it does not stop gradient recording, which frameworks control with a separate no-gradient context. Validation must never update the weights." },
    },
    {
      id: "budget-and-checkpoints", title: "Plan compute and save recoverable checkpoints", minutes: 45,
      intro: "Training consumes memory and time even before a model is useful. Estimate the experiment's size, measure a short run, and save enough state to reproduce the chosen result.",
      sections: [
        { title: "Count updates and examples", body: "An epoch is one pass through the training dataset. An optimiser step updates the weights once. With gradient accumulation, several micro-batches contribute to one update. Report examples or valid tokens processed along with epoch counts; different batch sizes, sequence lengths and dropped batches can make equal epoch counts misleading. Under a drop-last policy, a dataset smaller than one micro-batch produces no updates at all, a silent failure that a line of arithmetic catches before the run starts." },
        { title: "Include memory beyond weights", body: "Parameter storage is only part of training memory. Gradients, optimiser state, activations and temporary workspaces also matter. A simple mixed-precision Adam estimate may include 2-byte weights, 2-byte gradients, 4-byte master weights and two 4-byte moments: roughly 16 bytes per parameter before activations. Your float64 engine uses 32 bytes per parameter for weights, gradients and Adam's two moments. Actual frameworks can store these differently, so profile the configuration you use." },
        { title: "Save for inference or for resuming", body: "An inference artefact needs model weights, architecture configuration and preprocessing or tokeniser information. Exact training resumption additionally needs optimiser and scheduler state, the training position and the relevant random-generator states. Store them as numbers and text, such as the generator's state written as JSON. Load only checkpoint files from sources you trust; NumPy's np.load refuses pickled objects by default because unpickling can run arbitrary code." },
      ],
      experiment: "training-budget",
      example: { title: "An update budget", body: "For 8,192 examples, micro-batch size 16 and accumulation over 4 micro-batches on one worker, there are 128 optimiser updates per complete epoch, each with an effective batch of 64. Three epochs give 384 updates. A different final-batch policy or distributed sampler can change this arithmetic." },
      code: { title: "What a resumable checkpoint holds", source: "arrays = {f\"param_{i}\": p.data for i, p in enumerate(model.parameters())}\narrays.update({f\"opt_{key}\": value for key, value in optimiser.state_dict().items()})\nnp.savez(path, step=step, rng_state=json.dumps(rng.bit_generator.state), **arrays)", explanation: "This is the body of save_checkpoint from notebook 3: weights, optimiser state, the step and the random generator's state, stored as numbers and JSON text so that loading never needs pickle." },
      notebook: notebook3,
      practice: ["Use the budget explorer, then type optimiser_updates and reproduce the 384 updates of the worked example.", "Time 50 steps in notebook 3 and read the memory estimate per parameter.", "Type save_checkpoint and load_checkpoint, prove exact resumption, then run 🧪 Experiment 3 and explain what each forgotten piece of state changes."],
      references: [
        { title: "Rajbhandari et al. (2019): ZeRO, memory optimisations toward training trillion-parameter models", url: "https://arxiv.org/abs/1910.02054" },
        { title: "NumPy: numpy.savez", url: "https://numpy.org/doc/stable/reference/generated/numpy.savez.html" },
        numpyLoad,
      ],
      quiz: { question: "What is missing from a weights-only file if you want to resume Adam training exactly?", options: ["The name of the output folder and the date of the run", "A larger test set for the final evaluation", "Optimiser state, step and random state"], correctIndex: 2, explanation: "Adam's running moments and step counter, the training position and the random state all affect the next update. Weights alone are sufficient for neither exact resumption nor reconstructing an unknown architecture." },
    },
    {
      id: "evaluate-and-reload", title: "Evaluate and reload the selected model", minutes: 45,
      intro: "A model is ready for evaluation when its development decisions are fixed. Test the actual checkpoint you intend to keep, including the preprocessing used at prediction time.",
      sections: [
        { title: "Select first, test afterwards", body: "Choose architecture, hyperparameters and checkpoint using training and validation data, keeping copies of the best weights because optimisers update them in place. Then evaluate once on the untouched test partition; in notebook 4 the test rows sit in an object that hands them out only once. If test results trigger further tuning, they have become development feedback, and you need fresh final evidence. Keep the original baseline and split definition in the report." },
        { title: "Look beyond one aggregate", body: "For classification, inspect the confusion matrix, per-class support, recall and precision, and important failure groups. For regression, attach units to the mean absolute error and inspect residuals. Read the most confident mistakes first: in notebook 4 they turn out to be flipped labels, while the least confident sit on the decision boundary. Variation across seeds is useful to report, but the best seed is not an unbiased summary. Small datasets require cautious claims." },
        { title: "Test the exported artefact", body: "Rebuild the architecture from saved configuration, load the selected weights, restore preprocessing and compare predictions on a fixed batch. A saved object that reloads successfully can still use the wrong class order or transform: without its standardiser, the notebook's model returns the same logits for every input. Record package versions in the artefact, and confirm that evaluation mode is restored before producing outputs, because dropout left on makes predictions random." },
      ],
      example: { title: "A checkpoint round trip", body: "Record logits for ten fixed inputs before saving. Reload the selected checkpoint into a fresh network, use the same inputs and compare the arrays with a reasonable numerical tolerance, such as 1e-12 for float64 on the same machine. This checks the artefact path; it does not replace held-out evaluation." },
      notebook: notebook4,
      practice: ["Type snapshot, train with validation-based checkpoint selection, and run 🧪 Experiment 1 to see the reference bug.", "Open the sealed test set once, then type confusion_matrix, recall_and_precision and most_confident_mistakes, and describe one failure pattern.", "Type load_for_inference, pass the round-trip check, and run the broken reloads of 🧪 Experiment 2."],
      references: [
        { title: "scikit-learn user guide: metrics and scoring, including the confusion matrix", url: "https://scikit-learn.org/stable/modules/model_evaluation.html" },
        { title: "Bouthillier et al. (2021): Accounting for variance in machine learning benchmarks", url: "https://arxiv.org/abs/2103.03098" },
        numpyLoad,
      ],
      quiz: { question: "Which checkpoint should receive the final test evaluation?", options: ["Whichever epoch reaches the best accuracy on the test set", "The one selected before any test result was seen", "Always the last epoch, regardless of the selection protocol"], correctIndex: 1, explanation: "The test evaluates a decision already made using development data. Choosing the checkpoint on the test set would compromise that independence." },
    },
    {
      id: "base-model-capstone", title: "Capstone: document and defend a base model", minutes: 60,
      intro: "Bring the pieces together in a small, complete experiment. The goal is a result someone else can reproduce and question, with every training and evaluation choice explained.",
      sections: [
        { title: "Write the protocol before the comparison", body: "State the task, baseline, split unit, primary validation metric and allowed experiment budget. Decide how a checkpoint wins, how a configuration wins and when training stops. Keep test rows out of exploratory work until the decision is made. For the generated rings task in notebook 4, describe how the three classes and the replaced labels arise, so that another reader understands why the task is easier than a real application." },
        { title: "Make one controlled improvement", body: "Compare the base configuration with one deliberate variation, such as a narrower network, dropout or weight decay. Keep the split, the budget and the evaluation code fixed, and run every configuration with the same seeds. Explain the mechanism you expect to change, then compare the difference in mean validation loss with the spread across seeds. Record successful and failed attempts; additional complexity is justified by evidence, not by a longer architecture description." },
        { title: "Hand over a usable experiment", body: "Bundle code, configuration, preprocessing, checkpoint and results. The capstone cells print a report with the protocol, the validation results, the decision, the test result and the library versions; restart the kernel and run the notebook again to confirm that the report is identical. Discuss mistakes, unavailable evidence and the next experiment. Passing this page's quiz records conceptual progress; review the capstone deliverables separately before considering the practical project finished." },
      ],
      example: { title: "A defensible conclusion", body: "“The narrower network matched the wider one: their mean validation losses differed by 0.001, far less than the spread of 0.04 across three seeds. We chose the configuration before opening the test set. Most remaining test errors involve the middle ring, which borders both other classes, and we have not tested any real-world population.” The exact numbers must come from your own run." },
      notebook: notebook4,
      practice: ["Edit the protocol cell before running the capstone, choosing exactly one controlled change.", "Run the comparison across three seeds, open the rings test set once, and inspect three mistakes with most_confident_mistakes.", "Restart the kernel, run the notebook again, confirm that the report is identical, and answer the six report prompts."],
      references: [
        { title: "Mitchell et al. (2019): Model cards for model reporting", url: "https://arxiv.org/abs/1810.03993" },
        { title: "Pineau et al. (2021): Improving reproducibility in machine learning research (JMLR)", url: "https://jmlr.org/papers/v22/20-303.html" },
      ],
      quiz: { question: "Which result best demonstrates a complete base-model project?", options: ["A reproducible checkpoint with an independent evaluation", "A screenshot of the lowest training loss reached during the run", "A model file with no configuration or split information"], correctIndex: 0, explanation: "A complete experiment connects the saved artefact to its configuration, data protocol and evaluation evidence. Training loss alone is not enough." },
    },
  ],
};
