import type { Course } from "./types";

export const tabularCourses: Course[] = [
  {
    id: "training",
    level: "Beginner",
    track: "Tabular ML",
    prerequisites: ["Read a CSV table; no programming required."],
    prerequisiteCourses: [],
    practiceMode: "workbench",
    nextCourseId: "evaluation",
    title: "Model Training",
    description: "Turn a prediction question into a reproducible training run. Learn what to select, what the model learns, and how to improve it.",
    outcomes: [
      "Choose a target and useful features",
      "Split data and prevent leakage",
      "Train a model and tune its settings",
      "Understand cross-validation",
    ],
    toolPath: "/app/training",
    toolLabel: "Open Data Training",
    lessons: [
      {
        id: "targets-and-features",
        title: "Start with a prediction question",
        minutes: 5,
        intro: "A training run starts with a question about a new row. The target is the answer you want the model to predict; features are the clues it can use.",
        sections: [
          {
            title: "Choose the kind of answer",
            body: "Classification predicts a category, such as whether a parcel arrives late. Regression predicts a quantity, such as delivery time in hours. Numbers can name categories: delivery_status values 0, 1 and 2 still call for classification if they mean pending, on time and late. Furina suggests a task from the target profile; you can override Auto on Data Training.",
          },
          {
            title: "Use clues available at prediction time",
            body: "For a prediction made when an order is placed, distance and shipping method may be valid features. The eventual arrival time is not available yet. Including future information or a copy of the answer is target leakage. An order ID can also encourage memorisation instead of learning a useful relationship.",
          },
          {
            title: "Read the dataset profile",
            body: "Check missing values, column types and class counts on Data Preprocessing. A row with no target cannot teach supervised learning; Furina removes those rows before fitting. The target is excluded from the feature list. Keep enough labelled examples of each class to make both learning and evaluation meaningful.",
          },
        ],
        example: {
          title: "Example: predict a late delivery",
          body: "Question: will this parcel arrive late? Target: is_late. Features: distance_km, parcel_weight and shipping_method. Task: classification, even if is_late contains 0 and 1. Exclude actual_arrival_time because it is known only after the delivery.",
        },
        practice: [
          "Write your prediction question and when the prediction will be made.",
          "Upload a CSV on Data Preprocessing and inspect the target's missing values and distinct values.",
          "Open Data Training, select the target, confirm the task, and remove features that reveal the answer or arrive too late.",
        ],
        quiz: {
          question: "You predict whether an order will be returned at checkout. Which feature would leak future information?",
          options: ["The item's category", "The shipping method chosen at checkout", "The date the return was received"],
          correctIndex: 2,
          explanation: "The return date exists only after the outcome. The model would be learning from information unavailable at checkout.",
        },
      },
      {
        id: "splits-and-leakage",
        title: "Give the model an honest test",
        minutes: 7,
        intro: "Training rows teach the model. Test rows let you measure how it handles unseen examples. Keeping those roles separate is the foundation of a useful evaluation.",
        sections: [
          {
            title: "Balance learning data and test evidence",
            body: "Furina starts with an 80% training / 20% test split. More training rows leave fewer examples for estimating performance. A tiny test set can produce a score that changes sharply after just one mistake. Stratification tries to preserve class proportions for classification; rare classes can still make a split impossible, so read the run's warnings.",
          },
          {
            title: "Learn preprocessing on training rows",
            body: "Furina's training pipeline learns numeric medians, categorical replacements, category encodings and optional scaling from the training split. It applies those fitted transformations to test rows. In contrast, a cleaning recipe applied before training changes the whole CSV. Fitting scaling, PCA or feature selection there can leak test information; the training pipeline cannot undo it. For this course, keep learned transformations inside the training pipeline.",
          },
          {
            title: "Know when a random split is unsuitable",
            body: "Furina shuffles rows for its train/test split. Repeated observations of the same person or product can land on both sides, and future records can inform a model intended to predict the past. Time-ordered or grouped problems need a corresponding evaluation split outside this interface. A fixed random seed makes a split repeatable for the same rows, order and settings; it does not make it representative.",
          },
        ],
        example: {
          title: "Example: 1,000 independent labelled rows",
          body: "An 80/20 split gives 800 training rows and 200 test rows. If 10% of rows belong to the rare class, a stratified test set has about 20 such examples. A high overall score can still rest on very little evidence about that class.",
        },
        experiment: "split",
        practice: [
          "Try the split explorer below and compare how many test examples remain.",
          "On Data Training, keep a record of the test percentage and random seed; enable stratification for a classification task when possible.",
          "Use the training page's feature scaling option if needed, and check that earlier cleaning has not learned from the whole dataset.",
        ],
        quiz: {
          question: "Which rows should determine the median used to fill missing numeric features in a holdout evaluation?",
          options: ["Only the training rows", "All rows before splitting", "Only the test rows"],
          correctIndex: 0,
          explanation: "Learn the median from training rows, then use that same value for training and test inputs. Test rows must not influence fitted preprocessing.",
        },
      },
      {
        id: "fit-a-model",
        title: "Choose an algorithm and train",
        minutes: 6,
        intro: "An algorithm specifies how to learn a pattern. Training fits its internal parameters to your rows. Hyperparameters are the settings you choose before that fit.",
        sections: [
          {
            title: "Start with a model you can explain",
            body: "Furina filters its algorithm list by task. Logistic Regression is a classification model despite its name; Linear Regression predicts numeric quantities. Decision Trees and Random Forests support both tasks. A simple first run gives you something to compare with later experiments. Also calculate a naive reference, such as predicting the most common class, so a plausible-looking score has context.",
          },
          {
            title: "Read the settings before changing them",
            body: "The hyperparameter form comes from each algorithm's schema. Untouched settings use library defaults. A tree's maximum depth limits how many successive decisions it can make; a larger forest adds trees and computation. Scaling matters especially for distance-based, margin-based and neural models. Furina can scale numeric features inside the fitted pipeline.",
          },
          {
            title: "Record a reproducible run",
            body: "Give the run a descriptive name, select features, set the split and seed, and press Train. Training runs on the server and can take time. Inspect warnings, including convergence problems or adjusted settings. Keep the dataset version, target, feature list, algorithm, effective settings and library versions with your report. A seed alone does not guarantee identical results across environments.",
          },
        ],
        example: {
          title: "Example: a controlled tree experiment",
          body: "Name one run “delivery-tree-depth-3” and another “delivery-tree-depth-6”. Use the same dataset, target, features, test fraction and seed, changing only maximum depth. Compare validation evidence and the training/test gap before deciding whether the extra complexity helped.",
        },
        practice: [
          "Pick a task-compatible algorithm and read the explanation beside each setting you plan to change.",
          "Name and train a first run. Read its warnings before interpreting the scores.",
          "Write down one hypothesis for a second run, such as limiting tree depth to reduce memorisation.",
        ],
        quiz: {
          question: "Which is a hyperparameter you choose before fitting a Decision Tree?",
          options: ["The test accuracy produced after training", "The maximum depth allowed for the tree", "The split rules learned from the rows"],
          correctIndex: 1,
          explanation: "Maximum depth constrains the learning process. The split rules are learned during fitting, and test accuracy is measured afterward.",
        },
      },
      {
        id: "validation-and-tuning",
        title: "Validate and improve carefully",
        minutes: 7,
        intro: "One split is one piece of evidence. Cross-validation fits several models on different subsets to show how performance varies across held-out folds.",
        sections: [
          {
            title: "Understand the fold scores",
            body: "In five-fold cross-validation, each fold is scored once using a model fitted on the other four. Furina refits the entire pipeline for each fold. It reports accuracy for classification and R² for regression, plus the mean and standard deviation. A small spread means similar scores on these folds; it is not a confidence interval or a guarantee about new populations. Accuracy can still hide poor results on rare classes.",
          },
          {
            title: "Know which data Furina cross-validates",
            body: "The optional cross-validation setting uses the whole usable dataset, including rows used by the separate train/test split. It is a diagnostic alongside that split. If you use these fold scores to select settings, the reported test split is no longer a fully independent final check of that selection. Reserve separate final test data outside this workflow, or use a workflow that tunes only within training data, before making a final performance claim.",
          },
          {
            title: "Change settings for a reason",
            body: "A strong training score with a weaker validation score suggests overfitting. Try limiting complexity, improving the feature set or gathering more representative data. Weak scores on both can indicate underfitting, uninformative inputs or noisy labels. Compare one deliberate change at a time. Repeatedly picking the best test score turns that test set into part of model selection.",
          },
        ],
        example: {
          title: "Example: read variability, not just the mean",
          body: "Illustrative fold accuracies of 0.80, 0.81, 0.79, 0.82 and 0.78 have a mean of 0.80. The mean alone does not tell you whether the rare class was found, whether groups overlapped across folds, or whether the dataset represents future use.",
        },
        practice: [
          "Enable cross-validation on Data Training and select a fold count the dataset can support.",
          "Inspect the actual folds, scoring metric and warnings in the evaluation; Furina may reduce folds for small classes.",
          "Record what changed between runs and what data informed the choice. Keep final evaluation data separate from that decision.",
        ],
        quiz: {
          question: "You use Furina's whole-dataset cross-validation scores to choose a model. What supports an independent final performance claim?",
          options: ["Reporting only the highest training score", "Trying seeds until the reported test score improves", "Evaluating on separate data untouched by model selection"],
          correctIndex: 2,
          explanation: "The final evaluation must use data that did not influence model or hyperparameter selection. Whole-dataset cross-validation includes the rows in Furina's reported test split.",
        },
      },
    ],
    references: [
      { title: "scikit-learn: common pitfalls and data leakage", url: "https://scikit-learn.org/stable/common_pitfalls.html" },
      { title: "scikit-learn: cross-validation", url: "https://scikit-learn.org/stable/modules/cross_validation.html" },
    ],
  },
  {
    id: "evaluation",
    level: "Beginner",
    track: "Tabular ML",
    prerequisites: ["Understand targets, features and a train/test split."],
    prerequisiteCourses: ["training"],
    practiceMode: "workbench",
    nextCourseId: "base-models",
    title: "Model Evaluation",
    description: "Read a model's results with context. Understand mistakes, compare runs fairly, and decide what the evidence supports.",
    outcomes: [
      "Recognise overfitting and weak baselines",
      "Read classification metrics and confusion matrices",
      "Interpret regression errors in the right units",
      "Compare runs and communicate limitations",
    ],
    toolPath: "/app/evaluation",
    toolLabel: "Open Model Evaluation",
    lessons: [
      {
        id: "read-the-evidence",
        title: "Read the evidence behind a score",
        minutes: 5,
        intro: "An evaluation measures a particular model on particular rows. Before deciding a score is good, check which dataset, target, split and task produced it.",
        sections: [
          {
            title: "Compare training and test performance",
            body: "Furina shows metrics on both splits. Training performance describes rows the model fitted; test performance describes held-out rows. For higher-is-better metrics, a much higher training score can signal overfitting. For errors such as MAE, the concerning pattern is much lower training error. There is no universal acceptable gap: sample size and the problem both matter.",
          },
          {
            title: "Ask what a simple rule would achieve",
            body: "If 95 of 100 parcels arrive on time, predicting “on time” for every parcel reaches 95% accuracy while missing every late one. For a numeric target, a constant prediction gives a useful reference error. Compare against such a baseline on the same evaluation rows and choose a metric that reflects which mistakes matter.",
          },
          {
            title: "Check warnings and missing evidence",
            body: "Confirm the report's dataset and run name, especially after changing the active dataset. Read warnings before the headline metric. A missing metric is not zero: the model may not expose probabilities, the split may omit a class, or the score may be undefined. Small samples and unrepresentative rows limit what even a high score can tell you.",
          },
        ],
        example: {
          title: "Example: a suspicious gap",
          body: "A tree gets 100% training accuracy and 72% test accuracy. This suggests it may have memorised details of its training rows. Check sample sizes, leakage and the baseline, then compare a less complex model using validation data. The gap alone does not prove the cause.",
        },
        practice: [
          "Open a completed run in Model Evaluation and confirm its dataset, target and row counts.",
          "Compare the same metric across training and test splits, remembering whether higher or lower is better.",
          "Write one simple baseline and one limitation to consider before calling the model useful.",
        ],
        quiz: {
          question: "A model predicts only the majority class and gets 95% accuracy. What should you inspect next?",
          options: ["The confusion matrix and per-class recall", "Only whether training accuracy is also 95%", "Whether the run has a more complex algorithm name"],
          correctIndex: 0,
          explanation: "The confusion matrix and per-class recall reveal whether the minority class is missed. Accuracy alone can hide that failure.",
        },
      },
      {
        id: "classification-metrics",
        title: "Understand classification mistakes",
        minutes: 8,
        intro: "A classifier can make different kinds of mistakes. A confusion matrix counts them, and precision and recall describe them from different viewpoints.",
        sections: [
          {
            title: "Read actual rows and predicted columns",
            body: "In Furina's matrix, rows are actual labels and columns are predictions. The diagonal contains correct labels. For a chosen positive class, a false positive is a false alarm and a false negative is a missed case. Precision asks how many positive predictions were right; recall asks how many actual positives were found. F1 combines both using their harmonic mean.",
          },
          {
            title: "Inspect classes as well as averages",
            body: "Furina's headline precision, recall, F1 and F2 use support-weighted averages across classes, so common classes influence them more. The per-class table reports each label separately; support is the number of actual test examples of that label. Balanced accuracy averages recall across classes. Check the class that matters to your question even when the overall result looks strong.",
          },
          {
            title: "Probabilities answer another question",
            body: "ROC AUC evaluates ranking across decision thresholds, while log loss evaluates predicted probabilities and penalises confident mistakes. Higher AUC and lower log loss are better. Neither replaces inspection of class-specific errors at the decisions you actually use. Furina can leave probability-based metrics unavailable when the estimator or test split does not support them.",
          },
        ],
        example: {
          title: "Example: finding late parcels",
          body: "Treat “late” as positive. Finding 16 of 20 late parcels gives 80% recall. If the model flags 24 parcels as late and 16 really are, precision is 16/24, or about 66.7%. These are positive-class scores; Furina's headline weighted averages also include the other class.",
        },
        experiment: "classification",
        practice: [
          "Use the example below to trade false alarms against missed late parcels.",
          "In a classification report, locate the off-diagonal cells and describe one kind of error in everyday words.",
          "Read precision, recall and support for each class, then decide which errors are more costly for your use case.",
        ],
        quiz: {
          question: "Out of 20 actual late parcels, a model finds 16. It flags 24 parcels in total. What is recall for “late”?",
          options: ["16 / 24, about 66.7%", "16 / 20, or 80%", "24 / 100, or 24%"],
          correctIndex: 1,
          explanation: "Recall divides found positives by actual positives: 16/20. Dividing by all positive predictions gives precision instead.",
        },
      },
      {
        id: "regression-metrics",
        title: "Measure numeric prediction errors",
        minutes: 7,
        intro: "Regression errors describe how far predictions are from real quantities. Interpret them in the target's units before deciding whether the model is accurate enough.",
        sections: [
          {
            title: "Keep the units attached",
            body: "MAE is the average absolute miss. RMSE squares misses before averaging and taking a square root, making it more sensitive to large errors. Both use the target's units; MSE uses squared units. For delivery hours, MAE of 2 means a two-hour average absolute error, not 2%. Maximum error is the worst observed miss in this test set, not a bound on future errors.",
          },
          {
            title: "Use R² and MAPE with context",
            body: "For a nonconstant target, R² compares squared error to predicting the evaluated targets' mean: 1 is perfect, 0 matches that reference and negative values are worse. It is not classification accuracy and can be below −1. MAPE measures relative absolute error; near-zero targets can dominate it, and Furina omits it when a true target is zero.",
          },
          {
            title: "Look at the shape of the mistakes",
            body: "The actual-versus-predicted plot should lie near the equality line for accurate predictions. Points consistently below that line mean underprediction. A residual is actual minus predicted; positive residuals mean the prediction was too low. Look for bias and unusually large misses. Furina plots a sample, so the chart may not show every test row.",
          },
        ],
        example: {
          title: "Example: three delivery estimates",
          body: "Actual times are 10, 20 and 30 hours; predictions are 12, 18 and 36. Absolute errors are 2, 2 and 6 hours. MAE is 10/3 ≈ 3.33 hours and RMSE is √(44/3) ≈ 3.83 hours. RMSE rises more when one prediction becomes a large miss.",
        },
        experiment: "regression",
        practice: [
          "Move the final prediction below and observe how one large error changes MAE and RMSE.",
          "In a regression report, write the target's units next to MAE and RMSE and compare with a practical tolerance.",
          "Inspect the plot for underprediction or overprediction and note whether a few large misses need closer review.",
        ],
        quiz: {
          question: "A delivery-time model has MAE = 3.5 with a target measured in hours. What does this mean?",
          options: ["Every prediction is at most 3.5 hours wrong", "The model is 96.5% accurate", "The average absolute test error is 3.5 hours"],
          correctIndex: 2,
          explanation: "MAE describes an average absolute error in the target's units. It is neither a maximum error nor an accuracy percentage.",
        },
      },
      {
        id: "compare-and-report",
        title: "Compare runs and explain your result",
        minutes: 6,
        intro: "A useful evaluation ends with a decision and an explanation. Combine metrics, error patterns and the conditions of the experiment before choosing your next step.",
        sections: [
          {
            title: "Compare like with like",
            body: "Use the Dashboard to compare runs of the same task, then verify they use the same target, dataset version and evaluation split. The Dashboard groups by task, but does not guarantee these other conditions match. Higher accuracy or R² is better; lower MAE or log loss is better. Keep split settings fixed for controlled comparisons and avoid selecting models by repeatedly chasing the best test score.",
          },
          {
            title: "Treat importance as a clue",
            body: "Feature importance or coefficient magnitude describes how this fitted model uses inputs. It does not prove an input causes an outcome. Encoded categories appear as expanded features; correlated inputs can share or obscure their contributions, and coefficient magnitudes depend on scaling. Some algorithms do not expose importance. Inspect suspiciously dominant features for leakage.",
          },
          {
            title: "Communicate what was actually measured",
            body: "Download the Markdown evaluation report and, when useful, the fitted .joblib pipeline. Record the dataset, target, split, effective settings, main metric, class-specific or residual findings and limitations. The prediction form helps you try plausible inputs, but a few sensible outputs do not establish generalisation. The saved pipeline is the model fitted on the training split, with its preprocessing included.",
          },
        ],
        example: {
          title: "Example: a short evaluation note",
          body: "“On our held-out delivery rows, the model's MAE was 3.3 hours and RMSE was 3.8 hours. Long deliveries tended to be underestimated. Both candidate runs used the same rows and seed. We still need to test on a later time period before using these estimates for new delivery schedules.” These are illustrative results, not a run from your workspace.",
        },
        practice: [
          "Compare two runs in the Dashboard, checking the target, dataset, split and metric direction yourself.",
          "Inspect important inputs or error patterns and write one plausible explanation to investigate.",
          "Download the evaluation report and add a short conclusion: what works, where it fails, and what evidence is still needed.",
        ],
        quiz: {
          question: "A feature has the highest importance in a Random Forest. What can you conclude?",
          options: ["It is influential in this fitted model, but importance does not establish causation", "Changing it will definitely cause the target to change", "It will be the most important feature in every future model"],
          correctIndex: 0,
          explanation: "Importance describes a fitted model's use of features. It can reflect correlations, data leakage and the model's own structure; it is not causal evidence.",
        },
      },
    ],
    references: [
      { title: "scikit-learn: metrics and scoring", url: "https://scikit-learn.org/stable/modules/model_evaluation.html" },
      { title: "scikit-learn: cross-validation and model selection", url: "https://scikit-learn.org/stable/modules/cross_validation.html" },
    ],
  },
];
