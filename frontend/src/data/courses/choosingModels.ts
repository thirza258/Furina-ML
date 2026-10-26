import type { Course } from "./types";

export const choosingModelsCourse: Course = {
  id: "choosing-models",
  level: "Intermediate",
  track: "Tabular ML",
  prerequisites: [
    "Train a run on Data Training and read its results in Model Evaluation.",
    "Python is optional: the code examples use scikit-learn and are there for readers who want to see the mechanism.",
  ],
  prerequisiteCourses: ["training", "evaluation"],
  practiceMode: "workbench", dependencyTier: 2,
  nextCourseId: "time-series",
  title: "Choosing a Model",
  description: "Compare model families on the same tabular data and choose one with evidence. Learn what each family assumes and what it costs, then run a fair, documented comparison in the workbench.",
  outcomes: [
    "Start every comparison from a Dummy Baseline",
    "Explain what each model family assumes and when it fits",
    "Know which models give probabilities, importances and fast predictions",
    "Run a fair comparison and justify the model you choose",
  ],
  toolPath: "/app/training",
  toolLabel: "Open Data Training",
  capstone: {
    title: "Capstone: a fair, documented model comparison",
    brief: "Pick a prediction question and a CSV with at least a few hundred labelled rows. Before training anything, fix your decision metric and a shortlist of candidates. Then compare a Dummy Baseline with at least four models from different families in the workbench, under identical conditions, and recommend one model with evidence about performance, stability, cost and interpretability. Furina's cross-validation also uses the test rows, so state that limitation instead of claiming an independent final estimate.",
    deliverables: [
      "A short statement of the question, target, task and decision metric, written before any training, explaining why the metric fits the decision",
      "A run table for the baseline and every candidate: run name, algorithm, changed settings, test and training scores, cross-validation mean and spread, and fit time, including runs you discarded",
      "Dashboard evidence, such as a screenshot of the ranking filtered to one task and dataset with training scores shown",
      "The two closest finalists re-run with two or three other seeds, with those scores added to the table",
      "Downloaded Markdown evaluation reports for the baseline and the recommended model",
      "A recommendation of one or two paragraphs covering the margin over the baseline, ties, cost, probabilities, interpretability and remaining limitations",
    ],
    rubric: [
      "Every run in the main comparison shares the dataset, target, features, test size, seed, stratification, scaling and fold count, and About this run confirms it",
      "The recommended model beats the baseline on the decision metric, and the margin is stated in that metric's units",
      "Close scores are treated as ties using cross-validation spread and the extra seeds, not a single test score",
      "The candidates and metric were fixed in advance, any tuning is described honestly, and the optimism of picking the best run is acknowledged",
      "Cost, probability support and interpretability are weighed explicitly, and the limitations are named",
    ],
  },
  lessons: [
    {
      id: "baseline-first",
      title: "Start every comparison with a baseline",
      minutes: 7,
      intro: "A score means little until you know what a model that ignores every feature would achieve on the same rows. Furina's Dummy Baseline gives you that reference, so each candidate has something concrete to beat.",
      sections: [
        {
          title: "What the Dummy Baseline does",
          body: "Dummy Baseline, in the Baselines family, never looks at the features. For classification it answers from the training labels alone; for regression it predicts one number for every row, the training mean or median. Train it with exactly the target, features, test size, seed and stratification you will use for the real candidates. Its test scores then show what knowing nothing except the training targets achieves on those same held-out rows.",
        },
        {
          title: "Same labels, different probabilities",
          body: "The default strategy, prior, and the most_frequent strategy both predict the most common class for every row, so accuracy, precision, recall and the confusion matrix are identical. Only the probabilities differ. Prior reports the training class shares, such as 0.7 and 0.3; most_frequent reports 1.0 for the majority class and 0 for the rest. Both give every row the same score, so ROC AUC is 0.5, but most_frequent is certain and wrong on every minority row, so its log loss is far worse. Use prior as the reference for probability metrics.",
        },
        {
          title: "What beating the baseline means",
          body: "A candidate should beat the baseline on the metric that matches your decision, not only on accuracy. In a two-class problem with 90% of rows in one class, the baseline reaches 0.90 accuracy, yet its balanced accuracy is 0.50 and it finds none of the rare class. The stratified and uniform strategies show what random guessing scores. For regression, the mean baseline's test R² sits near 0 and is often slightly negative, so judge candidates by how far they cut its MAE. A first model far above the baseline deserves a leakage check.",
        },
      ],
      example: {
        title: "Example: one prediction, two kinds of confidence",
        body: "In a small computed example, 70 of 100 training rows are class 0, and the 20 test rows split 14 to 6. Both strategies predict class 0 for every test row and both score 0.70 accuracy. Prior reports probabilities of 0.7 and 0.3 and has a test log loss of about 0.61. Most_frequent reports 1.0 and 0.0, so each of the six class 1 rows receives zero probability, and scikit-learn's clipped log loss is about 10.8.",
      },
      code: {
        title: "Compare two Dummy strategies",
        source: "import numpy as np\nfrom sklearn.dummy import DummyClassifier\nfrom sklearn.metrics import accuracy_score, log_loss\n\ny_train = np.array([0] * 70 + [1] * 30)\ny_test = np.array([0] * 14 + [1] * 6)\nX_train, X_test = np.zeros((100, 1)), np.zeros((20, 1))  # a baseline ignores features\n\nfor strategy in [\"prior\", \"most_frequent\"]:\n    baseline = DummyClassifier(strategy=strategy).fit(X_train, y_train)\n    proba = baseline.predict_proba(X_test)\n    accuracy = accuracy_score(y_test, baseline.predict(X_test))\n    print(strategy, accuracy, proba[0], round(log_loss(y_test, proba), 3))",
        explanation: "Both lines print the same accuracy. The probabilities and the log loss differ because only the prior strategy admits that class 1 exists. A baseline ignores the features, so an array of zeros is enough.",
      },
      practice: [
        "On Data Training, choose your target and features, keep a seed you will reuse, and train Dummy Baseline with its default Strategy. Name the run, for example “baseline-prior”.",
        "For a classification target, train it again with Strategy set to “Most common class, fully confident”. In Model Evaluation, check that the confusion matrices match while log loss differs.",
        "Record the baseline's accuracy, balanced accuracy and recall for the class you care about, or its MAE and R² for regression. Every later candidate must beat these on the same split.",
      ],
      references: [
        { title: "scikit-learn: DummyClassifier", url: "https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html" },
        { title: "scikit-learn: DummyRegressor", url: "https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html" },
        { title: "scikit-learn: metrics and scoring", url: "https://scikit-learn.org/stable/modules/model_evaluation.html" },
      ],
      quiz: {
        question: "Two Dummy Baseline runs on the same split use the prior and most_frequent strategies. Which metric can differ between them?",
        options: ["Accuracy", "Log loss", "Recall for the majority class"],
        correctIndex: 1,
        explanation: "Both strategies predict the same label for every row, so every label-based metric matches. Only the reported probabilities differ, and log loss scores probabilities.",
      },
    },
    {
      id: "linear-models",
      title: "Linear models and regularisation",
      minutes: 9,
      intro: "Linear models give each feature one weight and add up the results. They are fast and readable, and with a penalty they rarely overfit, as long as you understand how penalties and feature scaling shape the weights.",
      sections: [
        {
          title: "One weight per column",
          body: "Linear Regression predicts a number as an intercept plus a weighted sum of the features. Logistic Regression, despite its name, classifies: it turns the same kind of sum into class probabilities. Ridge Classifier is a fast linear classifier that reports no probabilities. Every one-hot column gets its own weight, so a category such as method_express shifts the prediction by a fixed amount. A linear model cannot invent curves or interactions; if the real relationship bends, it underfits, and its training and test scores are both weak.",
        },
        {
          title: "Penalties keep weights small",
          body: "Regularisation adds a cost for large weights. Ridge's L2 penalty shrinks every weight, Lasso's L1 penalty can set weak ones exactly to zero, and Elastic Net mixes the two through its L1 ratio. In Furina, Regularization strength (alpha) is the penalty, so a larger alpha gives a simpler model. Logistic Regression and the support vector machines use C, which works the other way round: a smaller C regularises harder. Bayesian Ridge Regression estimates its own penalty from the data by learning the noise level and the typical weight size.",
        },
        {
          title: "Outliers and scaling",
          body: "Squared error lets a few extreme targets drag the whole fit. Huber Regression uses squared error for small residuals and absolute error for large ones. Its Outlier threshold (epsilon), 1.35 by default, is measured against a scale it learns, so a handful of corrupted rows matters much less. Penalties are sensitive to units: a column with small numbers needs a large weight, which the penalty punishes, while a column with big numbers needs only a tiny one. Keep Standardise numeric features on for penalised models; Furina never scales one-hot columns.",
        },
      ],
      example: {
        title: "Example: one column, two units",
        body: "In a computed example with three informative columns, one column was divided by 100, as if it had been recorded in a much larger unit. Without scaling, Lasso at alpha 5 set that column's weight to exactly zero, and Ridge at alpha 10 shrank its importance to about 0.1. After standardising, both kept it, with importances close to those of plain Linear Regression, which has no penalty and gave identical importances either way. Furina multiplies each coefficient by its column's spread before ranking importances; that fixes the units, but it cannot undo a fit the penalty has already distorted.",
      },
      code: {
        title: "Huber Regression versus least squares",
        source: "import numpy as np\nfrom sklearn.linear_model import HuberRegressor, LinearRegression\n\nrng = np.random.default_rng(0)\nx = rng.uniform(0, 10, size=(200, 1))\ny = 3.0 * x.ravel() + rng.normal(0, 1, size=200)  # true slope 3, intercept 0\ny[x.ravel() > 9.5] -= 40  # corrupt the targets of the largest x values\n\nfor model in [LinearRegression(), HuberRegressor()]:\n    model.fit(x, y)\n    print(type(model).__name__, model.coef_.round(2), round(model.intercept_, 2))",
        explanation: "Fifteen rows at the largest x values have corrupted targets. Linear Regression's slope drops to about 1.5, while Huber Regression stays near 2.9, close to the true slope of 3.",
      },
      practice: [
        "Train Linear Regression, or Logistic Regression for a classification target, with Standardise numeric features on and your baseline's split and seed. Compare the two runs in the Dashboard.",
        "Train Ridge or Lasso Regression with Regularization strength (alpha) at 0.1, 1 and 10, or Logistic Regression with Regularization strength (C) at 0.1, 1 and 10, changing nothing else. Note how the train–test gap and the Feature importance list change.",
        "If your numeric target has extreme values, train Huber Regression and Bayesian Ridge Regression on the same split and compare their MAE and RMSE with those of Linear Regression.",
      ],
      references: [
        { title: "scikit-learn: linear models", url: "https://scikit-learn.org/stable/modules/linear_model.html" },
        { title: "scikit-learn: HuberRegressor", url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html" },
        { title: "scikit-learn: BayesianRidge", url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html" },
      ],
      quiz: {
        question: "In Furina's Logistic Regression, which change regularises the model more strongly?",
        options: ["Raising Regularization strength (C) from 1 to 10", "Turning off Standardise numeric features", "Lowering Regularization strength (C) from 1 to 0.1"],
        correctIndex: 2,
        explanation: "C is the inverse of the penalty strength, so a smaller C penalises large weights more. Turning scaling off changes which columns the penalty hits hardest, not its overall strength.",
      },
    },
    {
      id: "probabilistic-classifiers",
      title: "Naive Bayes and discriminant analysis",
      minutes: 9,
      intro: "These classifiers model how the features are distributed inside each class, then use Bayes' rule to turn that picture into class probabilities. Their assumptions are strong, which makes them fast, stable on small datasets and sometimes surprisingly competitive.",
      sections: [
        {
          title: "Naive Bayes treats evidence as independent",
          body: "Naive Bayes assumes the features are independent within each class, so it combines one simple model per feature. Gaussian Naive Bayes treats each column as a bell curve per class. Bernoulli Naive Bayes reduces every value to present or absent: values above the Yes/no threshold, 0 by default, count as present. Correlated features are counted as separate evidence, which tends to make the model overconfident. scikit-learn's guide calls naive Bayes a decent classifier but a bad estimator whose probabilities should not be taken too seriously, so check log loss as well as accuracy.",
        },
        {
          title: "One shared shape or one per class",
          body: "Linear Discriminant Analysis models each class as a bell-shaped cloud with its own centre and one covariance matrix shared by all classes. The shared shape makes the boundaries straight, and LDA exposes coefficients, so Furina reports feature importances for it. Quadratic Discriminant Analysis gives every class its own covariance matrix, so boundaries can curve, but it must estimate roughly one number per pair of features for every class. With few rows per class it overfits. It has no coefficients, so Furina shows no importances for it.",
        },
        {
          title: "Shrinkage stabilises the estimates",
          body: "A covariance estimate becomes unreliable, or impossible to invert, when features are collinear or outnumber the rows of a class. One-hot columns are an exact case: the columns made from one categorical feature always sum to one. Shrinkage blends the estimate with a simpler, spherical one, and “Automatic (Ledoit-Wolf)” chooses the amount from the data. Choosing shrinkage with LDA's default solver switches it to least squares and adds a warning to the run. Furina runs QDA with automatic shrinkage by default, because scikit-learn's default QDA stops with an error on one-hot columns.",
        },
      ],
      example: {
        title: "Example: what Bernoulli Naive Bayes sees",
        body: "With Standardise numeric features on, every numeric column is centred on its training-split mean, so the default threshold of 0 turns distance_km into “longer than the average training parcel”. One-hot columns are already 0 or 1 and keep their meaning. With scaling off, the same threshold asks whether a distance is above 0 km. Every parcel answers yes, so the column becomes a constant 1 and carries no information.",
      },
      code: {
        title: "Watch default QDA fail on one-hot columns",
        source: "import numpy as np\nfrom sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n\nrng = np.random.default_rng(0)\ncolour = rng.integers(0, 3, size=300)\nX = np.column_stack([rng.normal(size=300), np.eye(3)[colour]])  # 1 numeric + 3 one-hot columns\ny = (X[:, 0] + (colour == 2) + rng.normal(0, 0.5, size=300) > 0.5).astype(int)\n\ntry:\n    QuadraticDiscriminantAnalysis().fit(X, y)\nexcept np.linalg.LinAlgError as error:\n    print(\"Default QDA failed:\", error)\n\nshrunk = QuadraticDiscriminantAnalysis(solver=\"eigen\", shrinkage=\"auto\").fit(X, y)\nprint(\"Shrunk QDA probabilities:\", shrunk.predict_proba(X[:2]).round(3))",
        explanation: "The three one-hot columns always sum to 1, so each class covariance matrix is singular and the default fit raises an error. With the eigen solver and automatic shrinkage, the same data fits and returns probabilities. This is the configuration Furina uses for Quadratic Discriminant Analysis.",
      },
      practice: [
        "On a classification target, train Gaussian Naive Bayes and Bernoulli Naive Bayes with your baseline's split and seed. Compare log loss and per-class recall as well as accuracy.",
        "Train Bernoulli Naive Bayes once with Standardise numeric features on and once with it off, and explain any difference using the Yes/no threshold.",
        "Train Linear Discriminant Analysis, then again with Covariance shrinkage set to “Automatic (Ledoit-Wolf)”, and read the run's warning. Train Quadratic Discriminant Analysis and compare its train–test gap with LDA's.",
      ],
      references: [
        { title: "scikit-learn: naive Bayes", url: "https://scikit-learn.org/stable/modules/naive_bayes.html" },
        { title: "scikit-learn: linear and quadratic discriminant analysis", url: "https://scikit-learn.org/stable/modules/lda_qda.html" },
        { title: "scikit-learn: shrunk covariance estimation", url: "https://scikit-learn.org/stable/modules/covariance.html" },
      ],
      quiz: {
        question: "Why does Furina run Quadratic Discriminant Analysis with covariance shrinkage by default?",
        options: ["One-hot columns make each class covariance matrix singular, so the unshrunk fit fails", "Shrinkage makes QDA's boundaries straight, like LDA's", "QDA can only report class probabilities when shrinkage is on"],
        correctIndex: 0,
        explanation: "Columns made from one categorical feature always sum to one, so their covariance cannot be inverted. Shrinkage blends in a spherical covariance, which makes the estimate invertible; the boundaries can still curve, and QDA reports probabilities either way.",
      },
    },
    {
      id: "distance-based-models",
      title: "Nearest neighbours and centroids",
      minutes: 8,
      intro: "Distance-based models predict from the training rows, or class centres, that look most similar to a new row. Everything therefore depends on what “similar” means, and your features and their scales decide that.",
      sections: [
        {
          title: "k-NN keeps the training rows",
          body: "K-Nearest Neighbours learns almost nothing during fitting: it stores the training rows and, for each new row, finds the k closest, 5 by default. Classification takes a vote and regression averages the neighbours' targets. A small k follows local noise; a large k smooths towards the majority. Prediction searches the stored rows, so it slows down as the training data grows, unlike a linear model or a tree. Its training score flatters it: each training row is its own nearest neighbour, and with “Closer neighbours count more” training accuracy is close to perfect by construction.",
        },
        {
          title: "Nearest Centroid keeps one centre per class",
          body: "Nearest Centroid summarises each class by a centre and assigns a new row to the closest one. With the Euclidean distance the centre is the class mean. With Manhattan it is the feature-wise median, which resists outliers, but scikit-learn then gives no class probabilities, so ROC AUC and log loss are blank. Centroid shrinkage, under Advanced, pulls every centre towards the overall centre and drops features that barely separate the classes. The model is tiny, fast and easy to explain, but it assumes each class forms one compact cloud.",
        },
        {
          title: "Scale and dimension decide what near means",
          body: "Distances add up differences across columns, so a column measured in thousands drowns out one measured in fractions. With Standardise numeric features on, one unit is one standard deviation in every numeric column. Furina leaves one-hot columns unscaled, so two rows that differ in one category are about 1.4 apart, like a gap of 1.4 standard deviations in a number. As columns multiply, including the many that one-hot encoding creates, distances bunch together and every irrelevant feature adds noise. Give these models a short list of relevant features.",
        },
      ],
      example: {
        title: "Example: scaling and dimension, computed",
        body: "On scikit-learn's wine data, where the largest column spread is about 2,500 times the smallest, five-fold cross-validated k-NN accuracy was 0.68 without scaling and 0.97 with standardisation inside the pipeline. In a separate simulation with 1,000 random points, the farthest point from a query was typically over 100 times as far away as the nearest in 2 dimensions, about 4 times in 10 dimensions and only about 1.1 times in 1,000 dimensions.",
      },
      code: {
        title: "k-NN with and without scaling",
        source: "from sklearn.datasets import load_wine\nfrom sklearn.model_selection import StratifiedKFold, cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\nX, y = load_wine(return_X_y=True)\nspread = X.std(axis=0)\nprint(f\"largest column spread is {spread.max() / spread.min():.0f} times the smallest\")\n\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nraw = cross_val_score(KNeighborsClassifier(), X, y, cv=cv)\nscaled = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier()), X, y, cv=cv)\nprint(f\"unscaled accuracy {raw.mean():.3f}, scaled accuracy {scaled.mean():.3f}\")",
        explanation: "Both pipelines are scored on the same five folds. Because StandardScaler sits inside the pipeline, each fold learns its scaling from its own training rows, as Furina does.",
      },
      practice: [
        "Train K-Nearest Neighbours with Standardise numeric features on, then again with it off, keeping the split and seed fixed. Compare the test metric.",
        "Try Number of neighbours (k) at 1, 5 and 25 and record each run's test score. Judge k by test or cross-validation scores, not by the train–test gap.",
        "For a classification target, train Nearest Centroid with “Euclidean (straight line)” and with “Manhattan (city block)”, and note which probability metrics disappear.",
      ],
      references: [
        { title: "scikit-learn: nearest neighbours", url: "https://scikit-learn.org/stable/modules/neighbors.html" },
        { title: "scikit-learn: NearestCentroid", url: "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html" },
        { title: "scikit-learn: importance of feature scaling", url: "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html" },
      ],
      quiz: {
        question: "You train k-NN on income, in tens of thousands, and age, in tens, with Standardise numeric features off. What happens?",
        options: ["Age dominates, because its values change more often", "Income dominates the distance, so age barely affects which rows count as nearest", "k-NN rescales the columns itself before measuring distance"],
        correctIndex: 1,
        explanation: "k-NN uses raw differences. A typical income difference is thousands of times larger than an age difference, so income decides the neighbours unless the columns are standardised.",
      },
    },
    {
      id: "margins-and-kernels",
      title: "Support vector machines",
      minutes: 8,
      intro: "A support vector machine looks for the boundary with the widest margin between classes. Kernels let it draw curved boundaries, at a cost that grows quickly with the number of training rows.",
      sections: [
        {
          title: "Margins and the C setting",
          body: "Only the rows on or inside the margin, the support vectors, determine the boundary. Error penalty (C) sets how much margin violations cost: a small C accepts more training mistakes in exchange for a wider, simpler margin, and a large C chases every training row and risks overfitting. Both Furina versions work with distances or dot products between rows, so keep Standardise numeric features on. For regression, Support Vector Machine fits an SVR, which ignores errors smaller than its Ignored error band, 0.1 by default, in the target's units.",
        },
        {
          title: "Kernels and their cost",
          body: "The default RBF kernel measures a similarity that fades with distance, and Kernel reach (gamma) controls how quickly it fades. This lets the model draw curved boundaries without new features, but the solver works with similarities between pairs of rows. scikit-learn's documentation says SVC's fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands, so ten times the rows can mean a hundred times the training time or more. Prediction also slows as the number of support vectors grows.",
        },
        {
          title: "Linear SVM and missing probabilities",
          body: "Linear Support Vector Machine, available for classification only, uses a different solver that scales much better with rows, so it is the practical choice once kernel training becomes slow. It draws only straight boundaries, offers an L1 penalty that drops weak features, and exposes coefficients for importances. It has no predict_proba, so ROC AUC and log loss show n/a, and the run drops out of any Dashboard ranking by those metrics. Furina's kernel model gets probabilities from an extra internal cross-validated calibration, which slows fitting; switch off Estimate probabilities when you do not need them.",
        },
      ],
      example: {
        title: "Example: deciding between the two",
        body: "Illustrative: a kernel Support Vector Machine takes 20 seconds on 5,000 training rows. Because its cost grows at least quadratically, 50,000 rows could take more than half an hour. If Linear Support Vector Machine or Logistic Regression scores within your tolerance of it on the same split, the linear model is the cheaper and more scalable choice. If the curved model is clearly better, check whether a boosted-tree model reaches the same score at lower cost before accepting the slow one.",
      },
      code: {
        title: "Watch training time grow",
        source: "import time\nfrom sklearn.datasets import make_classification\nfrom sklearn.svm import SVC, LinearSVC\n\nfor n_rows in [2000, 4000, 8000]:\n    X, y = make_classification(n_samples=n_rows, n_features=20, flip_y=0.1, random_state=0)\n    for model in [SVC(), LinearSVC()]:\n        start = time.perf_counter()\n        model.fit(X, y)\n        print(n_rows, type(model).__name__, f\"{time.perf_counter() - start:.3f} s\")\n\nprint(\"LinearSVC has predict_proba:\", hasattr(LinearSVC(), \"predict_proba\"))",
        explanation: "Timings depend on your machine. On typical hardware, the kernel SVC's time grows much faster than the number of rows, while LinearSVC stays far smaller. The last line confirms that LinearSVC offers no predict_proba.",
      },
      practice: [
        "On a classification target, train Support Vector Machine with its defaults and note the training time on the result card. Train it again with Estimate probabilities off and compare the time and the probability metrics.",
        "Train Linear Support Vector Machine on the same split and compare accuracy, balanced accuracy and F1 with the kernel model. Confirm that ROC AUC and log loss show n/a.",
        "Try Error penalty (C) at 0.1, 1 and 10 on one of the two models, changing nothing else, and record the training and test scores of each run.",
      ],
      references: [
        { title: "scikit-learn: support vector machines", url: "https://scikit-learn.org/stable/modules/svm.html" },
        { title: "scikit-learn: SVC", url: "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html" },
        { title: "scikit-learn: LinearSVC", url: "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html" },
      ],
      quiz: {
        question: "A Linear Support Vector Machine run shows n/a for ROC AUC and log loss. Why?",
        options: ["Standardise numeric features was turned off", "The test split was too small for probability metrics", "The model gives labels and decision scores but no class probabilities"],
        correctIndex: 2,
        explanation: "LinearSVC has no predict_proba, so Furina has no probabilities to score. Compare it with other models on label-based metrics such as accuracy, balanced accuracy and F1.",
      },
    },
    {
      id: "trees-and-bagging",
      title: "Trees, bagging and random forests",
      minutes: 8,
      intro: "A decision tree asks a sequence of yes/no questions about the features. One deep tree tends to memorise its training rows; many randomised trees averaged together generalise far better.",
      sections: [
        {
          title: "One tree is flexible and unstable",
          body: "Each split compares one feature with a threshold, so trees capture curves and interactions without feature engineering. Only the order of values matters, so standardising does not change a scikit-learn tree's predictions. Left unlimited, a tree keeps splitting until its leaves are pure, which usually gives near-perfect training scores and much weaker test scores. Small changes in the training rows can also produce a very different tree. Limiting Maximum tree depth or raising Minimum rows per leaf reduces this variance, at the cost of some bias.",
        },
        {
          title: "Bagging and forests average away variance",
          body: "Bagging trains many full trees, each on a bootstrap sample of the rows drawn with replacement, and averages their votes or predictions. Their individual mistakes partly cancel, so variance falls while each tree stays flexible. Furina's Bagging uses 10 trees by default and reports no feature importances, because the bag itself exposes none. Random Forest also gives each split a random subset of features, the square root of the feature count for classification, which makes the trees more different. For regression it considers all features by default, which scikit-learn notes is equivalent to bagged trees.",
        },
        {
          title: "Extra Trees and reading a forest",
          body: "Extra Trees adds more randomness: it draws a random threshold for each candidate feature and keeps the best of those, and by default every tree sees all the rows. It usually trains faster and often generalises as well. Expect forests to score close to perfect on their training rows, so compare their test and cross-validation scores rather than the gap. Their importances measure impurity reduction on training data and favour features with many distinct values. Costs grow with the number of trees: memory, the size of the .joblib file and prediction time.",
        },
      ],
      example: {
        title: "Example: one tree against averaged trees, computed",
        body: "On scikit-learn's breast cancer data with the same five folds, one unlimited decision tree averaged 0.916 accuracy, Bagging with 10 trees 0.944, Random Forest with 100 trees 0.954 and Extra Trees 0.961. Training the same tree on standardised columns gave exactly the same predictions. These results come from one dataset and one set of folds; they show a typical pattern, not a ranking that holds everywhere.",
      },
      code: {
        title: "Compare a tree with three ensembles",
        source: "import numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier\nfrom sklearn.model_selection import StratifiedKFold, cross_val_score\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.tree import DecisionTreeClassifier\n\nX, y = load_breast_cancer(return_X_y=True)\nX_scaled = StandardScaler().fit_transform(X)  # only to compare predictions, not to evaluate\nraw_tree = DecisionTreeClassifier(random_state=0).fit(X, y)\nscaled_tree = DecisionTreeClassifier(random_state=0).fit(X_scaled, y)\nprint(\"same predictions:\", np.array_equal(raw_tree.predict(X), scaled_tree.predict(X_scaled)))\n\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nfor model in [DecisionTreeClassifier(random_state=0), BaggingClassifier(random_state=0),\n              RandomForestClassifier(random_state=0), ExtraTreesClassifier(random_state=0)]:\n    scores = cross_val_score(model, X, y, cv=cv)\n    print(f\"{type(model).__name__:24} mean {scores.mean():.3f}  spread {scores.std():.3f}\")",
        explanation: "The first check shows that scaling leaves a tree's predictions unchanged. The loop then scores four models on identical folds, so the differences come from the models rather than from the split.",
      },
      practice: [
        "Train Decision Tree with its defaults, then with Maximum tree depth set to 4, and compare the Train vs test table in Model Evaluation.",
        "Train Bagging, Random Forest and Extra Trees with the same split and seed, then rank them in the Dashboard with Show training scores on.",
        "Train Random Forest once with Standardise numeric features on and once with it off. Confirm that the scores match, and compare the training times.",
      ],
      references: [
        { title: "scikit-learn: decision trees", url: "https://scikit-learn.org/stable/modules/tree.html" },
        { title: "scikit-learn: ensembles, bagging and forests", url: "https://scikit-learn.org/stable/modules/ensemble.html" },
        { title: "scikit-learn: permutation feature importance", url: "https://scikit-learn.org/stable/modules/permutation_importance.html" },
      ],
      quiz: {
        question: "Why does a Random Forest usually beat a single unlimited decision tree on test data?",
        options: ["Averaging many decorrelated trees cancels much of each tree's variance", "Each tree in the forest is shallower, so the forest has less bias", "The forest rescales the features before splitting"],
        correctIndex: 0,
        explanation: "Forest trees are still deep and flexible by default. The gain comes from averaging trees that make different mistakes, which bootstrap samples and random feature subsets encourage.",
      },
    },
    {
      id: "boosting",
      title: "Boosted trees",
      minutes: 9,
      intro: "Boosting builds trees one after another, each correcting the mistakes of the ensemble so far. It is often the strongest family on tabular data, and it is also easy to overfit by adding too many rounds.",
      sections: [
        {
          title: "Each tree corrects the ensemble",
          body: "Gradient Boosting fits each new, usually shallow tree to the remaining errors of the current ensemble, more precisely to the gradient of the loss, and adds a fraction of it set by the learning rate. AdaBoost instead reweights the rows, so later trees concentrate on rows that earlier ones got wrong; by default its classifier uses one-split trees and its regressor uses trees of depth 3. A forest averages independent trees to cut variance; boosting mainly cuts bias, so every extra round can also fit more noise.",
        },
        {
          title: "Five implementations in Furina",
          body: "Gradient Boosting is scikit-learn's classic version, with learning rate 0.1 and depth 3, and it is slow on large tables. Histogram Gradient Boosting first buckets each numeric feature into at most 255 bins, which makes it much faster from about ten thousand rows; it reports no importances. XGBoost defaults to learning rate 0.3 and depth 6, and LightGBM to 0.1 with up to 31 leaves grown leaf by leaf. Comparing them at their defaults therefore compares these settings as much as the algorithms.",
        },
        {
          title: "Learning rate, rounds and early stopping",
          body: "The learning rate and the number of trees trade off: a lower rate needs more rounds for a similar fit, and scikit-learn's guide notes that small rates tend to give better test error. Too many rounds at a high rate overfit, which shows as a widening train–test gap. Early stopping holds back part of the training rows and stops adding trees once that held-out score stops improving. Among Furina's boosted models, only Histogram Gradient Boosting offers it, under Advanced; “Automatic” switches it on only when the training split has more than 10,000 rows.",
        },
      ],
      example: {
        title: "Example: one model at two learning rates, computed",
        body: "On a synthetic dataset of 3,000 rows with 10% of labels assigned at random, Gradient Boosting at learning rate 0.5 reached its best validation log loss, 0.387, after only 6 trees; by 500 trees it had risen to 0.767. At learning rate 0.05 the best, 0.372, came after 328 trees, and 500 trees still gave 0.375. The lower rate was slightly better and far less sensitive to the number of rounds.",
      },
      code: {
        title: "Track validation loss tree by tree",
        source: "import numpy as np\nfrom sklearn.datasets import make_classification\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.metrics import log_loss\nfrom sklearn.model_selection import train_test_split\n\nX, y = make_classification(n_samples=3000, n_features=20, n_informative=6, flip_y=0.1, random_state=0)\nX_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)\n\nfor rate in [0.5, 0.05]:\n    model = GradientBoostingClassifier(learning_rate=rate, n_estimators=500, random_state=0)\n    model.fit(X_train, y_train)\n    losses = [log_loss(y_val, proba) for proba in model.staged_predict_proba(X_val)]\n    best = int(np.argmin(losses)) + 1\n    print(f\"rate {rate}: best validation log loss {min(losses):.3f} at {best} trees, {losses[-1]:.3f} at 500\")",
        explanation: "staged_predict_proba replays the ensemble after each added tree, so one fit shows the whole curve. Choosing the best round on this validation split is a simple form of early stopping; no test data is involved.",
      },
      practice: [
        "Train Histogram Gradient Boosting, XGBoost and LightGBM with their defaults on the same split and seed. Record each one's test metric, training score and training time.",
        "Pick one and train a short ladder, changing nothing else: learning rate 0.1 with 100 trees or rounds, 0.05 with 200 and 0.02 with 500. Turn cross-validation on and compare the means and spreads.",
        "Open an XGBoost run in Model Evaluation. Hyperparameters actually used lists untouched settings as “default”, so write the library's real defaults (100 rounds, learning rate 0.3, depth 6) into your notes.",
      ],
      references: [
        { title: "scikit-learn: gradient boosting", url: "https://scikit-learn.org/stable/modules/ensemble.html" },
        { title: "scikit-learn: early stopping in gradient boosting", url: "https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html" },
        { title: "XGBoost: parameters", url: "https://xgboost.readthedocs.io/en/stable/parameter.html" },
        { title: "LightGBM: parameters tuning", url: "https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html" },
      ],
      quiz: {
        question: "A boosted model's training score keeps rising with more trees while its cross-validated score falls. What should you try?",
        options: ["Add more trees at the same learning rate", "Turn on Standardise numeric features", "Use fewer trees, or a lower learning rate, chosen with validation evidence"],
        correctIndex: 2,
        explanation: "Rising training scores with falling validation scores mean the extra rounds are fitting noise. Reduce the rounds, lower the learning rate with more rounds, or use early stopping. Scaling does not address this.",
      },
    },
    {
      id: "compare-and-choose",
      title: "Compare fairly and choose",
      minutes: 10,
      intro: "A comparison is only as fair as its weakest control. Fix the data and the split, give every candidate the same evidence, and then choose with cost and interpretability in mind as well as the score.",
      sections: [
        {
          title: "Change only the model",
          body: "Use one dataset, one target and one feature list. Keep the test size, seed and stratification identical, so every candidate is scored on the same held-out rows, and set the cross-validation folds once. Keep Standardise numeric features on for every run: scikit-learn's trees and XGBoost ignore it, but LightGBM's results can change slightly when it is switched. Name each run after its model and the one setting you changed. The Dashboard filters by task and dataset but does not check seeds, splits or features, so confirm them in About this run.",
        },
        {
          title: "Read rankings with their uncertainty",
          body: "Rank runs in the Dashboard by the metric that matches your decision, with Show training scores on. A lead of 0.005 accuracy on 200 test rows is a single row, so treat close scores as ties and compare cross-validation means and spreads. Furina's cross-validation always scores accuracy or R², and it uses every row, including the test split. Each time you pick the best of many runs by test score, that score becomes optimistic: with enough candidates, even coin flips look skilled. Re-run close finalists with two or three other seeds.",
        },
        {
          title: "Choose with cost and use in mind",
          body: "When scores tie, prefer the model that is cheaper, simpler or better suited to its job. Fit time in Furina also covers evaluation and any cross-validation, so compare it only between runs with the same settings. Consider prediction speed, which falls for k-NN and kernel models as the training data grows, and the size of the .joblib file. Ask whether you need probabilities, coefficients or importances, and never compare importance values across libraries: LightGBM counts splits, XGBoost reports gain shares. A model people can explain may beat a slightly stronger one they cannot audit.",
        },
      ],
      example: {
        title: "Example: a short comparison note",
        body: "Illustrative: the baseline scores 0.70 accuracy. Logistic Regression scores 0.81, with cross-validation 0.80 ± 0.02; Random Forest 0.83, with 0.81 ± 0.03 and a training score of 1.00; LightGBM 0.83, with 0.82 ± 0.02. All three clear the baseline, the forest and LightGBM tie within their spread, and Logistic Regression is close behind. A defensible note recommends LightGBM, or Logistic Regression if readable coefficients matter more, and records the reason.",
      },
      code: {
        title: "Why the best test score flatters",
        source: "import numpy as np\n\nrng = np.random.default_rng(0)\ny_test = rng.integers(0, 2, size=200)  # 200 held-out labels\n\nscores = []\nfor candidate in range(50):  # 50 \"models\" that are coin flips\n    guesses = rng.integers(0, 2, size=200)\n    scores.append((guesses == y_test).mean())\n\nprint(f\"best test accuracy {max(scores):.3f}, median {np.median(scores):.3f}\")",
        explanation: "Every candidate is a coin flip with a true accuracy of 0.5, yet the best of 50 scored 0.580 on these 200 test rows, while the median scored 0.492. Picking the winner on the test set rewards luck, and the more candidates you try, the more the winner's score overstates its real skill.",
      },
      practice: [
        "Choose four to six candidates from different families, plus Dummy Baseline. Train each with the same split, seed, features, scaling and cross-validation folds, and name every run.",
        "In the Dashboard, set Task and Dataset, choose your decision metric and turn on Show training scores. Note ties, large gaps and fit times.",
        "Open the top runs in Model Evaluation, confirm their settings in About this run, and download the Markdown evaluation report for the baseline and for your choice.",
      ],
      references: [
        { title: "scikit-learn: cross-validation", url: "https://scikit-learn.org/stable/modules/cross_validation.html" },
        { title: "scikit-learn: tuning hyper-parameters", url: "https://scikit-learn.org/stable/modules/grid_search.html" },
        { title: "scikit-learn: common pitfalls and data leakage", url: "https://scikit-learn.org/stable/common_pitfalls.html" },
      ],
      quiz: {
        question: "You trained 30 configurations and kept the one with the best test accuracy. What does that accuracy tell you?",
        options: ["It is an unbiased estimate of accuracy on new data", "It is probably optimistic, because the test rows were used to make the choice", "It equals the model's cross-validation mean"],
        correctIndex: 1,
        explanation: "Choosing the maximum of many noisy scores favours lucky runs. An independent estimate needs rows that played no part in any choice, set aside before you upload the data.",
      },
    },
  ],
  references: [
    { title: "scikit-learn: choosing the right estimator", url: "https://scikit-learn.org/stable/machine_learning_map.html" },
    { title: "scikit-learn: supervised learning", url: "https://scikit-learn.org/stable/supervised_learning.html" },
    { title: "scikit-learn: common pitfalls and data leakage", url: "https://scikit-learn.org/stable/common_pitfalls.html" },
  ],
};
