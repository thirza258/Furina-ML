/**
 * The app's teaching surface.
 *
 * Every knob the user can turn — a preprocessing operation, a metric on the
 * evaluation page, a training option — has an entry here, written for someone
 * who has never trained a model before. Pages render `short` in a tooltip and
 * `long` in a help panel; nothing here is decorative.
 *
 * Keys are the exact strings the API uses, not prettified labels, so a page
 * that already holds an op name (`"fill_median"`), a metric key out of
 * `run.metrics` (`"roc_auc"`) or a `TrainRequest` field (`"test_size"`) can
 * look help up directly with no translation table in between.
 */
import type { PreprocessOp } from "../interface/MlTypes";

export interface HelpEntry {
  /** Human title for the panel heading. */
  title: string;
  /** One sentence. Must fit in a tooltip. */
  short: string;
  /** A paragraph or two. Explains what it does, and when you would want it. */
  long: string;
}

/* ------------------------------------------------------------------ */
/* Preprocessing operations                                            */
/* The Record<PreprocessOp, …> annotation makes tsc reject the file if */
/* an op from the contract is ever missing.                            */
/* ------------------------------------------------------------------ */

const OP_HELP: Record<PreprocessOp, HelpEntry> = {
  remove_duplicates: {
    title: "Remove duplicate rows",
    short: "Deletes rows that are identical to a row above them.",
    long:
      "Two identical rows carry no more information than one, but the model " +
      "sees the repeated pattern twice and weighs it twice as heavily. " +
      "Duplicates also leak between the training and test split: the same row " +
      "can land on both sides, so the model is scored on an example it has " +
      "already memorised and your accuracy looks better than it is. Only " +
      "exact, whole-row matches are removed — a row that differs in a single " +
      "column is kept.",
  },
  drop_columns: {
    title: "Drop columns",
    short: "Removes the columns you pick from the dataset entirely.",
    long:
      "Use this on columns that cannot help a model: row ids, names, free-text " +
      "notes, or anything recorded after the thing you are predicting already " +
      "happened. That last case is the dangerous one — a column that secretly " +
      "encodes the answer gives you a near-perfect score in training and a " +
      "useless model in the real world. Dropping is permanent for the new " +
      "dataset, but your original upload is never modified.",
  },
  drop_missing_rows: {
    title: "Drop rows with missing values",
    short: "Deletes every row that has at least one empty cell.",
    long:
      "The bluntest way to deal with gaps in your data. It is safe when only a " +
      "handful of rows are affected and honest when the gaps are random. It is " +
      "a bad idea when one column is mostly empty — you would throw away " +
      "thousands of good rows because of one bad column, and it is usually " +
      "better to drop that column instead. Watch the row count in the report: " +
      "if it falls sharply, undo this and fill the gaps instead.",
  },
  fill_mean: {
    title: "Fill missing values with the mean",
    short: "Replaces empty numeric cells with that column's average.",
    long:
      "Keeps the row instead of deleting it, and keeps the column's average " +
      "unchanged. Best for roughly symmetric numbers such as height or " +
      "temperature. The mean is dragged around by extreme values, so on a " +
      "skewed column like income or house price the median is the safer " +
      "choice. Filling always narrows the spread of a column slightly — the " +
      "filled values sit exactly on the average and add no variation.",
  },
  fill_median: {
    title: "Fill missing values with the median",
    short: "Replaces empty numeric cells with the column's middle value.",
    long:
      "The median is the value in the middle when the column is sorted, so a " +
      "few enormous outliers cannot pull it around the way they pull the mean. " +
      "That makes it the sensible default for money, counts, durations and " +
      "anything else with a long tail. This is also exactly what the training " +
      "pipeline does on its own for numeric columns, so you only need to run " +
      "it here if you want the filled values saved into the dataset itself.",
  },
  fill_mode: {
    title: "Fill missing values with the mode",
    short: "Replaces empty cells with the most common value in the column.",
    long:
      "The mode is the value that appears most often, which makes it the only " +
      "sensible fill for categories — you cannot average \"red\" and \"blue\". " +
      "Use it for text and category columns with a few missing entries. Be " +
      "careful when the column is already dominated by one value: filling " +
      "makes that value even more dominant, and the model may learn to always " +
      "guess it.",
  },
  fill_constant: {
    title: "Fill missing values with a constant",
    short: "Replaces empty cells with a fixed value you type in.",
    long:
      "Use this when a blank cell actually means something specific — 0 sales, " +
      "\"none\", \"unknown\". It keeps that meaning explicit instead of hiding " +
      "it behind an average. It also lets the model learn from the fact that a " +
      "value was missing, which is genuinely predictive surprisingly often. " +
      "Pick a value that could never occur naturally so you can always tell " +
      "filled cells apart from real ones.",
  },
  fill_empty_string: {
    title: "Fill missing values with an empty string",
    short: "Turns empty text cells into \"\" so they are text, not gaps.",
    long:
      "Text columns often mix true blanks with missing markers, and the two " +
      "behave differently downstream. This normalises them all into one empty " +
      "string so the column is uniformly text and later text operations — " +
      "lowercasing, punctuation stripping, one-hot encoding — do not fail or " +
      "silently skip rows.",
  },
  strip_whitespace: {
    title: "Strip surrounding whitespace",
    short: "Removes leading and trailing spaces from text values.",
    long:
      "\"Jakarta\" and \"Jakarta \" are the same city to you and two entirely " +
      "different categories to a computer. That invisible trailing space " +
      "splits one group into two, halves the evidence behind each, and creates " +
      "categories the model has barely seen. Running this early — before " +
      "encoding or grouping — quietly fixes a whole class of bugs caused by " +
      "hand-typed data.",
  },
  lowercase: {
    title: "Lowercase text",
    short: "Converts every text value in the column to lower case.",
    long:
      "Same idea as stripping whitespace: \"Yes\", \"YES\" and \"yes\" are one " +
      "answer to a human and three categories to a model. Lowercasing merges " +
      "them. Only apply it where case carries no meaning — product codes and " +
      "identifiers sometimes distinguish upper from lower case, and " +
      "lowercasing those destroys real information.",
  },
  remove_punctuation: {
    title: "Remove punctuation",
    short: "Strips punctuation characters out of text values.",
    long:
      "Mostly used to tidy free text and messy category labels, so that " +
      "\"high-risk\", \"high risk\" and \"high_risk\" collapse into the same " +
      "value. Do not run it on columns where the punctuation is part of the " +
      "data — currency, decimal points, email addresses, URLs and dates all " +
      "lose their meaning without it.",
  },
  remove_outliers_iqr: {
    title: "Remove outliers (IQR rule)",
    short: "Drops rows outside 1.5x the interquartile range of a column.",
    long:
      "The interquartile range is the span covering the middle half of your " +
      "data. Anything more than 1.5 of those ranges below the lower quarter or " +
      "above the upper quarter is flagged as an outlier and the row is dropped. " +
      "This is the rule behind the whiskers on a box plot, and it needs no " +
      "assumption about the shape of the data. Be deliberate: an outlier can " +
      "be a typo worth deleting, or the rare fraud case you were trying to " +
      "predict in the first place.",
  },
  remove_outliers_zscore: {
    title: "Remove outliers (z-score rule)",
    short: "Drops rows more than N standard deviations from the mean.",
    long:
      "A z-score says how many standard deviations a value sits from the " +
      "column's average; the default threshold of 3.0 removes roughly the most " +
      "extreme 0.3% of a bell-shaped column. Because it is built from the mean " +
      "and the standard deviation, and both are themselves inflated by " +
      "outliers, it is less reliable than the IQR rule on skewed data. Raise " +
      "the threshold to remove fewer rows, lower it to remove more.",
  },
  minmax_scale: {
    title: "Min-max scaling",
    short: "Squeezes a numeric column onto a fixed 0-to-1 range.",
    long:
      "The smallest value in the column becomes 0, the largest becomes 1, and " +
      "everything else lands in between. It puts columns with wildly different " +
      "units — age in years next to income in millions — on comparable " +
      "footing, which matters a lot for distance-based and gradient-based " +
      "algorithms. Its weakness is outliers: one huge value pushes every other " +
      "row into a narrow band near zero.",
  },
  standard_scale: {
    title: "Standard scaling (z-score)",
    short: "Recentres a column to mean 0 with a standard deviation of 1.",
    long:
      "Each value is replaced by how far it sits from the column average, " +
      "measured in standard deviations. Unlike min-max scaling it has no fixed " +
      "bounds, so a single extreme value does not compress everything else. " +
      "This is the standard choice for linear models, SVMs, k-NN and neural " +
      "networks. The training page can do this for you inside the pipeline — " +
      "which is safer, because it learns the mean and deviation from the " +
      "training rows only.",
  },
  robust_scale: {
    title: "Robust scaling",
    short: "Scales using the median and interquartile range, not the mean.",
    long:
      "The same idea as standard scaling, but built from statistics that " +
      "outliers cannot drag around: it subtracts the median and divides by the " +
      "interquartile range. Reach for it when a column has genuine extreme " +
      "values that you want to keep rather than delete — the bulk of the data " +
      "still ends up on a sensible scale.",
  },
  one_hot_encode: {
    title: "One-hot encode (as a saved step)",
    short: "Turns one category column into a 0/1 column per category.",
    long:
      "A column holding \"red\", \"green\", \"blue\" becomes three columns, " +
      "each holding 1 or 0. This is how category data becomes numbers without " +
      "inventing a fake ordering. Note that the training pipeline already " +
      "one-hot encodes every category column for you, safely; run this step " +
      "only when you want the expanded columns saved into the dataset itself, " +
      "for example to inspect them. A column with hundreds of distinct values " +
      "will explode into hundreds of columns.",
  },
  label_encode: {
    title: "Label encode",
    short: "Replaces each category with an integer code (0, 1, 2, ...).",
    long:
      "Compact — one column stays one column — but it invents an order that " +
      "does not exist: if \"blue\" becomes 2 and \"red\" becomes 1, a linear " +
      "model will happily conclude blue is twice red. Safe for genuinely " +
      "ordered categories (\"low\", \"medium\", \"high\") and for tree-based " +
      "models, which split on values rather than adding them up. For " +
      "everything else, prefer one-hot encoding.",
  },
  pca: {
    title: "PCA (principal component analysis)",
    short: "Compresses many numeric columns into a few combined ones.",
    long:
      "PCA finds the directions along which your data varies the most and " +
      "rewrites every row using just those directions, keeping as much of the " +
      "spread as it can in the number of components you ask for. It is useful " +
      "when you have many correlated numeric columns, and it is excellent for " +
      "plotting high-dimensional data in two dimensions. The cost is " +
      "interpretability: the new columns are blends of the originals, so " +
      "\"component 1\" no longer has a plain-English meaning.",
  },
  select_k_best: {
    title: "Select K best features",
    short: "Keeps only the K columns most statistically related to the target.",
    long:
      "Each candidate column is scored against the target column you name, and " +
      "only the top K survive. Fewer, stronger columns mean a faster model that " +
      "is less prone to memorising noise. Two cautions: the scoring looks at " +
      "one column at a time, so it can discard a column that is only useful in " +
      "combination with another; and because it peeks at the target, doing it " +
      "over the whole dataset lets a little information from your test rows " +
      "leak into the choice.",
  },
  lda: {
    title: "LDA (linear discriminant analysis)",
    short: "Projects the data onto axes that separate the target classes best.",
    long:
      "Like PCA, LDA reduces many columns to a few. Unlike PCA, it uses the " +
      "labels: it looks for the directions that pull the classes furthest " +
      "apart. That makes it a strong dimensionality reduction for " +
      "classification, and it is often used purely to visualise how separable " +
      "your classes are. It is limited to at most (number of classes - 1) " +
      "components, so a two-class problem yields exactly one.",
  },
};

/* ------------------------------------------------------------------ */
/* Metrics — keys match the `metrics` map on a training run            */
/* ------------------------------------------------------------------ */

const METRIC_HELP: Record<string, HelpEntry> = {
  /* ---- classification ---- */
  accuracy: {
    title: "Accuracy",
    short: "The share of test rows the model predicted correctly.",
    long:
      "The most intuitive score: correct predictions divided by total " +
      "predictions, from 0 to 1. Its blind spot is imbalanced data. If 99% of " +
      "your rows are \"not fraud\", a model that never predicts fraud at all " +
      "scores 0.99 accuracy while being completely useless. Whenever your " +
      "classes are lopsided, read balanced accuracy, F1 and the confusion " +
      "matrix before you trust accuracy.",
  },
  balanced_accuracy: {
    title: "Balanced accuracy",
    short: "Accuracy averaged per class, so rare classes count equally.",
    long:
      "The model's recall is computed separately for each class and then " +
      "averaged, which gives a small class exactly as much weight as a huge " +
      "one. The always-guess-the-majority model that scores 0.99 plain " +
      "accuracy scores 0.50 here — a much more honest number. When balanced " +
      "accuracy is far below accuracy, your model is doing well only on the " +
      "common classes.",
  },
  precision: {
    title: "Precision",
    short: "Of the rows predicted positive, how many really were.",
    long:
      "Precision answers \"when this model raises its hand, how often is it " +
      "right?\" It is the metric you optimise when a false alarm is expensive: " +
      "flagging a legitimate transaction as fraud, or telling a healthy person " +
      "they are ill. This app reports it weighted across all classes, so " +
      "multi-class problems get a single comparable number.",
  },
  recall: {
    title: "Recall (sensitivity)",
    short: "Of the rows that really were positive, how many were caught.",
    long:
      "Recall answers \"how many of the real cases did we find?\" It is the " +
      "metric that matters when a miss is expensive — an undetected disease, " +
      "an undetected fraud. Precision and recall pull against each other: " +
      "predicting positive more freely raises recall and lowers precision. " +
      "Choose which one to favour based on which mistake costs you more.",
  },
  f1: {
    title: "F1 score",
    short: "The balanced average of precision and recall.",
    long:
      "The harmonic mean of precision and recall, which means a model has to " +
      "do well at both to score well — being excellent at one and terrible at " +
      "the other still gives a poor F1. It is the standard single-number " +
      "summary for imbalanced classification, and usually a better headline " +
      "than accuracy when the classes are uneven.",
  },
  f2: {
    title: "F2 score",
    short: "Like F1 but weights recall four times more than precision.",
    long:
      "Same recipe as F1 with the balance tilted towards recall. Use it when " +
      "missing a positive case is much worse than raising a false alarm — " +
      "medical screening, safety alerts, fraud detection. Reading F1 and F2 " +
      "side by side is informative: if F2 is well above F1, the model is " +
      "catching most real cases but paying for it in false alarms.",
  },
  roc_auc: {
    title: "ROC AUC",
    short: "How well the model ranks a positive above a negative (0.5 = coin flip).",
    long:
      "Pick one random positive row and one random negative row: AUC is the " +
      "probability that the model gives the positive one the higher score. " +
      "0.5 is random guessing, 1.0 is perfect separation. Because it judges " +
      "the ranking rather than the final yes/no, it is independent of where " +
      "you put the decision threshold. It is blank when the chosen algorithm " +
      "cannot produce probabilities, or when a class is missing from the test " +
      "split.",
  },
  log_loss: {
    title: "Log loss",
    short: "Penalises confident wrong answers. Lower is better.",
    long:
      "Unlike accuracy, log loss looks at the probability behind each " +
      "prediction. Being 99% sure and wrong is punished far more harshly than " +
      "being 55% sure and wrong, so a low log loss means the model's " +
      "confidence is trustworthy, not just its verdict. There is no fixed " +
      "upper bound and lower is always better; it is blank when the algorithm " +
      "cannot produce probabilities.",
  },
  matthews_corrcoef: {
    title: "Matthews correlation coefficient",
    short: "A balanced score from -1 to +1 that uses every part of the matrix.",
    long:
      "MCC is the correlation between the true labels and the predicted ones. " +
      "+1 is perfect, 0 is random guessing, and negative means the model is " +
      "systematically worse than chance. It is the hardest metric to fool, " +
      "because it only goes high when the model does well on every cell of the " +
      "confusion matrix at once — including the rare class. Many practitioners " +
      "treat it as the single most honest classification score.",
  },

  /* ---- regression ---- */
  r2: {
    title: "R² (coefficient of determination)",
    short: "Share of the variation in the target the model explains (1.0 = perfect).",
    long:
      "R² compares your model against the dumbest possible one — always " +
      "predicting the average. 1.0 means perfect predictions, 0.0 means you " +
      "did no better than that average, and a negative value means you did " +
      "worse, which usually points at a bug or a badly overfitted model. It is " +
      "unitless, so you can compare it across datasets in a way you cannot " +
      "compare an error measured in dollars.",
  },
  mae: {
    title: "MAE (mean absolute error)",
    short: "Average size of the error, in the target's own units.",
    long:
      "Take every prediction's distance from the truth, ignore the sign, and " +
      "average them. An MAE of 3.2 on a price in dollars means the model is " +
      "off by $3.20 on a typical row — the most directly explainable number on " +
      "this page. Because it does not square anything, a handful of very bad " +
      "predictions affect it far less than they affect RMSE.",
  },
  mse: {
    title: "MSE (mean squared error)",
    short: "Average of the squared errors; punishes big misses hardest.",
    long:
      "Squaring the errors before averaging means one prediction that is ten " +
      "times too far off counts a hundred times as much. That is the property " +
      "most regression algorithms actually optimise, which is why it appears " +
      "here. Its drawback for reading is the unit: if the target is in " +
      "dollars, MSE is in dollars squared, so use RMSE when you want to talk " +
      "about it out loud.",
  },
  rmse: {
    title: "RMSE (root mean squared error)",
    short: "The square root of MSE, back in the target's own units.",
    long:
      "The same outlier-sensitive measure as MSE, converted back into " +
      "readable units so it can sit next to MAE. RMSE is always at least as " +
      "large as MAE; the wider the gap between the two, the more your error is " +
      "concentrated in a few badly-missed rows rather than spread evenly. " +
      "Checking that gap is a fast way to spot a model that is fine on average " +
      "but occasionally catastrophic.",
  },
  mape: {
    title: "MAPE (mean absolute percentage error)",
    short: "Average error as a percentage of the true value.",
    long:
      "Expresses error relative to size, so being off by $10 on a $20 item and " +
      "on a $2,000 item are correctly treated as very different failures. That " +
      "makes it easy to communicate — \"about 8% off\". It is undefined when " +
      "any true value is zero, in which case this app reports it as blank, and " +
      "it becomes unstable when values are merely close to zero.",
  },
  explained_variance: {
    title: "Explained variance",
    short: "How much of the target's spread the model accounts for.",
    long:
      "Very close to R², and usually identical. The one difference is that " +
      "explained variance ignores a constant bias: a model that is reliably " +
      "5 units too high on every row still scores well here but is penalised " +
      "by R². If explained variance is noticeably higher than R², your model " +
      "has a systematic offset worth investigating.",
  },
  max_error: {
    title: "Max error",
    short: "The single worst prediction the model made.",
    long:
      "The largest distance between a prediction and the truth anywhere in the " +
      "test set. Averages hide disasters; this deliberately does not. It is " +
      "the metric to check when a single bad prediction is unacceptable — " +
      "pricing, dosing, capacity planning. A good average error paired with a " +
      "huge max error means the model has a blind spot on some kind of row.",
  },
};

/* ------------------------------------------------------------------ */
/* Training concepts — keyed by the TrainRequest field where possible  */
/* ------------------------------------------------------------------ */

const TEST_SIZE: HelpEntry = {
  title: "Train / test split",
  short: "The share of rows held back to score the model on data it never saw.",
  long:
    "Before training, the rows are cut in two. The model learns from the " +
    "training part and is then scored on the test part, which it has never " +
    "seen. This is the whole basis of trusting a score: a model can memorise " +
    "the rows it was trained on and look flawless while having learned " +
    "nothing general. 0.2 — a 20% test set — is the usual choice. Raise it " +
    "for a more reliable score on a big dataset; lower it when you have so " +
    "few rows that the model needs every one it can get.",
};

const RANDOM_STATE: HelpEntry = {
  title: "Random seed",
  short: "Fixes the randomness so the same settings give the same result.",
  long:
    "Splitting the data and many algorithms themselves involve random " +
    "choices. Fixing the seed makes those choices repeat exactly, so " +
    "re-running with the same settings reproduces the same model and the same " +
    "numbers — which is what lets you compare two runs and know the " +
    "difference came from your change, not from luck. Leaving it empty gives " +
    "a different split each time; running that way a few times is a good way " +
    "to see how much your score wobbles.",
};

const STRATIFY: HelpEntry = {
  title: "Stratify the split",
  short: "Keeps each class in the same proportion in both train and test sets.",
  long:
    "Without stratification the split is purely random, and on an imbalanced " +
    "dataset that can go badly: a rare class with 12 rows might land entirely " +
    "in the training half, leaving the test set with no examples of it to " +
    "score. Stratifying preserves the class mix on both sides. It applies to " +
    "classification only — there are no classes to balance in regression — " +
    "and it is switched off automatically, with a warning, when some class " +
    "has too few rows to divide.",
};

const SCALE_FEATURES: HelpEntry = {
  title: "Scale features",
  short: "Puts numeric columns on a comparable scale inside the pipeline.",
  long:
    "Standardises every numeric column to mean 0 and standard deviation 1, so " +
    "that income in millions does not simply drown out age in years. Linear " +
    "models, SVMs, k-NN and neural networks need this; tree-based models such " +
    "as random forests are indifferent to it. It happens inside the pipeline, " +
    "which matters: the mean and deviation are learned from the training rows " +
    "only and then applied to the test rows, so no information about the test " +
    "set leaks into training.",
};

const CV_FOLDS: HelpEntry = {
  title: "Cross-validation",
  short: "Repeats the train/test cycle over several folds and averages the scores.",
  long:
    "A single split gives a single score, and that score depends on which " +
    "rows happened to fall on which side. Cross-validation splits the data " +
    "into K equal folds, trains K times with a different fold held out each " +
    "time, and reports the average and the spread. The average is a far more " +
    "trustworthy estimate; the spread tells you how stable the model is. It " +
    "costs roughly K times the training time, so it is optional here — 5 " +
    "folds is the common choice.",
};

const TASK_TYPE: HelpEntry = {
  title: "Task type",
  short: "Whether you are predicting a category (classification) or a number (regression).",
  long:
    "Classification predicts which group a row belongs to — spam or not, " +
    "which species, which risk band. Regression predicts a quantity on a " +
    "continuous scale — a price, a temperature, a duration. The app infers " +
    "this from your target column: text and boolean targets become " +
    "classification, and a numeric target becomes classification too if its " +
    "values are whole numbers and there are only a few distinct ones, " +
    "otherwise regression. Set it explicitly when the guess is wrong — for " +
    "instance a 1-to-5 rating you would rather treat as five categories. Your " +
    "explicit choice always wins.",
};

const CONCEPT_HELP: Record<string, HelpEntry> = {
  test_size: TEST_SIZE,
  train_test_split: TEST_SIZE,
  random_state: RANDOM_STATE,
  random_seed: RANDOM_STATE,
  stratify: STRATIFY,
  scale_features: SCALE_FEATURES,
  cv_folds: CV_FOLDS,
  cross_validation: CV_FOLDS,
  task_type: TASK_TYPE,

  overfitting: {
    title: "Overfitting",
    short: "When a model memorises the training rows instead of learning the pattern.",
    long:
      "An overfitted model has learned the noise in your training data — the " +
      "accidents of these particular rows — rather than the rule underneath. " +
      "It scores brilliantly on data it has seen and poorly on anything new. " +
      "The tell is right on the evaluation page: a training score far above " +
      "the test score. The usual cures are more data, a simpler model (a " +
      "shallower tree, fewer neighbours, stronger regularisation), fewer " +
      "features, or cross-validation to catch it earlier.",
  },
  one_hot_encoding: {
    title: "One-hot encoding (in the pipeline)",
    short: "How the pipeline turns category columns into numbers automatically.",
    long:
      "Models do arithmetic, and \"Jakarta\" is not a number. One-hot encoding " +
      "replaces a category column with one 0/1 column per category, which " +
      "represents the categories without inventing an ordering between them. " +
      "Every training run does this for you inside the pipeline, and it is " +
      "configured to ignore unknown categories — so a value that shows up at " +
      "prediction time but never appeared in training turns into all zeros " +
      "rather than crashing the model.",
  },
  feature_importance: {
    title: "Feature importance",
    short: "How much each input column contributed to the model's predictions.",
    long:
      "A ranking of which columns the model actually leaned on. For tree-based " +
      "models it comes from how much each column improved the splits; for " +
      "linear models it comes from the size of the coefficients. Read it as a " +
      "sanity check — if an id column ranks first, something has leaked — and " +
      "as a guide to what to collect more of. Two warnings: importance shows " +
      "association, not cause; and when two columns say nearly the same thing, " +
      "they split the credit and both look weaker than they are.",
  },
};

/* ------------------------------------------------------------------ */
/* Public surface                                                      */
/* ------------------------------------------------------------------ */

/** Every help entry, keyed by op name, metric key, or training concept. */
export const HELP: Record<string, HelpEntry> = {
  ...OP_HELP,
  ...METRIC_HELP,
  ...CONCEPT_HELP,
};

/** Look up one entry. Returns undefined for a key with no help written yet. */
export function helpFor(key: string): HelpEntry | undefined {
  return HELP[key];
}

export default HELP;
