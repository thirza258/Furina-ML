import type { Course } from "./types";

export const dataPreparationCourse: Course = {
  id: "data-preparation",
  level: "Beginner",
  track: "Tabular ML",
  prerequisites: ["Complete Model Training, or understand targets, features and a train/test split."],
  prerequisiteCourses: ["training"],
  practiceMode: "workbench", dependencyTier: 2,
  nextCourseId: "choosing-models",
  title: "Data Preparation",
  description: "Prepare a CSV for an honest training run. Read the column profile, clean text and duplicates, handle gaps and outliers, and learn which steps belong in the preprocessing recipe and which belong inside training.",
  outcomes: [
    "Read a column profile and spot placeholders, identifiers and unusable targets",
    "Clean text and duplicates with a previewed recipe that keeps the original file",
    "Handle missing values and keep missingness when it carries meaning",
    "Know which models need scaling and how training encodes categories",
    "Recognise when PCA, LDA or feature selection lets test rows shape the result",
  ],
  toolPath: "/app/preprocessing",
  toolLabel: "Open Data Preprocessing",
  lessons: [
    {
      id: "read-the-profile",
      title: "Read the column profile first",
      minutes: 6,
      intro: "Every later decision depends on what the file really contains. Before you change anything, read the column profile on Data Preprocessing. It describes each column the way Furina's training will see it.",
      sections: [
        {
          title: "Check how each column was read",
          body: "The Kind column says how Furina interprets each column: numeric, boolean (two values such as 0 and 1, or true and false), categorical, or text (text with many distinct values, such as notes). Stored type shows how the values are held. If a measurement appears as categorical or text, at least one cell probably contains something that is not a number, such as “?” or “unknown”. Training would then one-hot encode the column and give every distinct number its own column. Furina does not convert dates when it reads a file, so dates also appear as categorical or text.",
        },
        {
          title: "Find the gaps and the codes that hide them",
          body: "The Missing column shows how many cells are empty in each column, as a count and a percentage. When a file is uploaded, empty cells and common placeholders such as NA, N/A, null and None are read as missing. Other codes are not: “?”, “-”, “unknown”, -1 and 999 count as ordinary values. Look for them among each column's most common values and in the minimum and maximum of numeric columns. The recipe has no step that turns a code into a gap, so replace such codes with empty cells in the source file and upload it again.",
        },
        {
          title: "Spot the target and the columns to set aside",
          body: "Under each column name, the profile says “Target would be classification”, “Target would be regression” or “Not usable as a target”. A column is not usable when it is empty, holds a single value, or is text in which every value is different. Sort by Unique as well. A column with a different value in every row is usually an identifier, such as an order ID: it describes a row rather than a pattern, so a model can only memorise it. Remove it with Drop columns, or untick it under Choose the features on Data Training.",
        },
      ],
      example: {
        title: "Example: a delivery file before cleaning",
        body: "In a hypothetical delivery file, order_id has as many unique values as there are rows, so it is an identifier. weight_kg is not profiled as numeric, because a few rows say “unknown”. customer_age has a maximum of 999, which is a code for “not given” rather than an age. shipping_method lists “Express”, “express” and “EXPRESS” as three separate values. The target, is_late, is boolean, and the profile says “Target would be classification”. Each finding becomes a task: drop the identifier, replace the codes in the source file, and clean the text.",
      },
      notebook: { path: "workbench/load_workbench_data.ipynb", title: "Load your workbench data into a notebook" },
      practice: [
        "Upload a CSV on Data Preprocessing, then sort the Column profile by Missing. Write down each column with gaps and a first guess at why its values are missing.",
        "Sort by Kind and then by Unique. Mark identifiers, free-text columns and any measurement that was not read as numeric, and look for codes such as “?”, -1 or 999 in its values, minimum and maximum.",
        "Find your intended target and check that it says “Target would be …” with the task you expect. If you found codes standing in for missing values, replace them with empty cells in the source file and upload it again.",
      ],
      references: [
        { title: "pandas: how missing values are recognised when reading a CSV", url: "https://pandas.pydata.org/docs/user_guide/io.html#na-and-missing-data-handling" },
        { title: "pandas: working with missing data", url: "https://pandas.pydata.org/docs/user_guide/missing_data.html" },
      ],
      quiz: {
        question: "A weight_kg column is profiled as categorical, and its stored type is str. What is the most likely explanation?",
        options: [
          "Furina profiles every measurement with a unit, such as kilograms, as categorical",
          "Some cells hold text such as “unknown”, so the whole column was read as text",
          "Some cells are empty, and empty cells turn a numeric column into text",
        ],
        correctIndex: 1,
        explanation: "A single non-numeric entry is enough to make a column text. Empty cells alone do not do this: a numeric column with gaps is still profiled as numeric, and its gaps are counted as missing.",
      },
    },
    {
      id: "clean-rows-and-text",
      title: "Clean text and duplicates safely",
      minutes: 7,
      intro: "Some cleaning steps follow a fixed rule instead of learning a value from the data. Because the rule does not depend on how the rows are later split, these steps are the safest to run on the whole file, provided you check what they change.",
      sections: [
        {
          title: "Make equal values look equal",
          body: "To a model, “ Leeds”, “leeds” and “LEEDS ” are three unrelated categories. Trim surrounding whitespace and Lowercase text merge them into one. Check the most common values first, because case sometimes carries meaning in codes. Remove punctuation deletes every standard punctuation character, including decimal points, minus signs, hyphens and slashes: “12.5” becomes “125”, “-3” becomes “3” and “A-1” merges with “A1”. Use it only on free text or labels in which those characters carry no meaning, and never on numbers or dates stored as text.",
        },
        {
          title: "Remove repeated records after cleaning text",
          body: "Remove duplicate rows keeps the first copy of each repeated row. Exact copies matter for evaluation: after the random split, one copy can land among the training rows and another among the test rows, so part of the test only checks what the model memorised. The recipe runs from top to bottom, and each newly ticked step joins the end of the list, so use the ↑ and ↓ buttons to put text cleaning first. Be careful when there is no identifier: two different customers can share identical values, and removing one changes the class counts.",
        },
        {
          title: "Protect the target and keep the original",
          body: "The preprocessing page does not know your target. A step left on “all applicable columns” can change the target too: a fill step can invent labels for unlabelled rows, One-hot encode can replace a text target with new columns, and a numeric step can rescale or absorb it. Open the Columns list of each such step and tick only feature columns. Apply preprocessing saves a new dataset, listed as Cleaned and named with “(processed)”, and makes it active. Your upload stays under Load a previous dataset, and Undo — go back switches back.",
        },
      ],
      example: {
        title: "Example: the order of steps matters",
        body: "Four rows hold the cities “ Leeds”, “leeds”, “LEEDS ” and “York”, and the first three share the same value in every other column. Remove duplicate rows on its own finds no duplicates, because the text differs. Run Trim surrounding whitespace, then Lowercase text, then Remove duplicate rows, and the preview reports two duplicate rows removed, leaving one “leeds” row and one “york” row.",
      },
      practice: [
        "Tick Trim surrounding whitespace and Lowercase text. Open each step's Columns list and select only text columns whose most common values you have checked; cleaning a text target this way is harmless and often helpful.",
        "Tick Remove duplicate rows last, or move it below the text steps with the ↓ button. Press Preview changes and read Rows before, Rows after and each step's message.",
        "Make sure every fill, encoding, scaling or outlier step lists only feature columns, then press Apply preprocessing. Find the original under Load a previous dataset, and use Undo — go back if the result is wrong.",
      ],
      references: [
        { title: "pandas: working with text data", url: "https://pandas.pydata.org/docs/user_guide/text.html" },
        { title: "pandas: DataFrame.drop_duplicates", url: "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html" },
      ],
      quiz: {
        question: "Your recipe runs Remove duplicate rows, then Trim surrounding whitespace, then Lowercase text. Why might obvious repeats survive?",
        options: [
          "Remove duplicate rows ignores text columns and compares only numeric ones",
          "Lowercase text always runs first, whatever the recipe order says",
          "The rows were compared before their text was cleaned, so they still differed",
        ],
        correctIndex: 2,
        explanation: "Steps run from top to bottom. Clean the text first, so that rows which differ only in spacing or case are identical by the time duplicates are compared.",
      },
    },
    {
      id: "handle-missing-values",
      title: "Decide what a gap means",
      minutes: 8,
      intro: "A missing value tells you something about how the data was collected. Before filling it, decide whether to drop the row, fill the gap with a rule, or leave the gap for training to handle.",
      sections: [
        {
          title: "Know what each choice assumes",
          body: "Fill missing values with the mean or with the median writes a typical value into every gap; the median resists extreme values and the mean does not. Fill missing values with the mode writes the most common value and suits categories. Every fill makes a column look more certain than it is. Drop rows with missing values removes evidence and can change who is represented; left on all applicable columns, it drops any row with a gap anywhere. Rows with gaps will still arrive once the model is in use, so dropping them now does not make them go away.",
        },
        {
          title: "Keep missingness when it is a signal",
          body: "Sometimes the gap itself predicts the outcome: an empty referral_source may mean that a customer arrived directly. Filling the gap with a typical value erases that signal, and Furina's training pipeline adds no “was missing” column; for categories, it folds gaps into the most common value. Fill missing values with a constant, set to “unknown”, keeps the gap as its own category, which training encodes separately. Avoid NA, null or None as the constant, because the saved file reads them back as missing. On a numeric column, a constant such as -1 is a fake measurement that most models treat as real.",
        },
        {
          title: "Let training fill the remaining gaps",
          body: "Data Training fills numeric gaps with the median and category gaps with the most common value. It learns those values from the training split only and stores them in the saved model, so the prediction form and a downloaded model fill gaps in new rows the same way. A mean, median or mode fill in the preprocessing tool computes its value from the whole file, including rows that later become test rows. The effect on the score is often small, but the test rows are no longer untouched. Never fill the target: Furina drops rows without a target before training.",
        },
      ],
      example: {
        title: "Example: one extreme value moves the mean",
        body: "A distance_km column holds 1, 2, 4, 100 and 3, plus one gap. Fill missing values with the mean writes 22 into the gap, which is larger than four of the five known distances. Fill missing values with the median writes 3. Both values come from every row in the file, so even if the row with 100 later lands in the test split, it has already shaped a value in the data.",
      },
      code: {
        title: "What “learned from the training split” means",
        source: [
          "import numpy as np",
          "from sklearn.impute import SimpleImputer",
          "",
          "train = np.array([[1.0], [2.0], [np.nan], [4.0]])  # training rows",
          "test = np.array([[np.nan], [100.0]])               # rows held out for testing",
          "",
          "imputer = SimpleImputer(strategy=\"median\").fit(train)",
          "print(imputer.statistics_)      # the median of the training rows only",
          "print(imputer.transform(test))  # the test gap receives that same value",
        ].join("\n"),
        explanation: "Furina's training pipeline follows this pattern. The imputer learns a median of 2 from the training rows; the 100 in a test row cannot change it, and the gap in a test row receives the same 2. Running the script is optional and needs Python with scikit-learn installed.",
      },
      practice: [
        "Sort the profile by Missing. For each column with gaps, write down whether a gap looks random, means “none”, or hints at something else.",
        "For a category column in which a gap carries meaning, tick Fill missing values with a constant, set Value to unknown, select only that column and press Preview changes.",
        "Leave numeric gaps to Data Training. After a run, read its warnings to see how many rows were dropped because the target was missing.",
      ],
      references: [
        { title: "scikit-learn: imputation of missing values", url: "https://scikit-learn.org/stable/modules/impute.html" },
        { title: "scikit-learn: marking imputed values", url: "https://scikit-learn.org/stable/modules/impute.html#marking-imputed-values" },
        { title: "pandas: working with missing data", url: "https://pandas.pydata.org/docs/user_guide/missing_data.html" },
      ],
      quiz: {
        question: "Which fill can run in the preprocessing tool without letting future test rows influence the value it writes?",
        options: [
          "Fill missing values with a constant, such as “unknown”",
          "Fill missing values with the median of each numeric column",
          "Fill missing values with the mode, the most common value",
        ],
        correctIndex: 0,
        explanation: "You choose a constant yourself; it is not calculated from the rows, so the split cannot affect it. The median and the mode are calculated from every row in the file, including rows that later become test rows.",
      },
    },
    {
      id: "scale-encode-outliers",
      title: "Scale, encode and question outliers",
      minutes: 7,
      intro: "Some models compare raw numbers directly, while others only ask whether a value lies above or below a threshold. Knowing which kind you are training tells you which transformations matter and where they belong.",
      sections: [
        {
          title: "Match scaling to the model",
          body: "K-Nearest Neighbours and Support Vector Machines compare distances or margins, and regularised or gradient-trained models such as Logistic Regression and the Neural Network (MLP) are sensitive to units too: a column in metres can outweigh one in kilometres. Decision trees, forests and gradient boosting split on thresholds and are largely unaffected by rescaling. Data Training's Standardise numeric features option, which is on by default, learns each column's mean and standard deviation from the training split. While it is on, scaling the same columns in the tool first is redundant, because training standardises them again.",
        },
        {
          title: "Leave categories as text for training",
          body: "Training one-hot encodes every non-numeric column and learns the categories from the training split; a category first seen at prediction time is encoded as zeros. The tool's One-hot encode works on the whole file and skips columns with more than 50 values, and training then treats each new 0/1 column as a category and encodes it again. Label encode replaces categories with numbers in sorted order, which suggests a ranking; with more than two codes, training treats them as quantities. Recipe steps are not saved with the model, so the prediction form expects transformed values.",
        },
        {
          title: "Treat outliers as questions",
          body: "An extreme value may be an entry error, a rare but real case, or the most important row in the file. Remove outliers (IQR) deletes rows lying more than 1.5 interquartile ranges outside the middle half of a column. Remove outliers (z-score) deletes rows lying more than a chosen number of standard deviations, 3 by default, from the mean. Both compute their limits from the whole file and also delete future test rows, so the score no longer covers extreme cases the model will still meet. Correct known errors in the source file instead.",
        },
      ],
      example: {
        title: "Example: two extremes can hide each other",
        body: "A column holds the values 1 to 10 and one value of 1,000. Both outlier rules remove the 1,000. Add a second value of 1,000, and the z-score rule at its default threshold of 3 removes nothing, because the two extremes inflate the standard deviation that defines what counts as extreme. The IQR rule, which is based on quartiles, still removes both rows. Preview changes shows each step's count before anything is saved.",
      },
      practice: [
        "On Data Training, check whether your algorithm relies on distances, margins or gradient-based fitting, or is tree-based. Keep Standardise numeric features on unless you have a specific reason to turn it off.",
        "Leave text categories as text so that training encodes them. If you try One-hot encode or Label encode in the tool, select feature columns only and check the Preview changes report for any column that was left as it is.",
        "Add an outlier step with only feature columns selected and press Preview changes. Compare Rows before and Rows after, and the new minimum and maximum in the preview's column profile, before deciding whether those rows are errors.",
      ],
      references: [
        { title: "scikit-learn: standardization, or mean removal and variance scaling", url: "https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling" },
        { title: "scikit-learn: encoding categorical features", url: "https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features" },
        { title: "scikit-learn: importance of feature scaling", url: "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html" },
      ],
      quiz: {
        question: "A K-Nearest Neighbours model uses annual income in pounds and age in years, with no scaling. What is likely to happen?",
        options: [
          "Nothing changes, because K-Nearest Neighbours rescales every column itself",
          "Income dominates the distances, because its numbers are far larger",
          "Age dominates the distances, because it has fewer distinct values",
        ],
        correctIndex: 1,
        explanation: "A distance adds up differences across columns, so a column with large numbers swamps one with small numbers. Standardising both columns, which Data Training does by default, puts them on a comparable scale.",
      },
    },
    {
      id: "reduce-features-honestly",
      title: "Reduce features without leaking the answer",
      minutes: 8,
      intro: "PCA, LDA and Select K best features shrink the set of columns that a model sees. They can make a model simpler, but they learn from the data, and two of them learn from the target itself.",
      sections: [
        {
          title: "Three ways to shrink the feature set",
          body: "PCA standardises the chosen numeric columns and replaces them with components, pc_1, pc_2 and so on, that keep as much of their variation as possible; the report states the percentage kept. It never looks at the target. LDA replaces the columns with axes named ld_1 and onwards, at most one fewer than the target has classes, chosen to separate those classes. Select K best features keeps the K numeric columns most related to the target, according to an F-test, and drops the others. All three are skipped while a selected column still contains gaps.",
        },
        {
          title: "Know which steps see the answer",
          body: "Every fitted step in the recipe, from a median fill to PCA, is computed on the whole file before Data Training splits it. PCA uses the test rows' feature values but not their labels, which is a milder influence; still choose its columns yourself, because on all applicable columns it can absorb a numeric target. LDA and Select K best features read the target in every row, including future test rows, so they can latch onto patterns that exist only by chance in those labels. Their Target column starts as the first column that could be a target, so set it explicitly.",
        },
        {
          title: "Compare fairly, then question the gain",
          body: "Press Preview changes first and read the report: components kept, variance kept, columns dropped and any skipped step. Then train the same algorithm, target, settings and seed on the original and on the derived dataset, and compare the runs on the Dashboard. These steps keep every row in its place, so both runs use the same split. If the LDA or Select K best features run looks better, do not treat that alone as evidence. For classification, the Linear Discriminant Analysis algorithm on Data Training is an honest alternative, because it learns from the training split only.",
        },
      ],
      example: {
        title: "Example: finding signal in pure noise",
        body: "A simulation ran Furina's recipe steps on files of 200 rows, 500 columns of random numbers and a coin-flip yes/no target, then trained Logistic Regression with Furina's default 80/20 stratified split and standardisation, 50 times over. With Select K best features (K = 10) on the whole file, average test accuracy was about 68.5%; selecting on the training rows only gave about 50%. LDA on the whole file averaged about 78%, against about 50% when fitted on the training rows. The labels were random, so the extra accuracy came from leakage.",
      },
      code: {
        title: "Feature selection inside the pipeline",
        source: [
          "import numpy as np",
          "from sklearn.feature_selection import SelectKBest, f_classif",
          "from sklearn.linear_model import LogisticRegression",
          "from sklearn.model_selection import cross_val_score",
          "from sklearn.pipeline import make_pipeline",
          "from sklearn.preprocessing import StandardScaler",
          "",
          "rng = np.random.default_rng(0)",
          "X = rng.normal(size=(200, 500))   # 500 columns of random numbers",
          "y = rng.integers(0, 2, size=200)  # coin-flip labels: nothing to learn",
          "",
          "# Leaky: every row's label helps choose the columns, then we evaluate",
          "X_chosen = SelectKBest(f_classif, k=10).fit_transform(X, y)",
          "leaky = make_pipeline(StandardScaler(), LogisticRegression())",
          "print(\"leaky :\", cross_val_score(leaky, X_chosen, y, cv=5).mean())",
          "",
          "# Honest: the selection is refitted on the training rows of each fold",
          "honest = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10), LogisticRegression())",
          "print(\"honest:\", cross_val_score(honest, X, y, cv=5).mean())",
        ].join("\n"),
        explanation: "Both scores use coin-flip labels, so an honest method should score near 0.5. When this was run with scikit-learn 1.9, the leaky version printed about 0.70 and the honest version about 0.46; across 30 different seeds, the leaky average stayed near 0.70 and the honest one near 0.50. Select K best features in Furina's recipe works like the leaky version: it sees every row's label before any split.",
      },
      practice: [
        "Tick PCA, open its Columns list and select only feature columns. Press Preview changes and note the percentage of variance the components keep.",
        "In a separate recipe, try Select K best features or LDA with Target column set to your real target and only gap-free feature columns selected. Apply it, then train the same algorithm, target, settings and seed on the original and on the derived dataset, naming each run clearly.",
        "Compare the two runs on the Dashboard, using the Dataset column to tell them apart. If the derived run wins, check the gain with a method that never sees test labels, such as the Linear Discriminant Analysis algorithm or features chosen in advance.",
      ],
      references: [
        { title: "scikit-learn: data leakage during pre-processing", url: "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage-during-pre-processing" },
        { title: "scikit-learn: univariate feature selection", url: "https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection" },
        { title: "scikit-learn: principal component analysis", url: "https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca" },
        { title: "scikit-learn: linear and quadratic discriminant analysis", url: "https://scikit-learn.org/stable/modules/lda_qda.html" },
      ],
      quiz: {
        question: "After Select K best features runs on the whole file, test accuracy rises from 0.71 to 0.78. What can you conclude?",
        options: [
          "The selected columns are now proven to generalise better to new data",
          "Feature selection always improves a model, so the step should stay",
          "Part of the gain may come from test labels used to pick the columns",
        ],
        correctIndex: 2,
        explanation: "The selection read every row's target, including the test rows, before the split. Confirm the gain with a method that never sees test labels, such as features chosen in advance or a model fitted on the training split only, before claiming an improvement.",
      },
    },
  ],
  references: [
    { title: "scikit-learn: common pitfalls and data leakage", url: "https://scikit-learn.org/stable/common_pitfalls.html" },
    { title: "scikit-learn: preprocessing data", url: "https://scikit-learn.org/stable/modules/preprocessing.html" },
    { title: "pandas: working with missing data", url: "https://pandas.pydata.org/docs/user_guide/missing_data.html" },
  ],
};
