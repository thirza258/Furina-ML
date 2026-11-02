export type CourseId = "math-foundations" | "neural-networks" | "autograd" | "training" | "evaluation" | "data-preparation" | "choosing-models" | "time-series" | "base-models" | "cuda-base-models" | "intermediate-models" | "transformers" | "llms" | "vision-language" | "voice" | "ml-system-design";
export type CourseLevel = "Beginner" | "Intermediate" | "Advanced";
export type CourseTrack = "Mathematics" | "Tabular ML" | "Time series" | "Deep learning" | "Language" | "Multimodal" | "Speech" | "ML systems";

/**
 * What a course lets the learner's own code depend on.
 *
 * 0: the Python standard library only. 1: adds NumPy and Matplotlib.
 * 2: adds pandas and scikit-learn. 3: adds PyTorch. A notebook may compare its
 * result with a higher tier only in cells tagged `comparison` or
 * `visualization`; notebooks/tools/check_notebooks.py enforces that.
 */
export type DependencyTier = 0 | 1 | 2 | 3;

/** A notebook template under the repository's `notebooks/` folder. */
export interface CourseNotebook {
  /** Relative to `notebooks/`, e.g. "math-foundations/01_python_math_foundations.ipynb". */
  path: string;
  title: string;
}

export interface CourseReference {
  title: string;
  url: string;
}

/** IDs are stable: they identify shared lesson URLs and saved progress. */
export interface CourseLesson {
  id: string;
  title: string;
  minutes: number;
  intro: string;
  sections: { title: string; body: string }[];
  example: { title: string; body: string };
  experiment?: "split" | "classification" | "regression" | "attention" | "training-budget" | "audio" | "system-architecture" | "system-budget";
  code?: { title: string; source: string; explanation: string };
  notebook?: CourseNotebook;
  practice: string[];
  references?: CourseReference[];
  quiz: {
    question: string;
    options: string[];
    correctIndex: number;
    explanation: string;
  };
}

export interface Course {
  id: CourseId;
  title: string;
  description: string;
  level: CourseLevel;
  track: CourseTrack;
  prerequisites: string[];
  prerequisiteCourses: CourseId[];
  outcomes: string[];
  practiceMode: "workbench" | "python" | "notebook";
  dependencyTier: DependencyTier;
  /**
   * Native code the course's notebooks compile, beyond the Python tier, e.g.
   * "C/C++ compiler; NVIDIA GPU optional for CUDA". Such
   * notebooks run a compiler, which the in-browser JupyterLite preview cannot
   * do, so they open on the notebook server, in Colab or locally instead.
   * notebooks/tools/check_notebooks.py checks the matching `toolchain` in
   * each notebook's metadata.
   */
  toolchain?: string;
  toolPath?: string;
  toolLabel?: string;
  nextCourseId?: CourseId;
  lab?: {
    filename: string;
    title: string;
    description: string;
    exercises: string[];
    /** Omit for the shared PyTorch requirements; null means standard library only. */
    requirementsFile?: string | null;
    guide?: { filename: string; title: string };
  };
  capstone?: {
    title: string;
    brief: string;
    deliverables: string[];
    rubric: string[];
  };
  lessons: CourseLesson[];
  references: CourseReference[];
}
