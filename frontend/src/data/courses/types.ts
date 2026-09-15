export type CourseId = "training" | "evaluation" | "base-models" | "intermediate-models" | "transformers" | "llms" | "vision-language" | "voice" | "ml-system-design";
export type CourseLevel = "Beginner" | "Intermediate" | "Advanced";
export type CourseTrack = "Tabular ML" | "Deep learning" | "Language" | "Multimodal" | "Speech" | "ML systems";

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
  practiceMode: "workbench" | "python";
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
