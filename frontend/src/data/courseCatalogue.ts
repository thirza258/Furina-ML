import type { Course, CourseLesson, DependencyTier } from "./courses/types";
import { mathFoundationsCourse } from "./courses/mathFoundations";
import { neuralNetworksCourse } from "./courses/neuralNetworks";
import { tabularCourses } from "./courses/tabular";
import { dataPreparationCourse } from "./courses/dataPreparation";
import { choosingModelsCourse } from "./courses/choosingModels";
import { timeSeriesCourse } from "./courses/timeSeries";
import { baseModelsCourse } from "./courses/baseModels";
import { intermediateCourse } from "./courses/intermediate";
import { transformersCourse } from "./courses/transformers";
import { llmsCourse } from "./courses/llms";
import { visionLanguageCourse } from "./courses/visionLanguage";
import { voiceCourse } from "./courses/voice";
import { mlSystemDesignCourse } from "./courses/mlSystemDesign";

export type { Course, CourseId, CourseLevel, CourseTrack, CourseLesson, DependencyTier } from "./courses/types";

export const COURSES: Course[] = [
  mathFoundationsCourse,
  ...tabularCourses, dataPreparationCourse, choosingModelsCourse, timeSeriesCourse,
  neuralNetworksCourse, baseModelsCourse, intermediateCourse,
  transformersCourse, llmsCourse, visionLanguageCourse, voiceCourse,
  mlSystemDesignCourse,
];

/** What a course lets the learner's own code import (see types.ts). */
export const DEPENDENCY_TIERS: Record<DependencyTier, string> = {
  0: "Python standard library only",
  1: "NumPy and Matplotlib",
  2: "pandas and scikit-learn",
  3: "PyTorch",
};

export function tierLabel(tier: DependencyTier): string {
  return `Tier ${tier} · ${DEPENDENCY_TIERS[tier]}`;
}

export function lessonPath(course: Course, lesson: CourseLesson): string {
  return `/app/courses/${course.id}/${lesson.id}`;
}

export function lessonKey(course: Course, lesson: CourseLesson): string {
  return `${course.id}/${lesson.id}`;
}

export function studyTime(minutes: number): string {
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return hours ? `${hours}h${remainder ? ` ${remainder}m` : ""}` : `${minutes} min`;
}
