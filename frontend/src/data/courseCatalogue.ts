import type { Course, CourseLesson } from "./courses/types";
import { tabularCourses } from "./courses/tabular";
import { baseModelsCourse } from "./courses/baseModels";
import { intermediateCourse } from "./courses/intermediate";
import { transformersCourse } from "./courses/transformers";
import { llmsCourse } from "./courses/llms";
import { visionLanguageCourse } from "./courses/visionLanguage";
import { voiceCourse } from "./courses/voice";
import { mlSystemDesignCourse } from "./courses/mlSystemDesign";

export type { Course, CourseId, CourseLevel, CourseTrack, CourseLesson } from "./courses/types";

export const COURSES: Course[] = [
  ...tabularCourses, baseModelsCourse, intermediateCourse,
  transformersCourse, llmsCourse, visionLanguageCourse, voiceCourse,
  mlSystemDesignCourse,
];

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
