import { useState } from "react";
import { COURSES, lessonKey } from "../data/courseCatalogue";

const STORAGE_KEY = "furina-ml.course-progress.v1";
const VALID_KEYS = new Set(COURSES.flatMap((course) =>
  course.lessons.map((lesson) => lessonKey(course, lesson))
));

interface ProgressState {
  completed: string[];
  canPersist: boolean;
}

function readProgress(): ProgressState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { completed: [], canPersist: true };
    const parsed: unknown = JSON.parse(raw);
    const completed = Array.isArray(parsed)
      ? [...new Set(parsed.filter((key): key is string =>
        typeof key === "string" && VALID_KEYS.has(key)
      ))]
      : [];
    return { completed, canPersist: true };
  } catch {
    // Corrupt or unavailable storage must not prevent someone taking a lesson.
    return { completed: [], canPersist: false };
  }
}

export function useCourseProgress() {
  const [progress, setProgress] = useState<ProgressState>(readProgress);

  const completeLesson = (key: string) => {
    if (!VALID_KEYS.has(key) || progress.completed.includes(key)) return;
    const completed = [...progress.completed, key];
    let canPersist = true;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(completed));
    } catch {
      canPersist = false;
    }
    setProgress({ completed, canPersist });
  };

  return { ...progress, completeLesson };
}
