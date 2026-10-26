import { useEffect, useState } from "react";
import { COURSES, lessonKey } from "../data/courseCatalogue";
import { useAuth } from "../context/authContext";
import services from "../services/services";
import type { ApiError } from "../interface/MlTypes";

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

function clearStoredProgress(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* Nothing stored, or storage blocked: either way nothing to clear. */
  }
}

function errorDetail(error: unknown): string {
  return (error as ApiError)?.detail ?? "Progress could not be saved to your account.";
}

/**
 * Passed knowledge checks. Signed out, they live in this browser. Signed in,
 * they live in the account: whatever was passed while signed out is uploaded
 * once and then removed from the browser, so it cannot also be credited to
 * the next person who signs in here.
 */
export function useCourseProgress() {
  const { session } = useAuth();
  const username = session?.authenticated ? session.user?.username ?? null : null;
  const [local, setLocal] = useState<ProgressState>(readProgress);
  const [account, setAccount] = useState<string[] | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);

  useEffect(() => {
    if (!username) {
      setAccount(null);
      setLocal(readProgress());
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const pending = readProgress().completed;
        const saved = pending.length
          ? await services.saveProgress(pending)
          : await services.progress();
        if (pending.length) clearStoredProgress();
        if (!cancelled) {
          setAccount(saved.completed.filter((key) => VALID_KEYS.has(key)));
          setLocal({ completed: [], canPersist: true });
          setSyncError(null);
        }
      } catch (error) {
        if (!cancelled) setSyncError(errorDetail(error));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [username]);

  const completeLesson = (key: string) => {
    if (!VALID_KEYS.has(key)) return;
    if (username) {
      if (account?.includes(key)) return;
      setAccount((previous) => [...(previous ?? []), key]);
      services.saveProgress([key]).then(
        () => setSyncError(null),
        (error: unknown) => setSyncError(errorDetail(error))
      );
      return;
    }
    if (local.completed.includes(key)) return;
    const completed = [...local.completed, key];
    let canPersist = true;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(completed));
    } catch {
      canPersist = false;
    }
    setLocal({ completed, canPersist });
  };

  return {
    completed: username ? account ?? [] : local.completed,
    canPersist: username ? true : local.canPersist,
    username,
    syncError,
    completeLesson,
  };
}
