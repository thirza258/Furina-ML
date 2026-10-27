/** Accounts, saved course progress, notebook launches and grading (`/api/auth/…`). */

export type AccountRole = "student" | "instructor" | "admin";

export interface AccountUser {
  username: string;
  email: string;
  roles: AccountRole[];
  date_joined: string;
}

/** `GET /api/auth/session/` — also what sign-in, sign-up and sign-out return. */
export interface SessionInfo {
  authenticated: boolean;
  user: AccountUser | null;
  registration_open: boolean;
  notebooks: {
    /** True when this deployment runs JupyterHub. */
    hub_enabled: boolean;
    hub_url: string | null;
    /** True when a grader worker grades handed-in lesson notebooks. */
    grading_enabled: boolean;
  };
}

export interface Credentials {
  username: string;
  password: string;
}

export interface Registration extends Credentials {
  email?: string;
}

export interface ProgressResponse {
  /** `course-id/lesson-id` keys of passed knowledge checks. */
  completed: string[];
}

export interface NotebookLaunch {
  path: string;
  /** The hub URL that opens the notebook in the student's own workspace. */
  url: string;
  /** The same notebook in the full JupyterLab interface. */
  lab_url: string;
}

export type SubmissionStatus = "queued" | "running" | "graded" | "error";

/** One check in a graded notebook: a practice()/check() call on an exercise. */
export interface SubmissionCheck {
  name: string;
  /** "not-run": the notebook never reached the check (an earlier error, a time-out). */
  status: "passed" | "failed" | "not-run";
  /** What the check printed, or the error that stopped it. */
  message: string;
}

/** `GET /api/submissions/{id}/` — a notebook handed in for grading. */
export interface Submission {
  id: string;
  path: string;
  status: SubmissionStatus;
  /** Fraction of checks passed, 0 to 1; null until graded. */
  score: number | null;
  passed: number | null;
  total: number | null;
  result: {
    items?: SubmissionCheck[];
    /** Exercise cells the student changed, and those still as shipped. */
    answered?: number;
    unanswered?: number;
    /** False when the notebook stopped before the end; the result is partial. */
    complete?: boolean;
  };
  /** Why grading failed, when status is "error". */
  error: string;
  created_at: string;
  finished_at: string | null;
}
