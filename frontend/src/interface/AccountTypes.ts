/** Accounts, saved course progress and notebook launches (`/api/auth/…`). */

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
