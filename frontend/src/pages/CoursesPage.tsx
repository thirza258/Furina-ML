import { useEffect, useRef } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import CourseExperiments from "../components/CourseExperiments";
import CourseQuiz from "../components/CourseQuiz";
import LessonNotebook from "../components/LessonNotebook";
import { CourseCapstone } from "../components/CourseResources";
import CourseLibrary from "./CourseLibrary";
import CourseOverview from "./CourseOverview";
import { COURSES, lessonKey, lessonPath } from "../data/courseCatalogue";
import { useCourseProgress } from "../hooks/useCourseProgress";

export default function CoursesPage() {
  const { courseId, lessonId } = useParams();
  const navigate = useNavigate();
  const { completed, canPersist, completeLesson, username, syncError } = useCourseProgress();
  const headingRef = useRef<HTMLHeadingElement>(null);
  const course = COURSES.find((item) => item.id === courseId);
  const lessonIndex = course?.lessons.findIndex((item) => item.id === lessonId) ?? -1;
  const lesson = course?.lessons[lessonIndex];

  useEffect(() => {
    headingRef.current?.focus({ preventScroll: true });
    headingRef.current?.closest("main")?.scrollTo({ top: 0 });
  }, [courseId, lessonId]);

  const warning = syncError !== null || !canPersist;
  const persistenceNote = (
    <p className={`text-small ${warning ? "text-state-warn" : "text-ink-muted"}`} role={warning ? "status" : undefined}>
      {syncError !== null
        ? `Progress could not be saved to your account: ${syncError}`
        : username
          ? `Completed lessons are saved to your account, ${username}.`
          : canPersist
            ? <>Completed lessons are saved in this browser. No account or dataset needed; <Link to="/login?next=/app/courses">sign in</Link> to keep them with an account.</>
            : "Progress is available for this visit, but browser storage could not be read or saved. It may be lost on reload."}
    </p>
  );

  if (!courseId) {
    return <CourseLibrary completed={completed} persistenceNote={persistenceNote} headingRef={headingRef} />;
  }

  if (course && !lessonId) {
    return <CourseOverview course={course} completed={completed} headingRef={headingRef} />;
  }

  if (!course || !lesson) {
    return (
      <div className="p-5 sm:p-8">
        <h1 ref={headingRef} tabIndex={-1} className="text-h1 mb-4 focus:outline-none">Lesson not found</h1>
        <p className="mb-6">This course or lesson does not exist. Choose a lesson from the course library.</p>
        <Link to="/app/courses" className="btn btn-primary">Browse courses</Link>
      </div>
    );
  }

  const count = course.lessons.filter((entry) => completed.includes(lessonKey(course, entry))).length;
  const previous = course.lessons[lessonIndex - 1];
  const next = course.lessons[lessonIndex + 1];
  const courseComplete = count === course.lessons.length;
  const nextCourse = COURSES.find((item) => item.id === course.nextCourseId);
  // Notebook courses read like Colab: the notebook is the lesson, so it comes
  // first and the explanation follows as notes.
  const notebookFirst = course.practiceMode === "notebook" && Boolean(lesson.notebook);

  return (
    <div className="p-5 sm:p-8">
      <nav aria-label="Course breadcrumb" className="text-small mb-6 flex flex-wrap gap-2">
        <Link to="/app/courses">All courses</Link>
        <span aria-hidden="true">/</span><Link to={`/app/courses/${course.id}`}>{course.title}</Link>
      </nav>
      <div className="grid xl:grid-cols-[13rem_minmax(0,1fr)] gap-8">
        <aside>
          <nav aria-label={`${course.title} lessons`}>
            <h2 className="text-h3 mb-3"><Link to={`/app/courses/${course.id}`} className="text-ink no-underline">{course.title}</Link></h2>
            <p className="text-small mb-2" aria-live="polite">{count} of {course.lessons.length} lessons complete</p>
            <progress value={count} max={course.lessons.length} aria-label={`${course.title} progress`} className="course-progress mb-4" />
            <label htmlFor="course-lesson" className="xl:hidden block text-small mb-2">Jump to a lesson</label>
            <select
              id="course-lesson" value={lesson.id}
              onChange={(event) => {
                const selected = course.lessons.find((entry) => entry.id === event.target.value);
                if (selected) navigate(lessonPath(course, selected));
              }}
              className="xl:hidden w-full min-h-11 border border-rule-strong bg-paper-raised text-small px-2 rounded-control"
            >
              {course.lessons.map((entry, index) => (
                <option key={entry.id} value={entry.id}>
                  {index + 1}. {entry.title}{completed.includes(lessonKey(course, entry)) ? " · Complete" : ""}
                </option>
              ))}
            </select>
            <ol className="hidden xl:grid grid-cols-1 gap-1">
              {course.lessons.map((entry, index) => (
                <li key={entry.id}>
                  <Link
                    to={lessonPath(course, entry)} aria-current={entry.id === lesson.id ? "page" : undefined}
                    className={`block text-small no-underline border-l-2 py-3 px-3 ${entry.id === lesson.id ? "border-primary bg-paper-raised font-semibold text-ink" : "border-transparent hover:bg-paper-sunken"}`}
                  >
                    {index + 1}. {entry.title}
                    <span className={`block mt-1 text-caption ${completed.includes(lessonKey(course, entry)) ? "text-state-good" : "text-ink-muted"}`}>
                      {completed.includes(lessonKey(course, entry)) ? "✓ Complete" : `${entry.minutes} min`}
                    </span>
                  </Link>
                </li>
              ))}
            </ol>
          </nav>
          <div className="mt-5">{persistenceNote}</div>
        </aside>

        <article className="min-w-0">
          <header className="border-b border-rule pb-6 mb-6">
            <p className="text-small text-ink-muted mb-2">{course.track} · {course.level}</p>
            <p className="field-label mb-3">Lesson {lessonIndex + 1} of {course.lessons.length} · {lesson.minutes} min</p>
            <h1 ref={headingRef} tabIndex={-1} className="text-h1 mb-4 focus:outline-none">{lesson.title}</h1>
            <p className="prose-page text-lead">{lesson.intro}</p>
          </header>
          {notebookFirst && <LessonNotebook key={lesson.id} course={course} lesson={lesson} />}
          {notebookFirst && <h2 className="panel-title mt-10">Lesson notes</h2>}
          {lesson.sections.map((section) => (
            <section key={section.title} className="mb-7">
              <h2 className="text-h3 mb-3">{section.title}</h2>
              <p className="prose-page">{section.body}</p>
            </section>
          ))}
          <section className="border-l-2 border-accent pl-5 my-8">
            <h2 className="text-h3 mb-3">{lesson.example.title}</h2>
            <p className="prose-page">{lesson.example.body}</p>
          </section>
          {lesson.code && (
            <section className="my-8" aria-label="Code walkthrough">
              <h2 className="text-h3 mb-3">{lesson.code.title}</h2>
              <pre className="bg-paper-sunken border border-rule p-4 text-small overflow-x-auto" tabIndex={0} aria-label={lesson.code.title}><code>{lesson.code.source}</code></pre>
              <p className="text-small text-ink-muted mt-3">{lesson.code.explanation}</p>
            </section>
          )}
          {lesson.experiment && <CourseExperiments key={lesson.id} kind={lesson.experiment} />}
          {lesson.notebook && !notebookFirst && <LessonNotebook key={lesson.id} course={course} lesson={lesson} />}
          <section className="my-8">
            <h2 className="panel-title">Put it into practice</h2>
            <p className="text-small text-ink-muted mb-3">{course.practiceMode === "workbench" ? "Use your own CSV and a running backend for these workbench exercises." : course.practiceMode === "notebook" ? "Work through these in the lesson's notebook at the top of the page. Each ✍️ check tells you when your code is right." : "Work through these exercises in your Python environment. The course overview includes setup instructions, a downloadable CPU lab and project criteria."}</p>
            <ol className="list-decimal pl-5 space-y-3">
              {lesson.practice.map((step) => <li key={step}>{step}</li>)}
            </ol>
            <div className="flex flex-wrap gap-3 mt-5">
              {course.toolPath && <Link to={course.toolPath} className="btn btn-secondary">{course.toolLabel}</Link>}
              {course.lab && (course.practiceMode === "notebook"
                ? <a href={`/course-labs/${course.lab.filename}`} download className="btn btn-secondary">Optional: PyTorch lab</a>
                : <a href={`/course-labs/${course.lab.filename}`} download className="btn btn-primary">Download Python lab</a>)}
              {course.lab?.guide && <a href={`/course-labs/${course.lab.guide.filename}`} download className="btn btn-secondary">{course.lab.guide.title}</a>}
              <Link to={`/app/courses/${course.id}`} className="btn btn-secondary">Syllabus &amp; resources</Link>
            </div>
          </section>
          <CourseQuiz
            key={lessonKey(course, lesson)} quiz={lesson.quiz}
            completed={completed.includes(lessonKey(course, lesson))}
            onComplete={() => completeLesson(lessonKey(course, lesson))}
          />
          {!next && <CourseCapstone course={course} />}
          {courseComplete && (
            <section className="border-l-2 border-state-good pl-5 mb-8" aria-label="Course complete">
              <h2 className="text-h3 mb-2">✓ {course.title} knowledge checks complete</h2>
              <p className="mb-4">You have passed all {course.lessons.length} knowledge checks. {course.capstone ? (course.practiceMode === "notebook" ? "Finish the capstone notebook and review it against the capstone criteria to complete the practical project." : course.lab ? "Complete the lab and review the capstone criteria to finish the practical work." : "Review your work against the capstone criteria to finish the practical project.") : "Apply the lessons to a run and revisit any topic when you need it."}</p>
              <Link to={nextCourse ? `/app/courses/${nextCourse.id}` : "/app/courses"} className="btn btn-primary">
                {nextCourse ? `Continue to ${nextCourse.title}` : "Back to all courses"}
              </Link>
            </section>
          )}
          <nav aria-label="Lesson navigation" className="flex flex-wrap justify-between gap-3 border-t border-rule pt-5">
            <Link to={previous ? lessonPath(course, previous) : "/app/courses"} className="btn btn-secondary">
              {previous ? "← Previous lesson" : "← All courses"}
            </Link>
            <Link to={next ? lessonPath(course, next) : "/app/courses"} className="btn btn-primary">
              {next ? "Next lesson →" : "Back to courses →"}
            </Link>
          </nav>
          <details className="mt-8 text-small">
            <summary className="cursor-pointer min-h-11 py-2">Primary sources &amp; further reading</summary>
            <ul className="list-disc pl-5 space-y-2 mt-2">
              {(lesson.references ?? course.references).map((reference) => <li key={reference.url}><a href={reference.url} target="_blank" rel="noreferrer">{reference.title} <span className="sr-only">(opens in a new tab)</span></a></li>)}
            </ul>
          </details>
        </article>
      </div>
    </div>
  );
}
