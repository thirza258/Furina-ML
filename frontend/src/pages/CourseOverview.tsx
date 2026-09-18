import type { RefObject } from "react";
import { Link } from "react-router-dom";
import { COURSES, lessonKey, lessonPath, studyTime } from "../data/courseCatalogue";
import type { Course } from "../data/courseCatalogue";
import { CourseCapstone, CourseLab } from "../components/CourseResources";
import { usePageSeo } from "../hooks/usePageSeo";

export default function CourseOverview({ course, completed, headingRef }: {
  course: Course; completed: string[]; headingRef: RefObject<HTMLHeadingElement | null>;
}) {
  const count = course.lessons.filter((lesson) => completed.includes(lessonKey(course, lesson))).length;
  const next = course.lessons.find((lesson) => !completed.includes(lessonKey(course, lesson))) ?? course.lessons[0];
  const totalMinutes = course.lessons.reduce((sum, lesson) => sum + lesson.minutes, 0);

  usePageSeo({
    title: `${course.title} Course — Syllabus & Labs`,
    description: course.description,
    path: `/app/courses/${course.id}`,
    jsonLd: {
      "@context": "https://schema.org",
      "@type": "Course",
      name: course.title,
      description: course.description,
      provider: {
        "@type": "Organization",
        name: "Furina ML",
        sameAs: "https://furina.nevatal.id",
      },
      hasCourseInstance: {
        "@type": "CourseInstance",
        courseMode: "Online",
        courseWorkload: studyTime(totalMinutes),
      },
    },
  });
  return (
    <div className="p-5 sm:p-8">
      <Link to="/app/courses" className="text-small">← All courses</Link>
      <header className="border-b border-rule pb-6 mt-6 mb-8">
        <p className="field-label mb-3">{course.track} · {course.level}</p>
        <h1 ref={headingRef} tabIndex={-1} className="text-h1 mb-4 focus:outline-none">{course.title}</h1>
        <p className="prose-page text-lead mb-4">{course.description}</p>
        <p className="text-small text-ink-muted mb-5">{course.lessons.length} lessons · About {studyTime(totalMinutes)} of study and exercises{course.lab ? " · Python lab + capstone" : " · Workbench exercises"}</p>
        <Link to={lessonPath(course, next)} className="btn btn-primary">{count === course.lessons.length ? "Review course" : count ? "Continue course" : "Start first lesson"}</Link>
        <p className="text-small mt-4 mb-2">{count} of {course.lessons.length} knowledge checks complete</p>
        <progress className="course-progress" value={count} max={course.lessons.length} aria-label={`${course.title} progress`} />
      </header>
      <div className="grid lg:grid-cols-2 gap-8 mb-8">
        <section><h2 className="panel-title">What you will learn</h2><ul className="list-disc pl-5 space-y-2">{course.outcomes.map((item) => <li key={item}>{item}</li>)}</ul></section>
        <section>
          <h2 className="panel-title">Before you start</h2>
          <ul className="list-disc pl-5 space-y-2">{course.prerequisites.map((item) => <li key={item}>{item}</li>)}</ul>
          {course.prerequisiteCourses.length > 0 && <p className="text-small mt-4">Suggested courses: {course.prerequisiteCourses.map((id, index) => <span key={id}>{index > 0 && ", "}<Link to={`/app/courses/${id}`}>{COURSES.find((item) => item.id === id)?.title}</Link></span>)}</p>}
        </section>
      </div>
      <section aria-labelledby="syllabus-title">
        <h2 id="syllabus-title" className="panel-title">Course syllabus</h2>
        <ol className="divide-y divide-rule">
          {course.lessons.map((lesson, index) => (
            <li key={lesson.id} className="py-4">
              <div className="flex flex-wrap justify-between gap-2">
                <h3 className="text-h3"><Link to={lessonPath(course, lesson)}>{index + 1}. {lesson.title}</Link></h3>
                <span className="text-small text-ink-muted">{completed.includes(lessonKey(course, lesson)) ? "✓ Complete" : `${lesson.minutes} min`}</span>
              </div>
              <p className="text-small text-ink-muted mt-2 max-w-measure">{lesson.intro}</p>
            </li>
          ))}
        </ol>
      </section>
      <CourseLab course={course} />
      <CourseCapstone course={course} />
      <section className="border-t border-rule pt-5 mt-8">
        <h2 className="text-h3 mb-3">Primary sources &amp; further reading</h2>
        <ul className="list-disc pl-5 space-y-2 text-small">{course.references.map((reference) => <li key={reference.url}><a href={reference.url} target="_blank" rel="noreferrer">{reference.title}<span className="sr-only"> (opens in a new tab)</span></a></li>)}</ul>
      </section>
    </div>
  );
}
