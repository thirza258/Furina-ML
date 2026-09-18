import type { ReactNode, RefObject } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { COURSES, lessonKey, lessonPath, studyTime } from "../data/courseCatalogue";
import { usePageSeo } from "../hooks/usePageSeo";

const TRACKS = [...new Set(COURSES.map((course) => course.track))];
const LEVELS = ["Beginner", "Intermediate", "Advanced"];
const SEARCH_INDEX = new Map(COURSES.map((course) => [course.id,
  [course.title, course.description, ...course.outcomes, ...course.lessons.flatMap((lesson) =>
    [lesson.title, lesson.intro, ...lesson.sections.map((section) => section.body)]
  )].join(" ").toLocaleLowerCase(),
]));

export default function CourseLibrary({ completed, persistenceNote, headingRef }: {
  completed: string[]; persistenceNote: ReactNode; headingRef: RefObject<HTMLHeadingElement | null>;
}) {
  usePageSeo({
    title: "Machine Learning & AI Courses — Hands-on Labs & Capstones",
    description:
      "Explore 9 self-paced courses and 72 lessons across Tabular ML, Base Models, Transformers, LLMs, Vision-Language, Voice, and ML System Design with downloadable Python labs.",
    path: "/app/courses",
    jsonLd: {
      "@context": "https://schema.org",
      "@type": "ItemList",
      name: "Furina ML Courses",
      itemListElement: COURSES.map((course, i) => ({
        "@type": "ListItem",
        position: i + 1,
        name: course.title,
        url: `https://furina.nevatal.id/app/courses/${course.id}`,
      })),
    },
  });

  const [search, setSearch] = useSearchParams();
  const query = search.get("q") ?? "";
  const level = search.get("level") ?? "";
  const track = search.get("track") ?? "";
  const words = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const filtered = COURSES.filter((course) => (!level || course.level === level)
    && (!track || course.track === track)
    && words.every((word) => SEARCH_INDEX.get(course.id)?.includes(word)));
  const totalLessons = COURSES.reduce((sum, course) => sum + course.lessons.length, 0);
  const updateFilter = (key: string, value: string) => setSearch((previous) => {
    const next = new URLSearchParams(previous);
    if (value) next.set(key, value); else next.delete(key);
    return next;
  }, { replace: true });

  return (
    <div className="p-5 sm:p-8">
      <header className="border-b border-rule pb-7 mb-8">
        <p className="field-label mb-3">Furina ML · Course library</p>
        <h1 ref={headingRef} tabIndex={-1} className="text-h1 mb-4 focus:outline-none">Learn to train, adapt &amp; evaluate models</h1>
        <p className="prose-page text-lead mb-5">Start with a first training run, build neural models, then explore language, vision, speech and the systems that bring models into production. Follow complete courses with worked examples, practical labs and project review criteria.</p>
        <div className="flex flex-wrap gap-x-6 gap-y-2 text-small mb-4"><span>{COURSES.length} courses</span><span>{totalLessons} lessons</span><span>{COURSES.filter((course) => course.lab).length} downloadable Python labs</span><span>Beginner to advanced</span></div>
        {persistenceNote}
      </header>
      <section className="mb-8" aria-label="Your learning progress">
        <div className="flex flex-wrap justify-between items-baseline gap-3 mb-2"><h2 className="text-h3">Your progress</h2><span className="text-small tabular">{completed.length} / {totalLessons} knowledge checks complete</span></div>
        <progress value={completed.length} max={totalLessons} aria-label="Total course progress" className="course-progress" />
      </section>
      <details className="border-y border-rule py-4 mb-8">
        <summary className="font-medium cursor-pointer min-h-11 flex items-center">Suggested learning path</summary>
        <ol className="grid sm:grid-cols-2 xl:grid-cols-4 gap-3 mt-4 text-small">
          {COURSES.map((course, index) => <li key={course.id}><Link to={`/app/courses/${course.id}`}>{index + 1}. {course.title}</Link></li>)}
        </ol>
        <p className="text-small text-ink-muted mt-4">Take the tabular and base-model courses first if you are new to ML. Vision-language and voice are specializations. System design builds on the training and evaluation foundations and can be taken before the neural-model courses. Follow each overview's prerequisites; every lesson remains open for reference.</p>
      </details>
      <section className="mb-8" aria-label="Find a course">
        <div className="grid sm:grid-cols-2 xl:grid-cols-[2fr_1fr_1fr] gap-4">
          <label className="block sm:col-span-2 xl:col-span-1"><span className="field-label block mb-2">Search courses &amp; lesson topics</span><input type="search" value={query} onChange={(event) => updateFilter("q", event.target.value)} placeholder="Try LoRA, CTC, attention, evaluation…" className="w-full min-h-11 border border-rule-strong bg-paper-raised px-3 rounded-control text-small" /></label>
          <label className="block"><span className="field-label block mb-2">Level</span><select value={level} onChange={(event) => updateFilter("level", event.target.value)} className="w-full min-h-11 border border-rule-strong bg-paper-raised px-3 rounded-control text-small"><option value="">All levels</option>{LEVELS.map((item) => <option key={item}>{item}</option>)}</select></label>
          <label className="block"><span className="field-label block mb-2">Topic</span><select value={track} onChange={(event) => updateFilter("track", event.target.value)} className="w-full min-h-11 border border-rule-strong bg-paper-raised px-3 rounded-control text-small"><option value="">All topics</option>{TRACKS.map((item) => <option key={item}>{item}</option>)}</select></label>
        </div>
        <div className="flex items-center justify-between gap-3 mt-4"><p role="status" className="text-small text-ink-muted">{filtered.length} of {COURSES.length} courses shown</p>{(query || level || track) && <button type="button" className="btn btn-secondary" onClick={() => setSearch({})}>Clear filters</button>}</div>
      </section>
      {filtered.length === 0 && <div className="callout"><h2 className="text-h3 mb-2">No courses match these filters</h2><p>Try a broader topic or clear the filters to see every course.</p></div>}
      <div className="grid lg:grid-cols-2 gap-x-8 gap-y-10">
        {filtered.map((course) => {
          const count = course.lessons.filter((lesson) => completed.includes(lessonKey(course, lesson))).length;
          const next = course.lessons.find((lesson) => !completed.includes(lessonKey(course, lesson))) ?? course.lessons[0];
          const done = count === course.lessons.length;
          return (
            <section key={course.id} className="border-t-2 border-primary pt-5 flex flex-col" aria-labelledby={`course-title-${course.id}`}>
              <p className="field-label mb-2">{course.track} · {course.level}</p>
              <h2 id={`course-title-${course.id}`} className="text-h2 mb-3"><Link to={`/app/courses/${course.id}`} className="text-ink no-underline">{course.title}</Link></h2>
              <p className="mb-4">{course.description}</p>
              <p className="text-small text-ink-muted mb-4">{course.lessons.length} lessons · {studyTime(course.lessons.reduce((sum, lesson) => sum + lesson.minutes, 0))} study &amp; exercises · {course.lab ? "Python lab + capstone" : "Workbench exercises"}</p>
              <ul className="list-disc pl-5 space-y-1 text-small mb-5">{course.outcomes.map((outcome) => <li key={outcome}>{outcome}</li>)}</ul>
              <details className="mb-5 border-y border-rule py-2">
                <summary className="cursor-pointer min-h-11 py-2 text-small">Explore {course.lessons.length} lessons</summary>
                <ol className="divide-y divide-rule text-small">{course.lessons.map((lesson, index) => <li key={lesson.id}><Link to={lessonPath(course, lesson)} className="block py-3">{index + 1}. {lesson.title}{completed.includes(lessonKey(course, lesson)) && <span className="text-state-good"> · Complete ✓</span>}</Link></li>)}</ol>
              </details>
              <div className="mt-auto">
                <p className="text-small mb-2">{done ? "✓ All knowledge checks complete" : `${count} of ${course.lessons.length} knowledge checks complete`}</p>
                <progress value={count} max={course.lessons.length} aria-label={`${course.title} progress`} className="course-progress mb-4" />
                <div className="flex flex-wrap gap-3">
                  <Link to={lessonPath(course, next)} className="btn btn-primary flex-grow" aria-label={`${done ? "Review" : count ? "Continue" : "Start"} ${course.title} course`}>{done ? "Review lessons" : count ? "Continue course" : "Start course"} →</Link>
                  <Link to={`/app/courses/${course.id}`} className="btn btn-secondary" aria-label={`${course.title} syllabus and resources`}>Syllabus &amp; resources</Link>
                </div>
              </div>
            </section>
          );
        })}
      </div>
      <p className="text-small text-ink-muted border-t border-rule pt-5 mt-10">Lessons and interactive examples work without a backend. Tabular exercises use Furina's workbench; neural, language, vision, voice and system-design labs run in an external Python environment. Study-time estimates include exercises and exclude large-model training time.</p>
    </div>
  );
}
