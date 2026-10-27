import type { Course } from "../data/courseCatalogue";

export function CourseLab({ course }: { course: Course }) {
  if (!course.lab) return null;
  const requirementsFile = course.lab.requirementsFile === undefined ? "requirements.txt" : course.lab.requirementsFile;
  const setup = requirementsFile
    ? ["python3 -m venv .venv", "# macOS / Linux:", "source .venv/bin/activate", "# Windows PowerShell: .venv\\Scripts\\Activate.ps1", `python -m pip install -r ${requirementsFile}`, `python ${course.lab.filename}`].join("\n")
    : ["# Python 3.10+; uses only the standard library", `python3 ${course.lab.filename}`, `# Windows: python ${course.lab.filename}`].join("\n");
  return (
    <section className="callout my-8" aria-labelledby="course-lab-title">
      <p className="field-label mb-2">{course.practiceMode === "notebook" ? "Optional · Downloadable PyTorch lab · CPU" : "Downloadable Python lab · CPU"}</p>
      <h2 id="course-lab-title" className="text-h3 mb-3">{course.lab.title}</h2>
      <p className="mb-4 max-w-measure">{course.lab.description}</p>
      <div className="flex flex-wrap gap-3 mb-5">
        <a href={`/course-labs/${course.lab.filename}`} download className="btn btn-primary">Download {course.lab.filename}</a>
        {requirementsFile && <a href={`/course-labs/${requirementsFile}`} download className="btn btn-secondary">Download requirements</a>}
        <a href="/course-labs/README.md" download className="btn btn-secondary">Lab instructions</a>
        {course.lab.guide && <a href={`/course-labs/${course.lab.guide.filename}`} download className="btn btn-secondary">{course.lab.guide.title}</a>}
      </div>
      <h3 className="font-medium font-sans mb-2">Run in your Python environment</h3>
      <p className="text-small text-ink-muted mb-3">{requirementsFile ? "Save the script and requirements in the same folder." : "Save the script and run it with Python 3.10 or later. No packages or model downloads are needed."} These labs run on your machine; Furina's workbench trains tabular scikit-learn models.</p>
      <pre className="bg-paper-sunken border border-rule p-4 text-small overflow-x-auto" tabIndex={0} aria-label="Lab setup commands"><code>{setup}</code></pre>
      <p className="text-small text-ink-muted mt-3">The scripts create small teaching datasets locally and write checkpoints and JSON reports. Larger pretrained-model exercises need a separately chosen model, permitted data and suitable hardware.</p>
      <h3 className="font-medium font-sans mt-5 mb-2">Lab extensions</h3>
      <ol className="list-decimal pl-5 space-y-2 text-small">{course.lab.exercises.map((exercise) => <li key={exercise}>{exercise}</li>)}</ol>
    </section>
  );
}

export function CourseCapstone({ course }: { course: Course }) {
  if (!course.capstone) return null;
  return (
    <section className="border-t border-rule pt-6 my-8" aria-labelledby="capstone-title">
      <p className="field-label mb-2">Practical project · Self-reviewed</p>
      <h2 id="capstone-title" className="text-h2 mb-3">{course.capstone.title}</h2>
      <p className="prose-page mb-5">{course.capstone.brief}</p>
      <div className="grid md:grid-cols-2 gap-6">
        <div><h3 className="text-h3 mb-3">What to deliver</h3><ul className="list-disc pl-5 space-y-2">{course.capstone.deliverables.map((item) => <li key={item}>{item}</li>)}</ul></div>
        <div><h3 className="text-h3 mb-3">Review criteria</h3><ul className="list-disc pl-5 space-y-2">{course.capstone.rubric.map((item) => <li key={item}>{item}</li>)}</ul></div>
      </div>
      <p className="text-small text-ink-muted mt-5">Saved lesson progress tracks knowledge checks. {course.practiceMode === "notebook" ? "Review your notebook against these project criteria yourself; its checks test your code, but nothing grades the project." : course.lab ? "Review the lab outputs and these project criteria separately; the browser does not grade your training run." : "Review your runs against these project criteria yourself; the browser does not grade them."}</p>
    </section>
  );
}
