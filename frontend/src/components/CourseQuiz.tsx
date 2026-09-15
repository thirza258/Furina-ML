import { useState } from "react";
import type { CourseLesson } from "../data/courseCatalogue";

export default function CourseQuiz({ quiz, completed, onComplete }: {
  quiz: CourseLesson["quiz"];
  completed: boolean;
  onComplete: () => void;
}) {
  const [selected, setSelected] = useState<number | null>(null);
  const [submitted, setSubmitted] = useState<number | null>(null);
  const correct = submitted === quiz.correctIndex;

  return (
    <section className="callout my-8" aria-label="Knowledge check">
      <h2 className="text-h3 mb-2">Check your understanding</h2>
      <p className="text-small text-ink-muted mb-5">
        {completed ? "✓ Lesson complete. You can revisit this question any time." : "Answer correctly to complete this lesson. You can try again as often as you like."}
      </p>
      <form onSubmit={(event) => {
        event.preventDefault();
        if (selected === null) return;
        setSubmitted(selected);
        if (selected === quiz.correctIndex) onComplete();
      }}>
        <fieldset>
          <legend className="font-medium mb-3">{quiz.question}</legend>
          <div className="space-y-2">
            {quiz.options.map((option, index) => (
              <label key={option} className={`flex items-start gap-3 p-3 border cursor-pointer ${selected === index ? "border-primary bg-paper-sunken" : "border-rule hover:bg-paper"}`}>
                <input
                  type="radio" name="lesson-answer" value={index} checked={selected === index}
                  onChange={() => { setSelected(index); setSubmitted(null); }}
                  className="mt-1.5 accent-primary shrink-0"
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <button type="submit" className="btn btn-primary mt-5" disabled={selected === null}>
          Check answer
        </button>
        <div aria-live="polite" aria-atomic="true">
          {submitted !== null && (
            <div className={`mt-4 border-l-2 pl-4 ${correct ? "border-state-good" : "border-state-warn"}`}>
              <p className={`font-medium ${correct ? "text-state-good" : "text-state-warn"}`}>
                {correct ? "✓ Correct — lesson complete." : "Not quite — try again."}
              </p>
              <p className="text-small mt-1">{quiz.explanation}</p>
            </div>
          )}
        </div>
      </form>
    </section>
  );
}
