import { Link } from "react-router-dom";

export default function CourseLink({ courseId }: { courseId: "training" | "evaluation" }) {
  const training = courseId === "training";
  return (
    <aside className="border-l-2 border-primary pl-4 mb-6 text-small">
      <Link to={`/app/courses/${courseId}`}>
        {training ? "Take the Model Training course" : "Take the Model Evaluation course"}
      </Link>
      <p className="text-ink-muted mt-1">
        {training
          ? "Learn about targets, splits, model settings and cross-validation."
          : "Learn to read metrics, spot overfitting and compare your runs."}
        {" "}Lessons and examples are available before you train a model.
      </p>
    </aside>
  );
}
