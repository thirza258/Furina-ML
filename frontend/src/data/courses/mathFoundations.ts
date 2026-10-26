import type { Course } from "./types";
import { mathFoundationsLessonsA } from "./mathFoundationsLessonsA";
import { mathFoundationsLessonsB } from "./mathFoundationsLessonsB";

export const mathFoundationsCourse: Course = {
  id: "math-foundations", title: "Mathematics for Machine Learning from First Principles", level: "Beginner", track: "Mathematics",
  description: "Learn the mathematics behind machine learning by implementing it yourself in plain Python: numbers and rounding, vectors and matrices, probability and statistics, derivatives and gradients, and gradient descent. No numerical libraries do the work for you.",
  prerequisites: ["Basic Python: variables, functions, loops and lists", "School algebra; no calculus or statistics course needed"],
  prerequisiteCourses: [], practiceMode: "notebook", dependencyTier: 0, nextCourseId: "neural-networks",
  outcomes: [
    "Write vector, matrix and statistics functions from scratch and test them against hand calculations",
    "Compute derivatives numerically and check analytic gradients",
    "Implement gradient descent and diagnose learning rates, divergence and stopping",
    "Recognise floating-point error and use numerically stable formulas",
  ],
  references: [
    { title: "Mathematics for Machine Learning, by Deisenroth, Faisal and Ong (book website)", url: "https://mml-book.github.io/" },
    { title: "Python tutorial: floating-point arithmetic, issues and limitations", url: "https://docs.python.org/3/tutorial/floatingpoint.html" },
  ],
  lessons: [...mathFoundationsLessonsA, ...mathFoundationsLessonsB],
};
