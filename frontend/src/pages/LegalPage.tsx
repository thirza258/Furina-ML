import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function LegalPage() {
  const location = useLocation();
  const [tab, setTab] = useState<'privacy' | 'terms'>(
    location.pathname.includes('terms') ? 'terms' : 'privacy'
  );

  return (
    <div className="min-h-screen bg-canvas text-ink py-12 px-6 sm:px-12 font-sans selection:bg-amber-100 selection:text-ink">
      <div className="max-w-4xl mx-auto">
        {/* Navigation bar */}
        <header className="pb-6 mb-8 border-b border-rule flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/" className="flex items-center gap-2.5 text-ink hover:text-ink-muted transition-colors">
              <img src="/logo_furina.png" alt="Furina ML Logo" className="w-7 h-7 object-contain" />
              <span className="font-serif text-lg font-bold">Furina ML</span>
            </Link>
            <span className="text-caption text-ink-muted">/ Legal &amp; Policies</span>
          </div>
          <Link
            to="/"
            className="text-small text-ink-muted hover:text-ink font-medium transition-colors"
          >
            &larr; Back to Home
          </Link>
        </header>

        {/* Header Title */}
        <div className="mb-8">
          <p className="text-caption uppercase tracking-wider text-ink-muted font-mono mb-2">
            SELF-HOSTED PLATFORM TRANSPARENCY
          </p>
          <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight mb-3">
            Privacy Policy &amp; Terms of Service
          </h1>
          <p className="text-body text-ink-muted">
            Information regarding Furina ML&apos;s self-hosted deployment architecture, local data storage, and external AI integrations.
          </p>

          <div className="mt-6 flex gap-3">
            <button
              onClick={() => setTab('privacy')}
              className={`px-4 py-2 text-small rounded-md font-medium transition-colors ${
                tab === 'privacy'
                  ? 'bg-ink text-canvas shadow-sm'
                  : 'bg-surface border border-rule text-ink hover:bg-canvas'
              }`}
            >
              Privacy Policy
            </button>
            <button
              onClick={() => setTab('terms')}
              className={`px-4 py-2 text-small rounded-md font-medium transition-colors ${
                tab === 'terms'
                  ? 'bg-ink text-canvas shadow-sm'
                  : 'bg-surface border border-rule text-ink hover:bg-canvas'
              }`}
            >
              Terms of Service
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {tab === 'privacy' ? (
          <div className="space-y-6 text-body leading-relaxed">
            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">1. 100% Self-Hosted &amp; Private Infrastructure</h2>
              <p className="mb-3 text-ink-soft">
                Furina ML is an open-source, fully self-hosted machine learning educational workbench.
                All uploaded CSV datasets, cleaning pipelines, transformed datasets, scikit-learn models,
                fitted .joblib binaries, test evaluation metrics, and user accounts reside strictly within your
                self-hosted containers and PostgreSQL database.
              </p>
              <p className="text-ink-soft">
                There are zero central telemetry servers, tracking scripts, or data brokers.
                We do not sell, rent, monetize, or harvest your personal datasets or model training runs.
              </p>
            </section>

            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">2. In-App AI Assistant via OpenRouter</h2>
              <p className="mb-3 text-ink-soft">
                When enabled, the in-app machine learning assistant connects to third-party language models via <strong>OpenRouter</strong>.
              </p>
              <p className="mb-3 text-ink-soft">
                When you explicitly ask a question using the assistant:
              </p>
              <ul className="list-disc pl-5 mb-4 space-y-1 text-ink-soft">
                <li>Only the prompt question and high-level tool context (e.g., current stage name or estimator parameter metadata) are dispatched to OpenRouter for inference.</li>
                <li>Your raw private datasets, confidential CSV files, database credentials, and host environment configs are never transmitted.</li>
              </ul>
              <div className="bg-canvas border border-rule rounded-lg p-4 text-small">
                <p className="font-mono text-caption uppercase text-ink-muted mb-2 font-semibold">
                  OpenRouter Terms &amp; Privacy Policies:
                </p>
                <div className="flex flex-wrap gap-4">
                  <a
                    href="https://openrouter.ai/privacy"
                    target="_blank"
                    rel="noreferrer noopener"
                    className="text-ink font-medium underline underline-offset-2 hover:text-ink-muted"
                  >
                    OpenRouter Privacy Policy &rarr;
                  </a>
                  <a
                    href="https://openrouter.ai/terms"
                    target="_blank"
                    rel="noreferrer noopener"
                    className="text-ink font-medium underline underline-offset-2 hover:text-ink-muted"
                  >
                    OpenRouter Terms of Service &rarr;
                  </a>
                </div>
              </div>
            </section>

            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">3. Data Retention &amp; Host Control</h2>
              <p className="text-ink-soft">
                Because Furina ML runs on your own instance, you retain complete administrative control over all records.
                Datasets and trained runs can be permanently deleted at any time through the workbench interface or by clearing docker storage volumes.
              </p>
            </section>

            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">4. Privacy Contact</h2>
              <p className="text-ink-soft">
                For questions regarding data privacy or server policies, please contact:{" "}
                <a href="mailto:nevatalapp@gmail.com" className="font-medium underline hover:text-ink-muted">
                  nevatalapp@gmail.com
                </a>
              </p>
            </section>
          </div>
        ) : (
          <div className="space-y-6 text-body leading-relaxed">
            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">1. Terms of Use</h2>
              <p className="text-ink-soft">
                By deploying, running, or accessing Furina ML, you agree to these Terms of Service.
                Furina ML is an educational and experimentation tool designed to understand classical machine learning workflows and evaluation metrics.
              </p>
            </section>

            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">2. Model Predictions Disclaimer</h2>
              <p className="text-ink-soft">
                All models trained using scikit-learn in Furina ML are statistical estimators fitted on user-supplied data.
                Maintainers make no warranty regarding model accuracy, fitness for a particular industrial use, or generalization beyond the provided training split.
                Critical real-world decisions should undergo independent rigorous validation.
              </p>
            </section>

            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">3. OpenRouter AI Guidelines</h2>
              <p className="text-ink-soft">
                When using the AI assistant, users agree to abide by the usage guidelines set by upstream foundation model providers:
              </p>
              <div className="mt-3 flex flex-wrap gap-4 text-small">
                <a
                  href="https://openrouter.ai/terms"
                  target="_blank"
                  rel="noreferrer noopener"
                  className="text-ink font-medium underline underline-offset-2 hover:text-ink-muted"
                >
                  OpenRouter Terms of Service &rarr;
                </a>
                <a
                  href="https://openrouter.ai/privacy"
                  target="_blank"
                  rel="noreferrer noopener"
                  className="text-ink font-medium underline underline-offset-2 hover:text-ink-muted"
                >
                  OpenRouter Privacy Policy &rarr;
                </a>
              </div>
            </section>

            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">4. Disclaimer of Warranty</h2>
              <p className="text-ink-soft">
                Furina ML is provided under the MIT license on an &ldquo;AS IS&rdquo; basis without warranties of any kind.
                In no event shall the authors or contributors be liable for any claim, damages, or other liability.
              </p>
            </section>

            <section className="bg-surface border border-rule rounded-xl p-6 sm:p-8">
              <h2 className="font-serif text-xl font-bold mb-3">5. Terms Contact</h2>
              <p className="text-ink-soft">
                For questions regarding terms and licensing, reach out to:{" "}
                <a href="mailto:nevatalapp@gmail.com" className="font-medium underline hover:text-ink-muted">
                  nevatalapp@gmail.com
                </a>
              </p>
            </section>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-12 pt-6 border-t border-rule text-caption text-ink-muted flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            <p>Furina ML · Self-Hosted Machine Learning Platform</p>
            <p className="text-[11px] text-ink-faint">This project is under nevatal.id for educational purposes.</p>
          </div>
          <p>Contact: nevatalapp@gmail.com</p>
        </footer>
      </div>
    </div>
  );
}
