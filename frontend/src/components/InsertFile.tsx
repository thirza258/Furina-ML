import React, { useCallback, useRef, useState } from "react";

import services from "../services/services";
import type {
  DelimiterChoice,
  InsertDataFileProps,
} from "../interface/PreprocessingTypes";
import {
  ACCEPTED_EXTENSIONS,
  DELIMITER_OPTIONS,
  MAX_UPLOAD_BYTES,
} from "../interface/PreprocessingTypes";

/** Service methods reject with a normalized ApiError; render its `detail`. */
function errorDetail(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "string" && detail.length > 0) return detail;
  }
  if (error instanceof Error) return error.message;
  return "The upload failed for an unknown reason.";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Client-side gate so an obviously bad file never costs a round trip. */
function validate(file: File): string | null {
  const lower = file.name.toLowerCase();
  if (!ACCEPTED_EXTENSIONS.some((extension) => lower.endsWith(extension))) {
    return `"${file.name}" is not a delimited text file. Upload a ${ACCEPTED_EXTENSIONS.join(", ")} file.`;
  }
  if (file.size === 0) {
    return `"${file.name}" is empty.`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `"${file.name}" is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_UPLOAD_BYTES)}.`;
  }
  return null;
}

const InsertDataFile: React.FC<InsertDataFileProps> = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [delimiter, setDelimiter] = useState<DelimiterChoice>("auto");
  const [uploading, setUploading] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const accept = useCallback((file: File) => {
    const problem = validate(file);
    if (problem) {
      setError(problem);
      setSelectedFile(null);
      return;
    }
    setError(null);
    setProgress(0);
    setSelectedFile(file);
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) accept(file);
    // Allow re-picking the same file after a failed attempt.
    event.target.value = "";
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (uploading) return;
    const file = event.dataTransfer.files?.[0];
    if (file) accept(file);
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Choose a file first.");
      return;
    }

    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const dataset = await services.uploadDataset(selectedFile, {
        name: selectedFile.name,
        delimiter,
        onProgress: (pct) => setProgress(pct),
      });
      setSelectedFile(null);
      setProgress(100);
      onUploadSuccess(dataset);
    } catch (uploadError) {
      setError(errorDetail(uploadError));
      setProgress(0);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="panel">
      <h2 className="font-serif text-h3 text-ink mb-1">Load your data</h2>
      <p className="text-sm text-ink-muted mb-4">
        A delimited text file ({ACCEPTED_EXTENSIONS.join(", ")}), up to{" "}
        {formatBytes(MAX_UPLOAD_BYTES)}. The first row is read as the header.
      </p>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          if (!uploading) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => {
          if (!uploading) inputRef.current?.click();
        }}
        onKeyDown={(event) => {
          if (event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          if (!uploading) inputRef.current?.click();
        }}
        role="button"
        tabIndex={0}
        aria-label="Choose a data file, or drop one here"
        className={`cursor-pointer rounded-control border-2 border-dashed p-6 text-center transition-colors ${
          dragging
            ? "border-primary bg-primary/10"
            : "border-rule bg-paper-sunken hover:border-primary"
        } ${uploading ? "cursor-not-allowed opacity-60" : ""}`}
      >
        <p className="text-ink font-semibold">
          Drag a file here, or click to browse
        </p>
        <p className="text-sm text-ink-muted mt-1">
          {selectedFile
            ? `${selectedFile.name} — ${formatBytes(selectedFile.size)}`
            : "No file chosen yet"}
        </p>
        <input
          id="file-upload"
          ref={inputRef}
          type="file"
          className="hidden"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          onChange={handleFileChange}
          onClick={(event) => event.stopPropagation()}
          disabled={uploading}
        />
      </div>

      <div className="mt-4">
        <label
          htmlFor="delimiter"
          className="block text-sm font-semibold text-ink mb-1"
        >
          Column separator
        </label>
        <select
          id="delimiter"
          value={delimiter}
          onChange={(event) =>
            setDelimiter(event.target.value as DelimiterChoice)
          }
          disabled={uploading}
          className="w-full p-3 bg-paper-sunken border border-rule rounded-control text-ink focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-60"
        >
          {DELIMITER_OPTIONS.map((option) => (
            <option key={option.label} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-ink-muted mt-1">
          Leave on auto-detect unless the preview comes back as a single column.
        </p>
      </div>

      {uploading && (
        <div className="mt-4" aria-live="polite">
          <div className="flex justify-between text-sm text-ink-muted mb-1">
            <span>Uploading…</span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-paper-sunken overflow-hidden">
            <div
              className="h-2 rounded-full bg-primary transition-all"
              style={{ width: `${progress}%` }}
              role="progressbar"
              aria-valuenow={progress}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </div>
      )}

      {error && (
        <div
          className="mt-4 rounded-control border border-state-bad/40 bg-state-bad/10 p-3"
          role="alert"
        >
          <div className="flex items-start justify-between gap-4">
            <p className="text-sm text-state-bad">
              <span className="font-semibold">Upload failed — </span>
              {error}
            </p>
            <button
              type="button"
              onClick={() => setError(null)}
              className="text-sm font-semibold text-state-bad hover:underline"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      <button
        type="button"
        className="btn btn-primary text-body py-3 w-full mt-6"
        onClick={() => void handleSubmit()}
        disabled={!selectedFile || uploading}
      >
        {uploading ? "Uploading…" : "Upload dataset"}
      </button>
    </div>
  );
};

export default InsertDataFile;
