import Paper from "@mui/material/Paper";
import { DataGrid } from "@mui/x-data-grid";
import type { GridColDef } from "@mui/x-data-grid";

import type { TableViewerProps } from "../interface/PreprocessingTypes";

/**
 * Missing data is the whole reason to look at a preview, so it is drawn
 * rather than left blank.
 */
function MutedChip({ label, title }: { label: string; title: string }) {
  return (
    <span
      title={title}
      className="inline-block rounded-control px-2 py-0.5 text-xs font-semibold uppercase tracking-wide bg-paper-sunken text-ink-muted"
    >
      {label}
    </span>
  );
}

export default function TableViewer({
  preview,
  title = "Preview",
  subtitle,
  totalRows,
}: TableViewerProps) {
  const headers = preview.headers;
  const rawRows = preview.rows;

  // Synthetic field ids: a real header may be "1st Name", "class", or repeated,
  // none of which survive being used directly as an object key in the grid.
  const columns: GridColDef[] = headers.map((header, index) => ({
    field: `c${index}`,
    headerName: header,
    minWidth: 140,
    flex: 1,
    sortable: true,
    renderCell: (params) => {
      const value: unknown = params.value;
      if (value === null || value === undefined) {
        return <MutedChip label="null" title="Missing value" />;
      }
      if (value === "") {
        return (
          <MutedChip label="(empty)" title="Present, but an empty string" />
        );
      }
      return <span className="text-ink">{String(value)}</span>;
    },
  }));

  const rows: Record<string, string | number | null>[] = rawRows.map(
    (row, rowIndex) => {
      const record: Record<string, string | number | null> = { id: rowIndex };
      headers.forEach((_header, columnIndex) => {
        record[`c${columnIndex}`] = row[columnIndex] ?? null;
      });
      return record;
    }
  );

  const shownRows = rows.length;

  return (
    <div>
      <div className="mb-4">
        <h2 className="font-serif text-h3 text-ink">{title}</h2>
        <p className="text-sm text-ink-muted">
          {subtitle ??
            (totalRows !== undefined
              ? `First ${shownRows} of ${totalRows.toLocaleString()} rows · ${headers.length} columns`
              : `${shownRows} rows · ${headers.length} columns`)}
        </p>
      </div>

      {headers.length === 0 || shownRows === 0 ? (
        <div className="py-8 text-center">
          <p className="font-semibold text-ink">Nothing to show</p>
          <p className="text-sm text-ink-muted mt-1">
            {headers.length === 0
              ? "This file parsed to zero columns. Try a different column separator."
              : "This dataset has no rows left. A cleaning step may have removed them all."}
          </p>
        </div>
      ) : (
        <Paper sx={{ height: 420, width: "100%" }} elevation={0}>
          <DataGrid
            rows={rows}
            columns={columns}
            initialState={{
              pagination: { paginationModel: { page: 0, pageSize: 10 } },
            }}
            pageSizeOptions={[5, 10, 25]}
            disableRowSelectionOnClick
            density="compact"
            sx={{ border: 0 }}
          />
        </Paper>
      )}
    </div>
  );
}
