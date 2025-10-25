import Paper from "@mui/material/Paper";
import { DataGrid } from "@mui/x-data-grid";
import Typography from "@mui/material/Typography";

interface TableViewerProps {
  headers: string[];
  rows: string[][];
}

export default function TableViewer({ headers, rows }: TableViewerProps) {
  return (
    <div>
      {/* Header Summary */}
      <div style={{ marginBottom: "1.5rem" }}>
        <Typography variant="h6" fontWeight={600} color="secondary">
          All Columns of the Data
        </Typography>
        <Typography variant="body2" color="textSecondary">
          {headers.join(" | ")}
        </Typography>
      </div>

      {/* Table Preview */}
      <div style={{ marginBottom: "2rem" }}>
        <Typography variant="h6" fontWeight={600} color="secondary">
          First 5 Rows Preview
        </Typography>

        <Paper sx={{ height: 400, width: "100%", mt: 2 }}>
          <DataGrid
            rows={rows.map((row, index) => ({
              id: index,
              ...headers.reduce(
                (obj, header, i) => ({
                  ...obj,
                  [header]: row[i],
                }),
                {}
              ),
            }))}
            columns={headers.map((header) => ({
              field: header,
              headerName: header,
              width: 130,
              flex: 1,
            }))}
            initialState={{
              pagination: {
                paginationModel: { page: 0, pageSize: 5 },
              },
            }}
            pageSizeOptions={[5, 10]}
            checkboxSelection
            sx={{ border: 0 }}
          />
        </Paper>
      </div>
    </div>
  );
}
