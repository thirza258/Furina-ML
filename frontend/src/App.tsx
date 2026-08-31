/**
 * Routing, and the workbench shell.
 *
 * Two things live at the top level: the public page at `/`, which has no
 * chrome because none of it means anything before a dataset exists, and the
 * workbench at `/app`, which has all of it.
 *
 * Inside the workbench the URL is the source of truth for which tool is
 * showing, rather than a `useState` in this file. That is what makes each step
 * deep-linkable and makes the browser's back button behave: switching tools is
 * a navigation, so going back returns you to the previous tool instead of
 * dropping you out of the app entirely.
 */
import { useCallback } from 'react';
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';

import AboutPage from './pages/AboutPage';
import Dashboard from './pages/Dashboard';
import HomePage from './pages/HomePage';
import LandingPage from './pages/LandingPage';
import DataPreprocessing from './pages/traditionalMl/DataPreprocessing';
import DataTraining from './pages/traditionalMl/DataTraining';
import EvaluationReport from './pages/traditionalMl/ModelEvaluation';

import AssistantFab from './components/AssistantFab';
import NavBar from './components/NavBar';
import Sidebar from './components/Sidebar';
import WorkspaceProvider from './context/WorkspaceProvider';

/**
 * The tool names the sidebar and the pages pass around, paired with the path
 * each one lives at. The names are the contract — `goToTool("Data Training")`
 * is called from several pages — so a typo here is a dead link.
 */
const TOOL_ROUTES: { tool: string; path: string }[] = [
  { tool: 'Landing Page', path: '/app' },
  { tool: 'Dashboard', path: '/app/dashboard' },
  { tool: 'Data Preprocessing', path: '/app/preprocessing' },
  { tool: 'Data Training', path: '/app/training' },
  { tool: 'Model Evaluation', path: '/app/evaluation' },
  { tool: 'About Us', path: '/about' },
];

const PATH_FOR_TOOL = new Map(TOOL_ROUTES.map(({ tool, path }) => [tool, path]));
const TOOL_FOR_PATH = new Map(TOOL_ROUTES.map(({ tool, path }) => [path, tool]));

const DEFAULT_TOOL = 'Landing Page';

function pathForTool(tool: string): string {
  return PATH_FOR_TOOL.get(tool) ?? '/app';
}

/** The chrome: top bar, tool sidebar, and whichever tool the URL names. */
function Workbench() {
  const { pathname } = useLocation();
  const navigate = useNavigate();

  // Trailing slashes are the one variation worth absorbing: /app/training/
  // and /app/training have to mean the same thing.
  const normalized =
    pathname.length > 1 && pathname.endsWith('/')
      ? pathname.slice(0, -1)
      : pathname;
  const selectedTool = TOOL_FOR_PATH.get(normalized) ?? DEFAULT_TOOL;

  const handleToolSelection = useCallback(
    (tool: string) => {
      navigate(pathForTool(tool));
    },
    [navigate]
  );

  const renderSelectedPage = () => {
    switch (selectedTool) {
      case 'Data Preprocessing':
        return <DataPreprocessing />;
      case 'Data Training':
        return <DataTraining />;
      case 'Model Evaluation':
        return <EvaluationReport />;
      case 'Dashboard':
        return <Dashboard />;
      case 'About Us':
        return <AboutPage />;
      case 'Landing Page':
      default:
        return <LandingPage />;
    }
  };

  return (
    <div className="h-dvh flex flex-col bg-paper">
      <NavBar selectedTool={selectedTool} />
      <div className="flex flex-grow overflow-hidden">
        <Sidebar
          selectedTool={selectedTool}
          setSelectedTool={handleToolSelection}
        />
        <main className="flex-grow overflow-y-auto">
          {/* A cap on the measure: full-bleed panels on a wide monitor put
              120 characters on a line, which nobody reads. */}
          <div className="mx-auto max-w-7xl">{renderSelectedPage()}</div>
        </main>
      </div>
      {/* Sits above the whole workbench so it is reachable from every tool. */}
      <AssistantFab tool={selectedTool} />
    </div>
  );
}

function App() {
  const navigate = useNavigate();

  // The provider sits above the routes so a dataset survives a trip out to the
  // public page and back; if it were inside Workbench, every navigation would
  // remount it and re-run the restore.
  const goToTool = useCallback(
    (tool: string) => {
      navigate(pathForTool(tool));
    },
    [navigate]
  );

  return (
    <WorkspaceProvider onNavigate={goToTool}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/app" element={<Workbench />} />
        <Route path="/app/dashboard" element={<Workbench />} />
        <Route path="/app/preprocessing" element={<Workbench />} />
        <Route path="/app/training" element={<Workbench />} />
        <Route path="/app/evaluation" element={<Workbench />} />
        <Route path="/about" element={<Workbench />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </WorkspaceProvider>
  );
}

export default App;
