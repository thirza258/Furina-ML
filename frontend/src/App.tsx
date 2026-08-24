import { useCallback, useState } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';

import  AboutPage  from './pages/AboutPage';

import NavBar from './components/NavBar';
import Sidebar from './components/Sidebar';
import WorkspaceProvider from './context/WorkspaceProvider';

import LandingPage from './pages/LandingPage';
import DataPreprocessing from './pages/traditionalMl/DataPreprocessing';
import DataTraining from './pages/traditionalMl/DataTraining';
import EvaluationReport from './pages/traditionalMl/ModelEvaluation';

function App() {
  const [selectedTool, setSelectedTool] = useState("Landing Page");
  const navigate = useNavigate();

  const handleToolSelection = useCallback((tool: string) => {
    setSelectedTool(tool);
    navigate('/');
  }, [navigate]);

  const renderSelectedPage = () => {
    switch (selectedTool) {
      case "Landing Page":
        return <LandingPage />;
      case "About Us":
        return <AboutPage />;
      case "Data Preprocessing":
        return <DataPreprocessing />;
      case "Data Training":
        return <DataTraining />;
      case "Model Evaluation":
        return <EvaluationReport />;
      default: return <LandingPage />;
    }
  };


  return (
    <WorkspaceProvider onNavigate={handleToolSelection}>
      <div className="h-screen flex flex-col bg-gray-100">
        <NavBar
          selectedTool={selectedTool}
        />
        <div className="flex flex-grow overflow-hidden">
          <Sidebar
            selectedTool={selectedTool}
            setSelectedTool={handleToolSelection}
          />
          <main className="flex-grow overflow-y-auto p-6">
            <Routes>
              <Route path="/" element={renderSelectedPage()} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </WorkspaceProvider>
  );
}

export default App;
