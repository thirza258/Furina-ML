import React from "react";

interface SidebarProps {
  selectedTool: string;
  setSelectedTool: (tool: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  selectedTool,
  setSelectedTool,
}) => {
  const tools = [
    {
      name: "Applications",
      tools: [
        "Landing Page",
        "About Us"

      ],
    },
    {
      name: "Traditional Machine Learning",
      tools: [
        "Data Preprocessing",
        "Data Training",
        "Model Evaluation",
      ],
    },
  ];

  return (
    <div className="w-1/4 bg-gray-200 h-full p-4 border-r-2 border-white z-10 shadow-lg flex-shrink-0 flex flex-col">
      <div className="flex-shrink-0">
        <h2 className="mb-4 text-xl font-semibold text-secondary">Tools</h2>
        <div className=" overflow-y-auto">
          {tools.map((toolGroup) => (
            <div key={toolGroup.name} className="mb-4">
              <details className="group">
                <summary className="flex justify-between items-center cursor-pointer font-medium text-secondary mb-2 hover:bg-gray-300 p-2 rounded border-b border-gray-300">
                  {toolGroup.name}
                  <span className="transition group-open:rotate-180">▼</span>
                </summary>
                <ul className="pl-4 text-sm text-secondary">
                  {toolGroup.tools.map((tool) => (
                    <li key={tool} className="mb-2">
                      <a
                        href="#"
                        className={`block p-2 rounded hover:bg-gray-300 cursor-pointer border-b border-gray-300 ${
                          selectedTool === tool ? "font-bold bg-gray-300" : ""
                        }`}
                        onClick={(e) => {
                          e.preventDefault();
                          setSelectedTool(tool);
                        }}
                      >
                        {tool}
                      </a>
                    </li>
                  ))}
                </ul>
              </details>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
