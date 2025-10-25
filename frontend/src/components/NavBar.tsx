import React from "react";
import { useNavigate } from "react-router-dom";

interface NavBarProps {
  selectedTool: string;
}

  const NavBar: React.FC<NavBarProps> = ({ selectedTool }) => {
   
    return (
    <nav className="w-full py-6 h-16 bg-gray-800 text-white flex items-center justify-between px-4 border-b-2 border-white z-10">
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-bold text-blue-300">Furina ML</h1>
       
      </div>
      <div className="flex items-center gap-4">
        <div className="flex gap-4">
          <span className="font-bold text-blue-400">
            {selectedTool} Page 
          </span>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;