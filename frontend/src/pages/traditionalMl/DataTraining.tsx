import React, { useState, useEffect } from 'react';

const TrainingData: React.FC = () => {
  const [splitValue, setSplitValue] = useState<number>(50);
  const [selectedModel, setSelectedModel] = useState<string>('logistic-regression');
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);

  // Model options for the dropdown
  const modelOptions = [
    { value: 'logistic-regression', label: 'Scikit learn - logistic regression' },
    { value: 'svm', label: 'Scikit learn - Support Vector Machine' },
    { value: 'neural-network', label: 'TensorFlow - Neural Network' },
    { value: 'random-forest', label: 'Scikit learn - Random Forest' },
  ];

  // Handler for the slider change
  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSplitValue(Number(event.target.value));
  };

  // Handler for the number input change
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    let value = Number(event.target.value);
    if (value < 0) value = 0;
    if (value > 100) value = 100;
    
    if (event.target.name === 'training') {
      setSplitValue(value);
    } else {
      setSplitValue(100 - value);
    }
  };

  // Handler for model selection change
  const handleModelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedModel(event.target.value);
  };

  // Simulate training progress when the button is clicked
  const handleTrainModel = () => {
    if (isTraining) return;
    
    setIsTraining(true);
    setProgress(0);
  };
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    if (isTraining && progress < 100) {
      timer = setTimeout(() => {
        setProgress(prev => prev + 1);
      }, 50); // Adjust timing to control progress speed
    } else if (progress >= 100) {
      setIsTraining(false);
    }
    return () => clearTimeout(timer);
  }, [isTraining, progress]);

  return (
    <div className="bg-gray-100 min-h-screen p-8 w-full">
      <div className="w-full bg-white p-8 rounded-lg shadow-lg">
        <h1 className="text-4xl font-bold text-secondary mb-8">Training Data</h1>

        {/* Data Split Section */}
        <div className="mb-10">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Split the data</h2>
          <div className="relative flex items-center">
            <input
              type="range"
              min="0"
              max="100"
              value={splitValue}
              onChange={handleSliderChange}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>
          <div className="flex justify-between items-center mt-4">
            <div className="flex items-center space-x-2">
              <input
                type="number"
                name="training"
                value={splitValue}
                onChange={handleInputChange}
                className="w-20 p-2 border border-gray-300 rounded-md text-center bg-gray-200"
              />
              <span className="font-medium text-gray-700">% Training</span>
            </div>
            <div className="flex items-center space-x-2">
               <input
                type="number"
                name="testing"
                value={100 - splitValue}
                onChange={handleInputChange}
                className="w-20 p-2 border border-gray-300 rounded-md text-center bg-gray-200"
              />
              <span className="font-medium text-gray-700">% Testing</span>
            </div>
          </div>
        </div>

        <div>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Choose the target column</h2>
            <select
                id="target-select"
               
                className="w-full p-3 bg-gray-200 border border-gray-300 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary"
            >
            </select>
            <span className="font-medium text-gray-700">Target Column</span>
        </div>
        <div>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Choose the features</h2>
            <select
                id="features-select"
               
                className="w-full p-3 bg-gray-200 border border-gray-300 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary"
            >
            </select>
            <span className="font-medium text-gray-700">Features</span>
        </div>

        <div>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Set random seed</h2>
            <div className="flex items-center space-x-4">
                <input
                    type="number"
                    id="random-seed"
                    placeholder="Enter random seed (optional)"
                    className="w-full p-3 bg-gray-200 border border-gray-300 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary"
                />
                <button
                    className="px-4 py-3 bg-gray-300 hover:bg-gray-400 rounded-md text-gray-800 font-medium transition-colors"
                    onClick={() => {
                        const randomSeed = Math.floor(Math.random() * 1000);
                        const input = document.getElementById('random-seed') as HTMLInputElement;
                        if (input) input.value = randomSeed.toString();
                    }}
                >
                    Random
                </button>
            </div>
            <p className="text-sm text-gray-600 mt-2">Setting a random seed ensures reproducible results. Leave empty for random behavior.</p>
        </div>

        {/* Choose Model Section */}
        <div className="mb-10">
          <label htmlFor="model-select" className="block text-xl font-semibold text-gray-800 mb-4">
            Choose model
          </label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={handleModelChange}
            className="w-full p-3 bg-gray-200 border border-gray-300 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary"
          >
            {modelOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div className="mt-10">
          <label className="block text-xl font-semibold text-gray-800 mb-4">
            Hyperparameters
          </label>
          
          <div className="space-y-4">
            <div>
              <label htmlFor="learning-rate" className="block text-sm font-medium text-gray-700 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                id="learning-rate"
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="0.001"
                step="0.001"
                min="0.0001"
                max="1"
              />
            </div>

            <div>
              <label htmlFor="max-depth" className="block text-sm font-medium text-gray-700 mb-1">
                Max Depth
              </label>
              <input 
                type="number"
                id="max-depth"
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="3"
                min="1"
                max="10"
              />
            </div>

            <div>
              <label htmlFor="n-estimators" className="block text-sm font-medium text-gray-700 mb-1">
                Number of Estimators
              </label>
              <input
                type="number"
                id="n-estimators" 
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="100"
                min="10"
                max="1000"
              />
            </div>
          </div>
          
          <p className="text-sm text-gray-600 mt-2">
            Adjust these hyperparameters to optimize your model's performance. Hover over each field for more information.
          </p>
        </div>
        
        {/* Train Button and Progress Bar */}
        <div className="space-y-4">
           <button 
             onClick={handleTrainModel}
             disabled={isTraining}
             className="w-full bg-primary text-white py-3 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors disabled:bg-primary/50 disabled:cursor-not-allowed"
           >
            {isTraining ? 'Training...' : 'Train the model'}
           </button>
           
           <div className="w-full bg-gray-200 rounded-full h-8 overflow-hidden">
             <div 
               className="bg-accent h-8 flex items-center justify-center text-white font-bold transition-all duration-300 ease-linear"
               style={{ width: `${progress}%` }}
             >
              {progress > 0 && `${progress}% Progress`}
              {progress === 0 && 'Progress'}
             </div>
           </div>
        </div>

        
      </div>
    </div>
  );
};

export default TrainingData;