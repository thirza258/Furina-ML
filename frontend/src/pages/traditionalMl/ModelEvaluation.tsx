import React from 'react';

// Define a type for the metrics for better code organization
type EvaluationMetrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f2Score: number;
};

// Define a type for the confusion matrix values
type ConfusionMatrix = {
  truePositive: number;
  falseNegative: number;
  falsePositive: number;
  trueNegative: number;
};

const EvaluationReport: React.FC = () => {
  // Mock data for the report
  const metrics: EvaluationMetrics = {
    accuracy: 0.95,
    precision: 0.92,
    recall: 0.96,
    f2Score: 0.95,
  };

  const confusionMatrix: ConfusionMatrix = {
    truePositive: 150,
    falseNegative: 10,
    falsePositive: 8,
    trueNegative: 182,
  };
  
  // Handlers for the download buttons
  const handleDownloadReport = () => {
    // Placeholder for report download logic
    alert('Downloading evaluation report...');
  };

  const handleDownloadModel = () => {
    // Placeholder for model download logic
    alert('Downloading the model...');
  };

  const convertToPercentage = (value: number) => {
    return (value * 100).toFixed(2) + "%";
  };

  return (
    <div className="bg-gray-100 min-h-screen p-8">
      <div className="w-full bg-white p-8 rounded-lg shadow-lg text-secondary">
        <h1 className="text-4xl font-bold text-secondary mb-10">Evaluation Report</h1>

        {/* Metrics Section */}
        <div className="space-y-5 mb-10">
          {Object.entries(metrics).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between">
              <p className="text-lg font-semibold capitalize">
                {key.replace(/([A-Z])/g, ' $1').trim()}
              </p>
              <div className="w-1/2 bg-gray-200 p-3 rounded-md text-right font-mono">
                {value.toFixed(2)} / {convertToPercentage(value)}
              </div>
            </div>
          ))}
        </div>

        {/* Confusion Matrix Section */}
        <div className="mb-12">
          <h2 className="text-xl font-semibold mb-4">
            Confusion matrix <span className="text-gray-500 font-normal">(table based on column tested)</span>
          </h2>
          <div className="grid grid-cols-2 gap-4 text-center">
            <div className="bg-gray-200 p-6 rounded-lg">
              <p className="font-bold text-lg text-green-700">True Positive</p>
              <p className="text-2xl font-mono">{confusionMatrix.truePositive}</p>
            </div>
            <div className="bg-gray-200 p-6 rounded-lg">
              <p className="font-bold text-lg text-red-700">False Negative</p>
              <p className="text-2xl font-mono">{confusionMatrix.falseNegative}</p>
            </div>
            <div className="bg-gray-200 p-6 rounded-lg">
              <p className="font-bold text-lg text-red-700">False Positive</p>
              <p className="text-2xl font-mono">{confusionMatrix.falsePositive}</p>
            </div>
            <div className="bg-gray-200 p-6 rounded-lg">
              <p className="font-bold text-lg text-green-700">True Negative</p>
              <p className="text-2xl font-mono">{confusionMatrix.trueNegative}</p>
            </div>
          </div>
        </div>

        {/* Download Buttons */}
        <div className="space-y-4">
          <button 
            onClick={handleDownloadReport}
            className="w-full bg-primary text-white py-3 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors"
          >
            Download the evaluation report
          </button>
          <button 
            onClick={handleDownloadModel}
            className="w-full bg-gray-300 text-gray-800 py-3 px-4 rounded-md text-lg font-semibold hover:bg-gray-400 transition-colors"
          >
            Download the model
          </button>
        </div>

        <div className="my-10">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Try It Yourself</h2>
            <div className="space-y-4">
                <div>
                    <label htmlFor="feature1" className="block text-sm font-medium text-gray-700 mb-1">
                        Feature 1
                    </label>
                    <input
                        type="number"
                        id="feature1"
                        className="w-full p-2 border border-gray-300 rounded-md"
                        placeholder="Enter value for feature 1"
                    />
                </div>
                <div>
                    <label htmlFor="feature2" className="block text-sm font-medium text-gray-700 mb-1">
                        Feature 2
                    </label>
                    <input
                        type="number" 
                        id="feature2"
                        className="w-full p-2 border border-gray-300 rounded-md"
                        placeholder="Enter value for feature 2"
                    />
                </div>
                <button
                    className="w-full bg-primary text-white py-2 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors"
                >
                    Predict
                </button>
                <p className="text-sm text-gray-600 mt-1">The predicted result will be displayed here</p>
            </div>
        </div>
      </div>
    </div>
  );
};

export default EvaluationReport;