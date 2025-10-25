import React, { useState } from 'react';

// Since the services are not created, we can define a placeholder.
// import services from '../services/services';

interface InsertDataFileProps {
  onUploadSuccess: (documentName: string) => void;
}

const InsertDataFile: React.FC<InsertDataFileProps> = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [documentName, setDocumentName] = useState<string>('data.csv'); // Default file name

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];

      // Example validation for CSV files
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        setSelectedFile(file);
        setError('');
        setDocumentName(file.name);
      } else {
        setError('Invalid file type. Please upload a CSV file.');
        setSelectedFile(null);
        // Reset to default or clear the name
        setDocumentName('data.csv'); 
        event.target.value = ''; // Clear the input
      }
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError('Please select a CSV file to submit.');
      return;
    }

    setUploading(true);
    setError('');

    try {
      // The service call is commented out as it is not yet created.
      // await services.insertDataFile(selectedFile);
      
      // Simulate a successful upload for demonstration purposes
      console.log(`Simulating upload for: ${documentName}`);
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay

      onUploadSuccess(documentName);

    } catch (err) {
      setError('An error occurred during file submission. Please try again.');
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-white p-6 w-full">
      <h2 className="text-xl font-bold text-secondary mb-4">Load Your Data</h2>
      
      <div className="mb-4">
        <label htmlFor="file-upload" className="sr-only">
          Choose file
        </label>
        <input
          id="file-upload"
          type="file"
          className="w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-primary/20 file:text-secondary
            hover:file:bg-primary/30"
          onChange={handleFileChange}
          accept=".csv, text/csv"
        />
      </div>

      <div className="mb-6">
        <p className="text-gray-700">
          Current file: <span className="font-semibold">{documentName}</span>
        </p>

        {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
        
        <button
          className="w-full mt-4 bg-primary text-white py-2 px-4 rounded-md hover:bg-opacity-90 transition-colors disabled:bg-primary/50"
          onClick={handleSubmit}
          disabled={!selectedFile || uploading}
        >
          {uploading ? 'Submitting...' : 'Submit'}
        </button>
      </div>
    </div>
    )
}
export default InsertDataFile;