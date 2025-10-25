import React, { useState } from "react";
import TableViewer from "../../components/TableViewer";
import InsertDataFile from "../../components/InsertFile";
import PreprocessingCheckbox from "../../components/PreprocessingCheckbox";
import type { CheckboxStates } from "../../interface/PreprocessingTypes";

const DataPreprocessing: React.FC = () => {
  const [checkboxes, setCheckboxes] = useState<CheckboxStates>({
    removeDuplicates: false,
    fillWithMean: false,
    fillWithEmptyString: false,
    fillWithMode: false,
    lowercaseData: false,
    deleteOutlier: false,
    minMaxScaling: false,
    zScore: false,
    oneHotEncoding: false,
    labelEncoding1: false,
    labelEncoding2: false,
    lowercasing: false,
    tokenization: false,
    removePunctuation: false,
    pca: false,
    featureSelection: false,
    lda: false,
  });

  const [selectedColumn, setSelectedColumn] = useState<string>("column1");

  const handleCheckboxChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    setCheckboxes({
      ...checkboxes,
      [name]: checked,
    });
  };

  const handleSelectChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedColumn(event.target.value);
  };

  const mockData = {
    headers: ["Name", "Age", "City", "Salary", "Department"],
    rows: [
      ["John Smith", "32", "New York", "75000", "Engineering"],
      ["Sarah Johnson", "28", "San Francisco", "82000", "Marketing"],
      ["Michael Williams", "41", "Chicago", "65000", "Sales"], 
      ["Emily Brown", "35", "Boston", "70000", "Engineering"],
      ["James Wilson", "29", "Seattle", "68000", "Product"],
      ["Jessica Taylor", "38", "Austin", "72000", "Marketing"],
      ["David Miller", "44", "Denver", "80000", "Sales"],
      ["Lisa Anderson", "31", "Portland", "69000", "Engineering"],
      ["Robert Thomas", "37", "Miami", "71000", "Product"],
      ["Jennifer Garcia", "33", "Los Angeles", "77000", "Marketing"]
    ]
  };

  const sections = [
    {
      title: "Data Cleaning",
      options: [
        { id: "removeDuplicates", label: "remove duplicates" },
        { id: "fillWithMean", label: "fill missing values with mean" },
        {
          id: "fillWithEmptyString",
          label: "fill missing values with empty string",
        },
        { id: "fillWithMode", label: "fill missing values with mode" },
        { id: "lowercaseData", label: "lowercase the data" },
        { id: "deleteOutlier", label: "Delete the outlier" },
      ],
    },
    {
      title: "Data Transformation",
      options: [
        { id: "minMaxScaling", label: "MinMax Scaling" },
        { id: "zScore", label: "Z-Score" },
      ],
    },
    {
      title: "Data Encoding",
      options: [
        { id: "oneHotEncoding", label: "one hot encoding" },
        { id: "labelEncoding1", label: "Label encoding" },
        { id: "labelEncoding2", label: "Label encoding" },
      ],
    },
    {
      title: "Text cleaning",
      options: [
        { id: "lowercasing", label: "Lowercasing" },
        { id: "tokenization", label: "Tokenization" },
        { id: "removePunctuation", label: "Remove punctuation" },
      ],
    },
    {
      title: "Feature Engineering",
      options: [
        { id: "pca", label: "PCA" },
        {
          id: "featureSelection",
          label: "Feature selection",
          dropdown: {
            options: ["column 1", "column 2", "column 3"],
            value: selectedColumn,
            onChange: handleSelectChange,
          },
        },
        { id: "lda", label: "LDA" },
      ],
    },
  ];

  const handleUploadSuccess = (documentName: string) => {
    console.log(`File uploaded successfully: ${documentName}`);
  };

  return (
    <div className="bg-gray-100 min-h-screen p-8">
      <div className="">
        <h1 className="text-4xl font-bold text-secondary mb-6">
          Data Preprocessing
        </h1>

        <div className="mb-6">
          <InsertDataFile onUploadSuccess={handleUploadSuccess} />
        </div>
        <hr className="my-6 border-t border-gray-300" />

        <TableViewer headers={mockData.headers} rows={mockData.rows} />
        <hr className="my-6 border-t border-gray-300" />

        <PreprocessingCheckbox
          sections={sections}
          checkboxes={checkboxes}
          handleCheckboxChange={handleCheckboxChange}
        />
      </div>
      <div className="mt-8">
        <button
          className="w-full bg-primary text-white py-3 px-6 rounded-md hover:bg-opacity-90 transition-colors disabled:bg-primary/50 font-semibold text-lg"
          onClick={() => {
            // TODO: Implement preprocessing logic
            console.log("Preprocessing data...");
          }}
        >
          Preprocess Data
        </button>
        <button
          className="w-full mt-4 bg-gray-400 text-white py-3 px-6 rounded-md hover:bg-opacity-90 transition-colors disabled:bg-gray-400/50 font-semibold text-lg"
          onClick={() => {
            console.log("Downloading data...");
          }}
          disabled
        >
          Download Data
        </button>
        <button
          className="w-full mt-4 bg-gray-400 text-white py-3 px-6 rounded-md hover:bg-opacity-90 transition-colors disabled:bg-gray-400/50 font-semibold text-lg"
          onClick={() => {
            console.log("Continuing to training...");
          }}
          disabled
        >
          Continue to Training
        </button>
      </div>
    </div>
  );
};

export default DataPreprocessing;
