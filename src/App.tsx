import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  FileUp, 
  Brain, 
  Activity, 
  Loader2, 
  CheckCircle2, 
  AlertCircle,
  BarChart4
} from 'lucide-react';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<string>('dt');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
      setResult(null);
      setError(null);
    },
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please upload a CSV file');
      return;
    }

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    try {
      // Use fetch API with proper error handling
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        let errorMessage = `HTTP error! Status: ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData && errorData.error) {
            errorMessage = errorData.error;
          }
        } catch (jsonError) {
          // If JSON parsing fails, use the default error message
          console.warn('Failed to parse error response as JSON', jsonError);
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred during prediction. Please check your file format and try again.';
      console.error('Error during prediction:', errorMessage);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const models = [
    { value: 'dt', label: 'Decision Tree' },
    { value: 'knn', label: 'K-Nearest Neighbors' },
    { value: 'rf', label: 'Random Forest' },
    { value: 'nb', label: 'Naive Bayes' },
    { value: 'ab', label: 'AdaBoost' },
    { value: 'lr', label: 'Logistic Regression' },
    { value: 'svm', label: 'Support Vector Machine' },
    { value: 'nn1', label: 'Neural Network (1 Hidden Layer)' },
    { value: 'nn2', label: 'Neural Network (2 Hidden Layers)' },
    { value: 'nn3', label: 'Neural Network (3 Hidden Layers)' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex items-center">
          <Brain className="h-8 w-8 text-indigo-600 mr-3" />
          <h1 className="text-3xl font-bold text-gray-900">Diabetes Prediction Platform</h1>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-xl overflow-hidden">
          <div className="md:flex">
            <div className="md:w-1/2 p-8">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <Activity className="h-6 w-6 text-indigo-500 mr-2" />
                Upload Your Data
              </h2>
              
              <form onSubmit={handleSubmit}>
                <div className="mb-6">
                  <label className="block text-gray-700 text-sm font-bold mb-2">
                    Select Prediction Model
                  </label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    {models.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </div>
                
                <div className="mb-6">
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                      isDragActive
                        ? 'border-indigo-500 bg-indigo-50'
                        : 'border-gray-300 hover:border-indigo-400 hover:bg-indigo-50'
                    }`}
                  >
                    <input {...getInputProps()} />
                    <FileUp className="h-12 w-12 text-indigo-500 mx-auto mb-4" />
                    {file ? (
                      <div>
                        <p className="text-sm text-gray-600">Selected file:</p>
                        <p className="font-medium text-indigo-600">{file.name}</p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-gray-700 font-medium">
                          Drag & drop your CSV file here, or click to select
                        </p>
                        <p className="text-sm text-gray-500 mt-1">
                          File should contain: Pregnancies, Glucose, BloodPressure, SkinThickness, 
                          Insulin, BMI, DiabetesPedigreeFunction, Age
                        </p>
                      </div>
                    )}
                  </div>
                </div>
                
                <button
                  type="submit"
                  disabled={loading || !file}
                  className={`w-full py-2 px-4 rounded-md text-white font-medium flex items-center justify-center ${
                    loading || !file
                      ? 'bg-indigo-300 cursor-not-allowed'
                      : 'bg-indigo-600 hover:bg-indigo-700'
                  }`}
                >
                  {loading ? (
                    <>
                      <Loader2 className="animate-spin h-5 w-5 mr-2" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Brain className="h-5 w-5 mr-2" />
                      Predict Diabetes
                    </>
                  )}
                </button>
                
                {error && (
                  <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md flex items-start">
                    <AlertCircle className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" />
                    <span>{error}</span>
                  </div>
                )}
              </form>
            </div>
            
            <div className="md:w-1/2 bg-indigo-50 p-8">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <BarChart4 className="h-6 w-6 text-indigo-500 mr-2" />
                Prediction Results
              </h2>
              
              {result ? (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center mb-4">
                    <CheckCircle2 className="h-6 w-6 text-green-500 mr-2" />
                    <h3 className="text-xl font-medium text-gray-800">Analysis Complete</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-gray-500">MODEL USED</h4>
                      <p className="text-lg font-medium text-indigo-600">
                        {models.find(m => m.value === model)?.label}
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-500">ACCURACY</h4>
                      <p className="text-lg font-medium text-indigo-600">
                        {result.accuracy ? `${(result.accuracy * 100).toFixed(2)}%` : 'N/A'}
                      </p>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-500">PREDICTIONS</h4>
                      <div className="mt-2 max-h-60 overflow-y-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Row
                              </th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Prediction
                              </th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Probability
                              </th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {result.predictions && result.predictions.map((pred: any, idx: number) => (
                              <tr key={idx}>
                                <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                  {idx + 1}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap">
                                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                                    pred.prediction === 1 
                                      ? 'bg-red-100 text-red-800' 
                                      : 'bg-green-100 text-green-800'
                                  }`}>
                                    {pred.prediction === 1 ? 'Diabetic' : 'Non-Diabetic'}
                                  </span>
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                  {pred.probability ? `${(pred.probability * 100).toFixed(2)}%` : 'N/A'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
                  <Brain className="h-16 w-16 text-indigo-200 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-500 mb-1">No Results Yet</h3>
                  <p className="text-gray-400">
                    Upload a CSV file and select a model to see prediction results
                  </p>
                </div>
              )}
              
              <div className="mt-6 bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-medium text-gray-800 mb-3">About the Models</h3>
                <p className="text-sm text-gray-600 mb-4">
                  This platform uses various machine learning algorithms to predict diabetes based on 
                  patient data including pregnancies, glucose level, blood pressure, skin thickness, 
                  insulin, BMI, diabetes pedigree function, and age.
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-indigo-50 p-2 rounded">
                    <span className="font-medium">DT:</span> Decision Tree
                  </div>
                  <div className="bg-indigo-50 p-2 rounded">
                    <span className="font-medium">KNN:</span> K-Nearest Neighbors
                  </div>
                  <div className="bg-indigo-50 p-2 rounded">
                    <span className="font-medium">RF:</span> Random Forest
                  </div>
                  <div className="bg-indigo-50 p-2 rounded">
                    <span className="font-medium">NB:</span> Naive Bayes
                  </div>
                  <div className="bg-indigo-50 p-2 rounded">
                    <span className="font-medium">NN:</span> Neural Networks
                  </div>
                  <div className="bg-indigo-50 p-2 rounded">
                    <span className="font-medium">SVM:</span> Support Vector Machine
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;