import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, Activity, Shield, AlertTriangle, CheckCircle,
  BarChart, Target, Info, ChevronRight, Eye, Clock,
  FileText, Layers, TrendingUp, AlertCircle
} from 'lucide-react';

export default function ComprehensiveAnalysis({ result }) {
  const [activeView, setActiveView] = useState('overview');
  const [showDetails, setShowDetails] = useState({
    findings: true,
    reasoning: true,
    validation: true,
    medical: true
  });
  
  // Error checking
  if (!result) {
    return (
      <div className="text-center py-8">
        <p className="text-slate-600 dark:text-slate-400">No results available</p>
      </div>
    );
  }

  // Ensure all required data exists with fallbacks
  const summary = result.summary || {};
  const visualizations = result.visualizations || {};
  const comprehensiveAnalysis = result.comprehensive_analysis || {};
  const validation = result.validation || {};
  const medicalInterpretation = result.medical_interpretation || {};
  const analysisMetadata = result.analysis_metadata || {};
  
  // Safely access nested properties
  const aiGeneratedFindings = comprehensiveAnalysis.ai_generated_findings || {};
  const aiReasoningContext = comprehensiveAnalysis.ai_reasoning_clinical_context || {};
  const validationReliability = comprehensiveAnalysis.validation_reliability || {};
  const regionCharacteristics = comprehensiveAnalysis.region_characteristics || {};

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Title Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold mb-2">
          AI Brain Tumor Detection - Complete Explainable Analysis
        </h1>
        <div className="w-24 h-1 bg-gradient-to-r from-indigo-500 to-purple-600 mx-auto" />
      </motion.div>

      {/* Summary Cards */}
      <div className="grid md:grid-cols-3 gap-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="card-pro p-6 text-center"
        >
          <Brain className="w-12 h-12 mx-auto mb-3 text-indigo-600" />
          <h2 className="text-2xl font-bold">
            {summary.diagnosis || result.prediction?.label?.toUpperCase() || 'UNKNOWN'}
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">DIAGNOSIS</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="card-pro p-6 text-center"
        >
          <BarChart className="w-12 h-12 mx-auto mb-3 text-purple-600" />
          <h2 className="text-2xl font-bold">
            {summary.confidence || (result.prediction?.probability ? 
              `${(result.prediction.probability * 100).toFixed(1)}%` : 'N/A')}
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">CONFIDENCE</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="card-pro p-6 text-center"
        >
          <Shield className="w-12 h-12 mx-auto mb-3 text-green-600" />
          <h2 className="text-2xl font-bold">
            {summary.validation || (validation.is_valid ? 'PASSED' : 'PENDING')}
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">VALIDATION</p>
        </motion.div>
      </div>

      {/* Visualization Tabs */}
      <div className="card-pro p-6">
        <div className="flex flex-wrap gap-2 mb-6">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'sensitivity', label: 'AI Focus Areas' },
            { id: 'regions', label: 'Analyzed Regions' },
            { id: 'gradcam', label: 'Grad-CAM' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveView(tab.id)}
              className={`px-4 py-2 rounded-lg transition ${
                activeView === tab.id 
                  ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400' 
                  : 'hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Main Visualization */}
          <div>
            <AnimatePresence mode="wait">
              <motion.div
                key={activeView}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {activeView === 'overview' && visualizations.original && (
                  <div>
                    <h3 className="font-medium mb-3">Original MRI Scan</h3>
                    <img 
                      src={visualizations.original} 
                      alt="Original MRI"
                      className="w-full rounded-lg shadow-lg"
                    />
                  </div>
                )}
                
                {activeView === 'sensitivity' && visualizations.sensitivity_overlay && (
                  <div>
                    <h3 className="font-medium mb-3">AI Focus Areas (Sensitivity)</h3>
                    <img 
                      src={visualizations.sensitivity_overlay} 
                      alt="Sensitivity heatmap"
                      className="w-full rounded-lg shadow-lg"
                    />
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                      Heatmap shows regions that influenced the AI's decision. 
                      Brighter areas indicate higher importance.
                    </p>
                  </div>
                )}
                
                {activeView === 'regions' && visualizations.regions_marked && (
                  <div>
                    <h3 className="font-medium mb-3">Analyzed Regions</h3>
                    <img 
                      src={visualizations.regions_marked} 
                      alt="Regions marked"
                      className="w-full rounded-lg shadow-lg"
                    />
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                      Numbered regions show the top areas of interest. 
                      Red (1) is most important, followed by green (2) and blue (3).
                    </p>
                  </div>
                )}
                
                {activeView === 'gradcam' && visualizations.gradcam_overlay && (
                  <div>
                    <h3 className="font-medium mb-3">Grad-CAM Visualization</h3>
                    <img 
                      src={visualizations.gradcam_overlay} 
                      alt="Grad-CAM"
                      className="w-full rounded-lg shadow-lg"
                    />
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                      Grad-CAM highlights the most important features for classification.
                    </p>
                  </div>
                )}

                {/* Fallback if no visualization available */}
                {((activeView === 'overview' && !visualizations.original) ||
                  (activeView === 'sensitivity' && !visualizations.sensitivity_overlay) ||
                  (activeView === 'regions' && !visualizations.regions_marked) ||
                  (activeView === 'gradcam' && !visualizations.gradcam_overlay)) && (
                  <div className="flex items-center justify-center h-64 bg-slate-100 dark:bg-slate-800 rounded-lg">
                    <p className="text-slate-600 dark:text-slate-400">
                      Visualization not available
                    </p>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* AI Predictions Chart */}
          <div>
            <h3 className="font-medium mb-3">AI Predictions</h3>
            <div className="space-y-4">
              {result.all_predictions ? (
                Object.entries(result.all_predictions).map(([label, prob], idx) => {
                  const isHighest = label === result.prediction?.label;
                  return (
                    <div key={label}>
                      <div className="flex justify-between mb-1">
                        <span className={`text-sm font-medium ${isHighest ? 'text-indigo-600 dark:text-indigo-400' : ''}`}>
                          {label.charAt(0).toUpperCase() + label.slice(1).replace('_', ' ')}
                        </span>
                        <span className={`text-sm ${isHighest ? 'font-bold' : ''}`}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${prob * 100}%` }}
                          transition={{ duration: 0.5, delay: idx * 0.1 }}
                          className={`h-full rounded-full ${
                            isHighest 
                              ? 'bg-gradient-to-r from-indigo-500 to-purple-600' 
                              : 'bg-gray-400 dark:bg-gray-500'
                          }`}
                        />
                      </div>
                    </div>
                  );
                })
              ) : (
                <p className="text-slate-600 dark:text-slate-400">No prediction data available</p>
              )}
            </div>

            {/* Region Statistics */}
            {regionCharacteristics.total_regions !== undefined && (
              <div className="mt-6 p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
                <h4 className="font-medium mb-2">Region Statistics</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>Total Regions: {regionCharacteristics.total_regions || 0}</div>
                  <div>Affected Area: {aiReasoningContext.total_affected_area || 'N/A'}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* AI-Generated Findings */}
      <div className="card-pro p-6">
        <div 
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setShowDetails({...showDetails, findings: !showDetails.findings})}
        >
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Target className="w-5 h-5" />
            🔍 AI-GENERATED FINDINGS
          </h3>
          <ChevronRight className={`w-5 h-5 transition-transform ${showDetails.findings ? 'rotate-90' : ''}`} />
        </div>
        
        <AnimatePresence>
          {showDetails.findings && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="mt-4 space-y-4"
            >
              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-950/20">
                  <h4 className="font-medium mb-2">Diagnosis</h4>
                  <p className="text-2xl font-bold">
                    {aiGeneratedFindings.diagnosis || result.prediction?.label?.toUpperCase() || 'UNKNOWN'}
                  </p>
                  <p className="text-sm mt-1">
                    Confidence: {aiGeneratedFindings.confidence || 
                      (result.prediction?.probability ? `${(result.prediction.probability * 100).toFixed(1)}%` : 'N/A')}
                  </p>
                </div>
                
                <div className="p-4 rounded-xl bg-purple-50 dark:bg-purple-950/20">
                  <h4 className="font-medium mb-2">Pattern Analysis</h4>
                  <p className="text-sm">
                    {aiGeneratedFindings.pattern_analysis || 
                     result.interpretation?.pattern_description || 
                     'Pattern analysis not available'}
                  </p>
                </div>
              </div>
              
              <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-950/20">
                <h4 className="font-medium mb-2">Spatial Distribution</h4>
                <p className="text-sm">
                  {aiGeneratedFindings.spatial_distribution || 
                   result.interpretation?.location_analysis ||
                   'Spatial analysis not available'}
                </p>
              </div>
              
              {(aiGeneratedFindings.detailed_regional_analysis || 
                result.interpretation?.main_findings) && (
                <div className="p-4 rounded-xl bg-indigo-50 dark:bg-indigo-950/20">
                  <h4 className="font-medium mb-3">Detailed Regional Analysis</h4>
                  <ol className="space-y-2">
                    {(aiGeneratedFindings.detailed_regional_analysis || 
                      result.interpretation?.main_findings || []).map((finding, idx) => (
                      <li key={idx} className="text-sm flex items-start gap-2">
                        <span className="font-bold text-indigo-600">{idx + 1}.</span>
                        <span>{finding}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* AI Reasoning & Clinical Context */}
      <div className="card-pro p-6">
        <div 
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setShowDetails({...showDetails, reasoning: !showDetails.reasoning})}
        >
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5" />
            🧠 AI REASONING & CLINICAL CONTEXT
          </h3>
          <ChevronRight className={`w-5 h-5 transition-transform ${showDetails.reasoning ? 'rotate-90' : ''}`} />
        </div>
        
        <AnimatePresence>
          {showDetails.reasoning && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="mt-4 space-y-4"
            >
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
                <p className="text-sm">
                  {aiReasoningContext.confidence_interpretation || 
                   result.interpretation?.confidence_interpretation ||
                   'Confidence interpretation not available'}
                </p>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-3">Pattern Characteristics</h4>
                  {(aiReasoningContext.pattern_characteristics || 
                    result.interpretation?.characteristics || []).length > 0 ? (
                    <ul className="space-y-2">
                      {(aiReasoningContext.pattern_characteristics || 
                        result.interpretation?.characteristics || []).map((char, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm">
                          <div className="w-2 h-2 rounded-full bg-indigo-500 mt-1.5" />
                          <span>{char}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-600 dark:text-slate-400">No specific patterns identified</p>
                  )}
                </div>
                
                <div>
                  <h4 className="font-medium mb-3">Clinical Relevance</h4>
                  {(aiReasoningContext.clinical_relevance || 
                    result.interpretation?.clinical_relevance || []).length > 0 ? (
                    <ul className="space-y-2">
                      {(aiReasoningContext.clinical_relevance || 
                        result.interpretation?.clinical_relevance || []).map((rel, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm">
                          <div className="w-2 h-2 rounded-full bg-purple-500 mt-1.5" />
                          <span>{rel}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-600 dark:text-slate-400">No specific clinical relevance noted</p>
                  )}
                </div>
              </div>
              
              <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-900/50">
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Decision Basis:</span>
                    <p className="mt-1">{aiReasoningContext.decision_basis || 'Not specified'}</p>
                  </div>
                  <div>
                    <span className="font-medium">Total Affected Area:</span>
                    <p className="mt-1">{aiReasoningContext.total_affected_area || 'Not calculated'}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Regional Analysis Table */}
      {result.regional_analysis && result.regional_analysis.length > 0 && (
        <div className="card-pro p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Detailed Regional Analysis
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="text-left py-3 px-4">Region</th>
                  <th className="text-left py-3 px-4">Size (pixels)</th>
                  <th className="text-left py-3 px-4">Size %</th>
                  <th className="text-left py-3 px-4">Location</th>
                  <th className="text-left py-3 px-4">Importance</th>
                  <th className="text-left py-3 px-4">Eccentricity</th>
                  <th className="text-left py-3 px-4">Solidity</th>
                </tr>
              </thead>
              <tbody>
                {result.regional_analysis.map((region) => (
                  <tr key={region.region_number} className="border-b border-slate-100 dark:border-slate-800">
                    <td className="py-3 px-4 font-medium">Region {region.region_number}</td>
                    <td className="py-3 px-4">{region.size_pixels}</td>
                    <td className="py-3 px-4">{region.size_percentage}</td>
                    <td className="py-3 px-4 font-mono text-xs">{region.location}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div 
                            className="h-full bg-gradient-to-r from-amber-500 to-red-500 rounded-full"
                            style={{ width: `${parseFloat(region.importance) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs">{region.importance}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">{region.eccentricity}</td>
                    <td className="py-3 px-4">{region.solidity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Validation & Reliability */}
      <div className="card-pro p-6">
        <div 
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setShowDetails({...showDetails, validation: !showDetails.validation})}
        >
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Shield className="w-5 h-5" />
            ✅ VALIDATION & RELIABILITY
          </h3>
          <ChevronRight className={`w-5 h-5 transition-transform ${showDetails.validation ? 'rotate-90' : ''}`} />
        </div>
        
        <AnimatePresence>
          {showDetails.validation && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="mt-4 space-y-4"
            >
              {validation.message && (
                <div className={`p-4 rounded-xl ${
                  validation.is_valid 
                    ? 'bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-900' 
                    : 'bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-900'
                }`}>
                  <div className="flex items-start gap-3">
                    {validation.is_valid ? (
                      <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                    )}
                    <div className="flex-1">
                      <h4 className="font-medium mb-2">{validation.message}</h4>
                      {validation.original_confidence !== undefined && (
                        <div className="grid md:grid-cols-3 gap-4 mt-3 text-sm">
                          <div>
                            <p className="text-slate-600 dark:text-slate-400">Original confidence</p>
                            <p className="font-medium">{(validation.original_confidence * 100).toFixed(1)}%</p>
                          </div>
                          <div>
                            <p className="text-slate-600 dark:text-slate-400">After occlusion</p>
                            <p className="font-medium">{(validation.occluded_confidence * 100).toFixed(1)}%</p>
                          </div>
                          <div>
                            <p className="text-slate-600 dark:text-slate-400">Confidence drop</p>
                            <p className="font-medium">{(validation.confidence_drop * 100).toFixed(1)}%</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
                  <h4 className="font-medium mb-2">Analysis Quality Metrics</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Regions detected:</span>
                      <span className="font-medium">
                        {validationReliability.regions_detected || summary.detected_regions || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Average importance:</span>
                      <span className="font-medium">
                        {validationReliability.avg_importance || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Sensitivity validated:</span>
                      <span className="font-medium">
                        {validationReliability.sensitivity_validated !== undefined ? 
                          (validationReliability.sensitivity_validated ? '✓ Yes' : '✗ No') : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
                  <h4 className="font-medium mb-2">Analysis Performance</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Analysis time:</span>
                      <span className="font-medium">{summary.analysis_time || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Methods used:</span>
                      <span className="font-medium">{analysisMetadata.methods_used?.length || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Model type:</span>
                      <span className="font-medium">{analysisMetadata.model_type || 'Unknown'}</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Medical Interpretation */}
      <div className="card-pro p-6">
        <div 
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setShowDetails({...showDetails, medical: !showDetails.medical})}
        >
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <FileText className="w-5 h-5" />
            ⚕️ MEDICAL INTERPRETATION
          </h3>
          <ChevronRight className={`w-5 h-5 transition-transform ${showDetails.medical ? 'rotate-90' : ''}`} />
        </div>
        
        <AnimatePresence>
          {showDetails.medical && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="mt-4 space-y-4"
            >
              {medicalInterpretation.assessment && (
                <div className={`p-4 rounded-xl ${
                  result.prediction?.label === 'no_tumor' 
                    ? 'bg-green-50 dark:bg-green-950/20' 
                    : 'bg-amber-50 dark:bg-amber-950/20'
                }`}>
                  <h4 className="font-medium text-lg mb-3">{medicalInterpretation.assessment}</h4>
                  {medicalInterpretation.observations && medicalInterpretation.observations.length > 0 && (
                    <ul className="space-y-1">
                      {medicalInterpretation.observations.map((obs, idx) => (
                        <li key={idx} className="text-sm">{obs}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
              
              {medicalInterpretation.recommendations && medicalInterpretation.recommendations.length > 0 && (
                <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-950/20">
                  <h4 className="font-medium mb-3">⚕️ RECOMMENDED ACTION:</h4>
                  <ul className="space-y-2">
                    {medicalInterpretation.recommendations.map((rec, idx) => (
                      <li key={idx} className="text-sm flex items-start gap-2">
                        {rec.includes('✓') ? (
                          <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5" />
                        )}
                        <span>{rec.replace(/[✓⚠]/g, '').trim()}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5" />
                  <div className="text-sm text-red-700 dark:text-red-300">
                    <p className="font-medium mb-1">⚠️ IMPORTANT DISCLAIMER:</p>
                    <p>{analysisMetadata.disclaimer || 
                       "This AI analysis is for screening purposes only. Clinical correlation and professional medical evaluation are essential for diagnosis."}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="text-center text-sm text-slate-600 dark:text-slate-400 py-8 border-t border-slate-200 dark:border-slate-800">
        <p className="mb-1">This analysis was generated dynamically based on AI model behavior.</p>
        <p>No predetermined medical knowledge was used in the interpretation.</p>
      </div>
    </div>
  );
}