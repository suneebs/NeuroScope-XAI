import React from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, Activity, Shield, AlertTriangle, CheckCircle,
  BarChart, Target, Info, ChevronRight, Eye, Clock
} from 'lucide-react';

export default function ComprehensiveResults({ result }) {
  const [activeView, setActiveView] = useState('overview');
  
  if (!result) return null;

  return (
    <div className="space-y-8">
      {/* Header Summary Card */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card-pro p-6"
      >
        <h2 className="text-2xl font-bold mb-4">
          AI BRAIN TUMOR DETECTION - COMPREHENSIVE ANALYSIS SUMMARY
        </h2>
        
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="text-center p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
            <Brain className="w-8 h-8 mx-auto mb-2 text-indigo-600" />
            <h3 className="text-2xl font-bold">{result.summary.diagnosis}</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400">DIAGNOSIS</p>
          </div>
          
          <div className="text-center p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
            <BarChart className="w-8 h-8 mx-auto mb-2 text-purple-600" />
            <h3 className="text-2xl font-bold">{result.summary.confidence}</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400">CONFIDENCE</p>
          </div>
          
          <div className="text-center p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
            <Shield className="w-8 h-8 mx-auto mb-2 text-green-600" />
            <h3 className="text-2xl font-bold">{result.summary.validation}</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400">VALIDATION</p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
          <Clock className="w-4 h-4" />
          Analysis completed in {result.summary.analysis_time}
        </div>
      </motion.div>

      {/* Visualization Tabs */}
      <div className="card-pro p-6">
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveView('overview')}
            className={`px-4 py-2 rounded-lg ${activeView === 'overview' ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveView('heatmap')}
            className={`px-4 py-2 rounded-lg ${activeView === 'heatmap' ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}
          >
            AI Focus Areas
          </button>
          <button
            onClick={() => setActiveView('regions')}
            className={`px-4 py-2 rounded-lg ${activeView === 'regions' ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}
          >
            Analyzed Regions
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Visualizations */}
          <div>
            {activeView === 'overview' && (
              <div>
                <h3 className="font-medium mb-3">Original MRI with Overlay</h3>
                <img 
                  src={result.visualizations.heatmap_overlay} 
                  alt="MRI with heatmap overlay"
                  className="w-full rounded-lg"
                />
              </div>
            )}
            
            {activeView === 'heatmap' && (
              <div>
                <h3 className="font-medium mb-3">AI Sensitivity Map</h3>
                <img 
                  src={result.visualizations.sensitivity_map} 
                  alt="Sensitivity heatmap"
                  className="w-full rounded-lg"
                />
              </div>
            )}
            
            {activeView === 'regions' && (
              <div>
                <h3 className="font-medium mb-3">Detected Regions Analysis</h3>
                <img 
                  src={result.visualizations.regions_marked} 
                  alt="Regions marked"
                  className="w-full rounded-lg"
                />
              </div>
            )}
          </div>

          {/* Predictions Chart */}
          <div>
            <h3 className="font-medium mb-3">AI Predictions</h3>
            <div className="space-y-3">
              {Object.entries(result.all_predictions).map(([label, prob]) => (
                <div key={label}>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">{label}</span>
                    <span className="text-sm">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        label === result.prediction.label 
                          ? 'bg-gradient-to-r from-indigo-500 to-purple-600' 
                          : 'bg-gray-400'
                      }`}
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* AI Generated Findings */}
      <div className="card-pro p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Target className="w-5 h-5" />
          AI-GENERATED FINDINGS
        </h3>
        
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-950/20">
            <h4 className="font-medium mb-2">Pattern Analysis</h4>
            <p className="text-sm">{result.comprehensive_analysis.ai_generated_findings.pattern_analysis}</p>
          </div>
          
          <div className="p-4 rounded-xl bg-purple-50 dark:bg-purple-950/20">
            <h4 className="font-medium mb-2">Spatial Distribution</h4>
            <p className="text-sm">{result.comprehensive_analysis.ai_generated_findings.spatial_distribution}</p>
          </div>
          
          <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-950/20">
            <h4 className="font-medium mb-2">Detailed Regional Analysis</h4>
            <ol className="space-y-2">
              {result.comprehensive_analysis.ai_generated_findings.detailed_regional_analysis.map((finding, idx) => (
                <li key={idx} className="text-sm flex items-start gap-2">
                  <span className="font-medium">{idx + 1}.</span>
                  <span>{finding}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </div>

      {/* AI Reasoning & Clinical Context */}
      <div className="card-pro p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5" />
          AI REASONING & CLINICAL CONTEXT
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">Pattern Characteristics</h4>
            <ul className="space-y-2">
              {result.comprehensive_analysis.ai_reasoning_clinical_context.pattern_characteristics.map((char, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <ChevronRight className="w-4 h-4 text-indigo-500 mt-0.5" />
                  <span>{char}</span>
                </li>
              ))}
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">Clinical Relevance</h4>
            <ul className="space-y-2">
              {result.comprehensive_analysis.ai_reasoning_clinical_context.clinical_relevance.map((rel, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <ChevronRight className="w-4 h-4 text-purple-500 mt-0.5" />
                  <span>{rel}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        
        <div className="mt-4 p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
          <p className="text-sm">
            <strong>Decision Basis:</strong> {result.comprehensive_analysis.ai_reasoning_clinical_context.decision_basis}
          </p>
          <p className="text-sm mt-2">
            <strong>Total Affected Area:</strong> {result.comprehensive_analysis.ai_reasoning_clinical_context.total_affected_area}
          </p>
        </div>
      </div>

      {/* Regional Analysis Table */}
      {result.regional_analysis && result.regional_analysis.length > 0 && (
        <div className="card-pro p-6">
          <h3 className="text-lg font-semibold mb-4">Detailed Regional Analysis</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="text-left py-2">Region</th>
                  <th className="text-left py-2">Size (pixels)</th>
                  <th className="text-left py-2">Size %</th>
                  <th className="text-left py-2">Location</th>
                  <th className="text-left py-2">Importance</th>
                  <th className="text-left py-2">Shape</th>
                </tr>
              </thead>
              <tbody>
                {result.regional_analysis.map((region) => (
                  <tr key={region.region_number} className="border-b border-slate-100 dark:border-slate-800">
                    <td className="py-2">Region {region.region_number}</td>
                    <td className="py-2">{region.size_pixels}</td>
                    <td className="py-2">{region.size_percentage}</td>
                    <td className="py-2">{region.location}</td>
                    <td className="py-2">{region.importance}</td>
                    <td className="py-2">
                      Ecc: {region.eccentricity}, Sol: {region.solidity}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Medical Interpretation */}
      <div className="card-pro p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          MEDICAL INTERPRETATION
        </h3>
        
        <div className={`p-4 rounded-xl mb-4 ${
          result.prediction.label === 'no_tumor' 
            ? 'bg-green-50 dark:bg-green-950/20' 
            : 'bg-amber-50 dark:bg-amber-950/20'
        }`}>
          <h4 className="font-medium mb-2">{result.medical_interpretation.assessment}</h4>
          <ul className="space-y-1">
            {result.medical_interpretation.observations.map((obs, idx) => (
              <li key={idx} className="text-sm">{obs}</li>
            ))}
          </ul>
        </div>
        
        <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-950/20">
          <h4 className="font-medium mb-2">Recommended Actions</h4>
          <ul className="space-y-1">
            {result.medical_interpretation.recommendations.map((rec, idx) => (
              <li key={idx} className="text-sm flex items-start gap-2">
                {result.prediction.label === 'no_tumor' ? (
                  <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                ) : (
                  <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5" />
                )}
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
        
        <div className="mt-4 p-4 rounded-xl bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900">
          <p className="text-sm text-red-700 dark:text-red-300">
            <strong>⚠️ IMPORTANT DISCLAIMER:</strong> This AI analysis is for screening purposes only. 
            Clinical correlation and professional medical evaluation are essential for diagnosis.
          </p>
        </div>
      </div>

      {/* Validation & Reliability */}
      <div className="card-pro p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5" />
          VALIDATION & RELIABILITY
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className={`p-4 rounded-xl ${
            result.validation.is_valid 
              ? 'bg-green-50 dark:bg-green-950/20' 
              : 'bg-yellow-50 dark:bg-yellow-950/20'
          }`}>
            <h4 className="font-medium mb-2">Validation Status</h4>
            <p className="text-sm">{result.validation.message}</p>
            <div className="mt-2 text-sm space-y-1">
              <p>Original confidence: {(result.validation.original_confidence * 100).toFixed(1)}%</p>
              <p>After occlusion: {(result.validation.occluded_confidence * 100).toFixed(1)}%</p>
              <p>Confidence drop: {(result.validation.confidence_drop * 100).toFixed(1)}%</p>
            </div>
          </div>
          
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/50">
            <h4 className="font-medium mb-2">Analysis Quality</h4>
            <div className="text-sm space-y-1">
              <p>Regions detected: {result.comprehensive_analysis.validation_reliability.regions_detected}</p>
              <p>Average importance: {result.comprehensive_analysis.validation_reliability.avg_importance}</p>
              <p>Analysis time: {result.summary.analysis_time}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Note */}
      <div className="text-center text-sm text-slate-600 dark:text-slate-400 py-4">
        <p>This analysis was generated dynamically based on AI model behavior.</p>
        <p>No predetermined medical knowledge was used in the interpretation.</p>
      </div>
    </div>
  );
}