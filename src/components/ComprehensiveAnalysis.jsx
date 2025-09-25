import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, Activity, Shield, AlertTriangle, CheckCircle,
  BarChart, Target, Info, ChevronRight, Eye, Clock,
  FileText, Layers, TrendingUp, AlertCircle, Terminal,
  Download, Maximize2
} from 'lucide-react';

export default function FullComprehensiveAnalysis({ result }) {
  const [expandedSections, setExpandedSections] = useState({
    visualizations: true,
    findings: true,
    reasoning: true,
    validation: true,
    medical: true,
    printSummary: false
  });
  const [fullscreenImage, setFullscreenImage] = useState(null);
  
  if (!result) {
    return (
      <div className="text-center py-8">
        <p className="text-slate-600 dark:text-slate-400">No results available</p>
      </div>
    );
  }

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const openFullscreen = (imageSrc, title) => {
    setFullscreenImage({ src: imageSrc, title });
  };

  // Safe data access
  const visualizations = result.visualizations || {};
  const summary = result.summary || {};
  const validation = result.validation || {};
  const comprehensiveAnalysis = result.comprehensive_analysis || {};
  const medicalInterpretation = result.medical_interpretation || {};

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Title */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold mb-2">
          AI Brain Tumor Detection - Complete Explainable Analysis
        </h1>
        <div className="w-24 h-1 bg-gradient-to-r from-indigo-500 to-purple-600 mx-auto mb-4" />
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Complete 12-panel analysis matching TrueExplainableAI output
        </p>
      </motion.div>

      {/* Main Summary Cards */}
      <div className="grid md:grid-cols-3 gap-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="card-pro p-6 text-center"
        >
          <Brain className="w-12 h-12 mx-auto mb-3 text-indigo-600" />
          <h2 className="text-2xl font-bold">{summary.diagnosis || 'UNKNOWN'}</h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">🧠 DIAGNOSIS</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="card-pro p-6 text-center"
        >
          <BarChart className="w-12 h-12 mx-auto mb-3 text-purple-600" />
          <h2 className="text-2xl font-bold">{summary.confidence || 'N/A'}</h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">📊 CONFIDENCE</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="card-pro p-6 text-center"
        >
          <Shield className="w-12 h-12 mx-auto mb-3 text-green-600" />
          <h2 className="text-2xl font-bold">{summary.validation || 'PENDING'}</h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">✅ VALIDATION</p>
        </motion.div>
      </div>

      {/* Comprehensive Visualizations (12 panels) */}
      <div className="card-pro p-6">
        <div 
          className="flex items-center justify-between cursor-pointer mb-4"
          onClick={() => toggleSection('visualizations')}
        >
          <h3 className="text-lg font-semibold">Comprehensive Visualizations</h3>
          <ChevronRight className={`w-5 h-5 transition-transform ${
            expandedSections.visualizations ? 'rotate-90' : ''
          }`} />
        </div>
        
        <AnimatePresence>
          {expandedSections.visualizations && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="grid md:grid-cols-3 lg:grid-cols-4 gap-4"
            >
              {/* Panel 1: Original Image */}
              {visualizations.original && (
                <div className="relative group">
                  <h4 className="text-sm font-medium mb-2">1. Original MRI Scan</h4>
                  <img 
                    src={visualizations.original} 
                    alt="Original MRI"
                    className="w-full rounded-lg shadow cursor-pointer"
                    onClick={() => openFullscreen(visualizations.original, 'Original MRI Scan')}
                  />
                  <Maximize2 className="absolute top-8 right-2 w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition" />
                </div>
              )}
              
              {/* Panel 2: Sensitivity Overlay */}
              {visualizations.sensitivity_overlay && (
                <div className="relative group">
                  <h4 className="text-sm font-medium mb-2">2. AI Focus Areas</h4>
                  <img 
                    src={visualizations.sensitivity_overlay} 
                    alt="Sensitivity overlay"
                    className="w-full rounded-lg shadow cursor-pointer"
                    onClick={() => openFullscreen(visualizations.sensitivity_overlay, 'AI Focus Areas (Sensitivity)')}
                  />
                  <Maximize2 className="absolute top-8 right-2 w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition" />
                </div>
              )}
              
              {/* Panel 3: Analyzed Regions */}
              {visualizations.regions_marked && (
                <div className="relative group">
                  <h4 className="text-sm font-medium mb-2">3. Analyzed Regions</h4>
                  <img 
                    src={visualizations.regions_marked} 
                    alt="Regions marked"
                    className="w-full rounded-lg shadow cursor-pointer"
                    onClick={() => openFullscreen(visualizations.regions_marked, 'Analyzed Regions')}
                  />
                  <Maximize2 className="absolute top-8 right-2 w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition" />
                </div>
              )}
              
              {/* Panel 4: Predictions Chart */}
              {visualizations.predictions_chart && (
                <div className="relative group">
                  <h4 className="text-sm font-medium mb-2">4. AI Predictions</h4>
                  <img 
                    src={visualizations.predictions_chart} 
                    alt="Predictions chart"
                    className="w-full rounded-lg shadow cursor-pointer"
                    onClick={() => openFullscreen(visualizations.predictions_chart, 'AI Predictions')}
                  />
                  <Maximize2 className="absolute top-8 right-2 w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition" />
                </div>
              )}
              
              {/* Panel 5-6: AI-Generated Findings (Text) */}
              {visualizations.findings_text && (
                <div className="md:col-span-2 p-4 rounded-lg bg-blue-50 dark:bg-blue-950/20">
                  <h4 className="text-sm font-bold mb-2">{visualizations.findings_text.title}</h4>
                  <div className="text-sm space-y-1">
                    <p><strong>Diagnosis:</strong> {visualizations.findings_text.diagnosis}</p>
                    <p><strong>Confidence:</strong> {visualizations.findings_text.confidence}</p>
                    <p><strong>Pattern Analysis:</strong> {visualizations.findings_text.pattern_analysis}</p>
                    <p><strong>Spatial Distribution:</strong> {visualizations.findings_text.spatial_distribution}</p>
                    {visualizations.findings_text.detailed_regional_analysis?.length > 0 && (
                      <div>
                        <strong>Detailed Regional Analysis:</strong>
                        <ol className="ml-4 mt-1">
                          {visualizations.findings_text.detailed_regional_analysis.map((item, idx) => (
                            <li key={idx} className="text-xs">{idx + 1}. {item}</li>
                          ))}
                        </ol>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* Panel 7-8: AI Reasoning (Text) */}
              {visualizations.reasoning_text && (
                <div className="md:col-span-2 p-4 rounded-lg bg-yellow-50 dark:bg-yellow-950/20">
                  <h4 className="text-sm font-bold mb-2">{visualizations.reasoning_text.title}</h4>
                  <div className="text-sm space-y-1">
                    <p className="mb-2">{visualizations.reasoning_text.confidence_interpretation}</p>
                    {visualizations.reasoning_text.pattern_characteristics?.length > 0 && (
                      <div>
                        <strong>Pattern Characteristics:</strong>
                        <ul className="ml-4">
                          {visualizations.reasoning_text.pattern_characteristics.map((char, idx) => (
                            <li key={idx}>• {char}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    <p className="mt-2"><strong>Decision Basis:</strong> {visualizations.reasoning_text.decision_basis}</p>
                    <p><strong>Total Affected Area:</strong> {visualizations.reasoning_text.total_affected_area}</p>
                  </div>
                </div>
              )}
              
              {/* Panel 9: Region Scatter Plot */}
              {visualizations.region_scatter && (
                <div className="relative group">
                  <h4 className="text-sm font-medium mb-2">9. Region Characteristics</h4>
                  <img 
                    src={visualizations.region_scatter} 
                    alt="Region characteristics"
                    className="w-full rounded-lg shadow cursor-pointer"
                    onClick={() => openFullscreen(visualizations.region_scatter, 'Region Characteristics')}
                  />
                  <Maximize2 className="absolute top-8 right-2 w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition" />
                </div>
              )}
              
              {/* Panel 10: Validation Status (Text) */}
              {visualizations.validation_text && (
                <div className={`p-4 rounded-lg ${
                  visualizations.validation_text.is_valid 
                    ? 'bg-green-50 dark:bg-green-950/20' 
                    : 'bg-yellow-50 dark:bg-yellow-950/20'
                }`}>
                  <h4 className="text-sm font-bold mb-2">{visualizations.validation_text.title}</h4>
                  <div className="text-sm space-y-1">
                    {visualizations.validation_text.status_lines?.map((line, idx) => (
                      <p key={idx}>{line}</p>
                    ))}
                    <p>Regions detected: {visualizations.validation_text.regions_detected}</p>
                    <p>Avg importance: {visualizations.validation_text.avg_importance}</p>
                  </div>
                </div>
              )}
              
              {/* Panel 11-12: Medical Interpretation (Text) */}
              {visualizations.medical_text && (
                <div className="md:col-span-2 p-4 rounded-lg bg-amber-50 dark:bg-amber-950/20">
                  <h4 className="text-sm font-bold mb-2">{visualizations.medical_text.title}</h4>
                  <div className="text-sm space-y-2">
                    <p className="font-medium">{visualizations.medical_text.assessment}</p>
                    {visualizations.medical_text.observations?.length > 0 && (
                      <div>
                        {visualizations.medical_text.observations.map((obs, idx) => (
                          <p key={idx}>{obs}</p>
                        ))}
                      </div>
                    )}
                    {visualizations.medical_text.recommendations?.length > 0 && (
                      <div className="mt-2">
                        <strong>Recommendations:</strong>
                        {visualizations.medical_text.recommendations.map((rec, idx) => (
                          <p key={idx}>{rec}</p>
                        ))}
                      </div>
                    )}
                    {visualizations.medical_text.disclaimer?.length > 0 && (
                      <div className="mt-2 text-xs text-red-700 dark:text-red-400">
                        {visualizations.medical_text.disclaimer.map((line, idx) => (
                          <p key={idx}>{line}</p>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Print Summary (Terminal Style) */}
      {result.print_summary && (
        <div className="card-pro p-6">
          <div 
            className="flex items-center justify-between cursor-pointer mb-4"
            onClick={() => toggleSection('printSummary')}
          >
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Terminal className="w-5 h-5" />
              Complete Analysis Summary (Terminal Output)
            </h3>
            <ChevronRight className={`w-5 h-5 transition-transform ${
              expandedSections.printSummary ? 'rotate-90' : ''
            }`} />
          </div>
          
          <AnimatePresence>
            {expandedSections.printSummary && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="bg-slate-900 text-green-400 p-4 rounded-lg font-mono text-xs overflow-x-auto"
              >
                <pre className="whitespace-pre-wrap">
                  {result.print_summary.join('\n')}
                </pre>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Detailed Sections */}
      
      {/* Pattern Analysis */}
      <div className="card-pro p-6">
        <h3 className="text-lg font-semibold mb-4">🔍 PATTERN ANALYSIS</h3>
        <div className="space-y-2 text-sm">
          <p>{summary.pattern_analysis || 'Not available'}</p>
          <p>{summary.location_analysis || 'Not available'}</p>
        </div>
      </div>

      {/* Detected Regions Table */}
      {result.regions && result.regions.length > 0 && (
        <div className="card-pro p-6">
          <h3 className="text-lg font-semibold mb-4">📍 DETECTED REGIONS: {result.regions.length}</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="text-left py-2">Region</th>
                  <th className="text-left py-2">Area</th>
                  <th className="text-left py-2">Location</th>
                  <th className="text-left py-2">Importance</th>
                  <th className="text-left py-2">Shape</th>
                </tr>
              </thead>
              <tbody>
                {result.regions.slice(0, 10).map((region, idx) => {
                  const area_pct = (region.area / (224 * 224)) * 100;
                  return (
                    <tr key={idx} className="border-b border-slate-100 dark:border-slate-800">
                      <td className="py-2">{idx + 1}</td>
                      <td className="py-2">{region.area} ({area_pct.toFixed(1)}%)</td>
                      <td className="py-2">({region.centroid[1].toFixed(0)}, {region.centroid[0].toFixed(0)})</td>
                      <td className="py-2">{region.mean_intensity.toFixed(2)}</td>
                      <td className="py-2">Ecc: {region.eccentricity.toFixed(2)}, Sol: {region.solidity.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Key Characteristics & Clinical Relevance */}
      <div className="grid md:grid-cols-2 gap-6">
        {summary.key_characteristics?.length > 0 && (
          <div className="card-pro p-6">
            <h3 className="text-lg font-semibold mb-4">🎯 KEY CHARACTERISTICS</h3>
            <ul className="space-y-2">
              {summary.key_characteristics.map((char, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <span className="text-indigo-500">•</span>
                  <span>{char}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        
        {summary.clinical_relevance?.length > 0 && (
          <div className="card-pro p-6">
            <h3 className="text-lg font-semibold mb-4">🏥 CLINICAL RELEVANCE</h3>
            <ul className="space-y-2">
              {summary.clinical_relevance.map((rel, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <span className="text-purple-500">•</span>
                  <span>{rel}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* AI Reasoning */}
      <div className="card-pro p-6">
        <h3 className="text-lg font-semibold mb-4">💡 AI REASONING</h3>
        <p className="text-sm">{summary.ai_reasoning || 'Not available'}</p>
      </div>

      {/* Recommended Actions */}
      {medicalInterpretation.recommendations?.length > 0 && (
        <div className="card-pro p-6">
          <h3 className="text-lg font-semibold mb-4">⚕️ RECOMMENDED ACTION</h3>
          <ul className="space-y-2">
            {medicalInterpretation.recommendations.map((rec, idx) => (
              <li key={idx} className="flex items-start gap-2 text-sm">
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

      {/* Footer */}
      <div className="text-center text-sm text-slate-600 dark:text-slate-400 py-8 border-t border-slate-200 dark:border-slate-800">
        <p className="mb-1">This analysis was generated dynamically based on AI model behavior.</p>
        <p>No predetermined medical knowledge was used in the interpretation.</p>
        <p className="mt-2 text-xs">Analysis completed in {summary.analysis_time}</p>
      </div>

      {/* Fullscreen Image Modal */}
      <AnimatePresence>
        {fullscreenImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
            onClick={() => setFullscreenImage(null)}
          >
            <div className="relative max-w-4xl max-h-[90vh]">
              <h3 className="text-white text-lg mb-4">{fullscreenImage.title}</h3>
              <img 
                src={fullscreenImage.src} 
                alt={fullscreenImage.title}
                className="max-w-full max-h-[80vh] object-contain rounded-lg"
              />
              <button
                className="absolute top-0 right-0 text-white p-2"
                onClick={() => setFullscreenImage(null)}
              >
                ✕
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}