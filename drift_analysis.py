import warnings
warnings.filterwarnings('ignore')

class DataDriftAnalyzer:
    def __init__(self, reference_data=None, threshold=0.05):
        self.threshold = threshold
        self.reference_stats = None
        self.reference_data = reference_data
        
        if reference_data is not None:
            self._calculate_reference_stats()
    
    def _calculate_reference_stats(self):
        if self.reference_data is None:
            return
        
        numerical_cols = ['Temperature', 'Rainfall', 'Days_to_Harvest']
        categorical_cols = ['Region', 'Soil_Type', 'Crop_Type', 'Weather_Condition']
        
        self.reference_stats = {'numerical': {}, 'categorical': {}}
        
        for col in numerical_cols:
            if col in self.reference_data.columns:
                data = self.reference_data[col].dropna()
                self.reference_stats['numerical'][col] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max()
                }
        
        for col in categorical_cols:
            if col in self.reference_data.columns:
                self.reference_stats['categorical'][col] = {
                    'value_counts': self.reference_data[col].value_counts(normalize=True).to_dict(),
                    'unique_values': self.reference_data[col].unique().tolist()
                }
    
    def check_drift(self, input_data):
        if self.reference_stats is None:
            return {
                'drift_detected': False,
                'message': '✅ No drift detected',
                'details': {'warnings_count': 0}
            }
        
        drift_results = {
            'drift_detected': False,
            'message': '✅ No drift detected',
            'details': {'warnings_count': 0},
            'warnings': []
        }
        
        numerical_mapping = {
            'temperature': 'Temperature',
            'rainfall': 'Rainfall',
            'days': 'Days_to_Harvest'
        }
        
        for input_key, ref_col in numerical_mapping.items():
            if ref_col in self.reference_stats['numerical'] and input_key in input_data:
                ref_stats = self.reference_stats['numerical'][ref_col]
                value = float(input_data[input_key])
                
                lower_bound = ref_stats['mean'] - 3 * ref_stats['std']
                upper_bound = ref_stats['mean'] + 3 * ref_stats['std']
                
                if value < lower_bound or value > upper_bound:
                    drift_results['drift_detected'] = True
                    drift_results['message'] = '⚠️ Drift detected'
                    drift_results['warnings'].append(f"{ref_col}: {value:.2f} outside range")
        
        drift_results['details']['warnings_count'] = len(drift_results['warnings'])
        return drift_results