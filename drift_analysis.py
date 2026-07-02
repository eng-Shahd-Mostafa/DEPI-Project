# drift_analysis.py
# ============================================
# 📊 DRIFT ANALYSIS - Standalone Script
# ============================================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('data/crop_yield.csv')

print("=" * 60)
print("📊 DRIFT ANALYSIS REPORT")
print("=" * 60)

# Check if Year column exists
if 'Year' in data.columns:
    print(f"\n✅ Year column found. Analyzing drift between early and late periods...")
    
    # Split data into two time periods
    half = len(data) // 2
    early_data = data.iloc[:half]
    late_data = data.iloc[half:]
    
    print(f"\n📅 Early Period: {early_data['Year'].min()} - {early_data['Year'].max()}")
    print(f"📅 Late Period:  {late_data['Year'].min()} - {late_data['Year'].max()}")
    
    # Numerical columns to analyze
    numeric_cols = ['Temperature_Celsius', 'Rainfall_mm', 'Days_to_Harvest', 'Yield_tons_per_hectare']
    
    print("\n" + "=" * 60)
    print("📊 DRIFT ANALYSIS RESULTS")
    print("=" * 60)
    
    drift_results = []
    
    for col in numeric_cols:
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(early_data[col], late_data[col])
        
        drift_detected = p_value < 0.05
        drift_status = "⚠️ DRIFT DETECTED!" if drift_detected else "✅ No significant drift"
        
        print(f"\n🔍 {col}:")
        print(f"   Early Mean: {early_data[col].mean():.3f}")
        print(f"   Late Mean:  {late_data[col].mean():.3f}")
        print(f"   Change:     {((late_data[col].mean() - early_data[col].mean()) / early_data[col].mean() * 100):.2f}%")
        print(f"   KS Test p-value: {p_value:.4f}")
        print(f"   {drift_status}")
        
        drift_results.append({
            'Feature': col,
            'Early_Mean': early_data[col].mean(),
            'Late_Mean': late_data[col].mean(),
            'P_Value': p_value,
            'Drift_Detected': drift_detected
        })
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(early_data[col], alpha=0.5, label='Early', color='#667eea', bins=30, edgecolor='black')
        axes[0].hist(late_data[col], alpha=0.5, label='Late', color='#f093fb', bins=30, edgecolor='black')
        axes[0].set_title(f'{col} - Distribution Comparison')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Boxplot
        box_data = [early_data[col], late_data[col]]
        axes[1].boxplot(box_data, labels=['Early', 'Late'], patch_artist=True)
        axes[1].set_title(f'{col} - Boxplot Comparison')
        axes[1].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig(f'graphs/drift_analysis_{col}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Summary Table
    print("\n" + "=" * 60)
    print("📊 DRIFT ANALYSIS SUMMARY")
    print("=" * 60)
    
    summary_df = pd.DataFrame(drift_results)
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv('graphs/drift_analysis_summary.csv', index=False)
    print("\n✅ Summary saved to 'graphs/drift_analysis_summary.csv'")
    
else:
    print("\n❌ No 'Year' column found in dataset.")
    print("💡 To perform drift analysis, please add a 'Year' column or use another time-based split.")
    
print("\n" + "=" * 60)
print("✅ DRIFT ANALYSIS COMPLETE")
print("=" * 60)