import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="HEI Rock Typing Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
EPS = 1e-10

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_log10(x):
    """Safe log10 function that handles zeros and negatives"""
    x = np.array(x, dtype=float)
    x = np.maximum(x, EPS)
    return np.log10(x)

def safe_logspace(start, stop, num):
    """Safe logspace that handles invalid values"""
    start = max(start, np.log10(EPS))
    stop = max(stop, start + np.log10(2))
    return np.logspace(start, stop, num)

def get_table_download_link(df, filename, text):
    """Generate a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_image_download_link(fig, filename, text):
    """Generate a download link for matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """
    Load data from uploaded CSV file and prepare it for analysis
    """
    df = pd.read_csv(uploaded_file)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common column names
    col_mapping = {
        'porosity': 'porosity',
        'phi': 'porosity',
        'perm': 'permeability',
        'permeability': 'permeability',
        'k': 'permeability',
        'sw': 'sw',
        'water saturation': 'sw',
        'saturation': 'sw',
        'depth': 'depth',
        'frf': 'frf',
        'formation resistivity factor': 'frf'
    }
    
    # Rename columns
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Check required columns
    required_cols = ['porosity', 'permeability']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()
    
    # Add sample number
    if 'sample no' not in df.columns:
        df['sample no'] = range(1, len(df) + 1)
    
    # Remove invalid data
    df = df[(df['porosity'] > 0) & (df['permeability'] > 0)]
    
    # Convert porosity to decimal
    if df['porosity'].max() > 1:
        df['porosity'] = df['porosity'] / 100
    
    # Handle Sw if present
    if 'sw' in df.columns:
        df = df[df['sw'] > 0]
        if df['sw'].max() > 1:
            df['sw'] = df['sw'] / 100
        df['sw'] = df['sw'].clip(0.01, 0.99)
    
    # Add depth if not present
    if 'depth' not in df.columns:
        df['depth'] = np.linspace(5000, 5500, len(df))
    
    return df

# ============================================================================
# PARAMETER CALCULATIONS
# ============================================================================

def estimate_frf(df):
    """Estimate Formation Resistivity Factor (FRF)"""
    if 'frf' in df.columns:
        return df['frf'].clip(lower=1)
    
    m, a = 2.0, 1.0
    
    if 'sw' in df.columns:
        sw_safe = df['sw'].clip(lower=0.01)
        sw_factor = 1 / (sw_safe ** 2)
        frf = a * (df['porosity'] ** (-m)) * sw_factor
    else:
        frf = a * (df['porosity'] ** (-m))
    
    return frf.clip(lower=1, upper=10000)

def calculate_rock_parameters(df):
    """Calculate all rock parameters from paper equations"""
    
    # Get FRF
    df['frf'] = estimate_frf(df)
    
    # Basic properties
    phi = np.maximum(df['porosity'], 0.001)
    k = np.maximum(df['permeability'], 0.0001)
    
    # Equation 6: Normalized porosity (φN)
    df['phi_N'] = phi / np.maximum(1 - phi, 0.001)
    
    # Equation 7: Reservoir Quality Index (RQI)
    df['RQI'] = 0.0314 * np.sqrt(k / phi)
    
    # Equation 8: Flow Zone Indicator (FZI)
    df['FZI'] = df['RQI'] / df['phi_N']
    
    # Equation 4: Electrical Quality Index (EQI)
    df['EQI'] = np.sqrt(1 / np.maximum(df['frf'] * phi, 0.001))
    
    # Modify EQI with Sw if available
    if 'sw' in df.columns:
        df['EQI'] = df['EQI'] * (1 - df['sw'])
    
    # Equation 5: Electrical Zone Indicator (EZI)
    df['EZI'] = df['EQI'] / df['phi_N']
    
    # Equation 12: φHEI
    denominator = np.maximum((1 - phi) ** 4, 1e-10)
    df['phi_HEI'] = 1014 * (phi ** 6) / denominator
    
    # K/F ratio
    df['K/F'] = k / np.maximum(df['frf'], 0.1)
    
    # HEI parameter
    df['HEI_param'] = df['K/F'] / np.maximum(df['phi_HEI'], 1e-10)
    
    # Modify HEI with Sw if available
    if 'sw' in df.columns:
        df['HEI_param'] = df['HEI_param'] * (1 - df['sw'])
    
    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Ensure all values are positive
    for col in ['HEI_param', 'phi_HEI', 'K/F', 'FZI', 'EZI', 'permeability', 'frf']:
        if col in df.columns:
            df[col] = df[col].clip(lower=EPS)
    
    return df

# ============================================================================
# ROCK TYPING
# ============================================================================

def assign_rock_types(df, n_rock_types=None):
    """Assign rock types using clustering"""
    
    hei_param = df['HEI_param'].dropna()
    
    if len(hei_param) < 5:
        df['Rock_Type'] = 1
        return df
    
    # Prepare data
    X = safe_log10(hei_param.values).reshape(-1, 1)
    X = X[np.isfinite(X).all(axis=1)]
    
    if len(X) < 5:
        df['Rock_Type'] = 1
        return df
    
    # Determine number of clusters
    if n_rock_types is None:
        n_samples = len(X)
        max_clusters = min(9, n_samples // 5)
        best_n = min(3, max_clusters)
        best_score = -1
        
        if max_clusters >= 2:
            for n_clusters in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_n = n_clusters
                except:
                    continue
    else:
        best_n = n_rock_types
    
    # Apply clustering
    kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Order rock types by increasing HEI
    label_means = {i: np.mean(hei_param.iloc[labels == i]) for i in range(best_n)}
    sorted_labels = sorted(label_means.keys(), key=lambda x: label_means[x])
    label_mapping = {old: new + 1 for new, old in enumerate(sorted_labels)}
    
    # Assign rock types
    df['Rock_Type'] = pd.Series(
        [label_mapping[l] for l in labels], 
        index=hei_param.index
    )
    
    return df

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_parameter_logs(df):
    """Create log plots for all parameters"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Parameter Logs - Hydraulic and Electrical Properties', 
                fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    depth = df['depth'].values
    y_label = 'Depth (ft)' if df['depth'].max() > 1000 else 'Depth (m)'
    
    # Plot 1: Porosity
    axes[0].plot(df['porosity'] * 100, depth, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Porosity (%)')
    axes[0].set_ylabel(y_label)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Porosity')
    
    # Plot 2: Permeability
    axes[1].semilogx(np.maximum(df['permeability'], EPS), depth, 'g-', linewidth=1.5)
    axes[1].set_xlabel('Permeability (mD)')
    axes[1].set_ylabel(y_label)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Permeability')
    
    # Plot 3: FRF
    axes[2].semilogx(np.maximum(df['frf'], EPS), depth, 'r-', linewidth=1.5)
    axes[2].set_xlabel('Formation Resistivity Factor')
    axes[2].set_ylabel(y_label)
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('FRF')
    
    # Plot 4: FZI
    axes[3].semilogx(np.maximum(df['FZI'], EPS), depth, 'm-', linewidth=1.5)
    axes[3].set_xlabel('Flow Zone Indicator')
    axes[3].set_ylabel(y_label)
    axes[3].invert_yaxis()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('FZI')
    
    # Plot 5: EZI
    axes[4].semilogx(np.maximum(df['EZI'], EPS), depth, 'c-', linewidth=1.5)
    axes[4].set_xlabel('Electrical Zone Indicator')
    axes[4].set_ylabel(y_label)
    axes[4].invert_yaxis()
    axes[4].grid(True, alpha=0.3)
    axes[4].set_title('EZI')
    
    # Plot 6: φHEI
    axes[5].semilogx(np.maximum(df['phi_HEI'], EPS), depth, 'orange', linewidth=1.5)
    axes[5].set_xlabel('φHEI')
    axes[5].set_ylabel(y_label)
    axes[5].invert_yaxis()
    axes[5].grid(True, alpha=0.3)
    axes[5].set_title('φHEI (Eq. 12)')
    
    # Plot 7: HEI Parameter
    axes[6].semilogx(np.maximum(df['HEI_param'], EPS), depth, 'purple', linewidth=1.5)
    axes[6].set_xlabel('HEI Parameter')
    axes[6].set_ylabel(y_label)
    axes[6].invert_yaxis()
    axes[6].grid(True, alpha=0.3)
    axes[6].set_title('HEI Parameter')
    
    # Plot 8: Rock Types
    unique_rts = sorted(df['Rock_Type'].unique())
    if unique_rts:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_rts)))
        for rt, color in zip(unique_rts, colors):
            mask = df['Rock_Type'] == rt
            axes[7].scatter([rt] * mask.sum(), depth[mask], 
                           c=[color], label=f'RT {rt}', 
                           alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        axes[7].set_xlabel('Rock Type')
        axes[7].set_ylabel(y_label)
        axes[7].set_xticks(unique_rts)
        axes[7].invert_yaxis()
        axes[7].grid(True, alpha=0.3)
        axes[7].set_title('Rock Type Distribution')
        axes[7].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_hei_rock_typing(df):
    """Plot K/F vs φHEI on log-log scale (Figures 2 and 3)"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Prepare data
    df_plot = df.copy()
    df_plot['phi_HEI'] = np.maximum(df_plot['phi_HEI'], EPS)
    df_plot['K/F'] = np.maximum(df_plot['K/F'], EPS)
    
    rock_types = sorted(df_plot['Rock_Type'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(rock_types)))
    
    # Plot data points
    equations = {}
    for rt, color in zip(rock_types, colors):
        mask = df_plot['Rock_Type'] == rt
        if mask.any():
            ax.loglog(df_plot.loc[mask, 'phi_HEI'], 
                     df_plot.loc[mask, 'K/F'], 
                     'o', color=color, label=f'Rock Type {rt}', 
                     alpha=0.7, markersize=8, 
                     markeredgecolor='black', markeredgewidth=0.5)
            
            if mask.sum() > 1:
                avg_fziezi = np.mean(df_plot.loc[mask, 'FZI']**2 * 
                                    df_plot.loc[mask, 'EZI']**2)
                equations[rt] = avg_fziezi
    
    # Add unit slope lines
    if df_plot['phi_HEI'].min() > 0 and df_plot['phi_HEI'].max() > 0:
        phi_min = np.log10(df_plot['phi_HEI'].min() * 0.9)
        phi_max = np.log10(df_plot['phi_HEI'].max() * 1.1)
        phi_range = safe_logspace(phi_min, phi_max, 50)
        
        for rt, color in zip(rock_types, colors):
            if rt in equations:
                kf_pred = equations[rt] * phi_range
                ax.loglog(phi_range, kf_pred, '--', color=color, 
                         alpha=0.7, linewidth=2, 
                         label=f'RT {rt}: y={equations[rt]:.4f}x')
    
    ax.set_xlabel('φHEI', fontsize=14)
    ax.set_ylabel('K/F', fontsize=14)
    ax.set_title('HEI Rock Typing - K/F vs φHEI (Log-Log Scale)', 
                fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add text box with equations
    if equations:
        eq_text = "Rock Type Equations:\n"
        for rt, avg in equations.items():
            eq_text += f"RT {rt}: K/F = {avg:.4f} × φHEI\n"
        
        ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig, equations

def plot_regression_analysis(df):
    """Plot regression lines and summary table (Tables 4 and 5)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare data
    df_plot = df.copy()
    df_plot['phi_HEI'] = np.maximum(df_plot['phi_HEI'], EPS)
    df_plot['K/F'] = np.maximum(df_plot['K/F'], EPS)
    
    rock_types = sorted(df_plot['Rock_Type'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(rock_types)))
    
    # Plot 1: Regression lines
    ax1 = axes[0]
    regression_results = []
    
    for rt, color in zip(rock_types, colors):
        mask = df_plot['Rock_Type'] == rt
        if mask.sum() < 2:
            continue
        
        x = safe_log10(df_plot.loc[mask, 'phi_HEI'])
        y = safe_log10(df_plot.loc[mask, 'K/F'])
        valid = np.isfinite(x) & np.isfinite(y)
        
        if valid.sum() < 2:
            continue
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x[valid], y[valid]
        )
        
        # Plot data points
        ax1.loglog(df_plot.loc[mask, 'phi_HEI'], 
                  df_plot.loc[mask, 'K/F'], 
                  'o', color=color, alpha=0.5, markersize=6)
        
        # Plot regression line
        x_plot = np.linspace(x[valid].min(), x[valid].max(), 50)
        y_plot = slope * x_plot + intercept
        ax1.loglog(10**x_plot, 10**y_plot, '-', color=color, linewidth=2,
                  label=f'RT {rt}: R²={r_value**2:.3f}')
        
        regression_results.append([
            f'RT {rt}',
            f'y = {10**intercept:.4f}x',
            f'{10**intercept:.4f}',
            f'{r_value**2:.4f}',
            str(mask.sum())
        ])
    
    ax1.set_xlabel('φHEI', fontsize=12)
    ax1.set_ylabel('K/F', fontsize=12)
    ax1.set_title('Regression Lines for Each Rock Type', 
                 fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Summary table
    ax2 = axes[1]
    ax2.axis('off')
    
    if regression_results:
        columns = ['Rock Type', 'Equation', 'Avg ((FZI)²·(EZI)²)', 'R²', 'Samples']
        table = ax2.table(cellText=regression_results, colLabels=columns,
                         cellLoc='center', loc='center',
                         colWidths=[0.12, 0.25, 0.20, 0.10, 0.10])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header
        for i, col in enumerate(columns):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('Rock Type Summary Statistics (Tables 4 & 5)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, regression_results

def predict_permeability(df):
    """Predict permeability using HEI method - Equation 11"""
    
    phi = df['porosity']
    
    # Equation 11
    df['K_pred'] = (1014 * (phi**6) / ((1 - phi)**4) * 
                    (df['FZI']**2) * (df['EZI']**2))
    
    if 'sw' in df.columns:
        df['K_pred'] = df['K_pred'] * (1 - df['sw'])**2
    
    df['K_pred'] = df['K_pred'].clip(lower=EPS, upper=10000)
    
    return df

def plot_permeability_crossplot(df):
    """Plot predicted vs actual permeability (Figures 4 and 5)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare data
    df_plot = df.copy()
    df_plot['permeability'] = np.maximum(df_plot['permeability'], EPS)
    df_plot['K_pred'] = np.maximum(df_plot['K_pred'], EPS)
    
    rock_types = sorted(df_plot['Rock_Type'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(rock_types)))
    
    # Calculate plot limits
    min_val = max(EPS, df_plot['permeability'].min(), df_plot['K_pred'].min())
    max_val = max(df_plot['permeability'].max(), df_plot['K_pred'].max())
    if min_val >= max_val:
        max_val = min_val * 10
    
    # Plot 1: Colored by rock type
    ax1 = axes[0]
    for rt, color in zip(rock_types, colors):
        mask = df_plot['Rock_Type'] == rt
        if mask.any():
            ax1.loglog(df_plot.loc[mask, 'permeability'], 
                      df_plot.loc[mask, 'K_pred'], 
                      'o', color=color, label=f'RT {rt}', 
                      alpha=0.7, markersize=8,
                      markeredgecolor='black', markeredgewidth=0.5)
    
    # 1:1 line
    ax1.loglog([min_val, max_val], [min_val, max_val], 
              'r--', linewidth=2, label='1:1 line')
    
    # Calculate R²
    mask = (df_plot['permeability'] > 0) & (df_plot['K_pred'] > 0)
    if mask.any():
        log_real = safe_log10(df_plot.loc[mask, 'permeability'])
        log_pred = safe_log10(df_plot.loc[mask, 'K_pred'])
        valid = np.isfinite(log_real) & np.isfinite(log_pred)
        if valid.any():
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_real[valid], log_pred[valid]
            )
            r2 = r_value**2
        else:
            r2 = np.nan
    else:
        r2 = np.nan
    
    ax1.set_xlabel('Measured Permeability (mD)', fontsize=12)
    ax1.set_ylabel('Predicted Permeability (mD)', fontsize=12)
    ax1.set_title(f'Permeability Prediction by Rock Type\nR² = {r2:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Density plot
    ax2 = axes[1]
    
    x = safe_log10(df_plot['permeability'])
    y = safe_log10(df_plot['K_pred'])
    valid = np.isfinite(x) & np.isfinite(y)
    
    if valid.any():
        h = ax2.hist2d(x[valid], y[valid], bins=20, cmap='viridis', alpha=0.7,
                      range=[[np.log10(min_val), np.log10(max_val)],
                            [np.log10(min_val), np.log10(max_val)]])
        plt.colorbar(h[3], ax=ax2, label='Count')
    
    ax2.loglog([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Measured Permeability (mD)', fontsize=12)
    ax2.set_ylabel('Predicted Permeability (mD)', fontsize=12)
    ax2.set_title('Permeability Prediction - Density Plot', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig, r2

def plot_cross_validation(df, n_folds=5):
    """Perform and plot cross validation results (Figures 6-15)"""
    
    if len(df) < n_folds:
        return None, [], np.nan, np.nan
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        if fold > 5:
            break
        
        train_data = df.iloc[train_idx].copy()
        val_data = df.iloc[val_idx].copy()
        
        # Recalculate rock types on training data
        train_data = assign_rock_types(train_data)
        
        # Assign rock types to validation data
        if len(train_data['Rock_Type'].unique()) > 0:
            X_train = safe_log10(train_data['HEI_param'].values).reshape(-1, 1)
            X_train = X_train[np.isfinite(X_train).all(axis=1)]
            
            if len(X_train) >= len(train_data['Rock_Type'].unique()):
                kmeans = KMeans(n_clusters=len(train_data['Rock_Type'].unique()),
                               random_state=42, n_init=10)
                kmeans.fit(X_train)
                
                X_val = safe_log10(val_data['HEI_param'].values).reshape(-1, 1)
                X_val = X_val[np.isfinite(X_val).all(axis=1)]
                
                if len(X_val) > 0:
                    val_labels = kmeans.predict(X_val)
                    val_indices = val_data.index[np.isfinite(
                        safe_log10(val_data['HEI_param'])
                    )]
                    val_data.loc[val_indices, 'Rock_Type'] = val_labels
        
        # Predict permeability
        val_data = predict_permeability(val_data)
        
        # Calculate R²
        mask = (val_data['permeability'] > 0) & (val_data['K_pred'] > 0)
        if mask.any():
            log_real = safe_log10(val_data.loc[mask, 'permeability'])
            log_pred = safe_log10(val_data.loc[mask, 'K_pred'])
            valid = np.isfinite(log_real) & np.isfinite(log_pred)
            if valid.any():
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_real[valid], log_pred[valid]
                )
                r2 = r_value**2
            else:
                r2 = np.nan
        else:
            r2 = np.nan
        
        fold_results.append({
            'fold': fold,
            'r2': r2,
            'n_train': len(train_data),
            'n_val': len(val_data)
        })
        
        # Plot
        ax = axes[fold-1]
        val_plot = val_data.copy()
        val_plot['permeability'] = np.maximum(val_plot['permeability'], EPS)
        val_plot['K_pred'] = np.maximum(val_plot['K_pred'], EPS)
        
        if 'Rock_Type' in val_plot.columns:
            rock_types_val = sorted(val_plot['Rock_Type'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(rock_types_val)))
            
            for rt, color in zip(rock_types_val, colors):
                rt_mask = val_plot['Rock_Type'] == rt
                if rt_mask.any():
                    ax.loglog(val_plot.loc[rt_mask, 'permeability'],
                             val_plot.loc[rt_mask, 'K_pred'],
                             'o', color=color, label=f'RT {rt}',
                             alpha=0.7, markersize=6,
                             markeredgecolor='black', markeredgewidth=0.5)
        
        # 1:1 line
        if mask.any():
            min_k = max(EPS, val_plot.loc[mask, 'permeability'].min(),
                       val_plot.loc[mask, 'K_pred'].min())
            max_k = max(val_plot.loc[mask, 'permeability'].max(),
                       val_plot.loc[mask, 'K_pred'].max())
            if min_k >= max_k:
                max_k = min_k * 10
            ax.loglog([min_k, max_k], [min_k, max_k], 'r--', linewidth=2)
        
        ax.set_title(f'Fold {fold} - R² = {r2:.4f}')
        ax.set_xlabel('Measured K (mD)')
        ax.set_ylabel('Predicted K (mD)')
        ax.grid(True, alpha=0.3, which='both')
        if fold == 1:
            ax.legend(loc='upper left', fontsize=8)
    
    # Hide unused subplots
    for i in range(fold, 6):
        if i < len(axes):
            axes[i].axis('off')
    
    plt.suptitle(f'{n_folds}-Fold Cross Validation Results (Figs 6-15)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Calculate statistics
    valid_r2 = [r['r2'] for r in fold_results if not np.isnan(r['r2'])]
    avg_r2 = np.mean(valid_r2) if valid_r2 else np.nan
    std_r2 = np.std(valid_r2) if valid_r2 else np.nan
    
    return fig, fold_results, avg_r2, std_r2

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app"""
    
    st.title("📊 HEI (Hydraulic-Electric Index) Rock Typing Analysis")
    st.markdown("""
    This app reproduces the methodology from the paper:
    **"Development of a new hydraulic electric index for rock typing in carbonate reservoirs"** 
    (Scientific Reports, 2024)
    """)
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Input")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="File must contain porosity and permeability columns"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Preview data
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.subheader("Data Preview")
                st.dataframe(preview_df.head(), use_container_width=True)
                st.caption(f"Total rows: {len(preview_df)}")
            except Exception as e:
                st.error(f"Could not read CSV file: {e}")
        
        st.header("⚙️ Analysis Settings")
        
        n_rock_types = st.slider(
            "Number of Rock Types (optional)",
            min_value=2,
            max_value=9,
            value=5,
            help="Select number of rock types or let it optimize automatically"
        )
        
        n_folds = st.slider(
            "Cross-validation Folds",
            min_value=3,
            max_value=10,
            value=5
        )
        
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file to begin analysis")
        
        # Show example format
        with st.expander("📋 Expected CSV Format"):
            st.markdown("""
            Your CSV should have at least these columns:
            - **porosity** (fraction or percentage)
            - **permeability** (mD)
            
            Optional columns:
            - **sw** (water saturation, fraction or percentage)
            - **depth** (depth values)
            - **frf** (formation resistivity factor)
            
            **Example:** porosity,permeability,sw
0.15,10.5,0.35
0.22,45.2,0.28
0.08,0.5,0.75""")
""")

st.stop()

if run_button:
with st.spinner("Processing data and generating plots..."):
try:
    # Load and prepare data
    df = load_and_prepare_data(uploaded_file)
    
    # Calculate parameters
    df = calculate_rock_parameters(df)
    
    if len(df) == 0:
        st.error("No valid samples remaining after data cleaning!")
        st.stop()
    
    # Assign rock types
    df = assign_rock_types(df, n_rock_types)
    
    # Predict permeability
    df = predict_permeability(df)
    
    # Store in session state
    st.session_state.df = df
    st.session_state.analysis_done = True
    
    st.success("✅ Analysis complete! Scroll down to see results.")
    st.rerun()
    
except Exception as e:
    st.error(f"Error during analysis: {str(e)}")
    st.stop()

# Display results if analysis is done
if st.session_state.analysis_done and st.session_state.df is not None:
df = st.session_state.df

# Summary statistics
st.header("📊 Data Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric("Total Samples", len(df))
with col2:
st.metric("Rock Types", len(df['Rock_Type'].unique()))
with col3:
st.metric("Porosity Range", 
         f"{df['porosity'].min()*100:.1f} - {df['porosity'].max()*100:.1f}%")
with col4:
st.metric("Permeability Range", 
         f"{df['permeability'].min():.2f} - {df['permeability'].max():.2f} mD")

# Tabs for different plots
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
"📈 Parameter Logs",
"🪨 HEI Rock Typing",
"📐 Regression Analysis",
"🎯 Permeability Prediction",
"🔄 Cross Validation",
"📋 Data Export"
])

with tab1:
st.header("Parameter Logs")
st.caption("Log plots of all calculated parameters")

fig = plot_parameter_logs(df)
st.pyplot(fig)
plt.close(fig)

# Download button
st.markdown(
    get_image_download_link(fig, "parameter_logs.png", "📥 Download Plot"),
    unsafe_allow_html=True
)

with tab2:
st.header("HEI Rock Typing (K/F vs φHEI)")
st.caption("Figures 2 and 3 from the paper")

fig, equations = plot_hei_rock_typing(df)
st.pyplot(fig)
plt.close(fig)

if equations:
    st.subheader("Rock Type Equations")
    eq_df = pd.DataFrame([
        {"Rock Type": rt, "Equation": f"K/F = {avg:.4f} × φHEI"}
        for rt, avg in equations.items()
    ])
    st.dataframe(eq_df, use_container_width=True)

# Download button
st.markdown(
    get_image_download_link(fig, "hei_rock_typing.png", "📥 Download Plot"),
    unsafe_allow_html=True
)

with tab3:
st.header("Regression Analysis")
st.caption("Tables 4 and 5 from the paper")

fig, regression_results = plot_regression_analysis(df)
st.pyplot(fig)
plt.close(fig)

if regression_results:
    st.subheader("Summary Table")
    table_df = pd.DataFrame(
        regression_results,
        columns=['Rock Type', 'Equation', 'Avg ((FZI)²·(EZI)²)', 'R²', 'Samples']
    )
    st.dataframe(table_df, use_container_width=True)

# Download button
st.markdown(
    get_image_download_link(fig, "regression_analysis.png", "📥 Download Plot"),
    unsafe_allow_html=True
)

with tab4:
st.header("Permeability Prediction")
st.caption("Figures 4 and 5 from the paper")

fig, r2 = plot_permeability_crossplot(df)
st.pyplot(fig)
plt.close(fig)

st.metric("Overall R²", f"{r2:.4f}")

# Download button
st.markdown(
    get_image_download_link(fig, "permeability_prediction.png", "📥 Download Plot"),
    unsafe_allow_html=True
)

with tab5:
st.header("Cross Validation Results")
st.caption(f"Figures 6-15 from the paper ({n_folds}-fold validation)")

fig, fold_results, avg_r2, std_r2 = plot_cross_validation(df, n_folds=n_folds)

if fig:
    st.pyplot(fig)
    plt.close(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average R²", f"{avg_r2:.4f}")
    with col2:
        st.metric("Standard Deviation", f"{std_r2:.4f}")
    
    if fold_results:
        st.subheader("Fold Results")
        fold_df = pd.DataFrame(fold_results)
        st.dataframe(fold_df, use_container_width=True)
    
    # Download button
    st.markdown(
        get_image_download_link(fig, "cross_validation.png", "📥 Download Plot"),
        unsafe_allow_html=True
    )
else:
    st.warning(f"Not enough samples for {n_folds}-fold validation")

with tab6:
st.header("Data Export")

# Processed data
st.subheader("Processed Data")
st.dataframe(df.head(100), use_container_width=True)
st.caption(f"Showing first 100 of {len(df)} rows")

# Download buttons
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        get_table_download_link(
            df, 
            "processed_data.csv", 
            "📥 Download Processed Data (CSV)"
        ),
        unsafe_allow_html=True
    )

with col2:
    # Create summary dataframe
    summary_data = {
        'Parameter': ['Total Samples', 'Number of Rock Types', 
                     'Porosity Min (%)', 'Porosity Max (%)',
                     'Permeability Min (mD)', 'Permeability Max (mD)',
                     'HEI Min', 'HEI Max'],
        'Value': [
            len(df),
            len(df['Rock_Type'].unique()),
            f"{df['porosity'].min()*100:.2f}",
            f"{df['porosity'].max()*100:.2f}",
            f"{df['permeability'].min():.4f}",
            f"{df['permeability'].max():.4f}",
            f"{df['HEI_param'].min():.4f}",
            f"{df['HEI_param'].max():.4f}"
        ]
    }
    
    if 'sw' in df.columns:
        summary_data['Parameter'].extend(['Sw Min (%)', 'Sw Max (%)'])
        summary_data['Value'].extend([
            f"{df['sw'].min()*100:.2f}",
            f"{df['sw'].max()*100:.2f}"
        ])
    
    summary_df = pd.DataFrame(summary_data)
    
    st.markdown(
        get_table_download_link(
            summary_df,
            "summary_statistics.csv",
            "📥 Download Summary Statistics"
        ),
        unsafe_allow_html=True
    )

# Rock type distribution
if 'Rock_Type' in df.columns:
    st.subheader("Rock Type Distribution")
    rt_dist = df['Rock_Type'].value_counts().sort_index().reset_index()
    rt_dist.columns = ['Rock Type', 'Count']
    st.dataframe(rt_dist, use_container_width=True)
    
    st.markdown(
        get_table_download_link(
            rt_dist,
            "rock_type_distribution.csv",
            "📥 Download Rock Type Distribution"
        ),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
main()


