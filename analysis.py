import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import json, warnings
import pandas as pd
import glob
import os

# Set plotting backend for Windows
matplotlib.use('TkAgg') 
warnings.filterwarnings('ignore')

# --- DATASET LOADER ---
data = {}
file_list = glob.glob("Physics IA Data*.csv")

for file_path in file_list:
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        
        time_col = [c for c in df.columns if 'Time' in c][0]
        ln_col = [c for c in df.columns if 'ln' in c][0]
        status_col = [c for c in df.columns if 'Status' in c][0]

        df = df[df[status_col].isin(['Valid', 'Cutoff Threshold'])]
        
        filename = os.path.basename(file_path)
        clean_name = filename.replace("Physics IA Data - ", "").replace(".csv", "")
        parts = clean_name.split(" ")
        key = f"{parts[0]}_{parts[-1]}"
        
        data[key] = {
            "t": df[time_col].astype(float).tolist(),
            "ln": df[ln_col].astype(float).tolist()
        }
    except Exception as e:
        print(f"Skipping {file_path}. Error: {e}")

# --- CALCULATION FUNCTIONS ---
def linreg(t, lnx):
    t, y = np.array(t, dtype=float), np.array(lnx, dtype=float)
    sl, ic, r, _, _ = stats.linregress(t, y)
    r2 = r**2
    residuals = y - (sl*t + ic)
    s2 = np.sum(residuals**2) / (len(t) - 2)
    Sxx = np.sum((t - t.mean())**2)
    sb = np.sqrt(s2 / Sxx)
    return -sl, r2, sb, len(t), ic

def instrument_unc_b(t, lnx, b):
    t, y = np.array(t, dtype=float), np.array(lnx, dtype=float)
    amp = np.exp(y)
    
    # NEW: Pixel uncertainty = 0.00053 m
    delta_ln_x = 0.00053 / amp  
    # Time uncertainty = 1/30 seconds (30 fps video)
    delta_t = 1/30  
    
    Sxx = np.sum((t - t.mean())**2)
    sum_dlnx2 = np.sum(delta_ln_x**2 * (t - t.mean())**2)
    instr_slope = np.sqrt(sum_dlnx2) / Sxx
    instr_time = b * delta_t / np.sqrt(len(t))
    
    return np.sqrt(instr_slope**2 + instr_time**2)

results = {k: None for k in data.keys()}
for key, d in data.items():
    b, r2, sb, n, ic = linreg(d["t"], d["ln"])
    ui = instrument_unc_b(d["t"], d["ln"], b)
    results[key] = {"b": b, "r2": r2, "sb": sb, "n": n, "u_instr": ui}

groups = {
    "9x9": {"area": 0.0081, "trials": [1,2,3,4,5], "p": "9"},
    "11x11": {"area": 0.0121, "trials": [1,2,3,4,5], "p": "11"},
    "13x13": {"area": 0.0169, "trials": [1,2,3,4,5], "p": "13"},
    "15x15": {"area": 0.0225, "trials": [1,2,3,4,5], "p": "15"},
    "17x17": {"area": 0.0289, "trials": [1,2,3,4,5], "p": "17"},
}

summary = {}

# --- TABLE GENERATION ---
print("\n" + "="*60)
print("TABLES GENERATED AND SAVED TO FOLDER")
print("="*60)

for gname, g in groups.items():
    p = g["p"]
    bs = np.array([results[f"{p}_{t}"]["b"] for t in g["trials"] if f"{p}_{t}" in results])
    
    if len(bs) == 0: continue
    
    group_rows = []
    for t in g["trials"]:
        key = f"{p}_{t}"
        if key in results:
            res = results[key]
            group_rows.append({
                "Trial": t,
                "Damping Coeff (b)": round(res['b'], 4),
                "R^2 Value": round(res['r2'], 4),
                "Instr. Unc (u_i)": round(res['u_instr'], 5)
            })
    
    df_area = pd.DataFrame(group_rows)
    print(f"\n--- Cross-Sectional Area: {gname} ---")
    print(df_area.to_string(index=False))
    df_area.to_csv(f"Results_{gname}.csv", index=False)

    # Calculate Total Y Uncertainty (SEM + Instr)
    uis = np.array([results[f"{p}_{t}"]["u_instr"] for t in g["trials"] if f"{p}_{t}" in results])
    sem = bs.std(ddof=1) / np.sqrt(len(bs))
    u_total = np.sqrt(sem**2 + uis.mean()**2)
    
   # --- UPDATED AREA UNCERTAINTY CALCULATION ---
u_L = 0.001  # Instrumental uncertainty of the scale in meters (1mm)

# Calculate for each group
for gname, g in groups.items():
    p = g["p"]
    bs = np.array([results[f"{p}_{t}"]["b"] for t in g["trials"] if f"{p}_{t}" in results])
    if len(bs) == 0: continue
    
    # 1. Calculate side length L from the area
    L = np.sqrt(g["area"]) 
    
    # 2. Propagation: Delta A = A * ( (Delta L / L) + (Delta W / W) )
    # Since L = W, this simplifies to: Delta A = 2 * L * Delta L
    u_area = 2 * L * u_L 
    
    # [Rest of your SEM and summary calculation logic...]
    uis = np.array([results[f"{p}_{t}"]["u_instr"] for t in g["trials"] if f"{p}_{t}" in results])
    sem = bs.std(ddof=1) / np.sqrt(len(bs))
    u_total = np.sqrt(sem**2 + uis.mean()**2)
    
    summary[gname] = {
        "area": g["area"], 
        "u_area": u_area, 
        "b_mean": bs.mean(), 
        "u_total": u_total
    }
# --- SUMMARY TABLE ---
df_summary = pd.DataFrame([
    {
        "Group": k, 
        "Area (m^2)": v['area'], 
        "Unc Area (m^2)": round(v['u_area'], 6), 
        "Mean b": round(v['b_mean'], 4), 
        "Total Unc b": round(v['u_total'], 4)
    }
    for k, v in summary.items()
])
df_summary.to_csv("Overall_Summary.csv", index=False)
print("\n--- FINAL SUMMARY ---")
print(df_summary.to_string(index=False))

# --- PLOTTING (ENHANCED FOR DOCUMENTATION) ---
plt.rcParams.update({'font.size': 12})
plt.figure(1, figsize=(10, 7))

areas = df_summary["Area (m^2)"]
u_areas = df_summary["Unc Area (m^2)"] # NEW: Horizontal Error Bars
bs = df_summary["Mean b"]
u_bs = df_summary["Total Unc b"]

slope1, intercept1, r1, _, _ = stats.linregress(areas, bs)

# Plot data points with BOTH x and y error bars
plt.errorbar(areas, bs, xerr=u_areas, yerr=u_bs, fmt='ko', capsize=6, elinewidth=1.5, 
             markeredgewidth=1.5, ms=7, label='Data Points')

plt.plot(areas, slope1*areas + intercept1, 'r--', linewidth=2,
         label=f'Linear Trendline (R²={r1**2:.4f})')

plt.title("Damping Coefficient vs Cross-Sectional Area", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Cross-Sectional Area ($A$ / $m^2$)", fontsize=14, labelpad=10)
plt.ylabel("Damping Coefficient ($b$ / $s^{-1}$)", fontsize=14, labelpad=10)
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper left', frameon=True)

plt.tight_layout() 
plt.savefig('result_plot.png', dpi=300, bbox_inches='tight') 

# --- COMBINED PLOT: LINEARIZATION + RESIDUALS ---
plt.figure(figsize=(10, 10))

gs = plt.GridSpec(4, 1)
ax1 = plt.subplot(gs[0:3, 0])
ax2 = plt.subplot(gs[3, 0], sharex=ax1)

ln_A = np.log(df_summary["Area (m^2)"])
ln_b = np.log(df_summary["Mean b"])
# Fractional error propagation for natural logs: \Delta ln(x) = \Delta x / x
err_ln_A = df_summary["Unc Area (m^2)"] / df_summary["Area (m^2)"] 
err_ln_b = df_summary["Total Unc b"] / df_summary["Mean b"]

slope, intercept, r, _, _ = stats.linregress(ln_A, ln_b)
fit_ln_b = slope * ln_A + intercept
residuals = ln_b - fit_ln_b

# 2. Main Plot (ax1) with x and y error bars
ax1.errorbar(ln_A, ln_b, xerr=err_ln_A, yerr=err_ln_b, fmt='ko', capsize=5, elinewidth=1, label='Log-Log Data')
ax1.plot(ln_A, fit_ln_b, 'b-', label=f'Best Fit (n={slope:.3f})', linewidth=2)

# NEW: Max/Min Gradients utilizing BOTH horizontal and vertical uncertainty bounds
x_first, x_last = ln_A.iloc[0], ln_A.iloc[-1]
y_first, y_last = ln_b.iloc[0], ln_b.iloc[-1]
dx_first, dx_last = err_ln_A.iloc[0], err_ln_A.iloc[-1]
dy_first, dy_last = err_ln_b.iloc[0], err_ln_b.iloc[-1]

# Steepest line: from bottom-right of first point to top-left of last point
m_max = ((y_last + dy_last) - (y_first - dy_first)) / ((x_last - dx_last) - (x_first + dx_first))
c_max = (y_last + dy_last) - m_max * (x_last - dx_last)

# Shallowest line: from top-left of first point to bottom-right of last point
m_min = ((y_last - dy_last) - (y_first + dy_first)) / ((x_last + dx_last) - (x_first - dx_first))
c_min = (y_last - dy_last) - m_min * (x_last + dx_last)

ax1.plot(ln_A, m_max * ln_A + c_max, 'g--', alpha=0.3, label=f'Max Gradient (n={m_max:.3f})')
ax1.plot(ln_A, m_min * ln_A + c_min, 'r--', alpha=0.3, label=f'Min Gradient (n={m_min:.3f})')

ax1.set_title("Log-Log Linearization and Residual Analysis", fontweight='bold', fontsize=15, pad=20)
ax1.set_ylabel("$\ln(b / s^{-1})$", fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.6)

# 3. Residuals Plot (ax2)
ax2.errorbar(ln_A, residuals, xerr=err_ln_A, yerr=err_ln_b, fmt='ro', capsize=5, markersize=5, label='Residuals')
ax2.axhline(0, color='black', linestyle='-', linewidth=1) 
ax2.set_ylabel("Residuals\n$\ln(b / s^{-1})$", fontsize=10)
ax2.set_xlabel("$\ln(Area / m^2)$", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.tight_layout()
plt.savefig('combined_analysis_residuals.png', dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(3)

unc_n = (m_max - m_min) / 2
print(f"Final Exponent: n = {slope:.3f} ± {unc_n:.3f}")

# --- PLOT 3: REPRESENTATIVE TRIAL WAVEFORM ---
rep_files = glob.glob("Physics IA Data*17*1*.csv")

if rep_files:
    rep_file = rep_files[0] 
    print(f"Generating waveform using: {rep_file}")
    
    plt.figure(3, figsize=(10, 6))
    df_rep = pd.read_csv(rep_file)
    df_rep.columns = [c.strip() for c in df_rep.columns]
    
    time_col = [c for c in df_rep.columns if 'Time' in c][0]
    pos_col = [c for c in df_rep.columns if 'Horizontal' in c or 'Pos' in c][0]
    ln_col = [c for c in df_rep.columns if 'ln' in c][0]
    
    status_col = [c for c in df_rep.columns if 'Status' in c][0]
    df_valid = df_rep[df_rep[status_col].isin(['Valid', 'Cutoff Threshold'])]

    t_raw = df_rep[time_col].values
    x_raw = df_rep[pos_col].values
    
    b_val, r2_val, _, _, intercept = linreg(df_valid[time_col], df_valid[ln_col])
    x0 = np.exp(intercept) 
    
    plt.plot(t_raw, x_raw, color='gray', alpha=0.4, label='Experimental Trace', linewidth=1)
    plt.scatter(t_raw, x_raw, color='black', s=8, alpha=0.6, label='Data Points')
    
    t_model = np.linspace(min(t_raw), max(t_raw), 500)
    envelope_upper = x0 * np.exp(-b_val * t_model)
    envelope_lower = -x0 * np.exp(-b_val * t_model)
    
    plt.plot(t_model, envelope_upper, 'r--', linewidth=2, label=f'Decay Envelope ($b$={b_val:.3f})')
    plt.plot(t_model, envelope_lower, 'r--', linewidth=2)
    
    plt.title("Representative Waveform: Displacement vs Time", fontweight='bold', fontsize=14)
    plt.xlabel("Time ($t$ / $s$)")
    plt.ylabel("Horizontal Displacement ($x$ / $m$)")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('representative_waveform.png', dpi=300)
    plt.show(block=False)
    plt.pause(3)
else:
    print("\n[Error] No 17x17 data files found. Check your file names in the folder.")