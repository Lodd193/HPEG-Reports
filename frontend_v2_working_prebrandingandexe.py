"""
UHB HPEG Front-End (pure Tkinter + ttk)
- Step 1: Process -> select CSV, output folder (default pre-filled), report date defaults to today.
- Step 2: Visualise -> select Exec Team + date range -> pick charts -> generate.

Dependencies: ONLY stdlib + matplotlib + pandas + numpy (already required by your scripts)
No ttkbootstrap. No Pillow. Logo loaded via tkinter.PhotoImage if 'uhb_logo.png' exists.

Files expected in the SAME folder:
    process_hpeg_v1.py
    visualisation_hpeg_v1_enhanced_trial_v1.py
Optional:
    uhb_logo.png    (your attached logo saved as this filename)

Packaging (PyInstaller):
    py -m pip install --upgrade pip
    py -m pip install pandas numpy matplotlib pyinstaller
    py -m PyInstaller --noconfirm --onefile --windowed --name HPEG-Frontend hpeg_frontend_tk.py
"""

import os
import sys
import traceback
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, datetime

# Force non-interactive matplotlib backend BEFORE importing your visualisation module
import matplotlib
matplotlib.use("Agg")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import importlib.util

def resolve_app_dir() -> Path:
    # Works both for Python script and PyInstaller EXE
    if getattr(sys, "frozen", False):  # running in PyInstaller bundle
        return Path(sys.executable).parent
    return Path(__file__).parent

def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ====== NHS colours ======
NHS_BLUE = "#005EB8"
NHS_DARK = "#003087"
NHS_TEXT = "#FFFFFF"

DEFAULT_OUTPUT_DIR = r"C:\Users\lod19\OneDrive\Desktop\Work Related\HPEGs\Master\2025 - 26\1. August\processed\outputs"
LOGO_FILE = "uhb_logo.png"  # Save your provided PNG as this name next to this script

# ---------- Utility: dynamic import of your two scripts by file path ----------
def import_module_from_file(module_name: str, file_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# ---------- App state ----------
@dataclass
class AppState:
    process_csv_path: Path | None = None
    processed_csv_path: Path | None = None
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR)
    report_date: date = field(default_factory=date.today)
    exec_team: str | None = None
    vis_csv_path: Path | None = None
    start_date: date | None = None
    end_date: date | None = None
    charts_selected: list[str] = field(default_factory=list)
    exec_team_options: list[str] = field(default_factory=lambda: [
        "BHH", "QEH", "GHH", "SH", "W&C", "CSS", "Corporate"
    ])

# ---------- Thread-safe logger ----------
class TkLogger:
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def install(self):
        sys.stdout = self
        sys.stderr = self

    def uninstall(self):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    def write(self, msg: str):
        if not isinstance(msg, str):
            msg = str(msg)
        self.queue.put(msg)

    def flush(self):  # for compatibility
        pass

    def pump(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.text_widget.insert(tk.END, msg)
                self.text_widget.see(tk.END)
        except queue.Empty:
            pass

# ---------- Core run functions ----------
def run_processing(state: AppState, process_module):
    """Run your processing pipeline using public functions in process_hpeg_v1.py."""
    from pandas import Timestamp, DataFrame

    def _ensure_df(obj, step_name: str) -> DataFrame:
        """Return a pandas DataFrame from a function result.
        Accepts a DataFrame or a tuple that contains a DataFrame.
        Raises a clear error otherwise."""
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, tuple):
            for item in obj:
                if isinstance(item, pd.DataFrame):
                    print(f"[WARN] {step_name} returned a tuple; using its DataFrame element.")
                    return item
        raise TypeError(f"{step_name} expected a DataFrame, got {type(obj).__name__}")

    csv_path = state.process_csv_path
    outdir = state.output_dir
    report_date = Timestamp(str(state.report_date)).normalize()

    if not csv_path or not csv_path.exists():
        raise FileNotFoundError("Please select a valid CSV to process.")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing CSV: {csv_path}")
    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Report date: {report_date.date()}")

    # Pipeline with defensive DataFrame normalisation after each call
    df = _ensure_df(process_module.load_dataframe(csv_path), "load_dataframe")
    df = _ensure_df(process_module.early_drop_preprocessing(df), "early_drop_preprocessing")
    df = _ensure_df(process_module.drop_columns(df), "drop_columns")
    df = _ensure_df(process_module.rename_columns(df), "rename_columns")
    df = _ensure_df(process_module.clean_location_column(df), "clean_location_column")
    df = _ensure_df(process_module.override_exec_team_wc(df), "override_exec_team_wc")

    if "ID" in df.columns:
        df["ID"] = df["ID"].astype("string")

    df = _ensure_df(process_module.to_datetime_cols(df), "to_datetime_cols")
    df = _ensure_df(process_module.normalise_type(df), "normalise_type")
    df = _ensure_df(process_module.add_closed_and_deadline(df, report_date=report_date), "add_closed_and_deadline")
    df = _ensure_df(process_module.add_six_months_flag(df, report_date=report_date), "add_six_months_flag")

    # --- Subjects/Sub-subjects: handle both processed and raw Radar labels ---
    subjects_candidates = [
        "Subjects",
        "Complaint - Patient Relations Team Management (Subjects (Complaint))",
    ]
    subsubjects_candidates = [
        "Sub-subjects",
        "Complaint - Patient Relations Team Management (Sub-Subjects (Complaint))",
    ]
    present_subjects = [c for c in subjects_candidates if c in df.columns]
    present_subsubjects = [c for c in subsubjects_candidates if c in df.columns]

    if present_subjects and present_subsubjects:
        df = _ensure_df(
            process_module.keep_first_subjects_and_subsubjects(
                df,
                subjects_list=present_subjects,
                subsubjects_list=present_subsubjects,
            ),
            "keep_first_subjects_and_subsubjects",
        )
    else:
        print(
            "[WARN] Subjects/Sub-subjects columns not found for collapsing. "
            f"Available columns include: {', '.join(list(df.columns)[:12])}..."
        )

    df = _ensure_df(process_module.add_potential_safeguarding(df), "add_potential_safeguarding")
    df = _ensure_df(process_module.relocate_closed_deadline(df), "relocate_closed_deadline")
    df = _ensure_df(process_module.merge_complexity(df), "merge_complexity")

    anomalies = process_module.quality_checks(df)

    processed_path = process_module.write_outputs(
        df=df,
        outdir=outdir,
        source_csv=csv_path,
        report_date=report_date,
        anomalies=anomalies,
        unmatched=None,
    )
    print(f"[DONE] Processed file saved: {processed_path}")
    return Path(processed_path)


def run_visualisations(state: AppState, vis_module):
    """Run selected charts from visualisation_hpeg_v1_enhanced_trial_v1.py with pre-checks."""
    import pandas as pd

    csv_path = state.vis_csv_path or state.processed_csv_path
    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError("Select a valid processed CSV for visualisation.")

    exec_team = (state.exec_team or "").strip()
    if not exec_team:
        raise ValueError("Select an Exec Team.")

    # Inclusive end-date safeguard (if loader uses < end_date)
    start_date = pd.to_datetime(state.start_date)
    end_date = pd.to_datetime(state.end_date)
    end_date_inclusive = end_date + pd.Timedelta(days=1)

    output_dir = Path(state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(" GENERATING EXECUTIVE CHARTS")
    print("=" * 60)
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Exec Team: {exec_team}")
    print(f"[INFO] Date range: {start_date.date()} to {end_date.date()}")
    print(f"[INFO] Output dir: {output_dir}")

    # --- Pre-check: list exec teams and date span in file ---
    df_preview = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

    if "Exec Team" in df_preview.columns:
        teams_present = (
            df_preview["Exec Team"].astype(str).str.strip().replace({"": None}).dropna().value_counts()
        )
        if not teams_present.empty:
            print("[INFO] Exec Teams in file (by count):")
            for t, n in teams_present.items():
                print(f"   - {t}: {n}")
        else:
            print("[WARN] No Exec Team values in the processed file.")

    date_col = None
    for c in ["First Received", "Due Date", "Completed Date"]:
        if c in df_preview.columns:
            date_col = c
            break
    if date_col is not None:
        dseries = pd.to_datetime(df_preview[date_col], errors="coerce", dayfirst=True).dropna()
        if len(dseries):
            print(f"[INFO] File date span ({date_col}): {dseries.min().date()} to {dseries.max().date()}")

    # Load filtered data
    df = vis_module.load_and_prepare_data(csv_path, exec_team, start_date, end_date_inclusive)
    if len(df) == 0:
        print("\n⚠ No data found for the selected filters.")
        # Extra hint: show rows that match exec team ignoring date
        if "Exec Team" in df_preview.columns:
            n_team = (df_preview["Exec Team"].astype(str).str.strip() == exec_team).sum()
            print(f"[HINT] Rows for '{exec_team}' ignoring date: {n_team}")
        return

    chart_funcs = {
        "Chart 1 – Received vs Ongoing by Month": lambda: vis_module.chart1_received_vs_ongoing_monthly(df, exec_team, start_date, end_date, output_dir),
        "Chart 2 – Top Locations":                lambda: vis_module.chart2_top_locations(df, exec_team, output_dir),
        "Chart 3 – Top Specialties (stacked)":    lambda: vis_module.chart3_top_specialties_stacked(df, exec_team, output_dir),
        "Chart 4 – Top Subjects (pie)":           lambda: vis_module.chart4_top_subjects_pie(df, exec_team, output_dir),
        "Chart 5 – Top Sub-Subjects (pie)":       lambda: vis_module.chart5_top_subsubjects_pie(df, exec_team, output_dir),
        "Chart 6 – Safeguarding analysis":        lambda: vis_module.chart6_safeguarding_analysis(df, exec_team, output_dir),
        "Chart 7 – 6 months rule analysis":       lambda: vis_module.chart7_six_months_analysis(df, exec_team, output_dir),
        "Chart 8 – Complexity analysis":          lambda: vis_module.chart8_complexity_analysis(df, exec_team, output_dir),
    }

    to_run = state.charts_selected or list(chart_funcs.keys())
    for name in to_run:
        func = chart_funcs.get(name)
        if func:
            print(f"[INFO] Generating {name} ...")
            func()
            print(f"[OK]  {name} saved.")

    try:
        summary = vis_module.generate_executive_summary(df, exec_team, output_dir)
        print("\nEXECUTIVE SUMMARY")
        print(summary)
    except Exception as e:
        print(f"[WARN] Executive summary failed: {e}")

    print(f"\n✓ Dashboard generation complete. Files in: {output_dir}")

# ---------- Simple date helpers ----------
def parse_date_yyyymmdd(text: str, fallback: date | None = None) -> date:
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except Exception:
        return fallback or date.today()

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UHB HPEG – Processing & Visualisation")
        self.geometry("1100x760")
        self.configure(bg="#f5f7f9")

        # try to set window icon if you later add 'nhs.ico'
        try:
            ico = Path(__file__).with_name("nhs.ico")
            if ico.exists():
                self.iconbitmap(default=str(ico))
        except Exception:
            pass

        self.state = AppState()
        self.process_module = None
        self.vis_module = None

        self._build_styles()
        self._build_header()
        self._build_body()
        self._load_modules()

        self.after(100, self._pump_logs)

    def _build_styles(self):
        style = ttk.Style(self)
        style.configure("TNotebook.Tab", padding=(12, 6))
        style.configure("NHS.TButton", foreground="white", background=NHS_BLUE)
        style.map("NHS.TButton",
                  background=[("active", NHS_DARK)],
                  foreground=[("active", "white")])
        style.configure("Header.TFrame", background="#e9eff4")
        style.configure("Header.TLabel", background="#e9eff4", foreground=NHS_DARK, font=("Segoe UI", 20, "bold"))
        style.configure("SubHeader.TLabel", background="#e9eff4", foreground=NHS_BLUE, font=("Segoe UI", 11))

    def _build_header(self):
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(side=tk.TOP, fill=tk.X)

        # Logo
        self.logo_img = None
        logo_path = Path(__file__).with_name(LOGO_FILE)
        if logo_path.exists():
            try:
                self.logo_img = tk.PhotoImage(file=str(logo_path))
                tk.Label(header, image=self.logo_img, bg="#e9eff4").pack(side=tk.LEFT, padx=16, pady=8)
            except Exception:
                pass

        # Title block
        title_frame = tk.Frame(header, bg="#e9eff4")
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(title_frame, text="HPEG Front-End", style="Header.TLabel").pack(anchor="w", pady=(20, 0))
        ttk.Label(title_frame, text="Process CSV → Generate Executive Visualisations", style="SubHeader.TLabel").pack(anchor="w")

    def _build_body(self):
        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_process = ttk.Frame(nb)
        self.tab_visual = ttk.Frame(nb)
        self.tab_logs = ttk.Frame(nb)

        nb.add(self.tab_process, text="1) Process")
        nb.add(self.tab_visual, text="2) Visualise")
        nb.add(self.tab_logs, text="Logs")

        self._build_process_tab(self.tab_process)
        self._build_visual_tab(self.tab_visual)
        self._build_logs_tab(self.tab_logs)

    def _build_process_tab(self, parent):
        frm = ttk.LabelFrame(parent, text="Process CSV")
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # CSV
        row1 = ttk.Frame(frm)
        row1.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(row1, text="Input CSV").pack(side=tk.LEFT)
        self.ent_csv = ttk.Entry(row1)
        self.ent_csv.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row1, text="Browse", command=self._browse_csv).pack(side=tk.LEFT)

        # Output dir
        row2 = ttk.Frame(frm)
        row2.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(row2, text="Output folder").pack(side=tk.LEFT)
        self.ent_out = ttk.Entry(row2)
        self.ent_out.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.ent_out.insert(0, DEFAULT_OUTPUT_DIR)
        ttk.Button(row2, text="Browse", command=self._browse_outdir).pack(side=tk.LEFT)

        # Report date (YYYY-MM-DD)
        row3 = ttk.Frame(frm)
        row3.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(row3, text="Report date (YYYY-MM-DD)").pack(side=tk.LEFT)
        self.ent_report = ttk.Entry(row3, width=14)
        self.ent_report.insert(0, date.today().strftime("%Y-%m-%d"))
        self.ent_report.pack(side=tk.LEFT, padx=8)

        # Run
        row4 = ttk.Frame(frm)
        row4.pack(fill=tk.X, padx=8, pady=8)
        self.btn_process = ttk.Button(row4, text="Run Processing", command=self._run_process_clicked)
        self.btn_process.pack(side=tk.LEFT)
        ttk.Button(row4, text="Open output folder", command=self._open_output_folder).pack(side=tk.LEFT, padx=8)

    def _build_visual_tab(self, parent):
        frm = ttk.LabelFrame(parent, text="Generate Visualisations")
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Processed CSV
        row1 = ttk.Frame(frm)
        row1.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(row1, text="Processed CSV").pack(side=tk.LEFT)
        self.ent_vis_csv = ttk.Entry(row1)
        self.ent_vis_csv.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row1, text="Browse", command=self._browse_vis_csv).pack(side=tk.LEFT)

        # Exec Team
        row2 = ttk.Frame(frm)
        row2.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(row2, text="Exec Team").pack(side=tk.LEFT)
        self.cbo_exec = ttk.Combobox(row2, values=AppState().exec_team_options, state="readonly", width=22)
        self.cbo_exec.pack(side=tk.LEFT, padx=8)
        self.cbo_exec.current(0)

        # Date range
        row3 = ttk.Frame(frm)
        row3.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(row3, text="Start date (YYYY-MM-DD)").pack(side=tk.LEFT)
        self.ent_start = ttk.Entry(row3, width=14)
        self.ent_start.insert(0, date.today().strftime("%Y-%m-01"))  # first of month
        self.ent_start.pack(side=tk.LEFT, padx=8)
        ttk.Label(row3, text="End date (YYYY-MM-DD)").pack(side=tk.LEFT)
        self.ent_end = ttk.Entry(row3, width=14)
        self.ent_end.insert(0, date.today().strftime("%Y-%m-%d"))
        self.ent_end.pack(side=tk.LEFT, padx=8)

        # Charts
        row4 = ttk.LabelFrame(frm, text="Charts")
        row4.pack(fill=tk.X, padx=8, pady=8)

        self.chart_vars = []
        chart_labels = [
            "Chart 1 – Received vs Ongoing by Month",
            "Chart 2 – Top Locations",
            "Chart 3 – Top Specialties (stacked)",
            "Chart 4 – Top Subjects (pie)",
            "Chart 5 – Top Sub-Subjects (pie)",
            "Chart 6 – Safeguarding analysis",
            "Chart 7 – 6 months rule analysis",
            "Chart 8 – Complexity analysis",
        ]
        chk_frame = ttk.Frame(row4)
        chk_frame.pack(fill=tk.X, padx=6, pady=6)
        for i, label in enumerate(chart_labels):
            var = tk.BooleanVar(value=True)
            self.chart_vars.append((label, var))
            ttk.Checkbutton(chk_frame, text=label, variable=var).grid(row=i // 2, column=i % 2, sticky="w", padx=6, pady=4)

        # Run
        row5 = ttk.Frame(frm)
        row5.pack(fill=tk.X, padx=8, pady=8)
        self.btn_visual = ttk.Button(row5, text="Run Visualisations", command=self._run_visual_clicked)
        self.btn_visual.pack(side=tk.LEFT)
        ttk.Button(row5, text="Open output folder", command=self._open_output_folder).pack(side=tk.LEFT, padx=8)

    def _build_logs_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, padx=8, pady=8)
        self.pbar = ttk.Progressbar(top, mode="indeterminate")
        self.pbar.pack(fill=tk.X, padx=8, pady=8)

        self.txt_log = tk.Text(parent, height=24, wrap="word", bg="#0A0A0A", fg="#E8E8E8")
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.logger = TkLogger(self.txt_log)
        self.logger.install()

    # Logging pump
    def _pump_logs(self):
        self.logger.pump()
        self.after(100, self._pump_logs)

    # Module load
    def _load_modules(self):
        here = Path(__file__).parent
        proc_path = here / "process_hpeg_v1.py"
        vis_path = here / "visualisation_hpeg_v1_enhanced_trial_v1.py"

        try:
            self.process_module = import_module_from_file("process_hpeg_v1", proc_path)
            print(f"[INIT] Loaded processing module: {proc_path}")
        except Exception as e:
            print(f"[ERROR] Could not load process module: {e}")

        try:
            self.vis_module = import_module_from_file("visualisation_hpeg_v1_enhanced_trial_v1", vis_path)
            print(f"[INIT] Loaded visualisation module: {vis_path}")
        except Exception as e:
            print(f"[ERROR] Could not load visualisation module: {e}")

    # Handlers
    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select Radar export CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.ent_csv.delete(0, tk.END)
            self.ent_csv.insert(0, path)

    def _browse_outdir(self):
        path = filedialog.askdirectory(
            title="Select output folder",
            initialdir=DEFAULT_OUTPUT_DIR
        )
        if path:
            self.ent_out.delete(0, tk.END)
            self.ent_out.insert(0, path)

    def _browse_vis_csv(self):
        path = filedialog.askopenfilename(
            title="Select processed CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.ent_vis_csv.delete(0, tk.END)
            self.ent_vis_csv.insert(0, path)

    def _open_output_folder(self):
        try:
            outdir = Path(self.ent_out.get().strip('"'))
            outdir.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(outdir)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open folder:\n{e}")

    def _run_process_clicked(self):
        if not self.process_module:
            messagebox.showerror("Error", "Processing module not loaded.")
            return

        # Collect inputs
        csv_path = self.ent_csv.get().strip().strip('"')
        outdir = self.ent_out.get().strip().strip('"')
        rep_date_txt = self.ent_report.get().strip() or date.today().strftime("%Y-%m-%d")

        if not csv_path:
            messagebox.showwarning("Missing input", "Please select an input CSV.")
            return

        self.state.process_csv_path = Path(csv_path)
        self.state.output_dir = Path(outdir) if outdir else Path(DEFAULT_OUTPUT_DIR)
        self.state.report_date = parse_date_yyyymmdd(rep_date_txt, fallback=date.today())

        self.pbar.start()
        self.btn_process.configure(state=tk.DISABLED)

        def worker():
            try:
                processed = run_processing(self.state, self.process_module)
                self.state.processed_csv_path = processed
                # Autofill visual tab
                self.ent_vis_csv.delete(0, tk.END)
                self.ent_vis_csv.insert(0, str(processed))
                print("[INFO] Visualisation CSV set to processed output.")
                # Try to infer date bounds and Exec Team options from CSV
                self._infer_dates_from_csv(processed)
                messagebox.showinfo("Done", "Processing complete.")
            except Exception as e:
                print(f"\n✗ Processing failed: {e}\n{traceback.format_exc()}")
                messagebox.showerror("Error", f"Processing failed:\n{e}")
            finally:
                self.pbar.stop()
                self.btn_process.configure(state=tk.NORMAL)

        threading.Thread(target=worker, daemon=True).start()

    def _infer_dates_from_csv(self, processed_csv_path: Path):
        try:
            import pandas as pd
            df = pd.read_csv(processed_csv_path, encoding="utf-8-sig", low_memory=False)

            # Dates: prefer First Received, else Due/Completed
            date_col = None
            for c in ["First Received", "Due Date", "Completed Date"]:
                if c in df.columns:
                    date_col = c
                    break
            if date_col:
                s = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dropna()
                if len(s):
                    self.ent_start.delete(0, tk.END)
                    self.ent_start.insert(0, s.min().date().strftime("%Y-%m-%d"))
                    self.ent_end.delete(0, tk.END)
                    self.ent_end.insert(0, s.max().date().strftime("%Y-%m-%d"))

            # Exec Teams: rebuild dropdown from data actually present
            if "Exec Team" in df.columns:
                exec_series = (
                    df["Exec Team"]
                    .astype(str)
                    .str.strip()
                    .replace({"": None})
                    .dropna()
                )
                counts = exec_series.value_counts()
                options = list(counts.index)
                if options:
                    self.cbo_exec.configure(values=options)
                    self.cbo_exec.set(options[0])  # default to most frequent
                    print(f"[INFO] Exec Teams detected: {', '.join(options)}")
                else:
                    print("[WARN] No Exec Team values detected in processed CSV.")
        except Exception as e:
            print(f"[WARN] Could not infer dates/exec teams from CSV: {e}")

    def _run_visual_clicked(self):
        if not self.vis_module:
            messagebox.showerror("Error", "Visualisation module not loaded.")
            return

        vis_csv = self.ent_vis_csv.get().strip().strip('"')
        self.state.vis_csv_path = Path(vis_csv) if vis_csv else self.state.processed_csv_path
        self.state.exec_team = self.cbo_exec.get().strip() or None
        self.state.start_date = parse_date_yyyymmdd(self.ent_start.get().strip(), fallback=date.today())
        self.state.end_date = parse_date_yyyymmdd(self.ent_end.get().strip(), fallback=date.today())
        self.state.output_dir = Path(self.ent_out.get().strip().strip('"') or DEFAULT_OUTPUT_DIR)

        selected = [label for label, var in self.chart_vars if var.get()]
        self.state.charts_selected = selected

        if not self.state.vis_csv_path or not self.state.vis_csv_path.exists():
            messagebox.showwarning("Missing input", "Please select a valid processed CSV.")
            return
        if not self.state.exec_team:
            messagebox.showwarning("Missing input", "Please select an Exec Team.")
            return

        self.pbar.start()
        self.btn_visual.configure(state=tk.DISABLED)

        def worker():
            try:
                run_visualisations(self.state, self.vis_module)
                messagebox.showinfo("Done", "Visualisations complete.")
            except Exception as e:
                print(f"\n✗ Visualisations failed: {e}\n{traceback.format_exc()}")
                messagebox.showerror("Error", f"Visualisations failed:\n{e}")
            finally:
                self.pbar.stop()
                self.btn_visual.configure(state=tk.NORMAL)

        threading.Thread(target=worker, daemon=True).start()

# ---------- entry ----------
if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)
