# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:55:22 2025

@author: risha
"""

"""
Mitra: Multi-scale Adaptive Network (MAN) Epidemic Model - Interactive Application

Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)

Copyright (c) 2025 Rishabh Mishra

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. 
  You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others 
from doing anything the license permits.

Notices:
You do not have to comply with the license for elements of the material in the public domain 
or where your use is permitted by an applicable exception or limitation.

No warranties are given. The license may not give you all of the permissions necessary for your intended use. 
For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

## DISCLAIMER OF LIABILITY

This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors disclaim all warranties, including, but not limited to, any implied warranties of merchantability, fitness for a particular purpose, and non-infringement.
In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
This software is intended for research and educational purposes only. Any decisions made based on the outputs of this model are the sole responsibility of the user. The software developers are not responsible for any actions taken based on the software's results or interpretations thereof.
This epidemic model is a simplified representation of complex real-world systems and should not be the sole basis for public health decision-making.
===========================================================================




A GUI application for the MAN epidemic model that can be packaged for Windows and Mac.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import queue
import pickle
import json
from datetime import datetime
import time
import networkx as nx

# Import the model code from man_epidemic_model.py
from man_epidemic_model import (
    MANEpidemicModel, Parameters, HealthState, 
    NetworkScale, ModelResolution, Individual
)

def configure_dpi_awareness():
    """Configure high DPI settings for Windows displays."""
    try:
        # For Windows
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError):
        # For other platforms or if function fails
        pass

class StdoutRedirector:
    """Redirects stdout to a queue for GUI display."""
    
    def __init__(self, text_queue):
        self.text_queue = text_queue
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.text_queue.put(message)

    def flush(self):
        self.terminal.flush()


class ModelThread(threading.Thread):
    """Thread for running the epidemic model simulation."""
    
    def __init__(self, model, params, days, update_interval=1.0, callback=None):
        """
        Initialize simulation thread.
        
        Args:
            model: The epidemic model instance
            params: Model parameters
            days: Simulation duration in days
            update_interval: How often to update the UI (in days)
            callback: Function to call after each update interval
        """
        threading.Thread.__init__(self)
        self.model = model
        self.params = params
        self.days = days
        self.update_interval = update_interval
        self.callback = callback
        self.daemon = True
        self.paused = False
        self.stopped = False
        self._pause_condition = threading.Condition(threading.Lock())
        
    def run(self):
        """Run the simulation with periodic updates to UI."""
        if not self.model.initialized:
            self.model.initialize()
            
        steps_per_interval = max(1, int(self.update_interval / self.params.time_step))
        total_steps = int(self.days / self.params.time_step)
        steps_completed = 0
        
        while steps_completed < total_steps and not self.stopped:
            # Check if paused
            with self._pause_condition:
                while self.paused and not self.stopped:
                    self._pause_condition.wait()
                    
            if self.stopped:
                break
                
            # Determine how many steps to run in this iteration
            steps_to_run = min(steps_per_interval, total_steps - steps_completed)
            
            # Run simulation steps
            for _ in range(steps_to_run):
                self.model.simulate_step()
                steps_completed += 1
                
                # Check if stopped during step
                if self.stopped:
                    break
            
            # Update the UI
            if self.callback and not self.stopped:
                self.callback()
                
        # Final update
        if self.callback and not self.stopped:
            self.callback()
            
    def pause(self):
        """Pause the simulation."""
        self.paused = True
        
    def resume(self):
        """Resume the simulation."""
        with self._pause_condition:
            self.paused = False
            self._pause_condition.notify()
            
    def stop(self):
        """Stop the simulation."""
        self.stopped = True
        with self._pause_condition:
            self._pause_condition.notify()


class ScenarioManager:
    """Manages saving and loading model scenarios."""
    
    def __init__(self, model=None):
        """
        Initialize scenario manager.
        
        Args:
            model: The epidemic model instance
        """
        self.model = model
        self.current_scenario_name = "Untitled Scenario"
        
    def set_model(self, model):
        """Set the current model instance."""
        self.model = model
        
    def save_scenario(self, filename, name=None, description=None):
        """
        Save current model state and parameters as a scenario.
        
        Args:
            filename: File to save to
            name: Scenario name
            description: Scenario description
        """
        if not self.model:
            raise ValueError("No model assigned to ScenarioManager")
            
        if name:
            self.current_scenario_name = name
            
        # Create scenario data
        scenario = {
            'name': self.current_scenario_name,
            'description': description or "",
            'created': datetime.now().isoformat(),
            'parameters': vars(self.model.params),
            'current_time': self.model.time,
            'model_state': {
                'case_counts': self.model.case_counts,
                'R_effective': self.model.R_effective,
                'info_history': self.model.behavior_model.info_history,
                'media_history': self.model.behavior_model.media_history,
                'policy_history': self.model.behavior_model.policy_history,
                'pathogen_history': self.model.pathogen.history
            }
            # We'd need more complex serialization for population, networks, etc.
            # This is simplified for the example
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(scenario, f)
            
        return True
    
    def load_scenario(self, filename):
        """
        Load scenario from file.
        
        Args:
            filename: File to load from
            
        Returns:
            Dict with scenario data
        """
        with open(filename, 'rb') as f:
            scenario = pickle.load(f)
            
        self.current_scenario_name = scenario['name']
        return scenario
    
    def apply_scenario(self, scenario):
        """
        Apply a loaded scenario to the current model.
        
        Args:
            scenario: Scenario data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.model:
            raise ValueError("No model assigned to ScenarioManager")
            
        try:
            # Create new parameters
            params = Parameters()
            for key, value in scenario['parameters'].items():
                if hasattr(params, key):
                    setattr(params, key, value)
                    
            # Create new model with these parameters
            model = MANEpidemicModel(params)
            
            # If the scenario has state data, we would load it here
            # This is simplified for the example
            
            return model, params
        except Exception as e:
            print(f"Error applying scenario: {e}")
            return None, None


class MANEpidemicApp(tk.Tk):
    """Main application window for the interactive epidemic model."""
    
    def __init__(self):
        """Initialize the application."""
        super().__init__()
        
        self.configure_fonts()
        
        self.title("Mitra: Multi-scale Adaptive Network (MAN) Epidemic Model")
        self.geometry("1280x800")
        self.minsize(1024, 768)
        
        # Set icon (uncomment and add your icon file)
        # self.iconbitmap("icon.ico") 
        
        # Initialize model and parameters
        self.params = Parameters()
        self.model = MANEpidemicModel(self.params)
        
        # Initialize scenario manager
        self.scenario_manager = ScenarioManager(self.model)
        
        # Control variables
        self.simulation_thread = None
        self.output_queue = queue.Queue()
        
        # Redirect stdout
        sys.stdout = StdoutRedirector(self.output_queue)
        
        # Create the menu
        self._create_menu()
        
        # Create the main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self._create_dashboard_tab()
        self._create_parameters_tab()
        self._create_network_tab()
        self._create_behavior_tab()
        self._create_data_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start periodic UI updates
        self.after(100, self._process_output_queue)
        
    def configure_fonts(self):
        """Configure improved fonts for better text rendering."""
        # Configure default font settings
        default_font = ('Segoe UI', 9)  # Windows
        
        # Platform-specific fonts
        import platform
        if platform.system() == 'Darwin':  # macOS
            default_font = ('SF Pro', 12)
        elif platform.system() == 'Linux':
            default_font = ('DejaVu Sans', 10)
        
        # Apply font settings
        from tkinter import font
        font_config = font.nametofont("TkDefaultFont")
        font_config.configure(family=default_font[0], size=default_font[1])
        
        # Configure styles
        style = ttk.Style()
        style.configure(".", font=default_font)
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=default_font)
        style.configure("TScale", sliderthickness=20)
        
    def _create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Scenario", command=self._new_scenario)
        file_menu.add_command(label="Open Scenario", command=self._open_scenario)
        file_menu.add_command(label="Save Scenario", command=self._save_scenario)
        file_menu.add_command(label="Save Scenario As...", command=self._save_scenario_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        sim_menu.add_command(label="Initialize Model", command=self._initialize_model)
        sim_menu.add_command(label="Run Simulation", command=self._run_simulation)
        sim_menu.add_command(label="Pause Simulation", command=self._pause_simulation)
        sim_menu.add_command(label="Resume Simulation", command=self._resume_simulation)
        sim_menu.add_command(label="Stop Simulation", command=self._stop_simulation)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Plot Results", command=self._plot_results)
        analysis_menu.add_command(label="Generate Report", command=self._generate_report)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self._show_user_guide)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
        
    
        
    def _create_dashboard_tab(self):
        """Create the main dashboard tab."""
        dashboard = ttk.Frame(self.notebook)
        self.notebook.add(dashboard, text="Dashboard")
        
        # Split into left control panel and right display area
        control_frame = ttk.Frame(dashboard, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        display_frame = ttk.Frame(dashboard)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel
        ttk.Label(control_frame, text="Simulation Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Days to simulate
        days_frame = ttk.Frame(control_frame)
        days_frame.pack(fill=tk.X, pady=5)
        ttk.Label(days_frame, text="Simulation Days:").pack(side=tk.LEFT)
        self.days_var = tk.IntVar(value=100)
        ttk.Spinbox(days_frame, from_=1, to=1000, textvariable=self.days_var, width=10).pack(side=tk.RIGHT)
        
        # Initialize button
        ttk.Button(control_frame, text="Initialize Model", command=self._initialize_model).pack(fill=tk.X, pady=5)
        
        # Simulation control buttons
        sim_buttons = ttk.Frame(control_frame)
        sim_buttons.pack(fill=tk.X, pady=5)
        ttk.Button(sim_buttons, text="Run", command=self._run_simulation).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sim_buttons, text="Pause", command=self._pause_simulation).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sim_buttons, text="Resume", command=self._resume_simulation).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sim_buttons, text="Stop", command=self._stop_simulation).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Status display
        status_frame = ttk.LabelFrame(control_frame, text="Simulation Status")
        status_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.time_var = tk.StringVar(value="Day: 0")
        ttk.Label(status_frame, textvariable=self.time_var).pack(anchor=tk.W, pady=2)
        
        self.cases_var = tk.StringVar(value="Cases: 0")
        ttk.Label(status_frame, textvariable=self.cases_var).pack(anchor=tk.W, pady=2)
        
        self.recovered_var = tk.StringVar(value="Recovered: 0")
        ttk.Label(status_frame, textvariable=self.recovered_var).pack(anchor=tk.W, pady=2)
        
        self.deaths_var = tk.StringVar(value="Deaths: 0")
        ttk.Label(status_frame, textvariable=self.deaths_var).pack(anchor=tk.W, pady=2)
        
        self.r_effective_var = tk.StringVar(value="R_effective: 0")
        ttk.Label(status_frame, textvariable=self.r_effective_var).pack(anchor=tk.W, pady=2)
        
        # Intervention controls
        intervention_frame = ttk.LabelFrame(control_frame, text="Intervention Controls")
        intervention_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Use grid layout for better control
        # Define the label width to ensure consistency
        label_width = 15  # Adjust this value as needed
        
        # Social distancing slider
        sd_frame = ttk.Frame(intervention_frame)
        sd_frame.pack(fill=tk.X, pady=5)
        sd_frame.columnconfigure(1, weight=1)  # Make slider column expandable
        
        sd_label = ttk.Label(sd_frame, text="Social Distancing:", width=label_width)
        sd_label.grid(row=0, column=0, sticky="w")
        
        self.sd_var = tk.DoubleVar(value=0.0)
        self.sd_percent = tk.StringVar(value="0%")
        
        sd_slider = ttk.Scale(sd_frame, from_=0, to=1.0, variable=self.sd_var, 
                              command=lambda x: self._update_intervention_display("sd"))
        sd_slider.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        
        percent_label = ttk.Label(sd_frame, textvariable=self.sd_percent, width=5)
        percent_label.grid(row=0, column=2, sticky="e")
        
        # Mask policy slider
        mask_frame = ttk.Frame(intervention_frame)
        mask_frame.pack(fill=tk.X, pady=5)
        mask_frame.columnconfigure(1, weight=1)  # Make slider column expandable
        
        mask_label = ttk.Label(mask_frame, text="Mask Policy:", width=label_width)
        mask_label.grid(row=0, column=0, sticky="w")
        
        self.mask_var = tk.DoubleVar(value=0.0)
        self.mask_percent = tk.StringVar(value="0%")
        
        mask_slider = ttk.Scale(mask_frame, from_=0, to=1.0, variable=self.mask_var,
                               command=lambda x: self._update_intervention_display("mask"))
        mask_slider.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        
        percent_label = ttk.Label(mask_frame, textvariable=self.mask_percent, width=5)
        percent_label.grid(row=0, column=2, sticky="e")
        
        # Vaccination policy slider
        vax_frame = ttk.Frame(intervention_frame)
        vax_frame.pack(fill=tk.X, pady=5)
        vax_frame.columnconfigure(1, weight=1)  # Make slider column expandable
        
        vax_label = ttk.Label(vax_frame, text="Vaccination Policy:", width=label_width)
        vax_label.grid(row=0, column=0, sticky="w")
        
        self.vax_var = tk.DoubleVar(value=0.0)
        self.vax_percent = tk.StringVar(value="0%")
        
        vax_slider = ttk.Scale(vax_frame, from_=0, to=1.0, variable=self.vax_var,
                              command=lambda x: self._update_intervention_display("vax"))
        vax_slider.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        
        percent_label = ttk.Label(vax_frame, textvariable=self.vax_percent, width=5)
        percent_label.grid(row=0, column=2, sticky="e")
        
        # Output log
        log_frame = ttk.LabelFrame(control_frame, text="Output Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        self.log_text = ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Display area
        # Create plots with matplotlib
        self.fig = Figure(figsize=(8, 8))
        
        # Epidemic curve subplot
        self.ax1 = self.fig.add_subplot(311)
        self.ax1.set_title("Epidemic Curves")
        self.ax1.set_ylabel("Number of individuals")
        self.ax1.grid(True)
        
        # R effective subplot
        self.ax2 = self.fig.add_subplot(312)
        self.ax2.set_title("Effective Reproduction Number")
        self.ax2.set_ylabel("R effective")
        self.ax2.axhline(y=1, color='k', linestyle='--')
        self.ax2.grid(True)
        
        # Information and policy subplot
        self.ax3 = self.fig.add_subplot(313)
        self.ax3.set_title("Information and Policy Measures")
        self.ax3.set_xlabel("Time (days)")
        self.ax3.set_ylabel("Level")
        self.ax3.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(display_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
    
        
    def _create_parameters_tab(self):
        """Create the parameters configuration tab."""
        params_tab = ttk.Frame(self.notebook)
        self.notebook.add(params_tab, text="Parameters")
        
        # Create scrollable frame for parameters
        canvas = tk.Canvas(params_tab)
        scrollbar = ttk.Scrollbar(params_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Simulation parameters
        sim_frame = ttk.LabelFrame(scrollable_frame, text="Simulation Parameters")
        sim_frame.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        
        # Population size
        pop_frame = ttk.Frame(sim_frame)
        pop_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pop_frame, text="Population Size:").pack(side=tk.LEFT)
        self.pop_size_var = tk.IntVar(value=self.params.population_size)
        ttk.Spinbox(pop_frame, from_=100, to=1000000, textvariable=self.pop_size_var, width=10).pack(side=tk.RIGHT)
        
        # Time step
        ts_frame = ttk.Frame(sim_frame)
        ts_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ts_frame, text="Time Step (days):").pack(side=tk.LEFT)
        self.time_step_var = tk.DoubleVar(value=self.params.time_step)
        ttk.Spinbox(ts_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.time_step_var, width=10).pack(side=tk.RIGHT)
        
        # Disease parameters
        disease_frame = ttk.LabelFrame(scrollable_frame, text="Disease Parameters")
        disease_frame.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        
        # Base transmission rate
        tr_frame = ttk.Frame(disease_frame)
        tr_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tr_frame, text="Base Transmission Rate:").pack(side=tk.LEFT)
        self.trans_rate_var = tk.DoubleVar(value=self.params.base_transmission_rate)
        ttk.Spinbox(tr_frame, from_=0.01, to=2.0, increment=0.01, textvariable=self.trans_rate_var, width=10).pack(side=tk.RIGHT)
        
        # Base incubation rate
        ir_frame = ttk.Frame(disease_frame)
        ir_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ir_frame, text="Base Incubation Rate:").pack(side=tk.LEFT)
        self.incubation_rate_var = tk.DoubleVar(value=self.params.base_incubation_rate)
        ttk.Spinbox(ir_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.incubation_rate_var, width=10).pack(side=tk.RIGHT)
        
        # Base recovery rate
        rr_frame = ttk.Frame(disease_frame)
        rr_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rr_frame, text="Base Recovery Rate:").pack(side=tk.LEFT)
        self.recovery_rate_var = tk.DoubleVar(value=self.params.base_recovery_rate)
        ttk.Spinbox(rr_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.recovery_rate_var, width=10).pack(side=tk.RIGHT)
        
        # Base mortality rate
        mr_frame = ttk.Frame(disease_frame)
        mr_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mr_frame, text="Base Mortality Rate:").pack(side=tk.LEFT)
        self.mortality_rate_var = tk.DoubleVar(value=self.params.base_mortality_rate)
        ttk.Spinbox(mr_frame, from_=0.001, to=0.5, increment=0.001, textvariable=self.mortality_rate_var, width=10).pack(side=tk.RIGHT)
        
        # Asymptomatic fraction
        af_frame = ttk.Frame(disease_frame)
        af_frame.pack(fill=tk.X, pady=2)
        ttk.Label(af_frame, text="Asymptomatic Fraction:").pack(side=tk.LEFT)
        self.asymp_frac_var = tk.DoubleVar(value=self.params.asymptomatic_fraction)
        ttk.Spinbox(af_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.asymp_frac_var, width=10).pack(side=tk.RIGHT)
        
        # Asymptomatic transmission factor
        atf_frame = ttk.Frame(disease_frame)
        atf_frame.pack(fill=tk.X, pady=2)
        ttk.Label(atf_frame, text="Asymptomatic Transmission Factor:").pack(side=tk.LEFT)
        self.asymp_trans_var = tk.DoubleVar(value=self.params.asymptomatic_transmission_factor)
        ttk.Spinbox(atf_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.asymp_trans_var, width=10).pack(side=tk.RIGHT)
        
        # Pathogen evolution parameters
        evolution_frame = ttk.LabelFrame(scrollable_frame, text="Pathogen Evolution Parameters")
        evolution_frame.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        
        # Mutation rate
        mut_frame = ttk.Frame(evolution_frame)
        mut_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mut_frame, text="Mutation Rate:").pack(side=tk.LEFT)
        self.mutation_rate_var = tk.DoubleVar(value=self.params.mutation_rate)
        ttk.Spinbox(mut_frame, from_=0.0001, to=0.1, increment=0.0001, textvariable=self.mutation_rate_var, width=10).pack(side=tk.RIGHT)
        
        # Max transmissibility
        max_trans_frame = ttk.Frame(evolution_frame)
        max_trans_frame.pack(fill=tk.X, pady=2)
        ttk.Label(max_trans_frame, text="Maximum Transmissibility:").pack(side=tk.LEFT)
        self.max_trans_var = tk.DoubleVar(value=self.params.max_transmissibility)
        ttk.Spinbox(max_trans_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.max_trans_var, width=10).pack(side=tk.RIGHT)
        
        # Evolution noise scale
        noise_frame = ttk.Frame(evolution_frame)
        noise_frame.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, text="Evolution Noise Scale:").pack(side=tk.LEFT)
        self.noise_scale_var = tk.DoubleVar(value=self.params.evolution_noise_scale)
        ttk.Spinbox(noise_frame, from_=0.01, to=0.5, increment=0.01, textvariable=self.noise_scale_var, width=10).pack(side=tk.RIGHT)
        
        # Behavior parameters
        behavior_frame = ttk.LabelFrame(scrollable_frame, text="Behavior Parameters")
        behavior_frame.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        
        # Behavior adaptation rate
        ba_frame = ttk.Frame(behavior_frame)
        ba_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ba_frame, text="Behavior Adaptation Rate:").pack(side=tk.LEFT)
        self.behavior_adapt_var = tk.DoubleVar(value=self.params.behavior_adaption_rate)
        ttk.Spinbox(ba_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.behavior_adapt_var, width=10).pack(side=tk.RIGHT)
        
        # Information generation rate
        ig_frame = ttk.Frame(behavior_frame)
        ig_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ig_frame, text="Information Generation Rate:").pack(side=tk.LEFT)
        self.info_gen_var = tk.DoubleVar(value=self.params.information_generation_rate)
        ttk.Spinbox(ig_frame, from_=0.01, to=2.0, increment=0.01, textvariable=self.info_gen_var, width=10).pack(side=tk.RIGHT)
        
        # Information decay rate
        id_frame = ttk.Frame(behavior_frame)
        id_frame.pack(fill=tk.X, pady=2)
        ttk.Label(id_frame, text="Information Decay Rate:").pack(side=tk.LEFT)
        self.info_decay_var = tk.DoubleVar(value=self.params.information_decay_rate)
        ttk.Spinbox(id_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.info_decay_var, width=10).pack(side=tk.RIGHT)
        
        # Media influence factor
        mi_frame = ttk.Frame(behavior_frame)
        mi_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mi_frame, text="Media Influence Factor:").pack(side=tk.LEFT)
        self.media_influence_var = tk.DoubleVar(value=self.params.media_influence_factor)
        ttk.Spinbox(mi_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.media_influence_var, width=10).pack(side=tk.RIGHT)
        
        # Apply button
        ttk.Button(scrollable_frame, text="Apply Parameters", command=self._apply_parameters).pack(pady=10)
        
    def _create_network_tab(self):
        """Create the network visualization tab."""
        network_tab = ttk.Frame(self.notebook)
        self.notebook.add(network_tab, text="Network")
        
        # Network controls on the left
        control_frame = ttk.Frame(network_tab, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Network visualization on the right
        viz_frame = ttk.Frame(network_tab)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Network scale selection
        scale_frame = ttk.LabelFrame(control_frame, text="Network Scale")
        scale_frame.pack(fill=tk.X, pady=5)
        
        self.network_scale_var = tk.StringVar(value="Effective")
        ttk.Radiobutton(scale_frame, text="Micro (Household)", variable=self.network_scale_var, value="Micro", command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(scale_frame, text="Meso (Community)", variable=self.network_scale_var, value="Meso", command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(scale_frame, text="Macro (Regional)", variable=self.network_scale_var, value="Macro", command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(scale_frame, text="Effective (Combined)", variable=self.network_scale_var, value="Effective", command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        
        # Visualization options
        viz_options = ttk.LabelFrame(control_frame, text="Visualization Options")
        viz_options.pack(fill=tk.X, pady=5)
        
        self.show_nodes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_options, text="Show Nodes", variable=self.show_nodes_var, command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        
        self.color_by_var = tk.StringVar(value="Health State")
        ttk.Label(viz_options, text="Color Nodes By:").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(viz_options, text="Health State", variable=self.color_by_var, value="Health State", command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(viz_options, text="Age", variable=self.color_by_var, value="Age", command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(viz_options, text="Behavior", variable=self.color_by_var, value="Behavior", command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        
        self.show_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_options, text="Show Edges", variable=self.show_edges_var, command=self._update_network_viz).pack(anchor=tk.W, pady=2)
        
        self.edge_threshold_var = tk.DoubleVar(value=0.1)
        edge_frame = ttk.Frame(viz_options)
        edge_frame.pack(fill=tk.X, pady=2)
        ttk.Label(edge_frame, text="Edge Weight Threshold:").pack(side=tk.LEFT)
        ttk.Scale(edge_frame, from_=0, to=1.0, variable=self.edge_threshold_var, command=self._update_network_viz).pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Network statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Network Statistics")
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.network_stats_text = ScrolledText(stats_frame, height=10, width=30)
        self.network_stats_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figure for network visualization
        self.network_fig = Figure(figsize=(6, 6))
        self.network_ax = self.network_fig.add_subplot(111)
        
        self.network_canvas = FigureCanvasTkAgg(self.network_fig, master=viz_frame)
        self.network_canvas.draw()
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        network_toolbar_frame = ttk.Frame(viz_frame)
        network_toolbar_frame.pack(fill=tk.X)
        network_toolbar = NavigationToolbar2Tk(self.network_canvas, network_toolbar_frame)
        network_toolbar.update()
        
    def _create_behavior_tab(self):
        """Create the behavior analysis tab."""
        behavior_tab = ttk.Frame(self.notebook)
        self.notebook.add(behavior_tab, text="Behavior")
        
        # Split into control panel and visualization
        control_frame = ttk.Frame(behavior_tab, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        viz_frame = ttk.Frame(behavior_tab)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Behavior visualization options
        viz_options = ttk.LabelFrame(control_frame, text="Visualization Options")
        viz_options.pack(fill=tk.X, pady=5)
        
        self.behavior_viz_type_var = tk.StringVar(value="Time Series")
        ttk.Radiobutton(viz_options, text="Time Series", variable=self.behavior_viz_type_var, value="Time Series", command=self._update_behavior_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(viz_options, text="Distribution", variable=self.behavior_viz_type_var, value="Distribution", command=self._update_behavior_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(viz_options, text="Correlation with Cases", variable=self.behavior_viz_type_var, value="Correlation", command=self._update_behavior_viz).pack(anchor=tk.W, pady=2)
        
        # Behavior dimension selection
        dim_frame = ttk.LabelFrame(control_frame, text="Behavior Dimension")
        dim_frame.pack(fill=tk.X, pady=5)
        
        self.behavior_dim_var = tk.IntVar(value=0)
        ttk.Radiobutton(dim_frame, text="Caution Level", variable=self.behavior_dim_var, value=0, command=self._update_behavior_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(dim_frame, text="Mask Compliance", variable=self.behavior_dim_var, value=1, command=self._update_behavior_viz).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(dim_frame, text="Vaccination Willingness", variable=self.behavior_dim_var, value=2, command=self._update_behavior_viz).pack(anchor=tk.W, pady=2)
        
        # Individual selection for time series
        indiv_frame = ttk.LabelFrame(control_frame, text="Individual Selection")
        indiv_frame.pack(fill=tk.X, pady=5)
        
        self.show_individual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(indiv_frame, text="Show Individual Trajectories", variable=self.show_individual_var, command=self._update_behavior_viz).pack(anchor=tk.W, pady=2)
        
        self.num_individuals_var = tk.IntVar(value=5)
        num_indiv_frame = ttk.Frame(indiv_frame)
        num_indiv_frame.pack(fill=tk.X, pady=2)
        ttk.Label(num_indiv_frame, text="Number of Individuals:").pack(side=tk.LEFT)
        ttk.Spinbox(num_indiv_frame, from_=1, to=20, textvariable=self.num_individuals_var, width=5).pack(side=tk.RIGHT)
        
        # Create matplotlib figure for behavior visualization
        self.behavior_fig = Figure(figsize=(8, 6))
        self.behavior_ax = self.behavior_fig.add_subplot(111)
        
        self.behavior_canvas = FigureCanvasTkAgg(self.behavior_fig, master=viz_frame)
        self.behavior_canvas.draw()
        self.behavior_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        behavior_toolbar_frame = ttk.Frame(viz_frame)
        behavior_toolbar_frame.pack(fill=tk.X)
        behavior_toolbar = NavigationToolbar2Tk(self.behavior_canvas, behavior_toolbar_frame)
        behavior_toolbar.update()
        
    def _create_data_tab(self):
        """Create the data assimilation tab."""
        data_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="Data")
        
        # Split into control panel and visualization
        control_frame = ttk.Frame(data_tab, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        viz_frame = ttk.Frame(data_tab)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Data import controls
        import_frame = ttk.LabelFrame(control_frame, text="Import Data")
        import_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(import_frame, text="Import CSV", command=self._import_csv).pack(fill=tk.X, pady=2)
        ttk.Button(import_frame, text="Import JSON", command=self._import_json).pack(fill=tk.X, pady=2)
        
        # Data series selection
        series_frame = ttk.LabelFrame(control_frame, text="Data Series")
        series_frame.pack(fill=tk.X, pady=5)
        
        self.series_listbox = tk.Listbox(series_frame, height=6)
        self.series_listbox.pack(fill=tk.BOTH, expand=True, pady=2)
        self.series_listbox.insert(tk.END, "No data imported")
        
        # Assimilation controls
        assim_frame = ttk.LabelFrame(control_frame, text="Assimilation Controls")
        assim_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(assim_frame, text="Assimilate Selected Data", command=self._assimilate_data).pack(fill=tk.X, pady=2)
        
        self.auto_assimilate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(assim_frame, text="Auto-assimilate during simulation", variable=self.auto_assimilate_var).pack(anchor=tk.W, pady=2)
        
        # Ensemble settings
        ensemble_frame = ttk.LabelFrame(control_frame, text="Ensemble Settings")
        ensemble_frame.pack(fill=tk.X, pady=5)
        
        # Ensemble size
        es_frame = ttk.Frame(ensemble_frame)
        es_frame.pack(fill=tk.X, pady=2)
        ttk.Label(es_frame, text="Ensemble Size:").pack(side=tk.LEFT)
        self.ensemble_size_var = tk.IntVar(value=self.params.ensemble_size)
        ttk.Spinbox(es_frame, from_=5, to=100, textvariable=self.ensemble_size_var, width=5).pack(side=tk.RIGHT)
        
        # Observation noise
        on_frame = ttk.Frame(ensemble_frame)
        on_frame.pack(fill=tk.X, pady=2)
        ttk.Label(on_frame, text="Observation Noise SD:").pack(side=tk.LEFT)
        self.obs_noise_var = tk.DoubleVar(value=self.params.observation_noise_sd)
        ttk.Spinbox(on_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.obs_noise_var, width=5).pack(side=tk.RIGHT)
        
        ttk.Button(ensemble_frame, text="Reset Ensemble", command=self._reset_ensemble).pack(fill=tk.X, pady=2)
        
        # Create matplotlib figure for data visualization
        self.data_fig = Figure(figsize=(8, 6))
        self.data_ax = self.data_fig.add_subplot(111)
        
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, master=viz_frame)
        self.data_canvas.draw()
        self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        data_toolbar_frame = ttk.Frame(viz_frame)
        data_toolbar_frame.pack(fill=tk.X)
        data_toolbar = NavigationToolbar2Tk(self.data_canvas, data_toolbar_frame)
        data_toolbar.update()
        
    # Menu command methods
    def _new_scenario(self):
        """Create a new scenario."""
        if messagebox.askyesno("New Scenario", "This will reset all parameters and model state. Continue?"):
            self.params = Parameters()
            self.model = MANEpidemicModel(self.params)
            self.scenario_manager.set_model(self.model)
            self.scenario_manager.current_scenario_name = "Untitled Scenario"
            
            # Update UI
            self._apply_parameters()
            self._update_plots()
            self._update_status("Created new scenario")
            
    def _open_scenario(self):
        """Open a saved scenario."""
        filename = filedialog.askopenfilename(
            title="Open Scenario",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                scenario = self.scenario_manager.load_scenario(filename)
                new_model, new_params = self.scenario_manager.apply_scenario(scenario)
                
                if new_model and new_params:
                    self.model = new_model
                    self.params = new_params
                    self.scenario_manager.set_model(self.model)
                    
                    # Update UI
                    self._apply_parameters()
                    self._update_plots()
                    self._update_status(f"Loaded scenario: {self.scenario_manager.current_scenario_name}")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load scenario: {e}")
                
    def _save_scenario(self):
        """Save the current scenario."""
        if self.scenario_manager.current_scenario_name == "Untitled Scenario":
            self._save_scenario_as()
        else:
            # Generate filename from scenario name
            filename = f"{self.scenario_manager.current_scenario_name.replace(' ', '_')}.pkl"
            
            try:
                self.scenario_manager.save_scenario(filename)
                self._update_status(f"Saved scenario to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save scenario: {e}")
                
    def _save_scenario_as(self):
        """Save the current scenario with a new name."""
        filename = filedialog.asksaveasfilename(
            title="Save Scenario As",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            # Get scenario name
            name = os.path.splitext(os.path.basename(filename))[0].replace('_', ' ')
            
            try:
                self.scenario_manager.save_scenario(filename, name=name)
                self._update_status(f"Saved scenario as {name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save scenario: {e}")
                
    def _export_results(self):
        """Export simulation results to CSV."""
        if not self.model.initialized or not hasattr(self.model, 'case_counts') or not self.model.case_counts:
            messagebox.showerror("Error", "No simulation results to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Create DataFrame with time series data
                data = {
                    'Time': [t for t, _ in self.model.case_counts[HealthState.SUSCEPTIBLE]]
                }
                
                for state in HealthState:
                    counts = self.model.case_counts[state]
                    data[state.name] = [c for _, c in counts]
                    
                # Add R effective
                data['R_effective'] = [r for _, r in self.model.R_effective]
                
                # Create DataFrame and export
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                
                self._update_status(f"Exported results to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")
                
    # Simulation control methods
    def _initialize_model(self):
        """Initialize the epidemic model."""
        try:
            # Apply current parameter values
            self._apply_parameters()
            
            # Initialize model
            self.model.initialize()
            
            # Update plots
            self._update_plots()
            self._update_status("Model initialized")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize model: {e}")
            
    def _run_simulation(self):
        """Run the simulation."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            messagebox.showinfo("Simulation Running", "Simulation is already running")
            return
            
        days = self.days_var.get()
        
        try:
            # Apply parameters if they've changed
            self._apply_parameters()
            
            # Initialize model if not already done
            if not self.model.initialized:
                self.model.initialize()
                
            # Create and start simulation thread
            self.simulation_thread = ModelThread(
                self.model, 
                self.params, 
                days,
                update_interval=1.0,  # Update UI every day
                callback=self._update_plots
            )
            self.simulation_thread.start()
            
            self._update_status(f"Running simulation for {days} days")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run simulation: {e}")
            
    def _pause_simulation(self):
        """Pause the simulation."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            if not self.simulation_thread.paused:
                self.simulation_thread.pause()
                self._update_status("Simulation paused")
            else:
                messagebox.showinfo("Already Paused", "Simulation is already paused")
        else:
            messagebox.showinfo("No Simulation", "No simulation is currently running")
            
    def _resume_simulation(self):
        """Resume the paused simulation."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            if self.simulation_thread.paused:
                self.simulation_thread.resume()
                self._update_status("Simulation resumed")
            else:
                messagebox.showinfo("Not Paused", "Simulation is not paused")
        else:
            messagebox.showinfo("No Simulation", "No simulation is currently running")
            
    def _stop_simulation(self):
        """Stop the simulation."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.stop()
            self._update_status("Simulation stopped")
        else:
            messagebox.showinfo("No Simulation", "No simulation is currently running")
            
    # Analysis methods
    def _plot_results(self):
        """Generate and display plots of simulation results."""
        if not self.model.initialized or not hasattr(self.model, 'case_counts') or not self.model.case_counts:
            messagebox.showerror("Error", "No simulation results to plot")
            return
            
        # Update the plots
        self._update_plots()
        
        # Switch to dashboard tab
        self.notebook.select(0)
        
    def _generate_report(self):
        """Generate a comprehensive report of simulation results."""
        if not self.model.initialized or not hasattr(self.model, 'case_counts') or not self.model.case_counts:
            messagebox.showerror("Error", "No simulation results for report")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Generate basic HTML report
                with open(filename, 'w') as f:
                    f.write("<html>\n<head>\n")
                    f.write("<title>Epidemic Model Simulation Report</title>\n")
                    f.write("<style>body { font-family: Arial, sans-serif; margin: 30px; }\n")
                    f.write("h1, h2 { color: #2c3e50; }\n")
                    f.write("table { border-collapse: collapse; width: 100%; }\n")
                    f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
                    f.write("th { background-color: #f2f2f2; }\n")
                    f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
                    f.write("</style>\n</head>\n<body>\n")
                    
                    # Header
                    f.write(f"<h1>Epidemic Model Simulation Report</h1>\n")
                    f.write(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
                    f.write(f"<p>Scenario: {self.scenario_manager.current_scenario_name}</p>\n")
                    
                    # Summary statistics
                    f.write("<h2>Summary Statistics</h2>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
                    
                    # Get latest counts
                    latest_time = self.model.time
                    latest_counts = {state: counts[-1][1] for state, counts in self.model.case_counts.items()}
                    total_pop = sum(latest_counts.values())
                    
                    f.write(f"<tr><td>Simulation Duration</td><td>{latest_time:.1f} days</td></tr>\n")
                    f.write(f"<tr><td>Population Size</td><td>{total_pop}</td></tr>\n")
                    
                    for state in HealthState:
                        count = latest_counts[state]
                        percentage = 100 * count / total_pop if total_pop > 0 else 0
                        f.write(f"<tr><td>{state.name.capitalize()}</td><td>{count} ({percentage:.1f}%)</td></tr>\n")
                    
                    # Peak statistics
                    infectious_counts = [c for _, c in self.model.case_counts[HealthState.INFECTIOUS]]
                    peak_infectious = max(infectious_counts)
                    peak_day = self.model.case_counts[HealthState.INFECTIOUS][infectious_counts.index(peak_infectious)][0]
                    
                    f.write(f"<tr><td>Peak Infectious Cases</td><td>{peak_infectious}</td></tr>\n")
                    f.write(f"<tr><td>Peak Day</td><td>{peak_day:.1f}</td></tr>\n")
                    
                    if self.model.R_effective:
                        latest_r = self.model.R_effective[-1][1]
                        f.write(f"<tr><td>Current R Effective</td><td>{latest_r:.2f}</td></tr>\n")
                    
                    f.write("</table>\n")
                    
                    # Parameters
                    f.write("<h2>Model Parameters</h2>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Parameter</th><th>Value</th></tr>\n")
                    
                    for attr, value in vars(self.params).items():
                        if isinstance(value, dict):
                            f.write(f"<tr><td>{attr}</td><td>{str(value)}</td></tr>\n")
                        else:
                            f.write(f"<tr><td>{attr}</td><td>{value}</td></tr>\n")
                    
                    f.write("</table>\n")
                    
                    # End
                    f.write("</body>\n</html>")
                
                # Open the report in browser
                import webbrowser
                webbrowser.open(filename)
                
                self._update_status(f"Generated report: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate report: {e}")
                
    # Help methods
    def _show_user_guide(self):
        """Show the user guide."""
        guide_text = """
        Multi-scale Adaptive Network (MAN) Epidemic Model
        =================================================
        
        User Guide
        
        1. Getting Started
           - Create a new scenario or open an existing one from the File menu
           - Configure model parameters in the Parameters tab
           - Initialize the model using the button on the Dashboard
        
        2. Running Simulations
           - Set the number of days to simulate
           - Click "Run" to start the simulation
           - Use Pause/Resume/Stop to control the simulation
           - Watch the epidemic curves and statistics update in real-time
        
        3. Analyzing Results
           - The Dashboard shows the main epidemic curves
           - The Network tab visualizes contact networks
           - The Behavior tab analyzes behavioral dynamics
           - The Data tab allows importing and assimilating real-world data
        
        4. Interventions
           - Use the sliders on the dashboard to implement policy interventions
           - Observe how interventions affect behavioral dynamics and epidemic spread
        
        5. Saving and Exporting
           - Save your scenario to continue work later
           - Export results as CSV for further analysis
           - Generate HTML reports summarizing results
        """
        
        guide_window = tk.Toplevel(self)
        guide_window.title("User Guide")
        guide_window.geometry("600x500")
        
        text = ScrolledText(guide_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, guide_text)
        text.configure(state='disabled')
        
    def _show_about(self):
        """Show information about the application."""
        about_text = """
        Mitra: Multi-scale Adaptive Network (MAN) Epidemic Model
        =================================================
        
        Version 1.0
        
        An advanced epidemic modeling platform incorporating:
        
        - Hierarchical network structure
        - Adaptive parameters
        - Behavior-pathogen feedback loop
        - Computational efficiency through selective resolution
        - Bayesian data assimilation
        
        Developed as a novel approach to epidemic modeling that addresses
        limitations in traditional compartmental and agent-based models.
        
        Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)

        Copyright (c) 2025 Rishabh Mishra

        This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

        To view a copy of this license, visit:
        https://creativecommons.org/licenses/by-nc/4.0/
        or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

        You are free to:
        - Share — copy and redistribute the material in any medium or format
        - Adapt — remix, transform, and build upon the material

        Under the following terms:
        - Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. 
          You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
        - NonCommercial — You may not use the material for commercial purposes.

        No additional restrictions — You may not apply legal terms or technological measures that legally restrict others 
        from doing anything the license permits.

        Notices:
        You do not have to comply with the license for elements of the material in the public domain 
        or where your use is permitted by an applicable exception or limitation.

        No warranties are given. The license may not give you all of the permissions necessary for your intended use. 
        For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

        ## DISCLAIMER OF LIABILITY

        This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors disclaim all warranties, including, but not limited to, any implied warranties of merchantability, fitness for a particular purpose, and non-infringement.
        In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
        This software is intended for research and educational purposes only. Any decisions made based on the outputs of this model are the sole responsibility of the user. The software developers are not responsible for any actions taken based on the software's results or interpretations thereof.
        This epidemic model is a simplified representation of complex real-world systems and should not be the sole basis for public health decision-making.
        
        """
        
        messagebox.showinfo("About", about_text)
        
    # Utility methods
    def _process_output_queue(self):
        """Process and display stdout output in the log window."""
        try:
            while True:
                message = self.output_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
                self.output_queue.task_done()
        except queue.Empty:
            pass
        finally:
            # Schedule to run again
            self.after(100, self._process_output_queue)
            
    def _update_status(self, message):
        """Update the status bar message."""
        self.status_var.set(message)
        print(message)
        
    def _apply_parameters(self):
        """Apply parameter values from UI to the model."""
        # Create new parameters object
        new_params = Parameters()
        
        # Simulation parameters
        new_params.population_size = self.pop_size_var.get()
        new_params.time_step = self.time_step_var.get()
        
        # Disease parameters
        new_params.base_transmission_rate = self.trans_rate_var.get()
        new_params.base_incubation_rate = self.incubation_rate_var.get()
        new_params.base_recovery_rate = self.recovery_rate_var.get()
        new_params.base_mortality_rate = self.mortality_rate_var.get()
        new_params.asymptomatic_fraction = self.asymp_frac_var.get()
        new_params.asymptomatic_transmission_factor = self.asymp_trans_var.get()
        
        # Pathogen evolution parameters
        new_params.mutation_rate = self.mutation_rate_var.get()
        new_params.max_transmissibility = self.max_trans_var.get()
        new_params.evolution_noise_scale = self.noise_scale_var.get()
        
        # Behavior parameters
        new_params.behavior_adaption_rate = self.behavior_adapt_var.get()
        new_params.information_generation_rate = self.info_gen_var.get()
        new_params.information_decay_rate = self.info_decay_var.get()
        new_params.media_influence_factor = self.media_influence_var.get()
        
        # Data assimilation parameters
        new_params.ensemble_size = self.ensemble_size_var.get()
        new_params.observation_noise_sd = self.obs_noise_var.get()
        
        # Update model with new parameters
        self.params = new_params
        
        # If model is not yet initialized, create a new one
        if not hasattr(self.model, 'initialized') or not self.model.initialized:
            self.model = MANEpidemicModel(self.params)
            self.scenario_manager.set_model(self.model)
        else:
            # If model is already initialized, update its parameters
            self.model.params = self.params
            
        self._update_status("Applied parameter changes")
        
    def _update_plots(self):
        """Update all plots with current model state."""
        if not self.model.initialized:
            return
            
        # Update status variables
        self.time_var.set(f"Day: {self.model.time:.1f}")
        
        # Get latest counts if available
        if hasattr(self.model, 'case_counts') and self.model.case_counts:
            for state, counts in self.model.case_counts.items():
                if counts:
                    latest_count = counts[-1][1]
                    
                    if state == HealthState.INFECTIOUS:
                        self.cases_var.set(f"Cases: {latest_count}")
                    elif state == HealthState.RECOVERED:
                        self.recovered_var.set(f"Recovered: {latest_count}")
                    elif state == HealthState.DECEASED:
                        self.deaths_var.set(f"Deaths: {latest_count}")
            
        # Update R_effective
        if hasattr(self.model, 'R_effective') and self.model.R_effective:
            latest_r = self.model.R_effective[-1][1]
            self.r_effective_var.set(f"R_effective: {latest_r:.2f}")
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Epidemic curves
        self.ax1.set_title("Epidemic Curves")
        self.ax1.set_ylabel("Number of individuals")
        
        if hasattr(self.model, 'case_counts') and self.model.case_counts:
            for state in [HealthState.SUSCEPTIBLE, HealthState.EXPOSED, HealthState.INFECTIOUS, 
                         HealthState.ASYMPTOMATIC, HealthState.RECOVERED, HealthState.DECEASED]:
                data = self.model.case_counts[state]
                if data:
                    times, counts = zip(*data)
                    self.ax1.plot(times, counts, label=state.name.capitalize())
                    
            self.ax1.legend()
            self.ax1.grid(True)
            
        # R effective
        self.ax2.set_title("Effective Reproduction Number")
        self.ax2.set_ylabel("R effective")
        self.ax2.axhline(y=1, color='k', linestyle='--')
        
        if hasattr(self.model, 'R_effective') and self.model.R_effective:
            times, r_values = zip(*self.model.R_effective)
            self.ax2.plot(times, r_values, 'r-')
            self.ax2.grid(True)
            
        # Information and policy measures
        self.ax3.set_title("Information and Policy Measures")
        self.ax3.set_xlabel("Time (days)")
        self.ax3.set_ylabel("Level")
        
        if hasattr(self.model, 'behavior_model'):
            if hasattr(self.model.behavior_model, 'info_history') and self.model.behavior_model.info_history:
                info_times, info_values = zip(*self.model.behavior_model.info_history)
                self.ax3.plot(info_times, info_values, 'b-', label='Information')
                
            if hasattr(self.model.behavior_model, 'media_history') and self.model.behavior_model.media_history:
                media_times, media_values = zip(*self.model.behavior_model.media_history)
                self.ax3.plot(media_times, media_values, 'g-', label='Media')
                
            if hasattr(self.model.behavior_model, 'policy_history') and self.model.behavior_model.policy_history:
                policy_times, policy_values = zip(*self.model.behavior_model.policy_history)
                policy_values = np.array(policy_values)
                
                self.ax3.plot(policy_times, policy_values[:, 0], 'm--', label='Distancing Policy')
                self.ax3.plot(policy_times, policy_values[:, 1], 'c--', label='Mask Policy')
                self.ax3.plot(policy_times, policy_values[:, 2], 'y--', label='Vaccination Policy')
                
            self.ax3.legend()
            self.ax3.grid(True)
            
        # Redraw the figure
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update other tabs if they're visible
        current_tab = self.notebook.index(self.notebook.select())
        
        if current_tab == 1:  # Parameters tab
            # Nothing to update in parameters tab
            pass
        elif current_tab == 2:  # Network tab
            self._update_network_viz()
        elif current_tab == 3:  # Behavior tab
            self._update_behavior_viz()
        elif current_tab == 4:  # Data tab
            self._update_data_viz()
            
    def _update_interventions(self, *args):
        """Update policy interventions based on slider values."""
        if not hasattr(self.model, 'behavior_model'):
            return
            
        # Get values from sliders
        sd_value = self.sd_var.get()
        mask_value = self.mask_var.get()
        vax_value = self.vax_var.get()
        
        # Apply to model
        self.model.behavior_model.policy_interventions = np.array([sd_value, mask_value, vax_value])
        
        # Add to history
        self.model.behavior_model.policy_history.append((self.model.time, self.model.behavior_model.policy_interventions.copy()))
        
        # Update status message with percentages
        self._update_status(f"Updated interventions: SD={int(sd_value*100)}%, Mask={int(mask_value*100)}%, Vax={int(vax_value*100)}%")
        
    def _update_intervention_display(self, intervention_type):
        """Update the intervention display values and apply changes."""
        # Update percentage displays
        if intervention_type == "sd":
            value = self.sd_var.get()
            self.sd_percent.set(f"{int(value * 100)}%")
        elif intervention_type == "mask":
            value = self.mask_var.get()
            self.mask_percent.set(f"{int(value * 100)}%")
        elif intervention_type == "vax":
            value = self.vax_var.get()
            self.vax_percent.set(f"{int(value * 100)}%")
            
        # Apply interventions to model
        self._update_interventions()
        
    def _update_network_viz(self, *args):
        """Update the network visualization."""
        if not self.model.initialized or not hasattr(self.model, 'contact_network'):
            return
            
        # Clear previous plot completely
        self.network_fig.clear()
        self.network_ax = self.network_fig.add_subplot(111)
        
        # Get selected network scale
        scale = self.network_scale_var.get()
        
        if scale == "Micro":
            network = self.model.contact_network.networks[NetworkScale.MICRO]
            title = "Micro-scale Network (Household)"
        elif scale == "Meso":
            network = self.model.contact_network.networks[NetworkScale.MESO]
            title = "Meso-scale Network (Community)"
        elif scale == "Macro":
            network = self.model.contact_network.networks[NetworkScale.MACRO]
            title = "Macro-scale Network (Regional)"
        else:  # Effective
            # For effective network, we need to convert adjacency matrix to graph
            adj = self.model.contact_network.effective_network
            network = nx.from_numpy_array(adj)
            title = "Effective Contact Network"
            
        # Set title
        self.network_ax.set_title(title)
        
        # Compute layout (positions)
        # For stability, use individual locations if available
        pos = {}
        for i, ind in enumerate(self.model.population):
            pos[i] = ind.location
            
        # Node size and color
        node_sizes = []
        node_colors = []
        
        color_by = self.color_by_var.get()
        
        # Track unique health states for legend
        health_states_for_legend = set()
        
        for i, ind in enumerate(self.model.population):
            # Size based on age
            size = 10 + ind.age / 5
            node_sizes.append(size)
            
            # Color based on selection
            if color_by == "Health State":
                if ind.health_state == HealthState.SUSCEPTIBLE:
                    color = 'blue'
                elif ind.health_state == HealthState.EXPOSED:
                    color = 'orange'
                elif ind.health_state == HealthState.INFECTIOUS:
                    color = 'red'
                elif ind.health_state == HealthState.ASYMPTOMATIC:
                    color = 'yellow'
                elif ind.health_state == HealthState.RECOVERED:
                    color = 'green'
                else:  # DECEASED
                    color = 'black'
                    
                health_states_for_legend.add(ind.health_state)
                    
            elif color_by == "Age":
                # Blue (young) to red (old) color scale
                color = plt.cm.plasma(ind.age / 100)
            else:  # Behavior
                # Color based on first behavior dimension (caution)
                color = plt.cm.viridis(ind.behavior[0])
                
            node_colors.append(color)
            
        # Edge filtering and weights
        edge_weights = []
        filtered_edges = []
        threshold = self.edge_threshold_var.get()
        
        for i, j in network.edges():
            weight = network[i][j].get('weight', 1.0)
            if weight >= threshold:
                filtered_edges.append((i, j))
                edge_weights.append(weight * 2)  # Scale for visibility
        
        # Draw the network
        if self.show_nodes_var.get():
            nx.draw_networkx_nodes(
                network, pos, 
                node_size=node_sizes, 
                node_color=node_colors,
                alpha=0.7,
                ax=self.network_ax
            )
            
        if self.show_edges_var.get() and filtered_edges:
            nx.draw_networkx_edges(
                network, pos, 
                edgelist=filtered_edges,
                width=edge_weights,
                alpha=0.3,
                edge_color='gray',
                ax=self.network_ax
            )
        
        # Remove axis
        self.network_ax.set_axis_off()
        
        # Add legend based on coloring selection
        if color_by == "Health State":
            # Create a legend for health states
            legend_elements = []
            color_map = {
                HealthState.SUSCEPTIBLE: ('blue', 'Susceptible'),
                HealthState.EXPOSED: ('orange', 'Exposed'),
                HealthState.INFECTIOUS: ('red', 'Infectious'),
                HealthState.ASYMPTOMATIC: ('yellow', 'Asymptomatic'),
                HealthState.RECOVERED: ('green', 'Recovered'),
                HealthState.DECEASED: ('black', 'Deceased')
            }
            
            # Only add legend entries for health states that exist in the network
            for state in HealthState:
                if state in health_states_for_legend:
                    color, label = color_map[state]
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=10, label=label))
                                          
            self.network_ax.legend(handles=legend_elements, loc='upper right', title="Health States")
            
        elif color_by == "Age":
            # Create a colorbar for age
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 100))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=self.network_ax, orientation='vertical', shrink=0.8)
            cbar.set_label('Age (years)')
            
        else:  # Behavior (Caution level)
            # Create a colorbar for behavior
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=self.network_ax, orientation='vertical', shrink=0.8)
            cbar.set_label('Caution Level')
        
        # Update the plot
        self.network_canvas.draw()
        
        # Update network statistics
        self._update_network_stats(network)
        
    def _update_network_stats(self, network):
        """Update network statistics display."""
        self.network_stats_text.delete(1.0, tk.END)
        
        # Graph properties
        self.network_stats_text.insert(tk.END, f"Nodes: {network.number_of_nodes()}\n")
        self.network_stats_text.insert(tk.END, f"Edges: {network.number_of_edges()}\n")
        
        # Calculate average degree
        degrees = [d for _, d in network.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        self.network_stats_text.insert(tk.END, f"Avg. Degree: {avg_degree:.2f}\n")
        
        # Calculate average clustering coefficient (can be slow for large networks)
        if network.number_of_nodes() < 1000:  # Only for smaller networks
            avg_clustering = nx.average_clustering(network)
            self.network_stats_text.insert(tk.END, f"Avg. Clustering: {avg_clustering:.3f}\n")
            
        # Number of components
        components = list(nx.connected_components(network))
        self.network_stats_text.insert(tk.END, f"Components: {len(components)}\n")
        
        # Largest component size
        if components:
            largest_comp = max(components, key=len)
            self.network_stats_text.insert(tk.END, f"Largest Component: {len(largest_comp)} nodes\n")
            
        # Health state distribution
        if hasattr(self.model, 'population'):
            self.network_stats_text.insert(tk.END, "\nHealth States:\n")
            
            state_counts = {}
            for state in HealthState:
                state_counts[state] = 0
                
            for ind in self.model.population:
                state_counts[ind.health_state] += 1
                
            for state, count in state_counts.items():
                percentage = 100 * count / len(self.model.population)
                self.network_stats_text.insert(tk.END, f"{state.name}: {count} ({percentage:.1f}%)\n")
        
    def _update_behavior_viz(self, *args):
        """Update the behavior visualization."""
        if not self.model.initialized or not hasattr(self.model, 'population'):
            return
            
        # Clear previous plot
        self.behavior_ax.clear()
        
        # Get selected visualization type and dimension
        viz_type = self.behavior_viz_type_var.get()
        dim = self.behavior_dim_var.get()
        dim_names = ["Caution", "Mask Usage", "Vaccination Willingness"]
        
        if viz_type == "Time Series":
            self.behavior_ax.set_title(f"{dim_names[dim]} Over Time")
            self.behavior_ax.set_xlabel("Time (days)")
            self.behavior_ax.set_ylabel(dim_names[dim])
            
            # Calculate population average
            time_points = []
            avg_values = []
            
            if self.model.population and self.model.population[0].behavior_history:
                # Get all unique time points
                for ind in self.model.population:
                    for t, _ in ind.behavior_history:
                        if t not in time_points:
                            time_points.append(t)
                            
                time_points.sort()
                
                # Calculate average at each time point
                for t in time_points:
                    values = []
                    for ind in self.model.population:
                        # Find nearest time point in history
                        for hist_t, hist_b in ind.behavior_history:
                            if hist_t <= t:
                                behavior = hist_b
                            else:
                                break
                        values.append(behavior[dim])
                        
                    avg_values.append(np.mean(values))
                    
                # Plot population average
                self.behavior_ax.plot(time_points, avg_values, 'b-', label='Population Average')
                
                # Plot individual trajectories if requested
                if self.show_individual_var.get():
                    n_individuals = min(self.num_individuals_var.get(), len(self.model.population))
                    individuals = np.random.choice(self.model.population, size=n_individuals, replace=False)
                    
                    for i, ind in enumerate(individuals):
                        times, behaviors = zip(*ind.behavior_history)
                        values = [b[dim] for b in behaviors]
                        self.behavior_ax.plot(times, values, 'k-', alpha=0.3)
                        
            self.behavior_ax.set_ylim(0, 1)
            self.behavior_ax.grid(True)
            
        elif viz_type == "Distribution":
            self.behavior_ax.set_title(f"Distribution of {dim_names[dim]}")
            self.behavior_ax.set_xlabel(dim_names[dim])
            self.behavior_ax.set_ylabel("Frequency")
            
            # Get current behavior values
            values = [ind.behavior[dim] for ind in self.model.population]
            
            # Plot histogram
            self.behavior_ax.hist(values, bins=20, alpha=0.7, color='blue')
            self.behavior_ax.grid(True)
            
        elif viz_type == "Correlation":
            self.behavior_ax.set_title(f"Cases vs. {dim_names[dim]}")
            self.behavior_ax.set_xlabel("Infectious Cases (%)")
            self.behavior_ax.set_ylabel(f"Avg. {dim_names[dim]}")
            
            # Get case count and behavior history
            if hasattr(self.model, 'case_counts') and self.model.case_counts:
                # Initialize data
                case_times = []
                case_fractions = []
                behavior_avgs = []
                
                # Get infectious cases as percentage
                for t, count in self.model.case_counts[HealthState.INFECTIOUS]:
                    case_times.append(t)
                    total = sum(c for _, c in [(t, self.model.case_counts[s][i][1]) 
                                             for s in HealthState 
                                             for i, (time, _) in enumerate(self.model.case_counts[s]) 
                                             if time == t])
                    case_fractions.append(100 * count / total if total > 0 else 0)
                
                # Get average behavior at each time point
                for t in case_times:
                    values = []
                    for ind in self.model.population:
                        # Find nearest time point in history
                        behavior = None
                        for hist_t, hist_b in ind.behavior_history:
                            if hist_t <= t:
                                behavior = hist_b
                            else:
                                break
                                
                        if behavior is not None:
                            values.append(behavior[dim])
                            
                    behavior_avgs.append(np.mean(values) if values else 0)
                    
                # Plot scatter and trend line
                self.behavior_ax.scatter(case_fractions, behavior_avgs, alpha=0.5)
                
                # Add trend line if enough points
                if len(case_fractions) > 1:
                    z = np.polyfit(case_fractions, behavior_avgs, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(case_fractions), max(case_fractions), 100)
                    self.behavior_ax.plot(x_range, p(x_range), "r--")
                    
                    # Add correlation coefficient
                    corr = np.corrcoef(case_fractions, behavior_avgs)[0, 1]
                    self.behavior_ax.text(0.05, 0.95, f"Correlation: {corr:.2f}", 
                                      transform=self.behavior_ax.transAxes, 
                                      verticalalignment='top')
                    
                self.behavior_ax.grid(True)
                
        # Update the plot
        self.behavior_canvas.draw()
        
    def _update_data_viz(self):
        """Update the data assimilation visualization."""
        # Not implemented in this example
        pass
        
    def _import_csv(self):
        """Import data from CSV file."""
        filename = filedialog.askopenfilename(
            title="Import CSV Data",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Read CSV file
                df = pd.read_csv(filename)
                
                # Update series listbox
                self.series_listbox.delete(0, tk.END)
                for column in df.columns:
                    self.series_listbox.insert(tk.END, column)
                    
                # Store data for later use
                self.imported_data = df
                
                self._update_status(f"Imported data from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import CSV: {e}")
                
    def _import_json(self):
        """Import data from JSON file."""
        filename = filedialog.askopenfilename(
            title="Import JSON Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Read JSON file
                with open(filename, 'r') as f:
                    data = json.load(f)
                    
                # Convert to DataFrame if needed
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if all(isinstance(data[k], list) for k in data):
                        df = pd.DataFrame(data)
                    else:
                        # Nested structure - flatten
                        flat_data = []
                        for key, values in data.items():
                            if isinstance(values, dict):
                                values['_name'] = key
                                flat_data.append(values)
                            else:
                                flat_data.append({'_name': key, 'value': values})
                        df = pd.DataFrame(flat_data)
                else:
                    raise ValueError("Unsupported JSON structure")
                
                # Update series listbox
                self.series_listbox.delete(0, tk.END)
                for column in df.columns:
                    self.series_listbox.insert(tk.END, column)
                    
                # Store data for later use
                self.imported_data = df
                
                self._update_status(f"Imported data from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import JSON: {e}")
                
    def _assimilate_data(self):
        """Assimilate selected data into the model."""
        if not hasattr(self, 'imported_data'):
            messagebox.showinfo("No Data", "Please import data first")
            return
            
        selected = self.series_listbox.curselection()
        if not selected:
            messagebox.showinfo("No Selection", "Please select a data series to assimilate")
            return
            
        selected_column = self.series_listbox.get(selected[0])
        
        try:
            # Extract data
            data = self.imported_data[selected_column].values
            
            # Prepare for assimilation
            observed_data = {
                'name': selected_column,
                'values': data,
                'time': self.model.time
            }
            
            # Assimilate data
            self.model.data_assimilation.assimilate_data(observed_data)
            
            self._update_status(f"Assimilated data: {selected_column}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to assimilate data: {e}")
            
    def _reset_ensemble(self):
        """Reset the data assimilation ensemble."""
        if not hasattr(self.model, 'data_assimilation'):
            return
            
        try:
            # Update parameters first
            self.params.ensemble_size = self.ensemble_size_var.get()
            self.params.observation_noise_sd = self.obs_noise_var.get()
            self.model.params = self.params
            
            # Reinitialize ensemble
            self.model.data_assimilation.initialize_ensemble()
            
            self._update_status("Reset data assimilation ensemble")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset ensemble: {e}")


# Main entry point
if __name__ == "__main__":
    configure_dpi_awareness()
    app = MANEpidemicApp()
    # Configure matplotlib for better display
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
    rcParams['font.size'] = 9
    rcParams['figure.dpi'] = 120
    
    app.mainloop()