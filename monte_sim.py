from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from threading import Thread
import queue

class MonteCarloSimulationApp:
    def __init__(self, master):
        self.master = master
        master.title("Monte Carlo Simulation for RandomForest")
        
        # Queue for thread-safe communication
        self.queue = queue.Queue()
        
        # Layout
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        # Configure grid layout
        self.frame.columnconfigure(0, weight=1)
        
        # Control buttons frame
        self.controls_frame = ttk.Frame(self.frame)
        self.controls_frame.grid(row=0, column=0, sticky=W)

        self.start_button = ttk.Button(self.controls_frame, text="Start Simulation", command=self.start_simulation)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(self.controls_frame, text="Stop Simulation", command=self.stop_simulation, state=DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.frame, orient=HORIZONTAL, length=400, mode='determinate')
        self.progress.grid(row=1, column=0, pady=10)

        # Status labels
        self.status_label = ttk.Label(self.frame, text="Status: Ready")
        self.status_label.grid(row=2, column=0, pady=5)

        self.iteration_label = ttk.Label(self.frame, text="Iteration: 0/0")
        self.iteration_label.grid(row=3, column=0, pady=5)

        self.current_params_label = ttk.Label(self.frame, text="Current Params: None")
        self.current_params_label.grid(row=4, column=0, pady=5)

        # Results text area
        self.result_text = Text(self.frame, height=10, width=60)
        self.result_text.grid(row=5, column=0, pady=10)

        # Setup Matplotlib Figure with reduced update frequency
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        plt.tight_layout()

        # Accuracy Distribution Histogram
        self.ax1.set_title('Distribution of Model Accuracies')
        self.ax1.set_xlabel('Accuracy')
        self.ax1.set_ylabel('Frequency')
        self.hist_bins = 20
        self.hist_data = []

        # Best Accuracy Over Iterations
        self.ax2.set_title('Best Accuracy Over Iterations')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Best Accuracy')
        self.best_accuracy = []
        self.iterations = []

        # Embed Matplotlib Figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=6, column=0, pady=10)

        # Batch updates to reduce overhead
        self.update_interval = 10
        self.is_running = False

    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state=DISABLED)
            self.stop_button.config(state=NORMAL)
            self.status_label.config(text="Status: Running")
            
            # Start simulation in separate thread
            simulation_thread = Thread(target=self.run_simulation)
            simulation_thread.daemon = True
            simulation_thread.start()
            
            # Start checking queue for updates
            self.master.after(100, self.check_queue)

    def stop_simulation(self):
        if self.is_running:
            self.is_running = False
            self.status_label.config(text="Status: Stopping...")
            self.stop_button.config(state=DISABLED)

    def check_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg['type'] == 'update':
                    self.update_ui(msg)
                elif msg['type'] == 'complete':
                    self.complete_simulation(msg)
                    return
                self.queue.task_done()
        except queue.Empty:
            if self.is_running:
                self.master.after(100, self.check_queue)

    def update_ui(self, msg):
        i = msg['iteration']
        n_simulations = msg['total']
        params = msg['params']
        avg_score = msg['score']
        best_overall = msg['best']
        
        # Update UI elements
        if i % self.update_interval == 0 or i == n_simulations - 1:
            self.iteration_label.config(text=f"Iteration: {i+1}/{n_simulations}")
            self.current_params_label.config(text=f"Current Params: {params}")
            self.progress['value'] = (i + 1) / n_simulations * 100
            
            # Update plots less frequently
            self.hist_data.append(avg_score)
            self.best_accuracy.append(best_overall['accuracy'])
            self.iterations.append(i+1)
            
            self.ax1.cla()
            self.ax1.set_title('Distribution of Model Accuracies')
            self.ax1.set_xlabel('Accuracy')
            self.ax1.set_ylabel('Frequency')
            self.ax1.hist(self.hist_data, bins=self.hist_bins, alpha=0.7, color='blue', edgecolor='black')

            self.ax2.cla()
            self.ax2.set_title('Best Accuracy Over Iterations')
            self.ax2.set_xlabel('Iteration')
            self.ax2.set_ylabel('Best Accuracy')
            self.ax2.plot(self.iterations, self.best_accuracy, color='green')
            
            self.canvas.draw()

    def complete_simulation(self, msg):
        best_params = msg['best_params']
        self.result_text.delete(1.0, END)
        self.result_text.insert(END, f"Best Parameters:\n{best_params}\n")
        
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        self.status_label.config(text="Status: Ready")
        self.progress['value'] = 0
        self.iteration_label.config(text="Iteration: 0/0")
        self.current_params_label.config(text="Current Params: None")
        self.is_running = False

    def run_simulation(self):
        try:
            # Create synthetic data for demonstration
            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define ranges for hyperparameters
            param_ranges = {
                'n_estimators': list(range(100, 1001, 100)),
                'max_depth': list(range(5, 51, 5)),
                'min_samples_split': list(range(2, 21, 2)),
                'min_samples_leaf': list(range(1, 11)),
                'max_features': ['sqrt', 'log2', None]
            }

            n_simulations = 500  # Limit to 500 simulations

            results = []
            np.random.seed(42)
            best_overall = {'accuracy': -np.inf}

            for i in range(n_simulations):
                if not self.is_running:
                    break  # Stop simulation if stop button is pressed

                params = {
                    'n_estimators': np.random.choice(param_ranges['n_estimators']),
                    'max_depth': np.random.choice(param_ranges['max_depth']),
                    'min_samples_split': np.random.choice(param_ranges['min_samples_split']),
                    'min_samples_leaf': np.random.choice(param_ranges['min_samples_leaf']),
                    'max_features': np.random.choice(param_ranges['max_features']),
                }

                rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)  # Use all cores
                cv_scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
                avg_score = np.mean(cv_scores)

                result = {**params, 'accuracy': avg_score}
                results.append(result)

                if avg_score > best_overall['accuracy']:
                    best_overall = result.copy()

                self.queue.put({
                    'type': 'update',
                    'iteration': i,
                    'total': n_simulations,
                    'params': params,
                    'score': avg_score,
                    'best': best_overall
                })

            self.queue.put({
                'type': 'complete',
                'best_params': best_overall
            })

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.is_running = False

if __name__ == "__main__":
    root = Tk()
    app = MonteCarloSimulationApp(root)
    root.mainloop()
