import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ==========================================
# Default Configuration
# ==========================================
DEFAULT_START_COLUMN = 3261  # Column index for first X (0-based)
DEFAULT_STRIDE = 7           # Columns between points
DEFAULT_NUM_POINTS = 20      # Number of points to visualize
DEFAULT_START_POINT = 0      # Starting point ID


class CeilingProfileViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("KineVis3D")
        self.root.geometry("1600x900")
        
        self.df = None
        self.xyz_data = None
        self.point_indices = []
        self.file_path = None
        self.selected_point_indices = []  # Track selected points for profile
        self.picking_mode = False  # Track if in point picking mode
        self.current_view_mode = None  # Track current view mode for dynamic updates
        self.point_annotations = []  # Store text annotations for selected points
        
        # Configuration variables
        self.start_col = tk.IntVar(value=DEFAULT_START_COLUMN)
        self.stride = tk.IntVar(value=DEFAULT_STRIDE)
        self.num_points = tk.IntVar(value=DEFAULT_NUM_POINTS)
        self.start_point = tk.IntVar(value=DEFAULT_START_POINT)
        self.current_frame = tk.IntVar(value=0)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Control Panel Frame - LEFT SIDE with scrollbar
        control_container = tk.Frame(self.root, bg='lightgray', width=350)
        control_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_container.pack_propagate(False)  # Prevent frame from shrinking
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(control_container, bg='lightgray', width=330, highlightthickness=0)
        scrollbar = tk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        
        # Create frame inside canvas for all controls
        control_frame = tk.Frame(canvas, bg='lightgray', padx=10, pady=10)
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create window in canvas
        canvas_frame = canvas.create_window((0, 0), window=control_frame, anchor="nw")
        
        # Configure scroll region when frame changes size
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        control_frame.bind("<Configure>", on_frame_configure)
        
        # Bind mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # File Selection
        file_frame = tk.LabelFrame(control_frame, text="File Selection", padx=10, pady=5)
        file_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(file_frame, text="Load CSV File", command=self.load_file, 
                 width=20, bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.file_label = tk.Label(file_frame, text="No file loaded", fg='gray', wraplength=300, justify='left')
        self.file_label.pack(pady=5)
        
        # Configuration Frame
        config_frame = tk.LabelFrame(control_frame, text="Data Configuration", padx=10, pady=5)
        config_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(config_frame, text="Start Column (X):").grid(row=0, column=0, sticky='w', pady=2)
        tk.Entry(config_frame, textvariable=self.start_col, width=15).grid(row=0, column=1, pady=2, sticky='ew')
        
        tk.Label(config_frame, text="Stride:").grid(row=1, column=0, sticky='w', pady=2)
        tk.Entry(config_frame, textvariable=self.stride, width=15).grid(row=1, column=1, pady=2, sticky='ew')
        
        tk.Label(config_frame, text="Start Point:").grid(row=2, column=0, sticky='w', pady=2)
        tk.Entry(config_frame, textvariable=self.start_point, width=15).grid(row=2, column=1, pady=2, sticky='ew')
        
        tk.Label(config_frame, text="Detected Points:").grid(row=3, column=0, sticky='w', pady=2)
        self.detected_points_label = tk.Label(config_frame, text="--", fg='blue', font=('Arial', 10, 'bold'))
        self.detected_points_label.grid(row=3, column=1, pady=2, sticky='w')
        
        config_frame.columnconfigure(1, weight=1)
        
        tk.Button(config_frame, text="Auto-Detect & Process", command=self.process_data,
                 bg='#2196F3', fg='white', font=('Arial', 10, 'bold'), width=20).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Visualization Buttons Frame
        viz_frame = tk.LabelFrame(control_frame, text="Visualizations", padx=10, pady=5)
        viz_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(viz_frame, text="3D View", command=self.show_3d_view,
                 width=20, bg='#FF9800', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        tk.Button(viz_frame, text="2D Profile", command=self.show_2d_profile,
                 width=20, bg='#9C27B0', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        tk.Button(viz_frame, text="Animation", command=self.show_animation,
                 width=20, bg='#F44336', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        tk.Button(viz_frame, text="Save Animation", command=self.save_animation,
                 width=20, bg='#607D8B', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        # Point Selection Frame
        selection_frame = tk.LabelFrame(control_frame, text="Select Points for Profile", padx=10, pady=5)
        selection_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        tk.Button(selection_frame, text="Pick Points Interactively", command=self.start_point_picking,
                 width=20, bg='#00BCD4', fg='white', font=('Arial', 10, 'bold')).pack(pady=5, fill=tk.X)
        
        self.selected_count_label = tk.Label(selection_frame, text="Selected: 0 points", fg='blue')
        self.selected_count_label.pack(pady=3)
        
        tk.Button(selection_frame, text="Clear Selection", command=self.clear_selection,
                 width=20, bg='#FF5722', fg='white', font=('Arial', 9)).pack(pady=3, fill=tk.X)
        
        tk.Label(selection_frame, text="Show Profile", 
                font=('Arial', 9, 'bold')).pack(pady=(15,5))
        
        tk.Button(selection_frame, text="Show Selected Profile", command=self.show_selected_profile,
                 width=20, bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        tk.Button(selection_frame, text="Profile Time Evolution", command=self.show_profile_evolution,
                 width=20, bg='#8BC34A', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        tk.Button(selection_frame, text="Save Profile Evolution GIF", command=self.save_profile_evolution,
                 width=20, bg='#607D8B', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        tk.Button(selection_frame, text="Plot Time Series", command=self.plot_time_series,
                 width=20, bg='#FF6F00', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        tk.Label(selection_frame, text="Rigid Body Motion", 
                font=('Arial', 9, 'bold')).pack(pady=(15,5))
        
        tk.Button(selection_frame, text="Calculate 6-DOF Motion", command=self.calculate_rigid_body_motion,
                 width=20, bg='#E91E63', fg='white', font=('Arial', 10, 'bold')).pack(pady=3, fill=tk.X)
        
        # Frame Control
        frame_control = tk.LabelFrame(control_frame, text="Time Step Control", padx=10, pady=5)
        frame_control.pack(fill=tk.X, pady=5)
        
        self.frame_label = tk.Label(frame_control, text="0 / 0", font=('Arial', 10, 'bold'))
        self.frame_label.pack(pady=5)
        
        self.frame_slider = tk.Scale(frame_control, from_=0, to=100, orient=tk.HORIZONTAL,
                                     variable=self.current_frame, length=280, command=self.update_frame)
        self.frame_slider.pack(pady=5, fill=tk.X)
        
        # Status Bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Canvas Frame for matplotlib
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = None
        self.current_fig = None
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Displacement Data CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="d:\\"
        )
        
        if not file_path:
            return
        
        try:
            self.status_bar.config(text="Loading file...")
            self.root.update()
            
            self.df = pd.read_csv(file_path, header=None)
            self.file_path = file_path
            self.file_label.config(text=f"Loaded: {file_path.split('/')[-1]}", fg='green')
            self.status_bar.config(text=f"File loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Auto-process data
            self.process_data()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.status_bar.config(text="Error loading file")
    
    def process_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first!")
            return
        
        try:
            self.status_bar.config(text="Detecting valid points...")
            self.root.update()
            
            start_col = self.start_col.get()
            stride = self.stride.get()
            start_pt = self.start_point.get()
            
            # Auto-detect number of points with valid data (not all NaN)
            max_possible_points = (self.df.shape[1] - start_col) // stride
            valid_points_count = 0
            
            # Check each point to see if it has valid (non-NaN) data
            for i in range(max_possible_points):
                point_step = start_pt + i
                z_col = start_col + (point_step * stride) + 2
                
                if z_col >= self.df.shape[1]:
                    break
                    
                z_data = self.df.iloc[:, z_col].values
                # Check if at least some values are not NaN
                if not np.all(np.isnan(z_data)):
                    valid_points_count += 1
                else:
                    # Stop at first all-NaN column
                    break
            
            num_pts = valid_points_count
            self.status_bar.config(text=f"Auto-detected {num_pts} points with valid data")
            self.detected_points_label.config(text=str(num_pts))
            self.root.update()
            
            xyz_list = []
            self.point_indices = []
            
            for i in range(num_pts):
                point_step = start_pt + i
                x_col = start_col + (point_step * stride) + 0
                y_col = start_col + (point_step * stride) + 1
                z_col = start_col + (point_step * stride) + 2
                
                if z_col >= self.df.shape[1]:
                    break
                
                x_data = self.df.iloc[:, x_col].values
                y_data = self.df.iloc[:, y_col].values
                z_data = self.df.iloc[:, z_col].values
                
                # Handle NaN values - interpolate or fill
                x_data = self.handle_nan(x_data)
                y_data = self.handle_nan(y_data)
                z_data = self.handle_nan(z_data)
                
                xyz_list.append(np.column_stack([x_data, y_data, z_data]))
                self.point_indices.append(point_step + 1)
            
            # Shape: (num_frames, num_points, 3)
            self.xyz_data = np.array([np.array([xyz_list[j][i] for j in range(len(xyz_list))]) 
                                      for i in range(len(xyz_list[0]))])
            
            # Update frame slider
            self.frame_slider.config(to=len(self.xyz_data)-1)
            self.frame_label.config(text=f"0 / {len(self.xyz_data)-1}")
            
            self.status_bar.config(text=f"Processed {len(self.point_indices)} points with {len(self.xyz_data)} time steps")
            messagebox.showinfo("Success", f"Data processed successfully!\n{len(self.point_indices)} points, {len(self.xyz_data)} frames")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process data:\n{str(e)}")
            self.status_bar.config(text="Error processing data")
    
    def handle_nan(self, data):
        """Handle NaN values in data by interpolation and forward/backward fill"""
        data = data.copy()
        
        # Convert to pandas Series for easier handling
        series = pd.Series(data)
        
        # First, try linear interpolation
        series = series.interpolate(method='linear', limit_direction='both')
        
        # Forward fill any remaining NaN at the start
        series = series.ffill()
        
        # Backward fill any remaining NaN at the end
        series = series.bfill()
        
        # If still NaN (all values were NaN), fill with 0
        series = series.fillna(0)
        
        return series.values
    
    def parse_point_selection(self, text):
        """Parse point selection text like '1 5 10 15-20' into list of indices"""
        sequences = []
        for part in text.split():
            try:
                # Try converting the part to an integer (single number)
                num = int(part)
                sequences.append(num)
            except ValueError:
                # If conversion fails, assume it's a range like '15-20'
                if '-' in part:
                    start, end = map(int, part.split("-"))
                    sequences.extend(range(start, end+1))
        return sequences
    
    def clear_selection(self):
        """Clear all selected points"""
        self.selected_point_indices = []
        self.selected_count_label.config(text="Selected: 0 points")
        self.status_bar.config(text="Selection cleared")
    
    def start_point_picking(self):
        """Start interactive point picking mode"""
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        try:
            self.clear_canvas()
            self.picking_mode = True
            self.selected_point_indices = []
            self.selected_count_label.config(text="Selected: 0 points")
            self.point_annotations = []  # Reset annotations for new picking session
            
            self.status_bar.config(text="PICKING MODE: Click on points to select/deselect. Close window when done.")
            self.root.update()
            
            frame_idx = self.current_frame.get()
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            points = self.xyz_data[frame_idx]
            
            # Plot all points
            self.scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c='gray', s=100, alpha=0.6, edgecolors='black', linewidth=0.5, picker=True)
            
            # Store for highlighting selected points
            self.selected_scatter = ax.scatter([], [], [], 
                      c='red', s=200, alpha=1.0, edgecolors='yellow', linewidth=2)
            
            ax.set_xlabel('X Displacement (mm)')
            ax.set_ylabel('Y Displacement (mm)')
            ax.set_zlabel('Z Displacement (mm)')
            ax.set_title('PICK POINTS: Click on points to select (click again to deselect)', 
                        fontsize=14, fontweight='bold', color='red')
            
            # Set Z axis to start from zero
            z_max = np.max(self.xyz_data[:, :, 2])
            ax.set_zlim([0, z_max])
            
            # Store data for picking
            self.pick_points = points
            self.pick_ax = ax
            
            # Connect pick event
            def on_pick(event):
                if event.artist != self.scatter:
                    return
                
                # Get the index of the picked point
                ind = event.ind[0]
                point_id = self.point_indices[ind]
                
                # Toggle selection
                if ind in self.selected_point_indices:
                    self.selected_point_indices.remove(ind)
                else:
                    self.selected_point_indices.append(ind)
                
                # Update display
                self.update_selected_points_display()
                
            fig.canvas.mpl_connect('pick_event', on_pick)
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.current_fig = fig
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start point picking:\n{str(e)}")
            self.status_bar.config(text="Error in point picking mode")
    
    def update_selected_points_display(self):
        """Update the visual display of selected points"""
        if not hasattr(self, 'pick_points'):
            return
        
        # Clear existing annotations
        for annotation in self.point_annotations:
            annotation.remove()
        self.point_annotations = []
        
        # Update selected points scatter
        if self.selected_point_indices:
            selected_pts = self.pick_points[self.selected_point_indices]
            self.selected_scatter._offsets3d = (selected_pts[:, 0], selected_pts[:, 1], selected_pts[:, 2])
            
            # Add annotations for each selected point
            for idx in self.selected_point_indices:
                point = self.pick_points[idx]
                point_id = self.point_indices[idx]
                # Create 3D text annotation
                annotation = self.pick_ax.text(point[0], point[1], point[2], 
                                              f'  P{point_id}',
                                              fontsize=10, fontweight='bold', 
                                              color='red',
                                              bbox=dict(boxstyle='round,pad=0.3', 
                                                       facecolor='yellow', 
                                                       alpha=0.7, 
                                                       edgecolor='red'))
                self.point_annotations.append(annotation)
        else:
            self.selected_scatter._offsets3d = ([], [], [])
        
        # Update label
        count = len(self.selected_point_indices)
        self.selected_count_label.config(text=f"Selected: {count} points")
        self.status_bar.config(text=f"Selected {count} points. Click more or close to finish.")
        
        # Redraw
        self.canvas.draw()
    
    def update_frame(self, value):
        if self.xyz_data is None:
            return
        frame = int(value)
        self.frame_label.config(text=f"{frame} / {len(self.xyz_data)-1}")
        
        # Update profile view if in profile mode
        if self.current_view_mode == 'profile' and self.selected_point_indices:
            self.show_selected_profile()
    
    def clear_canvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
    
    def show_3d_view(self):
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        try:
            self.current_view_mode = None  # Clear view mode
            self.clear_canvas()
            self.status_bar.config(text="Generating 3D view... (Use mouse to rotate/zoom)")
            self.root.update()
            
            frame_idx = self.current_frame.get()
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Calculate displacement relative to first frame
            points_current = self.xyz_data[frame_idx]
            points_reference = self.xyz_data[0]
            points_displacement = points_current - points_reference
            
            # Use current XY position but show Z as displacement from frame 0
            display_points = points_current.copy()
            display_points[:, 2] = points_displacement[:, 2]
            
            # Plot visible lines connecting points
            ax.plot(display_points[:, 0], display_points[:, 1], display_points[:, 2], 
                   alpha=0.3, linewidth=1.0, color='gray')  # Visible lines
            
            # Create visible scatter plot for points
            scatter = ax.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2], 
                      c=display_points[:, 2], cmap='jet', s=100, alpha=1.0, edgecolors='black', linewidth=0.5)

            # Show point IDs beside each point in the 3D view
            for i, point_id in enumerate(self.point_indices):
                ax.text(display_points[i, 0], display_points[i, 1], display_points[i, 2],
                        f' P{point_id}', fontsize=8, color='black', alpha=0.9)
            
            ax.set_xlabel('X Position (mm)')
            ax.set_ylabel('Y Position (mm)')
            ax.set_zlabel('Z Displacement from Frame 0 (mm)')
            ax.set_title(f'3D Point Cloud View - Frame {frame_idx} (Reference: Frame 0)')
            
            # Set Z axis limits based on displacement range
            z_disp_all = self.xyz_data[:, :, 2] - self.xyz_data[0, :, 2]
            z_min = np.min(z_disp_all)
            z_max = np.max(z_disp_all)
            z_margin = (z_max - z_min) * 0.1
            ax.set_zlim([z_min - z_margin, z_max + z_margin])
            
            # Color bar
            plt.colorbar(scatter, ax=ax, label='Z Displacement (mm)', shrink=0.5)
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            
            # Add toolbar for zoom/pan controls
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
            toolbar.update()
            
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.current_fig = fig
            self.status_bar.config(text="3D view displayed - Use mouse to rotate, scroll to zoom")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate 3D view:\n{str(e)}")
            self.status_bar.config(text="Error generating 3D view")
    
    def show_2d_profile(self):
        """Show Z displacement profile of selected points at current frame"""
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        if not self.selected_point_indices:
            messagebox.showwarning("Warning", "Please pick points first using 'Pick Points Interactively' button!")
            return
        
        try:
            selected_indices = self.selected_point_indices
            selected_labels = [self.point_indices[i] for i in selected_indices]
            
            self.clear_canvas()
            self.status_bar.config(text=f"Generating Z profile for {len(selected_indices)} points...")
            self.root.update()
            
            frame_idx = self.current_frame.get()
            points = self.xyz_data[frame_idx][selected_indices]
            
            # Determine which direction has more variation (X or Y)
            x_range = np.ptp(points[:, 0])  # peak-to-peak range
            y_range = np.ptp(points[:, 1])
            
            if x_range > y_range:
                x_axis_values = points[:, 0]
                x_axis_label = 'X Position (mm)'
                direction = 'X'
            else:
                x_axis_values = points[:, 1]
                x_axis_label = 'Y Position (mm)'
                direction = 'Y'
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Use lowest point as reference (set to 0)
            z_values = points[:, 2]
            z_min = np.min(z_values)
            z_relative = z_values - z_min
            
            # Z displacement profile (height difference)
            ax.plot(x_axis_values, z_relative, 'o-', color='blue', linewidth=3, markersize=8)
            ax.set_ylabel(' Z Displacement (mm)', fontsize=14, fontweight='bold')
            ax.set_xlabel(x_axis_label, fontsize=14, fontweight='bold')
            ax.set_title(f'Height Profile (Z vs {direction}) - Frame {frame_idx} ({len(selected_indices)} points)', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(bottom=0)
            
            # Add value labels on points
            for i, (x_val, z) in enumerate(zip(x_axis_values, z_relative)):
                ax.annotate(f'{z:.2f}', (x_val, z), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
            
            plt.tight_layout()
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.current_fig = fig
            self.status_bar.config(text=f"Z profile displayed for {len(selected_indices)} points")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate 2D profile:\n{str(e)}")
            self.status_bar.config(text="Error generating 2D profile")
    
    def show_selected_profile(self):
        """Show profile of only selected points at current frame"""
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        if not self.selected_point_indices:
            messagebox.showwarning("Warning", "Please pick points first using 'Pick Points Interactively' button!")
            return
        
        try:
            # Set view mode for dynamic updates
            self.current_view_mode = 'profile'
            
            selected_indices = self.selected_point_indices
            selected_labels = [self.point_indices[i] for i in selected_indices]
            
            self.clear_canvas()
            self.status_bar.config(text=f"Generating profile for {len(selected_indices)} points...")
            self.root.update()
            
            frame_idx = self.current_frame.get()
            points = self.xyz_data[frame_idx][selected_indices]
            
            # Determine which direction has more variation (X or Y)
            x_range = np.ptp(points[:, 0])  # peak-to-peak range
            y_range = np.ptp(points[:, 1])
            
            if x_range > y_range:
                x_axis_values = points[:, 0]
                x_axis_label = 'X Position (mm)'
                direction = 'X'
            else:
                x_axis_values = points[:, 1]
                x_axis_label = 'Y Position (mm)'
                direction = 'Y'
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Calculate displacement relative to first frame (frame 0)
            points_initial = self.xyz_data[0][selected_indices]
            z_displacement = points[:, 2] - points_initial[:, 2]
            
            # Z displacement profile relative to first frame
            ax.plot(x_axis_values, z_displacement, 'o-', color='blue', linewidth=3, markersize=8)
            ax.set_ylabel('Z Displacement (mm) relative to Frame 0', fontsize=14, fontweight='bold')
            ax.set_xlabel(x_axis_label, fontsize=14, fontweight='bold')
            ax.set_title(f'Height Profile (Z vs {direction}) - Frame {frame_idx} (Reference: Frame 0, {len(selected_indices)} points)', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add value labels on points
            for i, (x_val, z) in enumerate(zip(x_axis_values, z_displacement)):
                ax.annotate(f'{z:.2f}', (x_val, z), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
            
            plt.tight_layout()
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.current_fig = fig
            self.status_bar.config(text=f"Profile displayed for {len(selected_indices)} points")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate selected profile:\n{str(e)}")
            self.status_bar.config(text="Error generating profile")
    
    def show_profile_evolution(self):
        """Show profile evolution with top view, trajectory plots, and toggle-slider control"""
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        if not self.selected_point_indices:
            messagebox.showwarning("Warning", "Please pick points first using 'Pick Points Interactively' button!")
            return
        
        try:
            selected_indices = self.selected_point_indices
            selected_labels = [self.point_indices[i] for i in selected_indices]
            
            self.clear_canvas()
            self.status_bar.config(text=f"Animating profile evolution for {len(selected_indices)} points...")
            self.root.update()
            
            # Determine which direction has more variation (X or Y) using first frame
            first_frame_points = self.xyz_data[0][selected_indices]
            x_range = np.ptp(first_frame_points[:, 0])
            y_range = np.ptp(first_frame_points[:, 1])
            
            if x_range > y_range:
                axis_index = 0
                x_axis_label = 'X Position (mm)'
                direction = 'X'
            else:
                axis_index = 1
                x_axis_label = 'Y Position (mm)'
                direction = 'Y'
            
            # --- Create figure: 2 rows x 4 columns ---
            # Row 0: Profile (cols 0-2), Top View (col 3)
            # Row 1: Z vs Y, Z vs X, X vs Y disp, (empty or merged)
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(
                2, 4,
                left=0.05, right=0.985, top=0.95, bottom=0.08,
                hspace=0.38, wspace=0.30
            )
            ax_profile = fig.add_subplot(gs[0, 0:2])      # Profile: top-left, equal width
            ax_topview = fig.add_subplot(gs[0, 2:4])      # Top view: top-right, equal width
            ax_zy = fig.add_subplot(gs[1, 0])             # Z vs Y trajectory
            ax_zx = fig.add_subplot(gs[1, 1])             # Z vs X trajectory
            ax_xy_disp = fig.add_subplot(gs[1, 2])        # X vs Y displacement trajectory
            ax_extra = fig.add_subplot(gs[1, 3])          # X-Y top-view zoom / info
            
            # Reference frame (frame 0) for all displacements
            points_reference = self.xyz_data[0][selected_indices]
            num_frames = len(self.xyz_data)
            
            # Precompute all displacements relative to frame 0
            all_x_disp = np.array([self.xyz_data[f, selected_indices, 0] - points_reference[:, 0] for f in range(num_frames)])
            all_y_disp = np.array([self.xyz_data[f, selected_indices, 1] - points_reference[:, 1] for f in range(num_frames)])
            all_z_disp = np.array([self.xyz_data[f, selected_indices, 2] - points_reference[:, 2] for f in range(num_frames)])
            
            z_disp_min, z_disp_max = all_z_disp.min(), all_z_disp.max()
            x_disp_min, x_disp_max = all_x_disp.min(), all_x_disp.max()
            y_disp_min, y_disp_max = all_y_disp.min(), all_y_disp.max()
            
            # --- Distinct colors per point ---
            colors_list = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
                           '#00ACC1', '#D81B60', '#6D4C41', '#FFB300', '#5E35B1',
                           '#00897B', '#C0CA33', '#F4511E', '#039BE5', '#7CB342',
                           '#FDD835', '#E91E63', '#00695C', '#9E9D24', '#6A1B9A']
            n_pts = len(selected_indices)
            if n_pts <= len(colors_list):
                colors = colors_list[:n_pts]
            else:
                colors = [colors_list[i % len(colors_list)] for i in range(n_pts)]
            
            def _axis_margin(vmin, vmax, frac=0.15):
                span = vmax - vmin if vmax != vmin else 1.0
                return vmin - span * frac, vmax + span * frac
            
            # ====== Profile plot (top-left) ======
            points = self.xyz_data[0][selected_indices]
            x_axis_values = points[:, axis_index]
            z_displacement = points[:, 2] - points_reference[:, 2]
            profile_line, = ax_profile.plot(x_axis_values, z_displacement, 'o-', color='blue', linewidth=3, markersize=8)
            
            ax_profile.set_ylabel('Z Disp. (mm) rel. Frame 0', fontsize=11, fontweight='bold')
            ax_profile.set_xlabel(x_axis_label, fontsize=11, fontweight='bold')
            profile_title = ax_profile.set_title(
                f'Profile Evolution (Z vs {direction}) — Frame 0 / {num_frames-1}  [{n_pts} pts]',
                fontsize=13, fontweight='bold')
            ax_profile.grid(True, alpha=0.3, linestyle='--')
            y_margin = (z_disp_max - z_disp_min) * 0.15 if z_disp_max != z_disp_min else 1.0
            ax_profile.set_ylim([z_disp_min - y_margin, z_disp_max + y_margin])
            
            # ====== Top View — transverse displacement along point positions (like profile) ======
            # axis_index: 0 = points along X, 1 = points along Y
            # transverse: if points along X → show Y disp; if along Y → show X disp
            if axis_index == 0:
                transverse_index = 1  # Y displacement
                all_transverse_disp = all_y_disp
                transverse_label = 'Y Disp. (mm)'
                topview_x_label = x_axis_label  # X Position
                topview_dir = 'Y'
            else:
                transverse_index = 0  # X displacement
                all_transverse_disp = all_x_disp
                transverse_label = 'X Disp. (mm)'
                topview_x_label = x_axis_label  # Y Position
                topview_dir = 'X'
            
            trans_disp_min, trans_disp_max = all_transverse_disp.min(), all_transverse_disp.max()
            
            # Connected line through all points positioned along their spatial axis
            topview_line, = ax_topview.plot(x_axis_values, np.zeros(n_pts), 'o-', color='#E91E63',
                                            linewidth=2.5, markersize=7, markeredgecolor='black',
                                            markeredgewidth=0.5, zorder=3)
            # Reference zero line
            ax_topview.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
            
            ax_topview.set_xlabel(topview_x_label, fontsize=11, fontweight='bold')
            ax_topview.set_ylabel(transverse_label, fontsize=11, fontweight='bold')
            topview_title = ax_topview.set_title(
                f'In-Plane Profile ({topview_dir} disp. vs {direction}) — Frame 0',
                fontsize=13, fontweight='bold')
            ax_topview.grid(True, alpha=0.3, linestyle='--')
            t_margin = (trans_disp_max - trans_disp_min) * 0.15 if trans_disp_max != trans_disp_min else 1.0
            ax_topview.set_ylim([trans_disp_min - t_margin, trans_disp_max + t_margin])
            
            # ====== Trajectory plots (bottom row) ======
            traj_lines_zy, traj_dots_zy = [], []
            traj_lines_zx, traj_dots_zx = [], []
            traj_lines_xy, traj_dots_xy = [], []
            
            for pi in range(n_pts):
                lbl = f'P{selected_labels[pi]}'
                c = colors[pi]
                # Z vs Y
                ln, = ax_zy.plot([], [], '-', color=c, linewidth=1.2, alpha=0.6)
                dt, = ax_zy.plot([], [], 'o', color=c, markersize=5, markeredgecolor='black', markeredgewidth=0.4)
                traj_lines_zy.append(ln); traj_dots_zy.append(dt)
                # Z vs X
                ln2, = ax_zx.plot([], [], '-', color=c, linewidth=1.2, alpha=0.6)
                dt2, = ax_zx.plot([], [], 'o', color=c, markersize=5, markeredgecolor='black', markeredgewidth=0.4)
                traj_lines_zx.append(ln2); traj_dots_zx.append(dt2)
                # X vs Y displacement
                ln3, = ax_xy_disp.plot([], [], '-', color=c, linewidth=1.2, alpha=0.6, label=lbl)
                dt3, = ax_xy_disp.plot([], [], 'o', color=c, markersize=5, markeredgecolor='black', markeredgewidth=0.4)
                traj_lines_xy.append(ln3); traj_dots_xy.append(dt3)
            
            ax_zy.set_xlabel('Y Disp. (mm)', fontsize=9, fontweight='bold')
            ax_zy.set_ylabel('Z Disp. (mm)', fontsize=9, fontweight='bold')
            ax_zy.set_title('Trajectory: Z vs Y', fontsize=10, fontweight='bold')
            ax_zy.grid(True, alpha=0.3, linestyle='--')
            ax_zy.set_xlim(*_axis_margin(y_disp_min, y_disp_max))
            ax_zy.set_ylim(*_axis_margin(z_disp_min, z_disp_max))
            
            ax_zx.set_xlabel('X Disp. (mm)', fontsize=9, fontweight='bold')
            ax_zx.set_ylabel('Z Disp. (mm)', fontsize=9, fontweight='bold')
            ax_zx.set_title('Trajectory: Z vs X', fontsize=10, fontweight='bold')
            ax_zx.grid(True, alpha=0.3, linestyle='--')
            ax_zx.set_xlim(*_axis_margin(x_disp_min, x_disp_max))
            ax_zx.set_ylim(*_axis_margin(z_disp_min, z_disp_max))
            
            ax_xy_disp.set_xlabel('X Disp. (mm)', fontsize=9, fontweight='bold')
            ax_xy_disp.set_ylabel('Y Disp. (mm)', fontsize=9, fontweight='bold')
            ax_xy_disp.set_title('Trajectory: X vs Y (disp.)', fontsize=10, fontweight='bold')
            ax_xy_disp.grid(True, alpha=0.3, linestyle='--')
            ax_xy_disp.set_xlim(*_axis_margin(x_disp_min, x_disp_max))
            ax_xy_disp.set_ylim(*_axis_margin(y_disp_min, y_disp_max))
            ax_xy_disp.legend(loc='best', fontsize=6, ncol=min(n_pts, 4))

            for ax in (ax_zy, ax_zx, ax_xy_disp, ax_extra):
                ax.set_box_aspect(1)

            fig.align_labels()
            
            # ====== Bottom-right: live displacement magnitude bar chart ======
            bar_colors = colors[:n_pts]
            bar_labels = [f'P{l}' for l in selected_labels]
            disp_bars = ax_extra.bar(range(n_pts), [0]*n_pts, color=bar_colors, edgecolor='black', linewidth=0.5)
            ax_extra.set_xticks(range(n_pts))
            ax_extra.set_xticklabels(bar_labels, fontsize=7, rotation=45)
            ax_extra.set_ylabel('|Disp.| (mm)', fontsize=9, fontweight='bold')
            ax_extra.set_title('Displacement Magnitude', fontsize=10, fontweight='bold')
            ax_extra.grid(True, alpha=0.3, linestyle='--', axis='y')
            # Precompute max magnitude for consistent y-axis
            all_mag = np.sqrt(all_x_disp**2 + all_y_disp**2 + all_z_disp**2)
            ax_extra.set_ylim(0, all_mag.max() * 1.15 if all_mag.max() > 0 else 1.0)
            
            # === Animation state ===
            evo_state = {
                'playing': True,
                'current_frame': 0,
                'ani': None,
                '_updating_slider': False,  # guard to prevent slider callback during animation
            }
            
            def _update_plots(frame):
                """Core update logic shared by animation and manual stepping."""
                evo_state['current_frame'] = frame
                
                # Profile line
                pts = self.xyz_data[frame][selected_indices]
                xvals = pts[:, axis_index]
                z_disp = pts[:, 2] - points_reference[:, 2]
                profile_line.set_xdata(xvals)
                profile_line.set_ydata(z_disp)
                profile_title.set_text(
                    f'Profile Evolution (Z vs {direction}) — Frame {frame} / {num_frames-1}  [{n_pts} pts]')
                
                # Top view — transverse displacement along point positions
                topview_line.set_xdata(xvals)
                topview_line.set_ydata(all_transverse_disp[frame, :])
                topview_title.set_text(
                    f'In-Plane Profile ({topview_dir} disp. vs {direction}) — Frame {frame}')
                
                trail_end = frame + 1
                
                # Trajectory trails
                for pi in range(n_pts):
                    traj_lines_zy[pi].set_data(all_y_disp[:trail_end, pi], all_z_disp[:trail_end, pi])
                    traj_dots_zy[pi].set_data([all_y_disp[frame, pi]], [all_z_disp[frame, pi]])
                    traj_lines_zx[pi].set_data(all_x_disp[:trail_end, pi], all_z_disp[:trail_end, pi])
                    traj_dots_zx[pi].set_data([all_x_disp[frame, pi]], [all_z_disp[frame, pi]])
                    traj_lines_xy[pi].set_data(all_x_disp[:trail_end, pi], all_y_disp[:trail_end, pi])
                    traj_dots_xy[pi].set_data([all_x_disp[frame, pi]], [all_y_disp[frame, pi]])
                
                # Update displacement magnitude bars
                for pi in range(n_pts):
                    mag = np.sqrt(all_x_disp[frame, pi]**2 + all_y_disp[frame, pi]**2 + all_z_disp[frame, pi]**2)
                    disp_bars[pi].set_height(mag)
                
                self.current_frame.set(frame)
                evo_state['_updating_slider'] = True
                evo_slider.config(command='')  # disable callback
                evo_slider.set(frame)
                evo_slider.config(command=on_slider_change)  # restore callback
                evo_state['_updating_slider'] = False
                evo_frame_label.config(text=f"Frame: {frame} / {num_frames-1}")
            
            def animate(frame):
                _update_plots(frame)
                artists = ([profile_line, profile_title, topview_line]
                           + traj_lines_zy + traj_dots_zy
                           + traj_lines_zx + traj_dots_zx
                           + traj_lines_xy + traj_dots_xy
                           + list(disp_bars))
                return artists
            
            # === Transport controls frame (pack BEFORE canvas so it stays visible) ===
            transport_frame = tk.Frame(self.canvas_frame, bg='#ECEFF1', pady=6)
            transport_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            evo_frame_label = tk.Label(transport_frame, text=f"Frame: 0 / {num_frames-1}",
                                       font=('Arial', 10, 'bold'), bg='#ECEFF1')
            evo_frame_label.pack(side=tk.TOP, pady=(0, 3))
            
            # === Embed figure (after transport so canvas fills remaining space) ===
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.current_fig = fig
            
            # Continue transport controls setup
            evo_frame_label_ref = evo_frame_label  # already packed above
            evo_slider = tk.Scale(transport_frame, from_=0, to=num_frames - 1,
                                  orient=tk.HORIZONTAL, length=500, showvalue=False)
            evo_slider.pack(side=tk.TOP, fill=tk.X, padx=20)
            
            def on_slider_change(val):
                if evo_state['_updating_slider']:
                    return  # skip when animation is driving the slider
                frame = int(val)
                # Auto-pause when user drags the slider
                if evo_state['playing']:
                    _pause()
                _update_plots(frame)
                self.canvas.draw_idle()
            
            evo_slider.config(command=on_slider_change)
            
            btn_frame = tk.Frame(transport_frame, bg='#ECEFF1')
            btn_frame.pack(side=tk.TOP, pady=4)
            
            # ---- Toggle slide switch (custom canvas widget) ----
            def _play():
                if evo_state['playing']:
                    return
                evo_state['playing'] = True
                _draw_toggle()
                start = evo_state['current_frame']
                remaining = list(range(start, num_frames)) + list(range(0, start))
                if evo_state['ani'] is not None:
                    evo_state['ani'].event_source.stop()
                evo_state['ani'] = animation.FuncAnimation(
                    fig, animate, frames=remaining,
                    interval=20, blit=False, repeat=True)
                self.current_ani = evo_state['ani']
                self.canvas.draw_idle()
            
            def _pause():
                if not evo_state['playing']:
                    return
                evo_state['playing'] = False
                _draw_toggle()
                if evo_state['ani'] is not None:
                    evo_state['ani'].event_source.stop()
            
            def _toggle_play_pause(event=None):
                if evo_state['playing']:
                    _pause()
                else:
                    _play()
            
            # --- Custom toggle slide button ---
            toggle_w, toggle_h = 80, 32
            toggle_canvas = tk.Canvas(btn_frame, width=toggle_w, height=toggle_h,
                                      bg='#ECEFF1', highlightthickness=0, cursor='hand2')
            
            def _draw_toggle():
                toggle_canvas.delete('all')
                r = toggle_h // 2  # radius of rounded ends
                if evo_state['playing']:
                    # ON state — green track, knob on right
                    track_color = '#4CAF50'
                    knob_x = toggle_w - r
                    label_text = 'PLAY'
                    label_x = toggle_w // 2 - 8
                else:
                    # OFF state — red track, knob on left
                    track_color = '#F44336'
                    knob_x = r
                    label_text = 'STOP'
                    label_x = toggle_w // 2 + 2
                
                # Draw rounded track
                toggle_canvas.create_oval(0, 0, toggle_h, toggle_h, fill=track_color, outline=track_color)
                toggle_canvas.create_oval(toggle_w - toggle_h, 0, toggle_w, toggle_h, fill=track_color, outline=track_color)
                toggle_canvas.create_rectangle(r, 0, toggle_w - r, toggle_h, fill=track_color, outline=track_color)
                # Label on track
                toggle_canvas.create_text(label_x, toggle_h // 2, text=label_text,
                                          fill='white', font=('Arial', 8, 'bold'))
                # Draw knob
                knob_r = r - 3
                toggle_canvas.create_oval(knob_x - knob_r, 3, knob_x + knob_r, toggle_h - 3,
                                          fill='white', outline='#888888', width=1)
            
            toggle_canvas.bind('<Button-1>', _toggle_play_pause)
            _draw_toggle()  # Draw initial state
            
            # Pack step buttons + toggle
            tk.Button(btn_frame, text='|<<', width=4, command=lambda: _step_fn(0, True),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text='<< 10', width=5, command=lambda: _step_fn(-10),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text='< 1', width=4, command=lambda: _step_fn(-1),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            
            toggle_canvas.pack(side=tk.LEFT, padx=10)
            
            tk.Button(btn_frame, text='1 >', width=4, command=lambda: _step_fn(1),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text='10 >>', width=5, command=lambda: _step_fn(10),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text='>>|', width=4, command=lambda: _step_fn(num_frames - 1, True),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            
            def _step_fn(delta, absolute=False):
                if evo_state['playing']:
                    _pause()
                if absolute:
                    f = delta
                else:
                    f = evo_state['current_frame'] + delta
                f = max(0, min(num_frames - 1, f))
                _update_plots(f)
                self.canvas.draw_idle()

            def _export_profile_evolution_hidpi():
                """Export a high-DPI static snapshot with full traces and max alignment highlight."""
                save_path = filedialog.asksaveasfilename(
                    title="Export Profile Evolution Snapshot (HiDPI)",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                               ("SVG files", "*.svg"), ("JPEG files", "*.jpg")]
                )

                if not save_path:
                    return

                try:
                    export_dpi = 600
                    top_n = min(10, num_frames)

                    def _top_misalignment_rankings(coord_a, coord_b, top_count):
                        """Return top misalignment frames and point pairs for a 2D plane."""
                        max_dist = np.zeros(num_frames)
                        pair_i = np.zeros(num_frames, dtype=int)
                        pair_j = np.zeros(num_frames, dtype=int)

                        for f_idx in range(num_frames):
                            da = coord_a[f_idx][:, None] - coord_a[f_idx][None, :]
                            db = coord_b[f_idx][:, None] - coord_b[f_idx][None, :]
                            dist_mat = np.sqrt(da ** 2 + db ** 2)
                            np.fill_diagonal(dist_mat, -np.inf)
                            ii, jj = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
                            pair_i[f_idx] = int(ii)
                            pair_j[f_idx] = int(jj)
                            max_dist[f_idx] = float(dist_mat[ii, jj])

                        top_frames = np.argsort(max_dist)[-top_count:][::-1]
                        return top_frames, max_dist, pair_i, pair_j

                    # Find frame and point pair with maximum Z misalignment.
                    z_spread_per_frame = np.max(all_z_disp, axis=1) - np.min(all_z_disp, axis=1)
                    max_align_frame = int(np.argmax(z_spread_per_frame))
                    max_align_diff = float(z_spread_per_frame[max_align_frame])
                    p_low_local = int(np.argmin(all_z_disp[max_align_frame, :]))
                    p_high_local = int(np.argmax(all_z_disp[max_align_frame, :]))

                    p_low_label = selected_labels[p_low_local]
                    p_high_label = selected_labels[p_high_local]

                    top_zy_frames, zy_max_dist, zy_pair_i, zy_pair_j = _top_misalignment_rankings(all_y_disp, all_z_disp, top_n)
                    top_zx_frames, zx_max_dist, zx_pair_i, zx_pair_j = _top_misalignment_rankings(all_x_disp, all_z_disp, top_n)
                    top_xy_frames, xy_max_dist, xy_pair_i, xy_pair_j = _top_misalignment_rankings(all_x_disp, all_y_disp, top_n)

                    # Find the frame with maximum in-plane misalignment for the top-right plot.
                    trans_spread_per_frame = np.max(all_transverse_disp, axis=1) - np.min(all_transverse_disp, axis=1)
                    inplane_align_frame = int(np.argmax(trans_spread_per_frame))
                    inplane_align_diff = float(trans_spread_per_frame[inplane_align_frame])
                    inplane_low_local = int(np.argmin(all_transverse_disp[inplane_align_frame, :]))
                    inplane_high_local = int(np.argmax(all_transverse_disp[inplane_align_frame, :]))
                    inplane_low_label = selected_labels[inplane_low_local]
                    inplane_high_label = selected_labels[inplane_high_local]

                    print("\n" + "=" * 88)
                    print("PROFILE EVOLUTION HIDPI EXPORT")
                    print(f"Max Z alignment difference: {max_align_diff:.6f} mm")
                    print(f"Frame: {max_align_frame} | Points: P{p_low_label} vs P{p_high_label}")
                    print(f"Max in-plane alignment difference: {inplane_align_diff:.6f} mm")
                    print(f"Frame: {inplane_align_frame} | Points: P{inplane_low_label} vs P{inplane_high_label}")
                    print(f"Top {top_n} misalignments (distance in each plane):")
                    print("  Z-Y: " + ", ".join([
                        f"F{int(f)}={zy_max_dist[int(f)]:.3f}"
                        for f in top_zy_frames
                    ]))
                    print("  Z-X: " + ", ".join([
                        f"F{int(f)}={zx_max_dist[int(f)]:.3f}"
                        for f in top_zx_frames
                    ]))
                    print("  X-Y: " + ", ".join([
                        f"F{int(f)}={xy_max_dist[int(f)]:.3f}"
                        for f in top_xy_frames
                    ]))
                    print("=" * 88)

                    fig_exp = plt.figure(figsize=(20, 10), dpi=export_dpi)
                    gs_exp = fig_exp.add_gridspec(
                        2, 4,
                        height_ratios=[1.05, 1.0],
                        left=0.05, right=0.985, top=0.95, bottom=0.08,
                        hspace=0.34, wspace=0.28
                    )

                    axp = fig_exp.add_subplot(gs_exp[0, 0:2])
                    axt = fig_exp.add_subplot(gs_exp[0, 2:4])
                    axzy_e = fig_exp.add_subplot(gs_exp[1, 0])
                    axzx_e = fig_exp.add_subplot(gs_exp[1, 1])
                    axxy_e = fig_exp.add_subplot(gs_exp[1, 2])
                    axbar_e = fig_exp.add_subplot(gs_exp[1, 3])

                    # Top-row traces for all frames (light), plus highlighted max-misalignment frame.
                    trace_step = max(1, num_frames // 150)
                    for f in range(0, num_frames, trace_step):
                        pts_f = self.xyz_data[f][selected_indices]
                        xvals_f = pts_f[:, axis_index]
                        axp.plot(xvals_f, all_z_disp[f, :], '-', color='#B0BEC5', linewidth=0.8, alpha=0.2)
                        axt.plot(xvals_f, all_transverse_disp[f, :], '-', color='#F8BBD0', linewidth=0.8, alpha=0.2)

                    pts_max = self.xyz_data[max_align_frame][selected_indices]
                    xvals_max = pts_max[:, axis_index]
                    zvals_max = all_z_disp[max_align_frame, :]
                    tvals_max = all_transverse_disp[inplane_align_frame, :]
                    pts_inplane = self.xyz_data[inplane_align_frame][selected_indices]
                    xvals_inplane = pts_inplane[:, axis_index]
                    tvals_inplane = all_transverse_disp[inplane_align_frame, :]

                    axp.plot(xvals_max, zvals_max, 'o-', color='blue', linewidth=3, markersize=7, alpha=1.0)
                    axp.scatter(
                        [xvals_max[p_low_local], xvals_max[p_high_local]],
                        [zvals_max[p_low_local], zvals_max[p_high_local]],
                        s=140, c='yellow', edgecolors='black', linewidths=1.0, zorder=6
                    )

                    axp.set_ylabel('Z Disp. (mm) rel. Frame 0', fontsize=11, fontweight='bold')
                    axp.set_xlabel(x_axis_label, fontsize=11, fontweight='bold')
                    axp.set_title(
                        f'Profile Evolution (all traces) - Max misalignment at frame {max_align_frame}',
                        fontsize=13, fontweight='bold'
                    )
                    axp.grid(True, alpha=0.3, linestyle='--')
                    axp.annotate(
                        f'Max Delta Z = {max_align_diff:.3f} mm\nP{p_low_label} vs P{p_high_label}',
                        xy=((xvals_max[p_low_local] + xvals_max[p_high_local]) / 2,
                            (zvals_max[p_low_local] + zvals_max[p_high_local]) / 2),
                        xytext=(12, 12),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black')
                    )

                    axt.plot(xvals_inplane, tvals_inplane, 'o-', color='#E91E63', linewidth=2.8, markersize=6, alpha=1.0)
                    axt.scatter(
                        [xvals_inplane[inplane_low_local], xvals_inplane[inplane_high_local]],
                        [tvals_inplane[inplane_low_local], tvals_inplane[inplane_high_local]],
                        s=100, c='yellow', edgecolors='black', linewidths=1.0, zorder=6
                    )
                    axt.set_xlabel(topview_x_label, fontsize=11, fontweight='bold')
                    axt.set_ylabel(transverse_label, fontsize=11, fontweight='bold')
                    axt.set_title(f'In-Plane Profile (all traces) - frame {inplane_align_frame}', fontsize=13, fontweight='bold')
                    axt.grid(True, alpha=0.3, linestyle='--')

                    # Full trajectories for all points in lower row.
                    for pi in range(n_pts):
                        c = colors[pi]
                        lbl = f'P{selected_labels[pi]}'
                        axzy_e.plot(all_y_disp[:, pi], all_z_disp[:, pi], '-', color=c, linewidth=1.6, alpha=0.75)
                        axzx_e.plot(all_x_disp[:, pi], all_z_disp[:, pi], '-', color=c, linewidth=1.6, alpha=0.75)
                        axxy_e.plot(all_x_disp[:, pi], all_y_disp[:, pi], '-', color=c, linewidth=1.6, alpha=0.75, label=lbl)
                        axzy_e.plot(all_y_disp[max_align_frame, pi], all_z_disp[max_align_frame, pi], 'o', color=c, markersize=4)
                        axzx_e.plot(all_x_disp[max_align_frame, pi], all_z_disp[max_align_frame, pi], 'o', color=c, markersize=4)
                        axxy_e.plot(all_x_disp[max_align_frame, pi], all_y_disp[max_align_frame, pi], 'o', color=c, markersize=4)

                    axzy_e.set_xlabel('Y Disp. (mm)', fontsize=9, fontweight='bold')
                    axzy_e.set_ylabel('Z Disp. (mm)', fontsize=9, fontweight='bold')
                    axzy_e.set_title('Trajectory: Z vs Y (full trace)', fontsize=10, fontweight='bold')
                    axzy_e.grid(True, alpha=0.3, linestyle='--')
                    axzy_e.set_xlim(*_axis_margin(y_disp_min, y_disp_max))
                    axzy_e.set_ylim(*_axis_margin(z_disp_min, z_disp_max))

                    axzx_e.set_xlabel('X Disp. (mm)', fontsize=9, fontweight='bold')
                    axzx_e.set_ylabel('Z Disp. (mm)', fontsize=9, fontweight='bold')
                    axzx_e.set_title('Trajectory: Z vs X (full trace)', fontsize=10, fontweight='bold')
                    axzx_e.grid(True, alpha=0.3, linestyle='--')
                    axzx_e.set_xlim(*_axis_margin(x_disp_min, x_disp_max))
                    axzx_e.set_ylim(*_axis_margin(z_disp_min, z_disp_max))

                    axxy_e.set_xlabel('X Disp. (mm)', fontsize=9, fontweight='bold')
                    axxy_e.set_ylabel('Y Disp. (mm)', fontsize=9, fontweight='bold')
                    axxy_e.set_title('Trajectory: X vs Y (full trace)', fontsize=10, fontweight='bold')
                    axxy_e.grid(True, alpha=0.3, linestyle='--')
                    axxy_e.set_xlim(*_axis_margin(x_disp_min, x_disp_max))
                    axxy_e.set_ylim(*_axis_margin(y_disp_min, y_disp_max))
                    axxy_e.legend(loc='best', fontsize=6, ncol=min(n_pts, 4))

                    rank_colors = plt.cm.tab10(np.linspace(0, 1, top_n))

                    def _draw_top_events(ax, frames, pair_i, pair_j, coord_x, coord_y, dist_values, title_tag):
                        for rank, fr in enumerate(frames, start=1):
                            i_local = int(pair_i[int(fr)])
                            j_local = int(pair_j[int(fr)])
                            x1 = float(coord_x[int(fr), i_local])
                            y1 = float(coord_y[int(fr), i_local])
                            x2 = float(coord_x[int(fr), j_local])
                            y2 = float(coord_y[int(fr), j_local])
                            c = rank_colors[rank - 1]

                            ax.scatter([x1, x2], [y1, y2], s=85, facecolors=[c], edgecolors='black', linewidths=1.0, zorder=7)

                    _draw_top_events(axzy_e, top_zy_frames, zy_pair_i, zy_pair_j, all_y_disp, all_z_disp, zy_max_dist, 'Z-Y')
                    _draw_top_events(axzx_e, top_zx_frames, zx_pair_i, zx_pair_j, all_x_disp, all_z_disp, zx_max_dist, 'Z-X')
                    _draw_top_events(axxy_e, top_xy_frames, xy_pair_i, xy_pair_j, all_x_disp, all_y_disp, xy_max_dist, 'X-Y')

                    mag_at_max = np.sqrt(
                        all_x_disp[max_align_frame, :] ** 2
                        + all_y_disp[max_align_frame, :] ** 2
                        + all_z_disp[max_align_frame, :] ** 2
                    )
                    bars = axbar_e.bar(range(n_pts), mag_at_max, color=colors[:n_pts], edgecolor='black', linewidth=0.5)
                    bars[p_low_local].set_edgecolor('black')
                    bars[p_low_local].set_linewidth(2.0)
                    bars[p_high_local].set_edgecolor('black')
                    bars[p_high_local].set_linewidth(2.0)
                    axbar_e.set_xticks(range(n_pts))
                    axbar_e.set_xticklabels([f'P{l}' for l in selected_labels], fontsize=7, rotation=45)
                    axbar_e.set_ylabel('|Disp.| (mm)', fontsize=9, fontweight='bold')
                    axbar_e.set_title(f'Disp. Magnitude at frame {max_align_frame}', fontsize=10, fontweight='bold')
                    axbar_e.grid(True, alpha=0.3, linestyle='--', axis='y')
                    axbar_e.set_ylim(0, all_mag.max() * 1.15 if all_mag.max() > 0 else 1.0)

                    for ax in (axzy_e, axzx_e, axxy_e, axbar_e):
                        ax.set_box_aspect(1)

                    fig_exp.align_labels()
                    fig_exp.tight_layout(rect=[0.02, 0.02, 0.99, 0.97])
                    fig_exp.savefig(save_path, dpi=export_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
                    plt.close(fig_exp)

                    messagebox.showinfo("Success", f"High-DPI profile evolution image saved to:\n{save_path}")
                    self.status_bar.config(text=f"Exported HiDPI profile evolution: {save_path}")

                except Exception as export_err:
                    messagebox.showerror("Error", f"Failed to export HiDPI profile evolution:\n{str(export_err)}")
                    self.status_bar.config(text="Error exporting HiDPI profile evolution")

            tk.Button(btn_frame, text='Export HiDPI', command=_export_profile_evolution_hidpi,
                      bg='#1565C0', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=10)
            
            # Start the animation
            evo_state['ani'] = animation.FuncAnimation(
                fig, animate, frames=num_frames,
                interval=20, blit=False, repeat=True)
            self.current_ani = evo_state['ani']
            
            self.status_bar.config(text=f"Profile evolution + top view — {n_pts} points — slide toggle to play/pause")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate profile evolution:\n{str(e)}")
            self.status_bar.config(text="Error generating profile evolution")
    
    def plot_time_series(self):
        """Plot time series displacement for selected points in a floating window"""
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        if not self.selected_point_indices:
            messagebox.showwarning("Warning", "Please pick points first using 'Pick Points Interactively' button!")
            return
        
        try:
            selected_indices = self.selected_point_indices
            selected_labels = [self.point_indices[i] for i in selected_indices]

            # Print peak differences across selected points to terminal.
            self.print_peak_difference_summary(selected_indices, selected_labels)
            
            self.status_bar.config(text=f"Opening time series plot for {len(selected_indices)} points...")
            self.root.update()
            
            # Create time array (frame numbers)
            time_steps = np.arange(len(self.xyz_data))
            
            # Create a new floating window
            time_series_window = tk.Toplevel(self.root)
            time_series_window.title(f"Time Series Displacement - {len(selected_indices)} Points")
            time_series_window.geometry("1200x800")
            
            # Set default save DPI to 600 for publication quality
            plt.rcParams['savefig.dpi'] = 600
            
            # Create matplotlib figure with editable/interactive backend
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, dpi=100)
            
            # Highly distinct colors for easy identification of multiple points
            # Using maximally different colors for clear visual separation
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['#E53935',  # Red
                          '#1E88E5',  # Blue
                          '#43A047',  # Green
                          '#FB8C00',  # Orange
                          '#8E24AA',  # Purple
                          '#00ACC1',  # Cyan
                          '#D81B60',  # Pink
                          '#6D4C41',  # Brown
                          '#FFB300',  # Amber
                          '#5E35B1',  # Deep Purple
                          '#00897B',  # Teal
                          '#C0CA33',  # Lime
                          '#F4511E',  # Deep Orange
                          '#039BE5',  # Light Blue
                          '#7CB342',  # Light Green
                          '#FDD835',  # Yellow
                          '#E91E63',  # Magenta
                          '#00695C',  # Dark Teal
                          '#9E9D24',  # Olive
                          '#6A1B9A']  # Violet
            if len(selected_indices) == 1:
                colors = ['#1E88E5']  # Single professional blue
            elif len(selected_indices) <= len(colors_list):
                colors = colors_list[:len(selected_indices)]
            else:
                # Use repeating pattern for many points
                colors = [colors_list[i % len(colors_list)] for i in range(len(selected_indices))]
            
            # Plot each point's time series
            for idx, (point_idx, label) in enumerate(zip(selected_indices, selected_labels)):
                # Extract time series for this point
                x_series = self.xyz_data[:, point_idx, 0]
                y_series = self.xyz_data[:, point_idx, 1]
                z_series = self.xyz_data[:, point_idx, 2]
                
                # Calculate relative displacement (first frame as reference)
                x_relative = x_series - x_series[0]
                y_relative = y_series - y_series[0]
                z_relative = z_series - z_series[0]
                
                # Alternate line styles: solid for even indices, dashed for odd indices
                linestyle = '-' if idx % 2 == 0 else '--'
                
                # Plot X displacement with alternating line styles
                axes[0].plot(time_steps, x_relative, linestyle, label=f'P{label}', 
                           color=colors[idx], linewidth=2, alpha=0.9)
                
                # Plot Y displacement with alternating line styles
                axes[1].plot(time_steps, y_relative, linestyle, label=f'P{label}', 
                           color=colors[idx], linewidth=2, alpha=0.9)
                
                # Plot Z displacement with alternating line styles
                axes[2].plot(time_steps, z_relative, linestyle, label=f'P{label}', 
                           color=colors[idx], linewidth=2, alpha=0.9)
            
            # Configure X displacement subplot
            axes[0].set_ylabel('Ux (mm)', fontsize=12, fontweight='bold')
            axes[0].set_title(f'Relative Time Series Displacement for {len(selected_indices)} Points (Reference: Frame 0)', 
                            fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].legend(loc='best', fontsize=8, ncol=min(len(selected_indices), 5))
            
            # Configure Y displacement subplot
            axes[1].set_ylabel('Uy (mm)', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].legend(loc='best', fontsize=8, ncol=min(len(selected_indices), 5))
            
            # Configure Z displacement subplot
            axes[2].set_ylabel('Uz (mm)', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Time Step (Frame Number)', fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3, linestyle='--')
            axes[2].legend(loc='best', fontsize=8, ncol=min(len(selected_indices), 5))
            
            # Add vertical line for current frame
            current_frame = self.current_frame.get()
            for ax in axes:
                ax.axvline(x=current_frame, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current Frame')
            
            fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.97])
            
            # Embed plot in the floating window with navigation toolbar
            canvas = FigureCanvasTkAgg(fig, master=time_series_window)
            canvas.draw()
            
            # Add navigation toolbar for zoom, pan, save, etc.
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = tk.Frame(time_series_window)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Add control buttons at the bottom
            control_frame = tk.Frame(time_series_window, bg='lightgray', pady=5)
            control_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            tk.Button(control_frame, text="Save Figure", 
                     command=lambda: self.save_time_series_figure(fig),
                     bg='#4CAF50', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=10)
            
            tk.Button(control_frame, text="Export Data", 
                     command=lambda: self.export_time_series_data(selected_indices, selected_labels),
                     bg='#2196F3', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=10)
            
            tk.Button(control_frame, text="Close Window", 
                     command=time_series_window.destroy,
                     bg='#F44336', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.RIGHT, padx=10)
            
            self.status_bar.config(text=f"Time series window opened for {len(selected_indices)} points")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate time series:\n{str(e)}")
            self.status_bar.config(text="Error generating time series")

    def print_peak_difference_summary(self, selected_indices, selected_labels):
        """Print peak displacement differences between selected points to terminal."""
        if self.xyz_data is None or len(selected_indices) == 0:
            return

        # Relative displacement with frame 0 as reference.
        rel_disp = self.xyz_data[:, selected_indices, :] - self.xyz_data[0, selected_indices, :]
        axis_names = ['X', 'Y', 'Z']

        print("\n" + "=" * 88)
        print("PEAK DISPLACEMENT DIFFERENCE SUMMARY (Reference: Frame 0)")
        print("Selected points: " + ", ".join([f"P{lbl}" for lbl in selected_labels]))
        print("-" * 88)

        # Overall spread among all selected points at each frame, then peak over time.
        for axis_idx, axis_name in enumerate(axis_names):
            spread_per_frame = np.max(rel_disp[:, :, axis_idx], axis=1) - np.min(rel_disp[:, :, axis_idx], axis=1)
            peak_frame = int(np.argmax(spread_per_frame))
            peak_value = float(spread_per_frame[peak_frame])
            print(f"{axis_name} overall max spread: {peak_value:.6f} mm (frame {peak_frame})")

        if len(selected_indices) >= 2:
            print("-" * 88)
            print("Pairwise peak absolute differences (mm):")
            for i in range(len(selected_indices) - 1):
                for j in range(i + 1, len(selected_indices)):
                    pair_diff = np.abs(rel_disp[:, i, :] - rel_disp[:, j, :])
                    peak_vals = np.max(pair_diff, axis=0)
                    peak_frames = np.argmax(pair_diff, axis=0)

                    print(
                        f"P{selected_labels[i]} vs P{selected_labels[j]} -> "
                        f"X: {peak_vals[0]:.6f} (frame {int(peak_frames[0])}), "
                        f"Y: {peak_vals[1]:.6f} (frame {int(peak_frames[1])}), "
                        f"Z: {peak_vals[2]:.6f} (frame {int(peak_frames[2])})"
                    )

        print("=" * 88)
    
    def save_time_series_figure(self, fig):
        """Save the time series figure to a file"""
        save_path = filedialog.asksaveasfilename(
            title="Save Time Series Figure",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("JPEG files", "*.jpg")]
        )
        
        if save_path:
            try:
                # Save with publication quality settings
                fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', 
                           edgecolor='none', format=None, metadata={'DPI': '600'})
                messagebox.showinfo("Success", f"Figure saved to:\n{save_path}")
                self.status_bar.config(text=f"Figure saved: {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save figure:\n{str(e)}")
    
    def export_time_series_data(self, selected_indices, selected_labels):
        """Export time series data to CSV file"""
        save_path = filedialog.asksaveasfilename(
            title="Export Time Series Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        
        if save_path:
            try:
                # Prepare data for export
                time_steps = np.arange(len(self.xyz_data))
                data_dict = {'Time_Step': time_steps}
                
                for idx, (point_idx, label) in enumerate(zip(selected_indices, selected_labels)):
                    x_series = self.xyz_data[:, point_idx, 0]
                    y_series = self.xyz_data[:, point_idx, 1]
                    z_series = self.xyz_data[:, point_idx, 2]
                    
                    # Calculate relative displacement (first frame as reference)
                    x_relative = x_series - x_series[0]
                    y_relative = y_series - y_series[0]
                    z_relative = z_series - z_series[0]
                    
                    data_dict[f'Point_{label}_X_Relative'] = x_relative
                    data_dict[f'Point_{label}_Y_Relative'] = y_relative
                    data_dict[f'Point_{label}_Z_Relative'] = z_relative
                
                df_export = pd.DataFrame(data_dict)
                
                if save_path.endswith('.xlsx'):
                    df_export.to_excel(save_path, index=False)
                else:
                    df_export.to_csv(save_path, index=False)
                
                messagebox.showinfo("Success", f"Data exported to:\n{save_path}")
                self.status_bar.config(text=f"Data exported: {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data:\n{str(e)}")
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (in degrees)"""
        # Using ZYX convention (Rz * Ry * Rx)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        return np.degrees([x, y, z])
    
    def compute_rigid_body_transformation(self, points_initial, points_current):
        """Compute rigid body transformation (translation + rotation) using Kabsch algorithm"""
        # Center the point clouds
        centroid_initial = np.mean(points_initial, axis=0)
        centroid_current = np.mean(points_current, axis=0)
        
        # Translation is the difference in centroids
        translation = centroid_current - centroid_initial
        
        # Center the points
        centered_initial = points_initial - centroid_initial
        centered_current = points_current - centroid_current
        
        # Compute the covariance matrix
        H = centered_initial.T @ centered_current
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) should be 1, not -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Extract Euler angles from rotation matrix
        rotation_angles = self.rotation_matrix_to_euler_angles(R)
        
        return translation, rotation_angles, R
    
    def calculate_rigid_body_motion(self):
        """Calculate 6-DOF rigid body motion from 4 selected points"""
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        if len(self.selected_point_indices) != 4:
            messagebox.showwarning("Warning", 
                "Please select exactly 4 points for rigid body calculation!\n"
                "Current selection: {} points".format(len(self.selected_point_indices)))
            return
        
        try:
            selected_indices = self.selected_point_indices
            selected_labels = [self.point_indices[i] for i in selected_indices]
            
            self.status_bar.config(text="Calculating 6-DOF rigid body motion...")
            self.root.update()
            
            # Reference frame (first frame)
            points_reference = self.xyz_data[0, selected_indices, :]
            
            # Calculate transformations for all frames
            num_frames = len(self.xyz_data)
            translations = np.zeros((num_frames, 3))
            rotations = np.zeros((num_frames, 3))
            
            for frame_idx in range(num_frames):
                points_current = self.xyz_data[frame_idx, selected_indices, :]
                translation, rotation_angles, R = self.compute_rigid_body_transformation(
                    points_reference, points_current)
                
                translations[frame_idx] = translation
                rotations[frame_idx] = rotation_angles
            
            # Create floating window for results
            rigid_body_window = tk.Toplevel(self.root)
            rigid_body_window.title(f"6-DOF Rigid Body Motion - Points {selected_labels}")
            rigid_body_window.geometry("1400x900")
            
            # Set default save DPI to 600 for publication quality
            plt.rcParams['savefig.dpi'] = 600
            
            # Create matplotlib figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=100)
            fig.suptitle(f'6-DOF Rigid Body Motion (4 Points: {selected_labels})', 
                        fontsize=16, fontweight='bold')
            
            time_steps = np.arange(num_frames)
            
            # Publication teal color for translations
            translation_color = '#00796B'
            
            # Plot translations
            axes[0, 0].plot(time_steps, translations[:, 0], '-', color=translation_color, linewidth=2.5)
            axes[0, 0].set_ylabel('Translation X (mm)', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('Translation X', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3, linestyle='--')
            axes[0, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            axes[0, 1].plot(time_steps, translations[:, 1], '-', color=translation_color, linewidth=2.5)
            axes[0, 1].set_ylabel('Translation Y (mm)', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('Translation Y', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, linestyle='--')
            axes[0, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            axes[0, 2].plot(time_steps, translations[:, 2], '-', color=translation_color, linewidth=2.5)
            axes[0, 2].set_ylabel('Translation Z (mm)', fontsize=11, fontweight='bold')
            axes[0, 2].set_title('Translation Z', fontsize=12, fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3, linestyle='--')
            axes[0, 2].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            # Plot rotations with distinct orange/red color
            rotation_color = '#FF6F00'  # Vibrant orange for rotations
            
            axes[1, 0].plot(time_steps, rotations[:, 0], '-', color=rotation_color, linewidth=2.5)
            axes[1, 0].set_ylabel('Rotation X (degrees)', fontsize=11, fontweight='bold')
            axes[1, 0].set_xlabel('Time Step (Frame)', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('Rotation about X-axis', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
            axes[1, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            axes[1, 1].plot(time_steps, rotations[:, 1], '-', color=rotation_color, linewidth=2.5)
            axes[1, 1].set_ylabel('Rotation Y (degrees)', fontsize=11, fontweight='bold')
            axes[1, 1].set_xlabel('Time Step (Frame)', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('Rotation about Y-axis', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, linestyle='--')
            axes[1, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            axes[1, 2].plot(time_steps, rotations[:, 2], '-', color=rotation_color, linewidth=2.5)
            axes[1, 2].set_ylabel('Rotation Z (degrees)', fontsize=11, fontweight='bold')
            axes[1, 2].set_xlabel('Time Step (Frame)', fontsize=11, fontweight='bold')
            axes[1, 2].set_title('Rotation about Z-axis', fontsize=12, fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3, linestyle='--')
            axes[1, 2].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            # Embed plot in the floating window
            canvas = FigureCanvasTkAgg(fig, master=rigid_body_window)
            canvas.draw()
            
            # Add navigation toolbar
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = tk.Frame(rigid_body_window)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Add control buttons
            control_frame = tk.Frame(rigid_body_window, bg='lightgray', pady=5)
            control_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            def save_rigid_body_figure():
                save_path = filedialog.asksaveasfilename(
                    title="Save Rigid Body Motion Figure",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                              ("SVG files", "*.svg")])
                if save_path:
                    # Save with publication quality settings
                    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', 
                               edgecolor='none', format=None, metadata={'DPI': '600'})
                    messagebox.showinfo("Success", f"Figure saved to:\n{save_path}")
            
            def export_rigid_body_data():
                save_path = filedialog.asksaveasfilename(
                    title="Export Rigid Body Motion Data",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
                if save_path:
                    df_export = pd.DataFrame({
                        'Time_Step': time_steps,
                        'Translation_X_mm': translations[:, 0],
                        'Translation_Y_mm': translations[:, 1],
                        'Translation_Z_mm': translations[:, 2],
                        'Rotation_X_deg': rotations[:, 0],
                        'Rotation_Y_deg': rotations[:, 1],
                        'Rotation_Z_deg': rotations[:, 2]
                    })
                    
                    if save_path.endswith('.xlsx'):
                        df_export.to_excel(save_path, index=False)
                    else:
                        df_export.to_csv(save_path, index=False)
                    
                    messagebox.showinfo("Success", f"Data exported to:\n{save_path}")
            
            tk.Button(control_frame, text="Save Figure", command=save_rigid_body_figure,
                     bg='#4CAF50', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=10)
            
            tk.Button(control_frame, text="Export Data", command=export_rigid_body_data,
                     bg='#2196F3', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=10)
            
            # Add statistics label
            stats_text = (
                f"Max Translation: X={np.max(np.abs(translations[:, 0])):.3f} mm, "
                f"Y={np.max(np.abs(translations[:, 1])):.3f} mm, "
                f"Z={np.max(np.abs(translations[:, 2])):.3f} mm | "
                f"Max Rotation: X={np.max(np.abs(rotations[:, 0])):.4f}°, "
                f"Y={np.max(np.abs(rotations[:, 1])):.4f}°, "
                f"Z={np.max(np.abs(rotations[:, 2])):.4f}°"
            )
            tk.Label(control_frame, text=stats_text, font=('Arial', 8), bg='lightgray').pack(side=tk.LEFT, padx=10)
            
            tk.Button(control_frame, text="Close Window", command=rigid_body_window.destroy,
                     bg='#F44336', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.RIGHT, padx=10)
            
            self.status_bar.config(text=f"6-DOF rigid body motion calculated for 4 points")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate rigid body motion:\n{str(e)}")
            self.status_bar.config(text="Error calculating rigid body motion")
    
    def show_animation(self):
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        try:
            self.current_view_mode = None  # Clear view mode
            self.clear_canvas()
            self.status_bar.config(text="Generating animation with heatmap surface...")
            self.root.update()
            
            from scipy.interpolate import griddata
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Reference frame for displacement calculation
            points_reference = self.xyz_data[0]
            num_frames = len(self.xyz_data)
            
            # Initial frame (displacement = 0)
            points_current = self.xyz_data[0]
            z_displacement = points_current[:, 2] - points_reference[:, 2]
            
            # Create grid for surface interpolation
            xi = np.linspace(points_current[:, 0].min(), points_current[:, 0].max(), 50)
            yi = np.linspace(points_current[:, 1].min(), points_current[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate Z displacement values on grid
            Zi = griddata((points_current[:, 0], points_current[:, 1]), z_displacement, (Xi, Yi), method='cubic')
            
            # Calculate global displacement range for consistent color scaling
            z_disp_all = self.xyz_data[:, :, 2] - points_reference[:, 2]
            z_disp_min = z_disp_all.min()
            z_disp_max = z_disp_all.max()
            
            # Plot surface with jet colormap (dark blue to bright red), no transparency
            surf = ax.plot_surface(Xi, Yi, Zi, cmap='jet', alpha=1.0, 
                                  edgecolor='none', antialiased=True, vmin=z_disp_min, vmax=z_disp_max)
            
            ax.set_xlabel('X Position (mm)')
            ax.set_ylabel('Y Position (mm)')
            ax.set_zlabel('Z Rel. Displacement (mm)')
            title = ax.set_title('3D Heatmap Surface — Frame 0 (Rel. Disp. from Frame 0)')
            
            # Set fixed limits — XY from point positions, Z from displacement range
            x_min, x_max = points_reference[:, 0].min(), points_reference[:, 0].max()
            y_min, y_max = points_reference[:, 1].min(), points_reference[:, 1].max()
            z_margin = (z_disp_max - z_disp_min) * 0.1 if z_disp_max != z_disp_min else 0.5
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_disp_min - z_margin, z_disp_max + z_margin])
            
            fig.tight_layout(pad=1.0)
            
            # Add colorbar
            cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Z Displacement (mm)', rotation=270, labelpad=15)
            
            # === Animation state ===
            anim_state = {
                'playing': True,
                'current_frame': 0,
                'ani': None,
                '_updating_slider': False,  # guard to prevent slider callback during animation
            }
            
            def _render_frame(frame):
                """Render a single frame on the 3D surface."""
                anim_state['current_frame'] = frame
                
                # Remove previous plot elements
                for collection in ax.collections[:]:
                    collection.remove()
                for line in ax.lines[:]:
                    line.remove()
                    
                points_current = self.xyz_data[frame]
                z_displacement = points_current[:, 2] - points_reference[:, 2]
                
                Zi = griddata((points_current[:, 0], points_current[:, 1]), z_displacement, (Xi, Yi), method='cubic')
                
                surf = ax.plot_surface(Xi, Yi, Zi, cmap='jet', alpha=1.0, 
                                      edgecolor='none', antialiased=True,
                                      vmin=z_disp_min, vmax=z_disp_max)
                
                # Re-apply fixed axis limits (matplotlib can reset them after replotting)
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                ax.set_zlim([z_disp_min - z_margin, z_disp_max + z_margin])
                
                ax.set_title(f'3D Heatmap Surface — Frame {frame} (Rel. Disp. from Frame 0)')
                
                self.current_frame.set(frame)
                anim_state['_updating_slider'] = True
                anim_slider.config(command='')  # disable callback
                anim_slider.set(frame)
                anim_slider.config(command=on_slider_change)  # restore callback
                anim_state['_updating_slider'] = False
                anim_frame_label.config(text=f"Frame: {frame} / {num_frames-1}")
                return surf,
            
            def animate(frame):
                return _render_frame(frame)
            
            # === Transport controls frame (pack BEFORE canvas so it stays visible) ===
            transport_frame = tk.Frame(self.canvas_frame, bg='#ECEFF1', pady=6)
            transport_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            anim_frame_label = tk.Label(transport_frame, text=f"Frame: 0 / {num_frames-1}",
                                        font=('Arial', 10, 'bold'), bg='#ECEFF1')
            anim_frame_label.pack(side=tk.TOP, pady=(0, 3))
            
            # === Embed figure (after transport so canvas fills remaining space) ===
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            
            # Add navigation toolbar for zoom, pan, rotate during animation
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = tk.Frame(self.canvas_frame)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            toolbar.update()
            
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.current_fig = fig
            
            anim_slider = tk.Scale(transport_frame, from_=0, to=num_frames - 1,
                                   orient=tk.HORIZONTAL, length=500, showvalue=False)
            anim_slider.pack(side=tk.TOP, fill=tk.X, padx=20)
            
            def on_slider_change(val):
                if anim_state['_updating_slider']:
                    return  # skip when animation is driving the slider
                frame = int(val)
                if anim_state['playing']:
                    _pause()
                _render_frame(frame)
                self.canvas.draw_idle()
            
            anim_slider.config(command=on_slider_change)
            
            btn_frame = tk.Frame(transport_frame, bg='#ECEFF1')
            btn_frame.pack(side=tk.TOP, pady=4)
            
            # ---- Toggle slide switch ----
            def _play():
                if anim_state['playing']:
                    return
                anim_state['playing'] = True
                _draw_toggle()
                start = anim_state['current_frame']
                remaining = list(range(start, num_frames)) + list(range(0, start))
                if anim_state['ani'] is not None:
                    anim_state['ani'].event_source.stop()
                anim_state['ani'] = animation.FuncAnimation(
                    fig, animate, frames=remaining,
                    interval=20, blit=False, repeat=True)
                self.current_ani = anim_state['ani']
                self.canvas.draw_idle()
            
            def _pause():
                if not anim_state['playing']:
                    return
                anim_state['playing'] = False
                _draw_toggle()
                if anim_state['ani'] is not None:
                    anim_state['ani'].event_source.stop()
            
            def _toggle_play_pause(event=None):
                if anim_state['playing']:
                    _pause()
                else:
                    _play()
            
            toggle_w, toggle_h = 80, 32
            toggle_canvas = tk.Canvas(btn_frame, width=toggle_w, height=toggle_h,
                                      bg='#ECEFF1', highlightthickness=0, cursor='hand2')
            
            def _draw_toggle():
                toggle_canvas.delete('all')
                r = toggle_h // 2
                if anim_state['playing']:
                    track_color = '#4CAF50'
                    knob_x = toggle_w - r
                    label_text = 'PLAY'
                    label_x = toggle_w // 2 - 8
                else:
                    track_color = '#F44336'
                    knob_x = r
                    label_text = 'STOP'
                    label_x = toggle_w // 2 + 2
                toggle_canvas.create_oval(0, 0, toggle_h, toggle_h, fill=track_color, outline=track_color)
                toggle_canvas.create_oval(toggle_w - toggle_h, 0, toggle_w, toggle_h, fill=track_color, outline=track_color)
                toggle_canvas.create_rectangle(r, 0, toggle_w - r, toggle_h, fill=track_color, outline=track_color)
                toggle_canvas.create_text(label_x, toggle_h // 2, text=label_text,
                                          fill='white', font=('Arial', 8, 'bold'))
                knob_r = r - 3
                toggle_canvas.create_oval(knob_x - knob_r, 3, knob_x + knob_r, toggle_h - 3,
                                          fill='white', outline='#888888', width=1)
            
            toggle_canvas.bind('<Button-1>', _toggle_play_pause)
            _draw_toggle()
            
            def _step_fn(delta, absolute=False):
                if anim_state['playing']:
                    _pause()
                if absolute:
                    f = delta
                else:
                    f = anim_state['current_frame'] + delta
                f = max(0, min(num_frames - 1, f))
                _render_frame(f)
                self.canvas.draw_idle()
            
            # Pack buttons
            tk.Button(btn_frame, text='|<<', width=4, command=lambda: _step_fn(0, True),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text='<< 10', width=5, command=lambda: _step_fn(-10),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text='< 1', width=4, command=lambda: _step_fn(-1),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            
            toggle_canvas.pack(side=tk.LEFT, padx=10)
            
            tk.Button(btn_frame, text='1 >', width=4, command=lambda: _step_fn(1),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text='10 >>', width=5, command=lambda: _step_fn(10),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text='>>|', width=4, command=lambda: _step_fn(num_frames - 1, True),
                      font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
            
            # Start animation
            anim_state['ani'] = animation.FuncAnimation(fig, animate, frames=num_frames,
                                        interval=20, blit=False, repeat=True)
            self.current_ani = anim_state['ani']
            self.status_bar.config(text="Heatmap animation — slide toggle to play/pause, drag slider to scrub")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate animation:\n{str(e)}")
            self.status_bar.config(text="Error generating animation")
    
    def _ask_save_range(self, title, max_frame):
        """Pop up a dialog asking the user for start frame, end frame, step, and FPS.
        Returns (start, end, step, fps) or None if cancelled."""
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.geometry("380x300")
        dlg.resizable(False, False)
        dlg.grab_set()  # modal

        result = {'ok': False}
        var_start = tk.IntVar(value=0)
        var_end   = tk.IntVar(value=max_frame)
        var_step  = tk.IntVar(value=1)
        var_fps   = tk.IntVar(value=30)

        pad = dict(padx=10, pady=4)
        tk.Label(dlg, text=f"Total frames available: 0 - {max_frame}",
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, **pad)

        tk.Label(dlg, text="Start Frame:").grid(row=1, column=0, sticky='w', **pad)
        tk.Entry(dlg, textvariable=var_start, width=12).grid(row=1, column=1, **pad)

        tk.Label(dlg, text="End Frame:").grid(row=2, column=0, sticky='w', **pad)
        tk.Entry(dlg, textvariable=var_end, width=12).grid(row=2, column=1, **pad)

        tk.Label(dlg, text="Frame Step:").grid(row=3, column=0, sticky='w', **pad)
        tk.Entry(dlg, textvariable=var_step, width=12).grid(row=3, column=1, **pad)
        tk.Label(dlg, text="(e.g. 5 = save every 5th frame)",
                 fg='gray', font=('Arial', 8)).grid(row=3, column=1, sticky='e', padx=(130, 5))

        tk.Label(dlg, text="FPS:").grid(row=4, column=0, sticky='w', **pad)
        tk.Entry(dlg, textvariable=var_fps, width=12).grid(row=4, column=1, **pad)

        info_label = tk.Label(dlg, text="", fg='#1565C0', font=('Arial', 10, 'bold'))
        info_label.grid(row=5, column=0, columnspan=2, **pad)

        def _update_info(*_):
            try:
                s, e, st = var_start.get(), var_end.get(), max(1, var_step.get())
                n = len(range(s, e + 1, st))
                info_label.config(text=f"-> {n} frames will be saved")
            except Exception:
                info_label.config(text="")

        var_start.trace_add('write', _update_info)
        var_end.trace_add('write', _update_info)
        var_step.trace_add('write', _update_info)
        _update_info()

        def _ok():
            try:
                s, e, st, f = var_start.get(), var_end.get(), var_step.get(), var_fps.get()
            except Exception:
                messagebox.showwarning("Invalid input", "Please enter valid integers.", parent=dlg)
                return
            if s < 0 or e > max_frame or s >= e or st < 1 or f < 1:
                messagebox.showwarning("Invalid range",
                    f"Please ensure 0 <= start < end <= {max_frame}, step >= 1, and FPS >= 1.",
                    parent=dlg)
                return
            result['ok'] = True
            result['start'] = s
            result['end'] = e
            result['step'] = st
            result['fps'] = f
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        btn_f = tk.Frame(dlg)
        btn_f.grid(row=6, column=0, columnspan=2, pady=12)
        tk.Button(btn_f, text="Save", width=10, bg='#4CAF50', fg='white',
                  font=('Arial', 10, 'bold'), command=_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_f, text="Cancel", width=10, bg='#F44336', fg='white',
                  font=('Arial', 10, 'bold'), command=_cancel).pack(side=tk.LEFT, padx=10)

        dlg.wait_window()
        if result['ok']:
            return result['start'], result['end'], result['step'], result['fps']
        return None

    def save_animation(self):
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        num_frames = len(self.xyz_data)
        rng = self._ask_save_range("Save Animation - Frame Range", num_frames - 1)
        if rng is None:
            return
        start_frame, end_frame, frame_step, fps = rng
        frame_list = list(range(start_frame, end_frame + 1, frame_step))
        
        save_path = filedialog.asksaveasfilename(
            title="Save Animation",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif"), ("MP4 files", "*.mp4")]
        )
        
        if not save_path:
            return
        
        try:
            from scipy.interpolate import griddata
            
            self.status_bar.config(text=f"Saving animation ({len(frame_list)} frames, step={frame_step})...")
            self.root.update()
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Reference frame for displacement calculation
            points_reference = self.xyz_data[0]
            
            # Create grid for surface interpolation
            xi = np.linspace(points_reference[:, 0].min(), points_reference[:, 0].max(), 50)
            yi = np.linspace(points_reference[:, 1].min(), points_reference[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Calculate global displacement range for consistent color scaling
            z_disp_all = self.xyz_data[:, :, 2] - points_reference[:, 2]
            z_disp_min = z_disp_all.min()
            z_disp_max = z_disp_all.max()
            
            # Initial frame
            z_displacement = self.xyz_data[start_frame][:, 2] - points_reference[:, 2]
            Zi = griddata((self.xyz_data[start_frame][:, 0], self.xyz_data[start_frame][:, 1]),
                          z_displacement, (Xi, Yi), method='cubic')
            
            surf = ax.plot_surface(Xi, Yi, Zi, cmap='jet', alpha=1.0, 
                                  edgecolor='none', antialiased=True, vmin=z_disp_min, vmax=z_disp_max)
            
            ax.set_xlabel('X Position (mm)')
            ax.set_ylabel('Y Position (mm)')
            ax.set_zlabel('Z Rel. Displacement (mm)')
            ax.set_title(f'3D Heatmap Surface — Frame {start_frame} (Rel. Disp. from Frame 0)')
            
            # Set fixed limits
            x_min, x_max = points_reference[:, 0].min(), points_reference[:, 0].max()
            y_min, y_max = points_reference[:, 1].min(), points_reference[:, 1].max()
            z_margin = (z_disp_max - z_disp_min) * 0.1 if z_disp_max != z_disp_min else 0.5
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_disp_min - z_margin, z_disp_max + z_margin])
            
            # Add colorbar
            cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Z Displacement (mm)', rotation=270, labelpad=15)
            fig.tight_layout(pad=1.0)
            
            def animate(frame):
                # Remove previous plot elements without clearing axes
                for collection in ax.collections[:]:
                    collection.remove()
                for line in ax.lines[:]:
                    line.remove()
                    
                points_current = self.xyz_data[frame]
                z_displacement = points_current[:, 2] - points_reference[:, 2]
                
                Zi = griddata((points_current[:, 0], points_current[:, 1]), z_displacement, (Xi, Yi), method='cubic')
                
                surf = ax.plot_surface(Xi, Yi, Zi, cmap='jet', alpha=1.0, 
                                      edgecolor='none', antialiased=True,
                                      vmin=z_disp_min, vmax=z_disp_max)
                
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                ax.set_zlim([z_disp_min - z_margin, z_disp_max + z_margin])
                ax.set_title(f'3D Heatmap Surface — Frame {frame} (Rel. Disp. from Frame 0)')
                return surf,
            
            ani = animation.FuncAnimation(fig, animate, frames=frame_list,
                                        interval=20, blit=False)
            
            writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
            ani.save(save_path, writer=writer, fps=fps)
            plt.close(fig)
            
            self.status_bar.config(text=f"Animation saved to {save_path}")
            messagebox.showinfo("Success",
                f"Animation saved to:\n{save_path}\n\nFrames {start_frame}–{end_frame}  |  FPS: {fps}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save animation:\n{str(e)}")
            self.status_bar.config(text="Error saving animation")

    def save_profile_evolution(self):
        """Save profile evolution animation as GIF (or MP4)."""
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please process data first!")
            return
        
        if not self.selected_point_indices:
            messagebox.showwarning("Warning", "Please pick points first using 'Pick Points Interactively' button!")
            return
        
        num_frames = len(self.xyz_data)
        rng = self._ask_save_range("Save Profile Evolution - Frame Range", num_frames - 1)
        if rng is None:
            return
        start_frame, end_frame, frame_step, fps = rng
        frame_list = list(range(start_frame, end_frame + 1, frame_step))
        
        save_path = filedialog.asksaveasfilename(
            title="Save Profile Evolution",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif"), ("MP4 files", "*.mp4")]
        )
        
        if not save_path:
            return
        
        try:
            self.status_bar.config(text=f"Saving profile evolution ({len(frame_list)} frames, step={frame_step})...")
            self.root.update()
            
            selected_indices = self.selected_point_indices
            selected_labels = [self.point_indices[i] for i in selected_indices]
            
            # Determine which direction has more variation (X or Y) using first frame
            first_frame_points = self.xyz_data[0][selected_indices]
            x_range = np.ptp(first_frame_points[:, 0])
            y_range = np.ptp(first_frame_points[:, 1])
            
            if x_range > y_range:
                axis_index = 0
                x_axis_label = 'X Position (mm)'
                direction = 'X'
            else:
                axis_index = 1
                x_axis_label = 'Y Position (mm)'
                direction = 'Y'
            
            # --- Create figure: 2 rows x 4 columns (same layout as interactive) ---
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.38)
            ax_profile = fig.add_subplot(gs[0, 0:3])
            ax_topview = fig.add_subplot(gs[0, 3])
            ax_zy = fig.add_subplot(gs[1, 0])
            ax_zx = fig.add_subplot(gs[1, 1])
            ax_xy_disp = fig.add_subplot(gs[1, 2])
            ax_extra = fig.add_subplot(gs[1, 3])
            
            # Reference frame
            points_reference = self.xyz_data[0][selected_indices]
            
            # Precompute all displacements relative to frame 0
            all_x_disp = np.array([self.xyz_data[f, selected_indices, 0] - points_reference[:, 0] for f in range(num_frames)])
            all_y_disp = np.array([self.xyz_data[f, selected_indices, 1] - points_reference[:, 1] for f in range(num_frames)])
            all_z_disp = np.array([self.xyz_data[f, selected_indices, 2] - points_reference[:, 2] for f in range(num_frames)])
            
            z_disp_min, z_disp_max = all_z_disp.min(), all_z_disp.max()
            x_disp_min, x_disp_max = all_x_disp.min(), all_x_disp.max()
            y_disp_min, y_disp_max = all_y_disp.min(), all_y_disp.max()
            
            # Distinct colors per point
            colors_list = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
                           '#00ACC1', '#D81B60', '#6D4C41', '#FFB300', '#5E35B1',
                           '#00897B', '#C0CA33', '#F4511E', '#039BE5', '#7CB342',
                           '#FDD835', '#E91E63', '#00695C', '#9E9D24', '#6A1B9A']
            n_pts = len(selected_indices)
            colors = [colors_list[i % len(colors_list)] for i in range(n_pts)]
            
            def _axis_margin(vmin, vmax, frac=0.15):
                span = vmax - vmin if vmax != vmin else 1.0
                return vmin - span * frac, vmax + span * frac
            
            # ====== Profile plot (top-left) ======
            points = self.xyz_data[0][selected_indices]
            x_axis_values = points[:, axis_index]
            z_displacement = points[:, 2] - points_reference[:, 2]
            profile_line, = ax_profile.plot(x_axis_values, z_displacement, 'o-', color='blue', linewidth=3, markersize=8)
            ax_profile.set_ylabel('Z Disp.(mm)', fontsize=10, fontweight='bold')
            ax_profile.set_xlabel(x_axis_label, fontsize=10, fontweight='bold')
            profile_title = ax_profile.set_title(
                f'Profile Evolution (Z vs {direction}) — Frame 0 / {num_frames-1}  [{n_pts} pts]',
                fontsize=12, fontweight='bold')
            ax_profile.grid(True, alpha=0.3, linestyle='--')
            y_margin = (z_disp_max - z_disp_min) * 0.15 if z_disp_max != z_disp_min else 1.0
            ax_profile.set_ylim([z_disp_min - y_margin, z_disp_max + y_margin])
            
            # ====== Top View ======
            if axis_index == 0:
                all_transverse_disp = all_y_disp
                transverse_label = 'Y Disp. (mm) rel. Frame 0'
                topview_dir = 'Y'
            else:
                all_transverse_disp = all_x_disp
                transverse_label = 'X Disp. (mm) rel. Frame 0'
                topview_dir = 'X'
            
            trans_disp_min, trans_disp_max = all_transverse_disp.min(), all_transverse_disp.max()
            topview_line, = ax_topview.plot(x_axis_values, np.zeros(n_pts), 'o-', color='#E91E63',
                                            linewidth=2.5, markersize=7, markeredgecolor='black',
                                            markeredgewidth=0.5, zorder=3)
            ax_topview.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
            ax_topview.set_xlabel(x_axis_label, fontsize=9, fontweight='bold')
            ax_topview.set_ylabel(transverse_label, fontsize=9, fontweight='bold')
            topview_title = ax_topview.set_title(
                f'In-Plane Profile ({topview_dir} disp. vs {direction}) — Frame 0',
                fontsize=10, fontweight='bold')
            ax_topview.grid(True, alpha=0.3, linestyle='--')
            t_margin = (trans_disp_max - trans_disp_min) * 0.15 if trans_disp_max != trans_disp_min else 1.0
            ax_topview.set_ylim([trans_disp_min - t_margin, trans_disp_max + t_margin])
            
            # ====== Trajectory plots (bottom row) ======
            traj_lines_zy, traj_dots_zy = [], []
            traj_lines_zx, traj_dots_zx = [], []
            traj_lines_xy, traj_dots_xy = [], []
            
            for pi in range(n_pts):
                lbl = f'P{selected_labels[pi]}'
                c = colors[pi]
                ln, = ax_zy.plot([], [], '-', color=c, linewidth=1.2, alpha=0.6)
                dt, = ax_zy.plot([], [], 'o', color=c, markersize=5, markeredgecolor='black', markeredgewidth=0.4)
                traj_lines_zy.append(ln); traj_dots_zy.append(dt)
                ln2, = ax_zx.plot([], [], '-', color=c, linewidth=1.2, alpha=0.6)
                dt2, = ax_zx.plot([], [], 'o', color=c, markersize=5, markeredgecolor='black', markeredgewidth=0.4)
                traj_lines_zx.append(ln2); traj_dots_zx.append(dt2)
                ln3, = ax_xy_disp.plot([], [], '-', color=c, linewidth=1.2, alpha=0.6, label=lbl)
                dt3, = ax_xy_disp.plot([], [], 'o', color=c, markersize=5, markeredgecolor='black', markeredgewidth=0.4)
                traj_lines_xy.append(ln3); traj_dots_xy.append(dt3)
            
            ax_zy.set_xlabel('Y Disp. (mm)', fontsize=9, fontweight='bold')
            ax_zy.set_ylabel('Z Disp. (mm)', fontsize=9, fontweight='bold')
            ax_zy.set_title('Trajectory: Z vs Y', fontsize=10, fontweight='bold')
            ax_zy.grid(True, alpha=0.3, linestyle='--')
            ax_zy.set_xlim(*_axis_margin(y_disp_min, y_disp_max))
            ax_zy.set_ylim(*_axis_margin(z_disp_min, z_disp_max))
            
            ax_zx.set_xlabel('X Disp. (mm)', fontsize=9, fontweight='bold')
            ax_zx.set_ylabel('Z Disp. (mm)', fontsize=9, fontweight='bold')
            ax_zx.set_title('Trajectory: Z vs X', fontsize=10, fontweight='bold')
            ax_zx.grid(True, alpha=0.3, linestyle='--')
            ax_zx.set_xlim(*_axis_margin(x_disp_min, x_disp_max))
            ax_zx.set_ylim(*_axis_margin(z_disp_min, z_disp_max))
            
            ax_xy_disp.set_xlabel('X Disp. (mm)', fontsize=9, fontweight='bold')
            ax_xy_disp.set_ylabel('Y Disp. (mm)', fontsize=9, fontweight='bold')
            ax_xy_disp.set_title('Trajectory: X vs Y (disp.)', fontsize=10, fontweight='bold')
            ax_xy_disp.grid(True, alpha=0.3, linestyle='--')
            ax_xy_disp.set_xlim(*_axis_margin(x_disp_min, x_disp_max))
            ax_xy_disp.set_ylim(*_axis_margin(y_disp_min, y_disp_max))
            ax_xy_disp.legend(loc='best', fontsize=6, ncol=min(n_pts, 4))
            
            # ====== Bottom-right: displacement magnitude bar chart ======
            bar_labels = [f'P{l}' for l in selected_labels]
            disp_bars = ax_extra.bar(range(n_pts), [0]*n_pts, color=colors[:n_pts], edgecolor='black', linewidth=0.5)
            ax_extra.set_xticks(range(n_pts))
            ax_extra.set_xticklabels(bar_labels, fontsize=7, rotation=45)
            ax_extra.set_ylabel('|Disp.| (mm)', fontsize=9, fontweight='bold')
            ax_extra.set_title('Displacement Magnitude', fontsize=10, fontweight='bold')
            ax_extra.grid(True, alpha=0.3, linestyle='--', axis='y')
            all_mag = np.sqrt(all_x_disp**2 + all_y_disp**2 + all_z_disp**2)
            ax_extra.set_ylim(0, all_mag.max() * 1.15 if all_mag.max() > 0 else 1.0)
            
            def animate(frame):
                pts = self.xyz_data[frame][selected_indices]
                xvals = pts[:, axis_index]
                z_disp = pts[:, 2] - points_reference[:, 2]
                profile_line.set_xdata(xvals)
                profile_line.set_ydata(z_disp)
                profile_title.set_text(
                    f'Profile Evolution (Z vs {direction}) — Frame {frame} / {num_frames-1}  [{n_pts} pts]')
                
                topview_line.set_xdata(xvals)
                topview_line.set_ydata(all_transverse_disp[frame, :])
                topview_title.set_text(
                    f'In-Plane Profile ({topview_dir} disp. vs {direction}) — Frame {frame}')
                
                trail_end = frame + 1
                for pi in range(n_pts):
                    traj_lines_zy[pi].set_data(all_y_disp[:trail_end, pi], all_z_disp[:trail_end, pi])
                    traj_dots_zy[pi].set_data([all_y_disp[frame, pi]], [all_z_disp[frame, pi]])
                    traj_lines_zx[pi].set_data(all_x_disp[:trail_end, pi], all_z_disp[:trail_end, pi])
                    traj_dots_zx[pi].set_data([all_x_disp[frame, pi]], [all_z_disp[frame, pi]])
                    traj_lines_xy[pi].set_data(all_x_disp[:trail_end, pi], all_y_disp[:trail_end, pi])
                    traj_dots_xy[pi].set_data([all_x_disp[frame, pi]], [all_y_disp[frame, pi]])
                
                for pi in range(n_pts):
                    mag = np.sqrt(all_x_disp[frame, pi]**2 + all_y_disp[frame, pi]**2 + all_z_disp[frame, pi]**2)
                    disp_bars[pi].set_height(mag)
                
                return ([profile_line, profile_title, topview_line]
                        + traj_lines_zy + traj_dots_zy
                        + traj_lines_zx + traj_dots_zx
                        + traj_lines_xy + traj_dots_xy
                        + list(disp_bars))
            
            ani = animation.FuncAnimation(fig, animate, frames=frame_list,
                                        interval=20, blit=False)
            
            writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
            ani.save(save_path, writer=writer, fps=fps, dpi=100)
            plt.close(fig)
            
            self.status_bar.config(text=f"Profile evolution saved to {save_path}")
            messagebox.showinfo("Success",
                f"Profile evolution saved to:\n{save_path}\n\nFrames {start_frame}–{end_frame}  |  FPS: {fps}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile evolution:\n{str(e)}")
            self.status_bar.config(text="Error saving profile evolution")


def load_and_process_data(file_path):
    print("Loading data... (this may take a moment for large files)")
    # Assuming no header or skipping standard header rows. 
    # If your file has a header, pandas usually handles it, but accessing by index is safer here.
    df = pd.read_csv(file_path, header=None)
    
    z_data_matrix = []
    point_indices = []

    print("Extracting Z-displacement columns...")
    for i in range(NUM_POINTS_TO_PLOT):
        # Calculate column index for Z component of the current point
        # Global Index = Start + (Point_Step * Stride) + Z_Offset
        point_step = START_POINT_ID + i
        col_idx = START_COLUMN_INDEX + (point_step * STRIDE) + Z_OFFSET
        
        # Check if column exists
        if col_idx < df.shape[1]:
            # Extract the entire time series for this point
            col_data = df.iloc[:, col_idx].values
            z_data_matrix.append(col_data)
            point_indices.append(point_step + 1)
        else:
            print(f"Warning: Column index {col_idx} out of bounds. Stopping at point {i}.")
            break
            
    # Transpose to shape (Time_Steps, Num_Points)
    return np.array(z_data_matrix).T, point_indices

def create_visualization(z_data, point_labels):
    print("Generating visualizations...")
    
    # Use a clean, publication-ready style
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    # --- 1. Animation Setup ---
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(point_labels, z_data[0, :], 'o-', lw=2, markersize=6, color='#1f77b4')
    
    # Set fixed axes limits based on min/max of entire dataset for stability
    z_min, z_max = np.min(z_data), np.max(z_data)
    buffer = (z_max - z_min) * 0.1
    ax.set_ylim(z_min - buffer, z_max + buffer)
    ax.set_xlabel('Sensor/Point Index along Line')
    ax.set_ylabel('Vertical Displacement Z (mm)')
    ax.set_title('Ceiling System Displacement Profile Over Time')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

    def init():
        line.set_ydata(z_data[0, :])
        time_text.set_text('Time Step: 0')
        return line, time_text

    def animate(i):
        line.set_ydata(z_data[i, :])
        time_text.set_text(f'Time Step: {i}')
        return line, time_text

    # Create Animation
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(z_data), interval=50, blit=True)
    
    # Save Animation
    print(f"Saving animation to {OUTPUT_ANIMATION}...")
    # 'pillow' writer handles gifs without extra dependencies. Use 'ffmpeg' for mp4.
    ani.save(OUTPUT_ANIMATION, writer='pillow', fps=15)
    plt.close(fig)

    # --- 2. Static Publication Plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Select a few evenly spaced timestamps to show evolution
    total_time = len(z_data)
    timestamps = [0, total_time//4, total_time//2, 3*total_time//4]
    
    for t in timestamps:
        if t < total_time:
            ax2.plot(point_labels, z_data[t, :], 'o-', label=f'Time T={t}', alpha=0.8)
    
    ax2.set_xlabel('Sensor/Point Index')
    ax2.set_ylabel('Vertical Displacement Z (mm)')
    ax2.set_title('Profile Evolution at Selected Timesteps')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    print(f"Saving static plot to {OUTPUT_STATIC_PLOT}...")
    plt.savefig(OUTPUT_STATIC_PLOT, dpi=600, bbox_inches='tight', facecolor='white', 
               edgecolor='none', format=None, metadata={'DPI': '600'})
    plt.close(fig2)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = CeilingProfileViewer(root)
    root.mainloop()