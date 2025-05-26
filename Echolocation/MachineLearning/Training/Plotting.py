#!/usr/bin/env python3
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, cpu_count
from MachineLearning.Training.TrainingConfig import DISTANCE_THRESHOLDS, PLOT_DPI, CLASSIFICATION_THRESHOLD
DPI = PLOT_DPI

def polar_to_cartesian(distances, angle_range=(-3*np.pi/4, 3*np.pi/4)):
    """
    Converts polar coordinates (distances and angles) to Cartesian coordinates (x, y).
    Parameters:
        distances (array-like): Array of distances from the origin for each point.
        angle_range (tuple, optional): Tuple specifying the start and end angles (in radians) for the points.
            Defaults to (-3*np.pi/4, 3*np.pi/4).
    Returns:
        tuple: Two numpy arrays representing the x and y Cartesian coordinates corresponding to the input distances.
    """
    
    angles = np.linspace(angle_range[0], angle_range[1], num=len(distances))
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return x, y

def plot_worker(queue, worker_id):
    """
    Worker function for plotting LiDAR prediction results in a multiprocessing environment.
    Continuously retrieves plotting tasks from a queue and generates plots based on the task type.
    Supports both 'cartesian' and 'scan_index' plot types for visualizing ground truth and predicted LiDAR data,
    as well as classification results. Saves the generated plots to the specified folder.
    Parameters:
        queue (multiprocessing.Queue): Queue from which plotting tasks are retrieved. Each task is a tuple
            containing the task type and associated data.
        worker_id (int): Identifier for the worker process, used for logging.
    Task Data Structure:
        Each task is a tuple: (task_type, data)
            - task_type (str): Type of plot to generate ('cartesian' or 'scan_index').
            - data (tuple): Contains the following elements:
                i (int): Sample index.
                Y_true_i (np.ndarray): Ground truth LiDAR data for the sample.
                Y_pred_i (np.ndarray): Predicted LiDAR data for the sample.
                classifications_i (np.ndarray): Classification results for each scan point.
                original_gt_i (np.ndarray): Original ground truth distances.
                num_epochs (int): Number of training epochs (unused in plotting).
                num_layers (int): Number of model layers (unused in plotting).
                lidar_plots_folder (str): Directory to save the generated plots.
                best_threshold (float): Threshold for classifying scan points as objects.
                chosen_dataset (str): Name of the dataset (unused in plotting).
                id (str or int): Unique identifier for the sample.
                distance (float): Distance threshold for ignoring ground truth points.
    Behavior:
        - For 'cartesian' tasks: Plots LiDAR points in Cartesian coordinates, highlighting ground truth,
          predictions, ignored points, and classified objects.
        - For 'scan_index' tasks: Plots LiDAR distances by scan index, highlighting ignored points and
          classified objects.
        - Saves each plot as a PNG file in the specified folder.
        - Shuts down gracefully when a None task is received.
    Raises:
        ValueError: If an invalid task type is provided.
        Exception: Any exception encountered during plotting is printed and re-raised.
    """
    
    while True:
        task = queue.get()
        if task is None:
            print(f"Worker {worker_id} shutting down")
            break
        try:
            task_type, data = task
            i, Y_true_i, Y_pred_i, classifications_i, original_gt_i, num_epochs, num_layers, lidar_plots_folder, best_threshold, chosen_dataset, id, distance = data

            fig, ax = plt.subplots(figsize=(8, 8) if task_type == 'cartesian' else (10, 6), dpi=DPI)
            if task_type == 'cartesian':
                #print(f"Worker {worker_id} plotting cartesian LiDAR for sample {i}...")
                #gt_x, gt_y = polar_to_cartesian(Y_true_i)
                gt_x, gt_y = polar_to_cartesian(original_gt_i)
                pred_x, pred_y = polar_to_cartesian(Y_pred_i)
                ignored_gt = original_gt_i > distance

                ignored_gt_x, ignored_gt_y = polar_to_cartesian(original_gt_i)
                ax.scatter(ignored_gt_x[ignored_gt], ignored_gt_y[ignored_gt], color='red', marker='o', label='GT Ignored', alpha=0.7, zorder=1)
                ax.scatter(gt_x[~ignored_gt], gt_y[~ignored_gt], label="GT LiDAR", marker='o', alpha=0.7, zorder=2)

                robot_circle = plt.Circle((0, 0), 0.2, color='gray', fill=True, alpha=0.5, label='Robot', zorder=2)
                ax.add_patch(robot_circle)

                # draw a line from origin to first scan point
                plt.plot([0, gt_x[0]], [0, gt_y[0]], color='blue', linestyle='--', alpha=0.5, zorder=3)
                plt.plot([0, pred_x[0]], [0, pred_y[0]], color='red', linestyle='--', alpha=0.5, zorder=3)
                # draw a line from origin to last scan point
                plt.plot([0, gt_x[-1]], [0, gt_y[-1]], color='blue', linestyle='--', alpha=0.5, zorder=3)
                plt.plot([0, pred_x[-1]], [0, pred_y[-1]], color='red', linestyle='--', alpha=0.5, zorder=3)
                # draw an arrow vector from origin to middle point(s)
                #plt.arrow(0, 0, gt_x[540], gt_y[540], head_width=0.15, head_length=0.25, fc='black', ec='black', alpha=1, zorder=9)
                plt.arrow(0, 0, 0.1, 0, head_width=0.1, head_length=0.2, fc='black', ec='black', alpha=1, zorder=10)
                
                
                classified_as_object = classifications_i > best_threshold

                classified_as_no_object = ~classified_as_object
                #print(f"classified_as_no_object: {classified_as_no_object}\n")
                #print(f"ignored_gt: {ignored_gt}\n")
                ax.scatter(pred_x[classified_as_object], pred_y[classified_as_object], color='green', marker='o', s=30, label='Object', zorder=6)
                ax.scatter(pred_x[classified_as_no_object], pred_y[classified_as_no_object], color='orange', marker='o', s=30, label='Not Object', zorder=5)

                ax.set_aspect('equal')
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_title(f"{id}")
                ax.grid(True)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            elif task_type == 'scan_index':
                #print(f"Worker {worker_id} plotting scan index LiDAR for sample {i}...")
                ax.plot(Y_true_i, label="Ground Truth LiDAR", marker="o")
                #ax.plot(Y_pred_i, label="Predicted LiDAR", linestyle="--", marker="x")
                ignored_gt = original_gt_i > distance
                ax.scatter(np.arange(len(original_gt_i))[ignored_gt], original_gt_i[ignored_gt], color='red', marker='o', label='Ignored GT')
                classified_as_object = classifications_i > best_threshold
                classified_as_no_object = ~classified_as_object
                ax.scatter(np.arange(len(Y_pred_i))[classified_as_object], Y_pred_i[classified_as_object], color='green', marker='o', s=50, label='Object', zorder=6)
                ax.scatter(np.arange(len(Y_pred_i))[classified_as_no_object], Y_pred_i[classified_as_no_object], color='orange', marker='o', s=50, label='Not Object', zorder=5)
                ax.set_xlabel("Scan Index")
                ax.set_ylabel("Distance (m)")
                ax.set_title(f"{id}")
                ax.grid(True)
                ax.legend()

            else:
                raise ValueError(f"Invalid task type: {task_type}")

            #plot_type = 'cartesian' if task_type == 'cartesian' else 'scan_index'
            filename = f"prediction_{task_type}_{id}.png"
            fig.savefig(os.path.join(lidar_plots_folder, filename), bbox_inches='tight', dpi=DPI)
            plt.close(fig)

        except Exception as e:
            print(f"Error in worker {worker_id} for task {i}: {e}")
            raise e

def start_multiprocessing_plotting(Y_true, Y_pred, classifications, original_distances_test, num_epochs, num_layers, cartesian_folder, 
                                   scan_index_folder, best_threshold, chosen_dataset, ids, distance):
    """
    Starts multiprocessing for plotting results of a machine learning model.
    This function distributes plotting tasks across multiple processes to speed up the
    generation of plots for model predictions and classifications. Each sample in the
    test set is plotted in both cartesian and scan index formats using separate worker
    processes.
    Args:
        Y_true (array-like): Ground truth values for the test set.
        Y_pred (array-like): Predicted values from the model for the test set.
        classifications (array-like): Classification results for each test sample.
        original_distances_test (array-like): Original distance values for the test set.
        num_epochs (int): Number of epochs used during model training.
        num_layers (int): Number of layers in the model.
        cartesian_folder (str): Directory path to save cartesian plots.
        scan_index_folder (str): Directory path to save scan index plots.
        best_threshold (float): Threshold value used for classification.
        chosen_dataset (str): Name or identifier of the dataset used.
        ids (array-like): Identifiers for each test sample.
        distance (float): Distance parameter used for plotting.
    Returns:
        None
    """
    start_time = time.time()
    num_workers = int(cpu_count()-2) if cpu_count() > 1 else 1
    print(f"Using {num_workers} multiprocessing workers for plotting")

    task_queue = Queue()
    workers = [Process(target=plot_worker, args=(task_queue, i)) for i in range(num_workers)]
    for w in workers:
        w.start()

    for i in range(len(Y_true)):
        task_queue.put(('cartesian', (i, Y_true[i], Y_pred[i], classifications[i], original_distances_test[i], num_epochs, num_layers, cartesian_folder, best_threshold, chosen_dataset, ids[i], distance)))
        #task_queue.put(('scan_index', (i, Y_true[i], Y_pred[i], classifications[i], original_distances_test[i], num_epochs, num_layers, scan_index_folder, best_threshold, chosen_dataset, ids[i], distance)))

    for _ in range(num_workers):
        task_queue.put(None)

    for w in workers:
        w.join()

    print(f"Multiprocessing plotting completed in {time.time() - start_time:.2f} seconds")
