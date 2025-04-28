#!/usr/bin/env python3
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, cpu_count
from Echolocation.MachineLearning.Training.TrainingConfig import DISTANCE_THRESHOLD

def polar_to_cartesian(distances, angle_range=(-3*np.pi/4, 3*np.pi/4)):
    angles = np.linspace(angle_range[0], angle_range[1], num=len(distances))
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return x, y

def plot_worker(queue, worker_id):
    while True:
        task = queue.get()
        if task is None:
            print(f"Worker {worker_id} shutting down")
            break
        try:
            task_type, data = task
            i, Y_true_i, Y_pred_i, classifications_i, original_gt_i, num_epochs, num_layers, lidar_plots_folder, best_threshold, chosen_dataset = data

            fig, ax = plt.subplots(figsize=(8, 8) if task_type == 'cartesian' else (10, 6))
            if task_type == 'cartesian':
                print(f"Worker {worker_id} plotting cartesian LiDAR for sample {i}...")
                gt_x, gt_y = polar_to_cartesian(Y_true_i)
                pred_x, pred_y = polar_to_cartesian(Y_pred_i)
                ignored_gt = original_gt_i > DISTANCE_THRESHOLD

                ignored_gt_x, ignored_gt_y = polar_to_cartesian(original_gt_i)
                ax.scatter(ignored_gt_x[ignored_gt], ignored_gt_y[ignored_gt], color='red', marker='o', label='Ignored GT', alpha=0.7, zorder=1)
                ax.plot(gt_x, gt_y, label="Ground Truth LiDAR", marker='o', linestyle='-', alpha=0.7, zorder=2)

                robot_circle = plt.Circle((0, 0), 0.2, color='gray', fill=True, alpha=0.5, label='Robot', zorder=2)
                ax.add_patch(robot_circle)

                # draw a line from origin to first scan point
                plt.plot([0, gt_x[0]], [0, gt_y[0]], color='blue', linestyle='--', alpha=0.5, zorder=3)
                plt.plot([0, pred_x[0]], [0, pred_y[0]], color='red', linestyle='--', alpha=0.5, zorder=3)
                # draw a line from origin to last scan point
                plt.plot([0, gt_x[-1]], [0, gt_y[-1]], color='blue', linestyle='--', alpha=0.5, zorder=3)
                plt.plot([0, pred_x[-1]], [0, pred_y[-1]], color='red', linestyle='--', alpha=0.5, zorder=3)
                # draw an arrow vector from origin to middle point(s)
                plt.arrow(0, 0, gt_x[540], gt_y[540], head_width=0.1, head_length=0.2, fc='black', ec='black', alpha=1, zorder=4)
                plt.arrow(0, 0, pred_x[540], pred_y[540], head_width=0.1, head_length=0.2, fc='black', ec='black', alpha=1, zorder=4)

                classified_as_object = classifications_i > best_threshold
                classified_as_no_object = ~classified_as_object
                ax.scatter(pred_x[classified_as_object], pred_y[classified_as_object], color='green', marker='o', s=30, label='Object')
                ax.scatter(pred_x[classified_as_no_object], pred_y[classified_as_no_object], color='orange', marker='o', s=30, label='Not Object')

                ax.set_aspect('equal')
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_title(f"{chosen_dataset}-{i}")
                ax.grid(True)
                ax.legend()
            else:
                print(f"Worker {worker_id} plotting scan index LiDAR for sample {i}...")
                ax.plot(Y_true_i, label="Ground Truth LiDAR", marker="o")
                ax.plot(Y_pred_i, label="Predicted LiDAR", linestyle="--", marker="x")
                ignored_gt = original_gt_i > DISTANCE_THRESHOLD
                ax.scatter(np.arange(len(original_gt_i))[ignored_gt], original_gt_i[ignored_gt], color='red', marker='o', label='Ignored GT')
                classified_as_object = classifications_i > best_threshold
                classified_as_no_object = ~classified_as_object
                ax.scatter(np.arange(len(Y_pred_i))[classified_as_object], Y_pred_i[classified_as_object], color='green', marker='o', s=50, label='Object')
                ax.scatter(np.arange(len(Y_pred_i))[classified_as_no_object], Y_pred_i[classified_as_no_object], color='orange', marker='o', s=50, label='Not Object')
                ax.set_xlabel("Scan Index")
                ax.set_ylabel("Distance (m)")
                ax.set_title(f"{chosen_dataset}-{i} \n Epochs: {num_epochs}, Layers: {num_layers}")
                ax.grid(True)
                ax.legend()

            plot_type = 'cartesian' if task_type == 'cartesian' else 'scan_index'
            filename = f"lidar_prediction_{plot_type}_{i}.png"
            fig.savefig(os.path.join(lidar_plots_folder, filename), bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Error in worker {worker_id} for task {i}: {e}")

def start_multiprocessing_plotting(Y_true, Y_pred, classifications, original_distances_test, num_epochs, num_layers, cartesian_folder, scan_index_folder, best_threshold, chosen_dataset):
    start_time = time.time()
    num_workers = int(cpu_count())
    print(f"Using {num_workers} multiprocessing workers for plotting")

    task_queue = Queue()
    workers = [Process(target=plot_worker, args=(task_queue, i)) for i in range(num_workers)]
    for w in workers:
        w.start()

    for i in range(len(Y_true)):
        task_queue.put(('cartesian', (i, Y_true[i], Y_pred[i], classifications[i], original_distances_test[i], num_epochs, num_layers, cartesian_folder, best_threshold, chosen_dataset)))
        task_queue.put(('scan_index', (i, Y_true[i], Y_pred[i], classifications[i], original_distances_test[i], num_epochs, num_layers, scan_index_folder, best_threshold, chosen_dataset)))

    for _ in range(num_workers):
        task_queue.put(None)

    for w in workers:
        w.join()

    print(f"Multiprocessing plotting completed in {time.time() - start_time:.2f} seconds")
