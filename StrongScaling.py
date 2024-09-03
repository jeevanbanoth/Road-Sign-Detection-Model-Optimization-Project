#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 00:21:39 2023

@author: bhanuprasadthota
"""
import matplotlib.pyplot as plt


serial_times = [
    178.00680899620056,
    176.5597710609436,
    180.0215609073639
]

parallel_times = [
    # Iteration 1
    [
        174.7447910308838, 167.88549494743347, 167.4947531223297, 165.08890199661255,
        169.44201827049255, 171.57433104515076, 168.91327214241028, 167.55857181549072,
        173.15953016281128, 176.92842078208923, 170.90809535980225, 173.57566332817078,
        172.31759595870972, 174.49727201461792, 171.39628386497498, 174.04913187026978,
        180.65356612205505, 186.76878476142883, 182.18892788887024, 174.17697620391846,
        178.3251247406006, 179.29870319366455, 180.00657892227173, 178.33247303962708,
        178.40558004379272, 181.04651594161987, 176.15590691566467, 172.7112169265747,
        189.27257204055786, 196.29644894599915
    ],
    # Iteration 2
    [
        170.45554184913635, 172.09356427192688, 172.50695300102234, 184.41161608695984,
        189.56986618041992, 182.61392521858215, 179.77614283561707, 192.9037868976593,
        183.21433424949646, 188.59113216400146, 185.9782440662384, 186.12694787979126,
        179.6565670967102, 183.86747980117798, 184.88260221481323, 175.25447821617126,
        170.13092684745789, 177.98629212379456, 172.17314982414246, 180.15226793289185,
        183.72449707984924, 184.60166096687317, 178.1765508651733, 189.8333032131195,
        181.82843804359436, 179.0817620754242, 178.36161184310913, 181.9782440662384,
        175.77614283561707, 177.77614283561707
    ],
    # Iteration 3
    [
        186.9880998134613, 175.30772805213928, 176.14331007003784, 177.26491284370422,
        177.6541759967804, 174.95713186264038, 176.7453908920288, 175.88008904457092,
        173.73473477363586, 176.54509592056274, 177.26160597801208, 177.60577988624573,
        179.85051369667053, 176.8400149345398, 179.05255722999573, 177.0462930202484,
        181.36566305160522, 183.69880723953247, 179.63936805725098, 181.84070801734924,
        181.99016499519348, 179.13007998466492, 185.35088109970093, 177.78096985816956,
        179.8942790031433, 179.61621189117432, 177.2784607410431, 178.62508416175842,
        178.5436270236969, 178.4235770702362
    ]
]

# Calculate the speedup for each worker configuration across all iterations
speedups = []
for worker_index in range(len(parallel_times[0])):
    worker_speedups = []
    for iteration_index, serial_time in enumerate(serial_times):
        speedup = serial_time / parallel_times[iteration_index][worker_index]
        worker_speedups.append(speedup)
    average_speedup = sum(worker_speedups) / len(worker_speedups)
    speedups.append(average_speedup)

# Display the results
for worker_count, speedup in enumerate(speedups, start=1):
    print(f"Average speedup for {worker_count} workers: {speedup:.2f}x")

# Find the maximum speedup and the corresponding number of workers
max_speedup = max(speedups)
optimal_workers = speedups.index(max_speedup) + 1
print(f"Maximum average speedup of {max_speedup:.2f}x was achieved with {optimal_workers} workers")


worker_counts = list(range(1, len(speedups) + 1))

plt.figure(figsize=(10, 5))
plt.plot(worker_counts, speedups, marker='o', linestyle='-', color='b')
plt.title('Average Speedup vs. Number of Workers')
plt.xlabel('Number of Workers')
plt.ylabel('Average Speedup')
plt.xticks(worker_counts)
plt.grid(True)

# Annotate the optimal point
plt.annotate(f'Max speedup: {max_speedup:.2f}x\nWorkers: {optimal_workers}',
             xy=(optimal_workers, max_speedup),
             xytext=(optimal_workers+3, max_speedup-0.5),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=10,
             color='green')

plt.show()

efficiencies = [speedup / worker_count for worker_count, speedup in enumerate(speedups, start=1)]

# Plot efficiency
plt.figure(figsize=(10, 5))
plt.plot(worker_counts, efficiencies, marker='o', linestyle='-', color='r')
plt.title('Efficiency vs. Number of Workers')
plt.xlabel('Number of Workers')
plt.ylabel('Efficiency')
plt.xticks(worker_counts)
plt.grid(True)
plt.show()


