# 🚚 Food Delivery Route Optimization

This project focuses on optimizing food delivery routes using **graph algorithms** — specifically **Dijkstra's Algorithm** for finding the shortest delivery path and **Kruskal's Algorithm** for creating an efficient delivery network (Minimum Spanning Tree). The project simulates a city map with synthetic data and helps visualize how deliveries can be optimized to save time and fuel.

## 📌 Problem Statement

In a typical food delivery system, multiple orders need to be delivered from restaurants to customers scattered across a city. Finding the most efficient routes reduces delivery time, improves customer satisfaction, and saves fuel costs. This project aims to:

- Find the **shortest path** from restaurant to customer using **Dijkstra's algorithm**.
- Construct a **minimum delivery network** using **Kruskal's algorithm**.
- Simulate city nodes and roads using **synthetic graph data**.

---

## 🧠 Algorithms Used

### 1. Dijkstra’s Algorithm
Used to find the shortest path from a single source node (e.g., a restaurant) to a target node (e.g., a customer) in a weighted graph.

### 2. Kruskal’s Algorithm
Used to build a **Minimum Spanning Tree (MST)** for all delivery locations to minimize the total length of delivery paths (useful for hub-based deliveries or pre-mapped planning).

---

## 🗺️ Data Simulation

- **Nodes** represent locations (e.g., restaurants, customers).
- **Edges** represent roads with weights as distances or travel times.
- Synthetic data is randomly generated using Python libraries like `random` and `networkx`.

---

## 📁 Project Structure

