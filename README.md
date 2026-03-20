# EH-OW IoT: Energy Harvesting Optical Wireless Networks

**EH-OW IoT** is a Python 3 framework for the simulation, analysis, and optimization of **Energy-Autonomous Optical Wireless IoT** networks. 

This repository provides the supplementary code associated with our upcoming publication. It features an end-to-end physical and electrical model for evaluating optical links, energy harvesting capabilities, and system-level optimization in realistic indoor environments.

---

## 📖 System Architecture

The network architecture models two primary node types:

* **Coordinator Nodes (CN):** * Utilize **Visible Light Communication (VLC)** for downlink data transmission.
    * Equipped with a **Photodiode (PD)** for uplink data reception.
* **Sensor Nodes (SN):** * Transmit uplink data using **Infrared (IR)** signals.
    * Equipped with dual-role **Photovoltaic (PV) panels** for both **data reception** and **energy harvesting (EH)**.

---

## ✨ Key Features

* **Realistic Illumination Modeling:** Supports the integration of natural daylight, artificial LED lighting, and diffuse reflections within the simulation environment.
* **Accurate PV Characteristics:** Models PV panels based on standard mono-crystalline parameters, accounting for efficiency degradation under low light and non-perpendicular angles of incidence.
* **System-Level Optimization:** Includes a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization framework to jointly determine:
    * The **minimum required PV panel area** ($A_\mathrm{PV}$) for energy self-sufficiency.
    * The **optimal 3D orientation** (tilt and azimuth) to maximize EH and satisfy target SNR constraints.
* **Robust Formulation:** Accounts for stochastic, real-world installation misalignments using a minimax robust optimization approach.

---

## 📂 Repository Structure

The repository is organized by optimization approach and validation scripts. Below is a guide to the core files:

| File Pattern | Description |
| :--- | :--- |
| `mod_*` | **Robust Optimization:** Implements the minimax robust CMA-ES framework to account for spatial uncertainty and angular misalignments. |
| `cma_simple_*` | **Standard Optimization:** Implements the baseline CMA-ES optimization without robust perturbation constraints. |
| `cma_def_orientation_*` | **Fixed Orientation:** Executes CMA-ES to minimize PV area while locking the PV orientation to a default alignment (e.g., facing the CN). |
| `run_opt_*` | **Validation Scripts:** Runs the simulation using the final optimized parameters to validate energy neutrality and SNR performance. |

---

## 📘 Publication Notice

This code is part of the supplementary materials for a forthcoming research publication. Full citation details, comprehensive documentation, and installation instructions will be updated here following formal acceptance and publication.

---

## 🧭 Acknowledgment

This code was developed as part of the **[OWIN6G MSCA](https://owin6g.eu)** project, which is supported and funded by the **European Union**.
