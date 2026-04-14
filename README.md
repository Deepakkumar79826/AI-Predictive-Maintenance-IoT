# AI-Predictive-Maintenance-IoT
AI-powered predictive maintenance system for IoT/industrial devices using machine sensor data, scikit-learn, virtual simulation, alert generation, and visualization.


# AI-Powered Predictive Maintenance for IoT Devices

## Overview
This project predicts machine/device failure before breakdown using industrial sensor data, machine learning, and virtual IoT simulation. It is designed as a beginner-friendly but industry-oriented predictive maintenance system.

## Problem Statement
Traditional maintenance is reactive:
- machine breaks
- production stops
- downtime increases
- repairs become expensive

This project demonstrates how AI can analyze machine telemetry and predict failure risk early.

## Industry Relevance
Predictive maintenance is widely used in:
- manufacturing plants
- factories
- power plants
- automotive production
- aviation maintenance
- industrial IoT systems

## Project Objective
Build an end-to-end predictive maintenance pipeline that:
- loads maintenance data
- cleans and preprocesses it
- engineers useful features
- trains a machine learning model
- predicts failure probability
- generates alerts
- visualizes results
- simulates live machine readings

## Tech Stack
- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Joblib
- Flask

## Dataset
This project uses the AI4I 2020 Predictive Maintenance dataset.

Main columns used:
- Type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine failure

## Architecture
Raw Data -> Cleaning -> Feature Engineering -> Model Training -> Prediction -> Alerting -> Visualization

## Folder Structure
```text
AI-Predictive-Maintenance-IoT/
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
├── images/
├── docs/
├── README.md
├── requirements.txt
├── .gitignore
└── main.py