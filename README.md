# telecom-ai-monitor
AI-Driven Network Anomaly Detector

Project Overview

This project is a smart monitoring system for telecom networks. It uses AI to watch 7 different network metrics at the same time. By looking at how these metrics change together, the system can instantly identify 7 specific types of network problems, ranging from security attacks to physical hardware failures.

Unlike traditional systems that use simple "high/low" alerts, our AI understands the complex patterns of a healthy network and flags anything that looks unusual.

The 7x7 Strategy

1. The 7 Metrics (What we watch)

We monitor these seven values to check the "pulse" of the network:

Traffic Volume: How much data is moving.

Latency: The delay in the connection.

Error Rate: How many packets are failing.

Jitter: How stable the connection timing is.

Throughput: The actual speed of data delivery.

Packet Loss: How much data is getting lost.

CPU Usage: How hard the network hardware is working.

2. The 7 Signatures (What we detect)

The AI can tell exactly what is wrong by matching patterns:

DDoS Attack: Huge traffic spike + slow connection + high CPU usage.

Fiber Cut: Physical cable break (Traffic and speed drop to zero).

Probe Attack: A sudden jump in errors (someone scanning for weaknesses).

Hardware Bottleneck: Slowness caused by old or overloaded equipment.

Resource Exhaustion: Hardware running out of power/memory.

Routing Loop: Data getting stuck in a circle (extreme delays).

VoIP Quality Drop: Poor voice/video call quality (high jitter and loss).

How it Works

The AI Engine: Isolation Forest

We use an algorithm called Isolation Forest. Think of it like a "Spot the Difference" game. The AI assumes that normal data is clustered together. It isolates "weird" data points because they stand out from the crowd.

Short Path = Anomaly: If a data point is easy to separate from others, the AI flags it as a problem.

PEAS Framework

Performance: We measure success by how fast and accurately we find bugs.

Environment: The "world" the AI lives in is the telecom network.

Actuators: How the AI reacts (sending alerts to a dashboard).

Sensors: Where the AI gets data (network logs and probes).

How to Run It

Install Requirements:
pip install pandas numpy scikit-learn matplotlib

Run the Simulation:
Run python main_project.py to see the AI analyze a live stream of data.

View the Dashboard:
Open index.html in your browser to see the high-tech visual control center.

Responsible AI

Explainability: The system tells you why it flagged an alert.

Fairness: It treats all parts of the network equally.

Human Control: The AI helps engineers; it doesn't replace them.
