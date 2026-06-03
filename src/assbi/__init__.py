"""ASSBI — AI-Powered Smart Surveillance & Business Intelligence platform.

A layered (clean-architecture) computer-vision BI system:

    domain/        framework-free entities, value objects and ports
    detection/     object detectors (YOLO adapter + synthetic)
    tracking/      multi-object tracker
    analytics/     line counting, crowd density, anomaly, forecasting
    persistence/   SQLite analytics warehouse
    video/         frame sources (OpenCV + synthetic + YouTube fetch)
    pipeline/      orchestration & composition root
    reporting/     KPIs and exportable BI reports
    chatbot/       natural-language analytics assistant
    dashboard/     Streamlit BI dashboard
"""

__version__ = "1.0.0"
