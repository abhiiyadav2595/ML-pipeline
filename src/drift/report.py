"""
Drift Report Generator.
Generates interactive HTML and JSON reports summarizing feature and prediction drift.
"""

from src.drift.detector import DriftResult


def generate_html_drift_report(result: DriftResult) -> str:
    """
    Generate clean, interactive HTML report with status cards and feature tables.
    """
    overall_status_color = "#e53e3e" if result.dataset_drift_detected else "#38a169"
    overall_status_text = "DRIFT DETECTED" if result.dataset_drift_detected else "STABLE (NO DRIFT)"

    rows_html = []
    for _feat_name, metric in result.feature_metrics.items():
        status_badge = (
            '<span style="background-color: #fed7d7; color: #9b2c2c; padding: 4px 8px; border-radius: 4px; font-weight: bold;">DRIFT</span>'
            if metric.drift_detected
            else '<span style="background-color: #c6f6d5; color: #22543d; padding: 4px 8px; border-radius: 4px; font-weight: bold;">STABLE</span>'
        )
        p_val_str = f"{metric.p_value:.5f}" if metric.p_value is not None else "N/A"
        rows_html.append(f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: 500;">{metric.feature_name}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{metric.feature_type.capitalize()}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{metric.test_name.upper()}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{metric.drift_score:.4f}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{p_val_str}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{metric.threshold:.2f}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{status_badge}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-size: 0.9em; color: #4a5568;">{metric.message}</td>
        </tr>
        """)

    pred_drift_html = ""
    if result.prediction_drift:
        pd = result.prediction_drift
        pd_badge = (
            '<span style="background-color: #fed7d7; color: #9b2c2c; padding: 4px 8px; border-radius: 4px; font-weight: bold;">DRIFT</span>'
            if pd.drift_detected
            else '<span style="background-color: #c6f6d5; color: #22543d; padding: 4px 8px; border-radius: 4px; font-weight: bold;">STABLE</span>'
        )
        pd_pval_str = f"{pd.p_value:.5f}" if pd.p_value is not None else "N/A"
        pred_drift_html = f"""
        <div style="background: white; border-radius: 8px; padding: 20px; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="margin-top:0; color: #2d3748;">Target / Prediction Probability Drift</h3>
            <p>Score (PSI): <strong>{pd.drift_score:.4f}</strong> | P-Value: <strong>{pd_pval_str}</strong> | Status: {pd_badge}</p>
            <p style="color: #4a5568;">{pd.message}</p>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML Model Real-Time Drift Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f7fafc; margin: 0; padding: 30px; color: #1a202c; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .card-title {{ font-size: 0.85em; text-transform: uppercase; color: #718096; margin-bottom: 5px; }}
        .card-value {{ font-size: 1.6em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th {{ background-color: #edf2f7; text-align: left; padding: 12px 10px; font-size: 0.85em; text-transform: uppercase; color: #4a5568; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin: 0; color: #2d3748;">ML Model Drift Detection Report</h1>
                <p style="margin: 5px 0 0 0; color: #718096;">Generated at {result.timestamp}</p>
            </div>
            <div>
                <span style="background-color: {overall_status_color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 1.1em;">
                    {overall_status_text}
                </span>
            </div>
        </div>

        <div class="card-grid">
            <div class="card">
                <div class="card-title">Inference Window Size</div>
                <div class="card-value">{result.sample_size}</div>
            </div>
            <div class="card">
                <div class="card-title">Drifted Features</div>
                <div class="card-value">{result.drifted_features_count} / {result.total_features_count}</div>
            </div>
            <div class="card">
                <div class="card-title">Drift Ratio</div>
                <div class="card-value">{(result.drift_ratio * 100):.1f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Dataset Drift Alert</div>
                <div class="card-value" style="color: {overall_status_color};">{'TRIGGERED' if result.dataset_drift_detected else 'CLEAR'}</div>
            </div>
        </div>

        {pred_drift_html}

        <div style="margin-top: 25px;">
            <h3 style="color: #2d3748; margin-bottom: 12px;">Feature-Level Drift Diagnostics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Type</th>
                        <th>Test</th>
                        <th>Score</th>
                        <th>P-Value</th>
                        <th>Threshold</th>
                        <th>Status</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
    """
    return html
