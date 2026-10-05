"""
Interactive Web Application Dashboard for ML Model Inference & Monitoring.
Provides real-time credit default risk prediction, live drift evaluation triggers,
system health metrics, and direct links to Swagger API documentation and Prometheus telemetry.
"""

def get_dashboard_html() -> str:
    """Return a production-grade, highly responsive, glassmorphic HTML/CSS/JS dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML RiskOps Platform | Enterprise Real-Time Inference & Drift Engine</title>
    <meta name="description" content="Production ML Model Deployment, Real-Time Inference Engine, and Statistical Drift Detection System.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #060913;
            --bg-surface: rgba(15, 23, 42, 0.75);
            --bg-card: rgba(30, 41, 59, 0.65);
            --bg-card-hover: rgba(51, 65, 85, 0.55);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.35);
            --accent-primary: #6366f1;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --radius-lg: 18px;
            --radius-md: 12px;
            --radius-sm: 8px;
            --shadow-glow: 0 0 35px -5px rgba(99, 102, 241, 0.25);
            --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-sans);
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            line-height: 1.5;
            background-image:
                radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 85% 75%, rgba(6, 182, 212, 0.10) 0%, transparent 50%);
            background-attachment: fixed;
            overflow-x: hidden;
        }

        /* Top Navigation Bar */
        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 36px;
            background: rgba(6, 9, 19, 0.85);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-subtle);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 40px;
            height: 40px;
            background: var(--accent-gradient);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--shadow-glow);
            font-weight: 800;
            font-size: 20px;
        }

        .brand-text h1 {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .brand-text p {
            font-size: 0.75rem;
            color: var(--text-dim);
            font-family: var(--font-mono);
        }

        .nav-links {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .nav-link {
            text-decoration: none;
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 500;
            padding: 8px 14px;
            border-radius: var(--radius-sm);
            border: 1px solid transparent;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .nav-link:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.05);
            border-color: var(--border-subtle);
        }

        .nav-link.github-btn {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.12);
            color: var(--text-main);
        }

        .nav-link.github-btn:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 9999px;
            font-size: 0.78rem;
            color: #34d399;
            font-weight: 600;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-success);
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        /* Container Layout */
        .container {
            max-width: 1320px;
            margin: 0 auto;
            padding: 32px 24px 64px;
        }

        /* Hero Banner */
        .hero {
            margin-bottom: 36px;
            text-align: center;
            padding: 24px 12px;
        }

        .hero-tag {
            display: inline-block;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #38bdf8;
            font-weight: 700;
            margin-bottom: 12px;
            background: rgba(56, 189, 248, 0.1);
            padding: 4px 12px;
            border-radius: 9999px;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }

        .hero h2 {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        .hero p {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 780px;
            margin: 0 auto;
        }

        /* Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 28px;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glassmorphic Cards */
        .glass-card {
            background: var(--bg-surface);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 28px;
            box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.4);
            transition: border-color 0.25s ease;
        }

        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.14);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 22px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-subtitle {
            font-size: 0.85rem;
            color: var(--text-dim);
            margin-top: 3px;
        }

        /* Presets Toolbar */
        .presets-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 24px;
            padding: 12px 14px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
        }

        .presets-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-right: 4px;
        }

        .preset-btn {
            font-family: var(--font-sans);
            font-size: 0.78rem;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-subtle);
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .preset-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-main);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .preset-btn.prime { border-color: rgba(16, 185, 129, 0.3); color: #6ee7b7; }
        .preset-btn.prime:hover { background: rgba(16, 185, 129, 0.15); }
        .preset-btn.subprime { border-color: rgba(239, 68, 68, 0.3); color: #fca5a5; }
        .preset-btn.subprime:hover { background: rgba(239, 68, 68, 0.15); }

        /* Form Inputs */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }

        @media (max-width: 640px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        @media (max-width: 640px) {
            .form-group.full-width {
                grid-column: span 1;
            }
        }

        .form-group label {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }

        .form-group label span.val-preview {
            color: #38bdf8;
            font-family: var(--font-mono);
            font-size: 0.78rem;
        }

        .form-input, .form-select {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            color: var(--text-main);
            padding: 10px 14px;
            font-family: var(--font-sans);
            font-size: 0.9rem;
            transition: all 0.2s ease;
            outline: none;
        }

        .form-input:focus, .form-select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .form-select option {
            background: #0f172a;
            color: var(--text-main);
        }

        /* Predict Action Button */
        .submit-btn {
            width: 100%;
            padding: 14px 24px;
            background: var(--accent-gradient);
            border: none;
            border-radius: var(--radius-md);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            font-family: var(--font-sans);
            cursor: pointer;
            box-shadow: var(--shadow-glow);
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.45);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Results Display Card */
        .result-box {
            margin-top: 24px;
            padding: 20px;
            border-radius: var(--radius-md);
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid var(--border-subtle);
            display: none;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .decision-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: -0.01em;
        }

        .decision-badge.approved {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.35);
            color: #34d399;
        }

        .decision-badge.declined {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #f87171;
        }

        .metric-chip {
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: var(--text-dim);
            background: rgba(255, 255, 255, 0.05);
            padding: 4px 10px;
            border-radius: var(--radius-sm);
        }

        /* Probability Progress Bar */
        .prob-section {
            margin-top: 12px;
        }

        .prob-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }

        .prob-bar-container {
            width: 100%;
            height: 12px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            overflow: hidden;
            position: relative;
        }

        .prob-bar-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1);
            background: linear-gradient(90deg, #10b981 0%, #f59e0b 60%, #ef4444 100%);
        }

        /* Right Column Architecture & Monitoring */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }

        .stat-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 16px;
        }

        .stat-card .stat-label {
            font-size: 0.75rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .stat-card .stat-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .stat-card .stat-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Action Buttons List */
        .actions-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 24px;
        }

        .action-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 18px;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            transition: all 0.2s ease;
        }

        .action-row:hover {
            border-color: rgba(255, 255, 255, 0.15);
            background: rgba(30, 41, 59, 0.5);
        }

        .action-info h4 {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .action-info p {
            font-size: 0.78rem;
            color: var(--text-dim);
        }

        .action-btn {
            font-family: var(--font-sans);
            font-size: 0.8rem;
            font-weight: 600;
            padding: 8px 14px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-subtle);
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-main);
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .action-btn:hover {
            background: var(--accent-primary);
            border-color: var(--accent-primary);
            color: #ffffff;
        }

        /* Live Response Preview Box */
        .live-preview {
            background: #090d16;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 12px;
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: #38bdf8;
            max-height: 180px;
            overflow-y: auto;
            white-space: pre-wrap;
            display: none;
            margin-top: 12px;
        }

        /* Footer */
        .footer {
            margin-top: 60px;
            text-align: center;
            border-top: 1px solid var(--border-subtle);
            padding-top: 24px;
            color: var(--text-dim);
            font-size: 0.82rem;
        }

        .footer a {
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.2s;
        }

        .footer a:hover {
            color: var(--text-main);
        }
    </style>
</head>
<body>

    <!-- Sticky Header Navigation -->
    <header class="navbar">
        <div class="brand-container">
            <div class="brand-logo">⚡</div>
            <div class="brand-text">
                <h1>ML RiskOps Platform</h1>
                <p>Enterprise Model Serving & Drift Monitoring</p>
            </div>
        </div>

        <nav class="nav-links">
            <div class="status-pill">
                <span class="status-dot"></span>
                <span>System Online</span>
            </div>
            <a href="/docs" class="nav-link" id="nav-docs" target="_blank" rel="noopener">
                <span>📖 API Docs</span>
            </a>
            <a href="/drift/report" class="nav-link" id="nav-drift" target="_blank" rel="noopener">
                <span>📊 Drift Report</span>
            </a>
            <a href="/metrics" class="nav-link" id="nav-metrics" target="_blank" rel="noopener">
                <span>📈 Metrics</span>
            </a>
            <a href="https://github.com/abhiiyadav2595/ML-pipeline" class="nav-link github-btn" id="nav-github" target="_blank" rel="noopener">
                <span>GitHub ↗</span>
            </a>
        </nav>
    </header>

    <main class="container">
        <!-- Hero Section -->
        <section class="hero">
            <div class="hero-tag">Automated Drift Detection & Kubernetes Ready</div>
            <h2>Credit Risk Scoring & Monitoring Service</h2>
            <p>
                Continuous dual-layer statistical monitoring utilizing Kolmogorov-Smirnov hypothesis testing for numerical distributions and Population Stability Index (PSI) for categorical covariate shift.
            </p>
        </section>

        <!-- Main Dashboard Grid -->
        <div class="dashboard-grid">
            <!-- Left Card: Interactive Prediction Engine -->
            <section class="glass-card">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">⚡ Real-Time Inference Playground</h3>
                        <p class="card-subtitle">Evaluate live credit default probability for single applications</p>
                    </div>
                </div>

                <!-- Fast Presets Toolbar -->
                <div class="presets-bar">
                    <span class="presets-label">Presets:</span>
                    <button type="button" class="preset-btn prime" id="btn-preset-prime" onclick="loadPreset('prime')">
                        🟢 Prime Applicant (Low Risk)
                    </button>
                    <button type="button" class="preset-btn subprime" id="btn-preset-subprime" onclick="loadPreset('subprime')">
                        🔴 High-Risk Applicant (Subprime)
                    </button>
                    <button type="button" class="preset-btn" id="btn-preset-moderate" onclick="loadPreset('moderate')">
                        🟡 Moderate Applicant
                    </button>
                </div>

                <!-- Input Form -->
                <form id="prediction-form" onsubmit="handleInference(event)">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="income">
                                Annual Income ($)
                                <span class="val-preview" id="preview-income">$75,000</span>
                            </label>
                            <input type="number" class="form-input" id="income" name="income" value="75000" min="0" max="1000000" step="1000" required oninput="updatePreview('income', '$' + Number(this.value).toLocaleString())">
                        </div>

                        <div class="form-group">
                            <label for="credit_score">
                                Credit Score (FICO)
                                <span class="val-preview" id="preview-credit">730</span>
                            </label>
                            <input type="number" class="form-input" id="credit_score" name="credit_score" value="730" min="300" max="850" required oninput="updatePreview('credit', this.value)">
                        </div>

                        <div class="form-group">
                            <label for="debt_to_income">
                                Debt-to-Income Ratio (DTI)
                                <span class="val-preview" id="preview-dti">0.24</span>
                            </label>
                            <input type="number" class="form-input" id="debt_to_income" name="debt_to_income" value="0.24" min="0" max="5" step="0.01" required oninput="updatePreview('dti', this.value)">
                        </div>

                        <div class="form-group">
                            <label for="loan_amount">
                                Loan Principal ($)
                                <span class="val-preview" id="preview-loan">$15,000</span>
                            </label>
                            <input type="number" class="form-input" id="loan_amount" name="loan_amount" value="15000" min="100" max="250000" step="500" required oninput="updatePreview('loan', '$' + Number(this.value).toLocaleString())">
                        </div>

                        <div class="form-group">
                            <label for="age">
                                Applicant Age
                                <span class="val-preview" id="preview-age">36 yrs</span>
                            </label>
                            <input type="number" class="form-input" id="age" name="age" value="36" min="18" max="100" required oninput="updatePreview('age', this.value + ' yrs')">
                        </div>

                        <div class="form-group">
                            <label for="employment_status">Employment Status</label>
                            <select class="form-select" id="employment_status" name="employment_status" required>
                                <option value="employed" selected>Employed</option>
                                <option value="self_employed">Self Employed</option>
                                <option value="unemployed">Unemployed</option>
                                <option value="retired">Retired</option>
                            </select>
                        </div>

                        <div class="form-group full-width">
                            <label for="loan_purpose">Loan Purpose</label>
                            <select class="form-select" id="loan_purpose" name="loan_purpose" required>
                                <option value="debt_consolidation" selected>Debt Consolidation</option>
                                <option value="home_improvement">Home Improvement</option>
                                <option value="business">Business Investment</option>
                                <option value="education">Education</option>
                                <option value="medical">Medical Expenses</option>
                            </select>
                        </div>
                    </div>

                    <button type="submit" class="submit-btn" id="submit-inference-btn">
                        <span>⚡ Run Model Prediction</span>
                    </button>
                </form>

                <!-- Dynamic Prediction Result Output -->
                <div class="result-box" id="prediction-result-box">
                    <div class="result-header">
                        <div class="decision-badge" id="decision-badge">
                            <span id="decision-icon">●</span>
                            <span id="decision-text">EVALUATING</span>
                        </div>
                        <span class="metric-chip" id="latency-chip">0.0 ms</span>
                    </div>

                    <div class="prob-section">
                        <div class="prob-meta">
                            <span style="color: var(--text-muted);">Estimated Default Risk</span>
                            <strong style="color: var(--text-main); font-family: var(--font-mono);" id="prob-percentage">0.0%</strong>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" id="prob-bar-fill" style="width: 0%;"></div>
                        </div>
                    </div>

                    <div style="margin-top: 14px; font-size: 0.78rem; color: var(--text-dim); display: flex; justify-content: space-between;">
                        <span>Model: <strong id="model-ver-span" style="color: var(--text-muted);">v1.0.0</strong></span>
                        <span>Streamed into Drift Window: <strong style="color: #34d399;">✓ Yes</strong></span>
                    </div>
                </div>
            </section>

            <!-- Right Column: Observability & Platform Actions -->
            <aside class="glass-card">
                <div class="card-header">
                    <div>
                        <h3 class="card-title">📊 Observability & System Ops</h3>
                        <p class="card-subtitle">Telemetry, statistical audits, and service probes</p>
                    </div>
                </div>

                <!-- System Stats -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Model Pipeline</div>
                        <div class="stat-value">GBoost</div>
                        <div class="stat-desc">Ensemble Classifier</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Baseline Reference</div>
                        <div class="stat-value">10,000</div>
                        <div class="stat-desc">Historical Records</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Drift Window</div>
                        <div class="stat-value">1,000</div>
                        <div class="stat-desc">Sliding FIFO Buffer</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Statistical Gate</div>
                        <div class="stat-value">KS + PSI</div>
                        <div class="stat-desc">p < 0.05, D ≥ 0.12</div>
                    </div>
                </div>

                <!-- Action Triggers -->
                <div class="actions-list">
                    <div class="action-row">
                        <div class="action-info">
                            <h4>Trigger Statistical Drift Audit</h4>
                            <p>Compute 2-sample KS & PSI against baseline</p>
                        </div>
                        <button type="button" class="action-btn" id="btn-trigger-drift" onclick="triggerDriftEval()">Evaluate</button>
                    </div>

                    <div class="action-row">
                        <div class="action-info">
                            <h4>Interactive HTML Drift Report</h4>
                            <p>Visual feature distribution & stability cards</p>
                        </div>
                        <a href="/drift/report" class="action-btn" target="_blank">View Report ↗</a>
                    </div>

                    <div class="action-row">
                        <div class="action-info">
                            <h4>Kubernetes Liveness / Readiness</h4>
                            <p>Check model memory & reference integrity</p>
                        </div>
                        <button type="button" class="action-btn" id="btn-check-health" onclick="checkHealth()">Probe /health</button>
                    </div>

                    <div class="action-row">
                        <div class="action-info">
                            <h4>Prometheus Metric Exporter</h4>
                            <p>Scrape-ready OpenMetrics telemetry stream</p>
                        </div>
                        <a href="/metrics" class="action-btn" target="_blank">Raw Metrics ↗</a>
                    </div>
                </div>

                <!-- Live JSON Terminal Preview -->
                <div class="live-preview" id="live-preview-box"></div>
            </aside>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <p>
                Deployed on <strong>Vercel Serverless Functions</strong> &nbsp;•&nbsp;
                Powered by <strong>FastAPI, Scikit-Learn, SciPy & Prometheus</strong> &nbsp;•&nbsp;
                <a href="https://github.com/abhiiyadav2595/ML-pipeline" target="_blank" rel="noopener">GitHub: abhiiyadav2595/ML-pipeline</a>
            </p>
        </footer>
    </main>

    <!-- Client Logic -->
    <script>
        const PRESETS = {
            prime: {
                income: 95000,
                credit_score: 790,
                debt_to_income: 0.15,
                loan_amount: 12000,
                age: 42,
                employment_status: "employed",
                loan_purpose: "home_improvement"
            },
            subprime: {
                income: 26000,
                credit_score: 510,
                debt_to_income: 0.58,
                loan_amount: 32000,
                age: 22,
                employment_status: "unemployed",
                loan_purpose: "debt_consolidation"
            },
            moderate: {
                income: 54000,
                credit_score: 650,
                debt_to_income: 0.32,
                loan_amount: 16000,
                age: 31,
                employment_status: "self_employed",
                loan_purpose: "personal"
            }
        };

        function updatePreview(id, val) {
            const el = document.getElementById('preview-' + id);
            if (el) el.innerText = val;
        }

        function loadPreset(type) {
            const p = PRESETS[type];
            if (!p) return;
            document.getElementById('income').value = p.income;
            document.getElementById('credit_score').value = p.credit_score;
            document.getElementById('debt_to_income').value = p.debt_to_income;
            document.getElementById('loan_amount').value = p.loan_amount;
            document.getElementById('age').value = p.age;
            document.getElementById('employment_status').value = p.employment_status;
            document.getElementById('loan_purpose').value = p.loan_purpose;

            updatePreview('income', '$' + p.income.toLocaleString());
            updatePreview('credit', p.credit_score);
            updatePreview('dti', p.debt_to_income);
            updatePreview('loan', '$' + p.loan_amount.toLocaleString());
            updatePreview('age', p.age + ' yrs');
        }

        async function handleInference(e) {
            e.preventDefault();
            const btn = document.getElementById('submit-inference-btn');
            const resultBox = document.getElementById('prediction-result-box');
            btn.disabled = true;
            btn.innerText = "Evaluating...";

            const payload = {
                features: {
                    income: parseFloat(document.getElementById('income').value),
                    credit_score: parseFloat(document.getElementById('credit_score').value),
                    debt_to_income: parseFloat(document.getElementById('debt_to_income').value),
                    loan_amount: parseFloat(document.getElementById('loan_amount').value),
                    age: parseInt(document.getElementById('age').value, 10),
                    employment_status: document.getElementById('employment_status').value,
                    loan_purpose: document.getElementById('loan_purpose').value
                }
            };

            const t0 = performance.now();
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Inference failed');
                }

                const data = await res.json();
                const latency = (performance.now() - t0).toFixed(1);

                resultBox.style.display = 'block';
                const badge = document.getElementById('decision-badge');
                const badgeText = document.getElementById('decision-text');
                const badgeIcon = document.getElementById('decision-icon');
                const isDefault = data.prediction === 1;

                if (isDefault) {
                    badge.className = 'decision-badge declined';
                    badgeText.innerText = 'HIGH DEFAULT RISK (DECLINED)';
                    badgeIcon.innerText = '✕';
                } else {
                    badge.className = 'decision-badge approved';
                    badgeText.innerText = 'LOW DEFAULT RISK (APPROVED)';
                    badgeIcon.innerText = '✓';
                }

                const probPct = (data.probability * 100).toFixed(1) + '%';
                document.getElementById('prob-percentage').innerText = probPct;
                document.getElementById('prob-bar-fill').style.width = probPct;
                document.getElementById('latency-chip').innerText = latency + ' ms';
                document.getElementById('model-ver-span').innerText = data.model_version || 'v1.0.0';

            } catch (err) {
                alert('Prediction Error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>⚡ Run Model Prediction</span>';
            }
        }

        async function triggerDriftEval() {
            const preview = document.getElementById('live-preview-box');
            preview.style.display = 'block';
            preview.innerText = "Running 2-Sample KS & PSI tests over sliding buffer...";
            try {
                const res = await fetch('/drift/evaluate', { method: 'POST' });
                const data = await res.json();
                preview.innerText = "// Statistical Drift Evaluation Result:\\n" + JSON.stringify(data, null, 2);
            } catch (err) {
                preview.innerText = "Error: " + err.message;
            }
        }

        async function checkHealth() {
            const preview = document.getElementById('live-preview-box');
            preview.style.display = 'block';
            preview.innerText = "Probing /health...";
            try {
                const res = await fetch('/health');
                const data = await res.json();
                preview.innerText = "// Health Probe Status:\\n" + JSON.stringify(data, null, 2);
            } catch (err) {
                preview.innerText = "Error: " + err.message;
            }
        }
    </script>
</body>
</html>"""
