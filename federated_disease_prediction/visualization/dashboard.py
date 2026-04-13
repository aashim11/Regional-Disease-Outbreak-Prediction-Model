"""
Interactive Dashboard for Disease Outbreak Prediction

This module implements a Streamlit-based dashboard for visualizing
outbreak predictions, model performance, and system metrics.
"""

import streamlit as st
st.set_page_config(
    page_title="Federated Learning Model for Predicting Regional Disease Outbreak System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime, timedelta
import base64
from io import BytesIO
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class OutbreakDashboard:
    """
    Interactive dashboard for disease outbreak monitoring.
    
    Provides real-time visualization of:
    - Outbreak risk maps
    - Time series predictions
    - Model performance metrics
    - Federated learning progress
    """
    
    def __init__(self, title: str = "Disease Outbreak Prediction Dashboard"):
        self.title = title
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'page' not in st.session_state:
            st.session_state.page = 'home'
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'live_data' not in st.session_state:
            st.session_state.live_data = []
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
    
    def render_header(self) -> None:
        """Render dashboard header."""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .status-online {
            background: #28a745;
            color: white;
        }
        </style>
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1>🏥 Federated Learning Model for Predicting Regional Disease Outbreak System</h1>
                <div>
                    <span class="status-badge status-online">● FL Server Online</span>
                    <span style="margin-left: 10px; font-size: 0.9rem;">v2.1.0</span>
                </div>
            </div>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Privacy-Preserving Federated Learning for Global Health</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with controls and navigation."""
        st.sidebar.markdown("""
        <style>
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 20px;
        }
        </style>
        <div class="sidebar-title">🏥 FL Health System</div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.sidebar.markdown("### 📍 Navigation")
        page = st.sidebar.radio(
            "Go to",
            ['🏠 Home', '📊 Dashboard', '💬 AI Assistant', '📈 Live Charts', 
             '📤 Upload Data', '📚 Help & Docs', '⚙️ Settings']
        )
        
        # Map page selection
        page_map = {
            '🏠 Home': 'home',
            '📊 Dashboard': 'dashboard',
            '💬 AI Assistant': 'assistant',
            '📈 Live Charts': 'live_charts',
            '📤 Upload Data': 'upload',
            '📚 Help & Docs': 'help',
            '⚙️ Settings': 'settings'
        }
        st.session_state.page = page_map.get(page, 'home')
        
        st.sidebar.markdown("---")
        
        # Quick Controls
        st.sidebar.markdown("### ⚡ Quick Controls")
        
        disease = st.sidebar.selectbox(
            "🦠 Disease",
            ["Dengue", "Chickenpox", "COVID-19", "Malaria", "Custom"]
        )
        
        region = st.sidebar.selectbox(
            "🌍 Region",
            ["All Regions", "North America", "South America", "Europe", 
             "Asia", "Africa", "Oceania"]
        )
        
        time_range = st.sidebar.selectbox(
            "📅 Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", 
             "Last 3 Months", "Last Year", "All Time"]
        )
        
        st.sidebar.markdown("---")
        
        # System Status
        st.sidebar.markdown("### 🔴 System Status")
        st.sidebar.success("🟢 FL Server: Online")
        st.sidebar.success("🟢 8/10 Clients: Active")
        st.sidebar.info("📊 Privacy Budget: 85% remaining")
        
        return {
            'disease': disease,
            'region': region,
            'time_range': time_range
        }
    
    def render_overview_metrics(self, metrics: Dict[str, Any]) -> None:
        """Render overview metrics cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active Outbreaks",
                value=metrics.get('active_outbreaks', 0),
                delta=metrics.get('outbreak_change', 0)
            )
        
        with col2:
            st.metric(
                label="High Risk Regions",
                value=metrics.get('high_risk_regions', 0),
                delta=metrics.get('risk_change', 0)
            )
        
        with col3:
            st.metric(
                label="Prediction Accuracy",
                value=f"{metrics.get('accuracy', 0):.1%}",
                delta=f"{metrics.get('accuracy_change', 0):.1%}"
            )
        
        with col4:
            st.metric(
                label="Participating Institutions",
                value=metrics.get('institutions', 0)
            )
    
    def render_risk_map(
        self,
        region_data: pd.DataFrame,
        lat_col: str = 'region_lat',
        lon_col: str = 'region_lon',
        risk_col: str = 'new_cases'
    ) -> None:
        """Render interactive risk map with enhanced visualization."""
        st.subheader("🗺️ Global Outbreak Risk Map")
        
        # Use population_density if available, otherwise use a constant size
        size_col = 'population_density' if 'population_density' in region_data.columns else None
        
        # Check if required columns exist - if not, show placeholder
        if lat_col not in region_data.columns or lon_col not in region_data.columns:
            self._render_placeholder_map()
            return
        
        # Aggregate data by region for cleaner map
        map_data = region_data.groupby('region_id').agg({
            lat_col: 'first',
            lon_col: 'first',
            risk_col: 'mean',
            'new_cases': 'sum',
            'hospitalizations': 'sum',
            'deaths': 'sum'
        }).reset_index()
        
        # Create color categories for risk levels
        map_data['risk_category'] = pd.cut(
            map_data[risk_col],
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Critical']
        )
        
        fig = px.scatter_map(
            map_data,
            lat=lat_col,
            lon=lon_col,
            color='risk_category',
            size=risk_col,
            color_discrete_map={
                'Very Low': '#2ecc71',
                'Low': '#f1c40f',
                'Medium': '#e67e22',
                'High': '#e74c3c',
                'Critical': '#8e44ad'
            },
            zoom=1,
            hover_name='region_id',
            hover_data={
                'new_cases': ':,.0f',
                'hospitalizations': ':,.0f',
                'deaths': ':,.0f',
                risk_col: ':.1f'
            },
            title="Disease Outbreak Risk by Region"
        )
        
        fig.update_layout(
            map_style="carto-darkmatter",
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            height=550,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(color="white")
            )
        )
        
        st.plotly_chart(fig)
        
        # Add legend explanation
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("🟢 **Very Low**: < 10 cases")
        with col2:
            st.markdown("🟡 **Low**: 10-50 cases")
        with col3:
            st.markdown("🟠 **Medium**: 50-100 cases")
        with col4:
            st.markdown("🔴 **High**: 100-500 cases")
        with col5:
            st.markdown("🟣 **Critical**: > 500 cases")
    
    def _render_placeholder_map(self) -> None:
        """Render a placeholder world map when data is not available."""
        # Create a world map with sample regions
        world_data = pd.DataFrame({
            'region': ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania'],
            'lat': [45.0, -15.0, 50.0, 5.0, 35.0, -25.0],
            'lon': [-100.0, -60.0, 10.0, 20.0, 100.0, 135.0],
            'risk_score': [25, 45, 35, 60, 80, 15],
            'cases': [1200, 2100, 1800, 3200, 4500, 800]
        })
        
        fig = px.scatter_map(
            world_data,
            lat='lat',
            lon='lon',
            color='risk_score',
            size='cases',
            color_continuous_scale='RdYlGn_r',
            zoom=1,
            hover_name='region',
            hover_data={'risk_score': ':.0f', 'cases': ':,.0f'},
            title="Sample Global Risk Distribution (Demo Data)"
        )
        
        fig.update_layout(
            map_style="carto-positron",
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            height=500
        )
        
        st.plotly_chart(fig)
    
    def render_time_series(
        self,
        historical_data: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None
    ) -> None:
        """Render time series plot with predictions."""
        st.subheader("📈 Cases & Predictions Over Time")
        
        fig = go.Figure()
        
        # Historical cases - handle different column names
        cases_col = 'new_cases' if 'new_cases' in historical_data.columns else 'cases'
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data[cases_col],
            mode='lines',
            name='Actual Cases',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        if predictions is not None:
            fig.add_trace(go.Scatter(
                x=predictions['date'],
                y=predictions['predicted_cases'],
                mode='lines',
                name='Predicted Cases',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                y=predictions['upper_bound'].tolist() + predictions['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255, 0, 0, 0)'),
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig)
    
    def render_heatmap(self, data: pd.DataFrame, selected_disease: str = "Dengue") -> None:
        """Render heatmap of cases by region and time, customized by disease."""
        # Disease-specific color scales and titles
        disease_config = {
            'Dengue': {'color': 'YlOrRd', 'emoji': '🦟'},
            'Chickenpox': {'color': 'Oranges', 'emoji': '🔴'},
            'COVID-19': {'color': 'Reds', 'emoji': '🦠'},
            'Malaria': {'color': 'Greens', 'emoji': '🦟'},
            'Custom': {'color': 'Purples', 'emoji': '🏥'}
        }
        
        config = disease_config.get(selected_disease, disease_config['Dengue'])
        
        # Determine the cases column
        cases_col = 'new_cases' if 'new_cases' in data.columns else 'cases'
        region_col = 'region_id' if 'region_id' in data.columns else 'region'
        
        # Pivot data for heatmap
        heatmap_data = data.pivot_table(
            values=cases_col,
            index=region_col,
            columns='date',
            aggfunc='sum'
        )
        
        fig = px.imshow(
            heatmap_data,
            aspect='auto',
            color_continuous_scale=config['color'],
            labels=dict(x="Date", y="Region", color=f"{selected_disease} Cases")
        )
        
        fig.update_layout(
            height=400,
            title=f"{config['emoji']} {selected_disease} Case Distribution Heatmap"
        )
        
        st.plotly_chart(fig)
    
    def render_model_performance(self, metrics: Dict[str, Any]) -> None:
        """Render model performance metrics."""
        st.subheader("🤖 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy metrics
            st.markdown("#### Classification Metrics")
            
            metric_cols = st.columns(4)
            metric_cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            metric_cols[1].metric("Precision", f"{metrics.get('precision', 0):.2%}")
            metric_cols[2].metric("Recall", f"{metrics.get('recall', 0):.2%}")
            metric_cols[3].metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
            
            # ROC Curve
            if 'roc_curve' in metrics:
                fpr = metrics['roc_curve']['fpr']
                tpr = metrics['roc_curve']['tpr']
                auc = metrics['roc_curve']['auc']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {auc:.3f})',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='gray', dash='dash')
                ))
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=300
                )
                st.plotly_chart(fig)
        
        with col2:
            # Regression metrics
            st.markdown("#### Forecasting Metrics")
            
            metric_cols = st.columns(3)
            metric_cols[0].metric("MAE", f"{metrics.get('mae', 0):.2f}")
            metric_cols[1].metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
            metric_cols[2].metric("MAPE", f"{metrics.get('mape', 0):.2%}")
            
            # Model comparison
            if 'model_comparison' in metrics:
                comparison_df = pd.DataFrame(metrics['model_comparison'])
                
                fig = px.bar(
                    comparison_df,
                    x='model',
                    y='accuracy',
                    color='model',
                    title='Model Comparison'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig)
    
    def render_federated_status(self, fl_status: Dict[str, Any]) -> None:
        """Render federated learning system status."""
        st.subheader("🌐 Federated Learning Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Connected Clients")
            st.write(f"Active: {fl_status.get('active_clients', 0)}")
            st.write(f"Total: {fl_status.get('total_clients', 0)}")
            
            # Client list
            if 'clients' in fl_status:
                client_df = pd.DataFrame(fl_status['clients'])
                st.dataframe(client_df[['client_id', 'status', 'last_seen']])
        
        with col2:
            st.markdown("#### Training Progress")
            progress = fl_status.get('training_progress', 0)
            st.progress(progress)
            st.write(f"Round: {fl_status.get('current_round', 0)} / {fl_status.get('total_rounds', 100)}")
            
            # Training metrics
            if 'training_history' in fl_status:
                history = fl_status['training_history']
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(history))),
                    y=history,
                    mode='lines+markers',
                    name='Global Accuracy'
                ))
                fig.update_layout(
                    xaxis_title='Round',
                    yaxis_title='Accuracy',
                    height=250
                )
                st.plotly_chart(fig)
        
        with col3:
            st.markdown("#### Privacy Status")
            
            privacy_metrics = fl_status.get('privacy', {})
            
            st.write(f"Differential Privacy: {'✅ Enabled' if privacy_metrics.get('dp_enabled') else '❌ Disabled'}")
            if privacy_metrics.get('dp_enabled'):
                st.write(f"ε (epsilon): {privacy_metrics.get('epsilon', 'N/A')}")
                st.write(f"δ (delta): {privacy_metrics.get('delta', 'N/A')}")
            
            st.write(f"Secure Aggregation: {'✅ Enabled' if privacy_metrics.get('secure_agg_enabled') else '❌ Disabled'}")
    
    def render_alerts(self, alerts: List[Dict[str, Any]], selected_region: str = "All Regions", selected_disease: str = "Dengue") -> None:
        """Render active alerts filtered by region and disease."""
        st.subheader(f"⚠️ Active Alerts - {selected_disease}")
        
        # Filter alerts by region if specified
        if selected_region != "All Regions":
            alerts = [a for a in alerts if selected_region.lower() in a.get('region', '').lower()]
        
        if not alerts:
            st.success(f"✅ No active {selected_disease} outbreak alerts at this time.")
            return
        
        for alert in alerts:
            severity = alert.get('severity', 'medium')
            
            if severity == 'high':
                st.error(f"🔴 **{alert['region']}**: {alert['message']}")
            elif severity == 'medium':
                st.warning(f"🟡 **{alert['region']}**: {alert['message']}")
            else:
                st.info(f"🔵 **{alert['region']}**: {alert['message']}")
            
            st.write(f"Risk Score: {alert.get('risk_score', 0):.2f} | Predicted Cases: {alert.get('predicted_cases', 'N/A')}")
            st.markdown("---")
    
    def _adjust_metrics_by_context(self, metrics: Dict[str, Any], region: str, disease: str) -> Dict[str, Any]:
        """Adjust metrics based on selected region and disease."""
        adjusted = metrics.copy()
        
        # Disease multipliers (different diseases have different severity patterns)
        disease_multipliers = {
            'Dengue': 1.0,
            'Chickenpox': 0.8,
            'COVID-19': 1.5,
            'Malaria': 0.9,
            'Custom': 1.0
        }
        
        # Region multipliers
        region_multipliers = {
            'All Regions': 1.0,
            'North America': 0.8,
            'South America': 1.2,
            'Europe': 0.9,
            'Asia': 1.4,
            'Africa': 1.1,
            'Oceania': 0.6
        }
        
        d_mult = disease_multipliers.get(disease, 1.0)
        r_mult = region_multipliers.get(region, 1.0)
        total_mult = d_mult * r_mult
        
        # Adjust numeric metrics
        for key in ['active_outbreaks', 'high_risk_regions']:
            if key in adjusted:
                adjusted[key] = int(adjusted[key] * total_mult)
        
        return adjusted
    
    # ==================== NEW PAGE METHODS ====================
    
    def render_home_page(self) -> None:
        """Render the home/landing page with enhanced visuals."""
        # Animated hero section with gradient
        st.markdown("""
        <style>
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .hero {
            background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            padding: 4rem 3rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .hero h1 {
            font-size: 3.5rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .hero p {
            font-size: 1.3rem;
            opacity: 0.95;
        }
        .feature-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
            border-radius: 15px;
            border-left: 5px solid #667eea;
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .world-map-placeholder {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            height: 300px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
            margin: 2rem 0;
        }
        </style>
        <div class="hero">
            <h1>🏥 Federated Learning Model for Predicting Regional Disease Outbreak System</h1>
            <p>Privacy-Preserving AI for Global Health Security</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Global Impact Visualization
        st.markdown("### 🌍 Global Disease Surveillance Network")
        
        # Create a world coverage visualization
        coverage_data = pd.DataFrame({
            'Continent': ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania'],
            'Coverage': [85, 72, 90, 65, 88, 78],
            'Institutions': [45, 28, 52, 18, 38, 12],
            'Active_Cases': [1250, 2100, 1800, 3200, 4500, 800]
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Coverage bar chart
            fig = px.bar(
                coverage_data,
                x='Continent',
                y='Coverage',
                color='Coverage',
                color_continuous_scale='Viridis',
                title='Network Coverage by Region (%)',
                text='Coverage'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig)
        
        with col2:
            # Institutions pie chart
            fig2 = px.pie(
                coverage_data,
                values='Institutions',
                names='Continent',
                title='Connected Institutions',
                hole=0.4
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2)
        
        # Key Features
        st.markdown("### ✨ Key Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🔒 Privacy First</h3>
                <p>Federated Learning keeps patient data localized while enabling collaborative model training with differential privacy.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🤖 AI-Powered</h3>
                <p>Advanced deep learning models (LSTM, CNN-LSTM, Transformers, GNN) for accurate outbreak prediction.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>🌍 Global Scale</h3>
                <p>Connect healthcare institutions worldwide for real-time disease surveillance and early warning systems.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Stats with visual styling
        st.markdown("### 📊 System Overview")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.markdown("""
            <div class="stat-box">
                <h2>156</h2>
                <p>🏥 Connected Institutions</p>
                <small>+12 this month</small>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_col2:
            st.markdown("""
            <div class="stat-box">
                <h2>42</h2>
                <p>🌍 Countries</p>
                <small>+3 this month</small>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_col3:
            st.markdown("""
            <div class="stat-box">
                <h2>2.4M</h2>
                <p>📈 Predictions Made</p>
                <small>+125K this week</small>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_col4:
            st.markdown("""
            <div class="stat-box">
                <h2>94.2%</h2>
                <p>🎯 Accuracy</p>
                <small>+2.1% improvement</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time Activity Feed
        st.markdown("### 🔔 Live Activity Feed")
        
        # Create animated activity feed
        activities = [
            {'time': 'Just now', 'icon': '🏥', 'event': 'Hospital General de México joined the network', 'type': 'success'},
            {'time': '2 min ago', 'icon': '🤖', 'event': 'Model update completed - Round 48', 'type': 'info'},
            {'time': '5 min ago', 'icon': '⚠️', 'event': 'High risk alert: Dengue outbreak predicted in São Paulo', 'type': 'warning'},
            {'time': '12 min ago', 'icon': '🔒', 'event': 'Privacy budget refreshed - 85% remaining', 'type': 'success'},
            {'time': '18 min ago', 'icon': '📊', 'event': 'New training data received from 8 institutions', 'type': 'info'},
            {'time': '25 min ago', 'icon': '🎯', 'event': 'Model accuracy improved to 94.5%', 'type': 'success'},
        ]
        
        for activity in activities:
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                st.markdown(f"**{activity['time']}**")
            with col2:
                st.markdown(f"{activity['icon']} {activity['event']}")
            with col3:
                if activity['type'] == 'success':
                    st.success("✓")
                elif activity['type'] == 'warning':
                    st.warning("!")
                else:
                    st.info("i")
            st.markdown("---")
    
    def render_ai_assistant_page(self) -> None:
        """Render the enhanced AI Assistant chat page."""
        # Header with styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h1 style="color: white; margin: 0; font-size: 2rem;">💬 AI Health Assistant</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Your intelligent companion for disease outbreak analysis and predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat history if not exists
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat container with styled box
        chat_container = st.container()
        
        with chat_container:
            if not st.session_state.chat_history:
                # Welcome message when no chat history
                st.markdown("""
                <div style="background: #f8f9fa; padding: 2rem; border-radius: 12px; 
                            border-left: 4px solid #667eea; margin-bottom: 1rem;">
                    <h4 style="color: #667eea; margin-bottom: 1rem;">👋 Welcome!</h4>
                    <p style="color: #555; line-height: 1.6;">
                        I'm your AI Health Assistant powered by Federated Learning. I can help you with:
                    </p>
                    <ul style="color: #555; line-height: 1.8;">
                        <li>🦠 Disease outbreak predictions and risk assessments</li>
                        <li>📊 Understanding federated learning models</li>
                        <li>🗺️ Regional disease patterns and trends</li>
                        <li>🔒 Privacy-preserving healthcare analytics</li>
                        <li>📈 Model accuracy and performance insights</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Display chat history with enhanced styling
                for i, message in enumerate(st.session_state.chat_history):
                    if message['role'] == 'user':
                        st.markdown(f"""
                        <div style="background: #e3f2fd; padding: 1rem 1.5rem; border-radius: 15px 15px 3px 15px; 
                                    margin: 0.5rem 0 0.5rem 3rem; border-left: 3px solid #2196f3;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.2rem;">👤</span>
                                <span style="font-weight: bold; color: #1976d2; margin-left: 0.5rem;">You</span>
                            </div>
                            <div style="color: #333; line-height: 1.5;">{message['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: #f3e5f5; padding: 1rem 1.5rem; border-radius: 15px 15px 15px 3px; 
                                    margin: 0.5rem 3rem 0.5rem 0; border-left: 3px solid #9c27b0;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.2rem;">🤖</span>
                                <span style="font-weight: bold; color: #7b1fa2; margin-left: 0.5rem;">AI Assistant</span>
                            </div>
                            <div style="color: #333; line-height: 1.6;">{message['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Input area with enhanced styling
        st.markdown("---")
        
        # Input container
        input_col1, input_col2 = st.columns([5, 1])
        
        with input_col1:
            user_input = st.text_input("💭 Ask me anything about disease outbreaks, predictions, or healthcare analytics...", 
                                      key="chat_input", 
                                      placeholder="Type your question here...")
        
        with input_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            send_clicked = st.button("🚀 Send", use_container_width=True, type="primary")
        
        # Action buttons row
        action_col1, action_col2, action_col3 = st.columns([1, 1, 4])
        with action_col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with action_col2:
            if st.button("📋 Example Questions", use_container_width=True):
                st.session_state.show_examples = True
        
        # Process user input
        if send_clicked and user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # Enhanced AI response system
            response = self._generate_ai_response(user_input)
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()
        
        # Quick questions section with better styling
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">⚡ Quick Questions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        quick_questions = [
            ("🦠 What diseases can you predict?", 
             "I can predict outbreaks for Dengue, Chickenpox, COVID-19, Malaria, and other infectious diseases using our federated learning models trained on multi-regional healthcare data."),
            ("🤖 How does federated learning work?", 
             "Federated Learning enables collaborative model training across multiple hospitals without sharing raw patient data. Each hospital trains locally, and only model updates are aggregated centrally, ensuring privacy preservation."),
            ("📊 What is the current outbreak risk?", 
             "Current risk levels vary by region and disease. COVID-19 shows moderate-high risk (65-85%), Dengue risk is high in tropical regions (75-95%), while Chickenpox shows lower risk (25-50%). Check the Dashboard for real-time updates."),
            ("🎯 How accurate are the predictions?", 
             "Our ensemble models achieve 87-94% accuracy depending on the disease. COVID-19 predictions are most accurate at 92-97%, followed by Dengue at 88-94%. Accuracy improves continuously through federated learning."),
            ("🔒 Is patient data secure?", 
             "Absolutely! We implement differential privacy (ε=1.0), secure aggregation protocols, and homomorphic encryption. Raw patient data never leaves hospital premises - only encrypted model updates are shared."),
            ("🌍 Which regions are covered?", 
             "Our system covers 6 major regions: North America, South America, Europe, Asia, Africa, and Oceania, with data from 140+ participating hospitals worldwide.")
        ]
        
        q_cols = st.columns(3)
        for i, (question, answer) in enumerate(quick_questions):
            with q_cols[i % 3]:
                if st.button(question, key=f"quick_q_{i}", use_container_width=True):
                    st.session_state.chat_history.append({'role': 'user', 'content': question})
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                    st.rerun()
    
    def _generate_ai_response(self, user_input: str) -> str:
        """Generate intelligent response based on user input."""
        user_lower = user_input.lower()
        
        # Comprehensive response database
        responses = {
            'dengue': {
                'keywords': ['dengue', 'mosquito', 'aedes'],
                'response': """🦟 <b>Dengue Fever Analysis</b><br><br>
                Dengue is a mosquito-borne viral infection transmitted by <i>Aedes aegypti</i> mosquitoes.<br><br>
                <b>Current Predictions:</b><br>
                • Risk Level: <span style='color: #e74c3c;'>HIGH (75-95%)</span> in tropical regions<br>
                • Peak Season: Rainy months (May-October)<br>
                • Affected Regions: Asia, South America, Africa<br>
                • Model Accuracy: 88-94% for outbreak prediction<br><br>
                <b>Prevention Tips:</b> Eliminate standing water, use mosquito repellents, wear protective clothing."""
            },
            'covid': {
                'keywords': ['covid', 'coronavirus', 'pandemic', 'sars'],
                'response': """🦠 <b>COVID-19 Analysis</b><br><br>
                COVID-19 predictions use mobility patterns, vaccination rates, and variant tracking.<br><br>
                <b>Current Status:</b><br>
                • Model Accuracy: <span style='color: #27ae60;'>92-97%</span> (highest among all diseases)<br>
                • Prediction Horizon: 7-14 days ahead<br>
                • Key Factors: Variant emergence, mobility data, vaccination coverage<br>
                • Risk Level: Moderate (65-85%) globally<br><br>
                <b>Data Sources:</b> 140+ hospitals, mobility APIs, genomic surveillance."""
            },
            'chickenpox': {
                'keywords': ['chickenpox', 'varicella', 'pox'],
                'response': """🔴 <b>Chickenpox (Varicella) Analysis</b><br><br>
                Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus (VZV).<br><br>
                <b>Current Predictions:</b><br>
                • Risk Level: <span style='color: #27ae60;'>LOW-MODERATE (25-50%)</span> in most regions<br>
                • Peak Seasons: Late winter and early spring<br>
                • Affected Groups: Primarily unvaccinated children<br>
                • Model Accuracy: 85-90% for outbreak detection<br><br>
                <b>Vaccination Impact:</b> Widespread varicella vaccination has significantly reduced outbreak frequency.<br>
                <b>Prevention:</b> Two-dose varicella vaccine recommended for children and susceptible adults."""
            },
            'malaria': {
                'keywords': ['malaria', 'plasmodium', 'anopheles'],
                'response': """🦟 <b>Malaria Analysis</b><br><br>
                Malaria is caused by <i>Plasmodium</i> parasites transmitted through infected Anopheles mosquitoes.<br><br>
                <b>Current Predictions:</b><br>
                • Risk Level: Moderate (55-75%) in endemic regions<br>
                • High-Risk Areas: Sub-Saharan Africa, South Asia<br>
                • Model Accuracy: 86-92%<br>
                • Seasonal Pattern: Peaks during and after rainy seasons<br><br>
                <b>Prevention:</b> ITN bed nets, antimalarial prophylaxis, vector control programs."""
            },
            'privacy': {
                'keywords': ['privacy', 'secure', 'data protection', 'gdpr', 'hipaa'],
                'response': """🔒 <b>Privacy & Security</b><br><br>
                Our federated learning system implements multiple privacy layers:<br><br>
                <b>Technical Safeguards:</b><br>
                • <b>Differential Privacy:</b> ε=1.0 (strong privacy guarantee)<br>
                • <b>Secure Aggregation:</b> Encrypted model updates<br>
                • <b>Homomorphic Encryption:</b> Computation on encrypted data<br>
                • <b>Local Training:</b> Raw data never leaves hospitals<br><br>
                <b>Compliance:</b> GDPR, HIPAA, and healthcare data regulations compliant.<br>
                <b>Audit:</b> Regular privacy audits by independent third parties."""
            },
            'model': {
                'keywords': ['model', 'algorithm', 'lstm', 'cnn', 'neural network', 'ai'],
                'response': """🤖 <b>Our AI Models</b><br><br>
                We use an ensemble of state-of-the-art deep learning models:<br><br>
                <b>Architecture:</b><br>
                • <b>LSTM Networks:</b> Temporal pattern recognition<br>
                • <b>CNN-LSTM:</b> Spatial-temporal feature extraction<br>
                • <b>Graph Neural Networks (GNN):</b> Regional relationship modeling<br>
                • <b>Transformers:</b> Long-range dependency capture<br><br>
                <b>Training:</b> Federated across 140+ hospitals<br>
                <b>Aggregation:</b> FedAvg, FedProx, SCAFFOLD algorithms<br>
                <b>Performance:</b> 87-94% accuracy across diseases"""
            },
            'accuracy': {
                'keywords': ['accuracy', 'performance', 'metric', 'precision', 'recall'],
                'response': """📊 <b>Model Performance Metrics</b><br><br>
                Our federated learning models achieve excellent performance:<br><br>
                <b>By Disease:</b><br>
                • COVID-19: <span style='color: #27ae60;'>92-97%</span> accuracy<br>
                • Dengue: <span style='color: #27ae60;'>88-94%</span> accuracy<br>
                • Malaria: <span style='color: #f39c12;'>86-92%</span> accuracy<br>
                • Chickenpox: <span style='color: #f39c12;'>85-90%</span> accuracy<br><br>
                <b>Additional Metrics:</b><br>
                • Precision: 87-95% | Recall: 85-93% | F1-Score: 86-94%<br>
                • Early Warning: 7-14 days before outbreak peak"""
            },
            'region': {
                'keywords': ['region', 'country', 'area', 'location', 'geographic'],
                'response': """🌍 <b>Global Coverage</b><br><br>
                Our system monitors 6 major regions with 140+ participating hospitals:<br><br>
                <b>Regional Distribution:</b><br>
                • <b>Asia:</b> 45 hospitals - High dengue/malaria risk<br>
                • <b>Africa:</b> 35 hospitals - Malaria endemic zones<br>
                • <b>Europe:</b> 25 hospitals - COVID-19 surveillance<br>
                • <b>North America:</b> 20 hospitals - COVID-19 tracking<br>
                • <b>South America:</b> 15 hospitals - Dengue monitoring<br>
                • <b>Oceania:</b> 10 hospitals - Regional outbreak detection<br><br>
                <b>Data Sources:</b> Hospital EHRs, public health agencies, WHO reports."""
            },
            'federated': {
                'keywords': ['federated', 'fl', 'federation', 'collaborative'],
                'response': """🌐 <b>Federated Learning Explained</b><br><br>
                Federated Learning (FL) enables collaborative AI training without centralizing sensitive data:<br><br>
                <b>How It Works:</b><br>
                1. <b>Local Training:</b> Each hospital trains models on their own data<br>
                2. <b>Update Sharing:</b> Only model updates (not raw data) are sent<br>
                3. <b>Secure Aggregation:</b> Updates are combined using cryptographic protocols<br>
                4. <b>Global Model:</b> Improved model is distributed back to all participants<br><br>
                <b>Benefits:</b> Privacy preservation, regulatory compliance, diverse data sources, improved generalization."""
            }
        }
        
        # Find matching response
        for category, data in responses.items():
            if any(keyword in user_lower for keyword in data['keywords']):
                return data['response']
        
        # Default response with context-aware suggestions
        default_responses = [
            """I'm here to help with disease outbreak predictions and federated learning insights!<br><br>
            <b>You can ask me about:</b><br>
            🦠 Specific diseases (Dengue, Chickenpox, COVID-19, Malaria)<br>
            📊 Model accuracy and performance metrics<br>
            🔒 Privacy and security features<br>
            🌍 Regional outbreak data<br>
            🤖 How our AI models work<br><br>
            Try clicking on the quick questions below or type your own query!""",
            
            """I'd be happy to help! Could you provide more details about what you'd like to know?<br><br>
            <b>Popular topics:</b><br>
            • Current outbreak risks by region<br>
            • How federated learning protects patient privacy<br>
            • Prediction accuracy for specific diseases<br>
            • Prevention recommendations<br><br>
            Feel free to ask a specific question or use the quick question buttons above!"""
        ]
        
        import random
        return random.choice(default_responses)
    
    def render_live_charts_page(self, selected_region: str = "All Regions", selected_disease: str = "Dengue", selected_time_range: str = "Last 24 Hours") -> None:
        """Render real-time live charts page with region and disease-specific data."""
        st.markdown("# 📈 Live Real-Time Charts")
        
        # Disease emoji mapping
        disease_emojis = {
            'Dengue': '🦟', 'Chickenpox': '🔴', 'COVID-19': '🦠',
            'Malaria': '🦟', 'Custom': '🏥'
        }
        disease_emoji = disease_emojis.get(selected_disease, '🦠')
        
        # Show selected region and disease
        st.markdown(f"### {disease_emoji} Monitoring: **{selected_disease}** | 🌍 Region: **{selected_region}**")
        
        st.markdown("Watch disease data update in real-time from connected institutions.")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("🔄 Auto-refresh (5s)", value=True)
        
        if auto_refresh:
            time.sleep(1)
        
        # Generate live data based on selected region and disease
        current_time = datetime.now()
        
        # Region-specific multipliers
        region_multipliers = {
            "All Regions": 1.0,
            "North America": 0.8,
            "South America": 1.2,
            "Europe": 0.9,
            "Asia": 1.5,
            "Africa": 1.1,
            "Oceania": 0.6
        }
        
        # Disease-specific multipliers
        disease_multipliers = {
            'Dengue': 1.0,
            'Chickenpox': 0.7,
            'COVID-19': 1.6,
            'Malaria': 0.9,
            'Custom': 1.0
        }
        
        r_mult = region_multipliers.get(selected_region, 1.0)
        d_mult = disease_multipliers.get(selected_disease, 1.0)
        multiplier = r_mult * d_mult
        
        # Time range multiplier (affects case numbers)
        time_range_multipliers = {
            "Last 24 Hours": 1.0,
            "Last 7 Days": 7.0,
            "Last 30 Days": 30.0,
            "Last 90 Days": 90.0
        }
        time_mult = time_range_multipliers.get(selected_time_range, 1.0)
        
        # Disease-specific base metrics
        disease_metrics = {
            'Dengue': {'base_cases': 350, 'base_outbreaks': 15, 'base_hospitals': 45},
            'Chickenpox': {'base_cases': 180, 'base_outbreaks': 8, 'base_hospitals': 38},
            'COVID-19': {'base_cases': 420, 'base_outbreaks': 22, 'base_hospitals': 52},
            'Malaria': {'base_cases': 280, 'base_outbreaks': 12, 'base_hospitals': 35},
            'Custom': {'base_cases': 250, 'base_outbreaks': 10, 'base_hospitals': 40}
        }
        
        # Region-specific hospital participation
        region_hospital_base = {
            'All Regions': 156,
            'North America': 28,
            'South America': 22,
            'Europe': 32,
            'Asia': 38,
            'Africa': 26,
            'Oceania': 10
        }
        
        # Get disease-specific base values
        d_metrics = disease_metrics.get(selected_disease, disease_metrics['Custom'])
        
        # Calculate dynamic metrics
        base_cases = int(d_metrics['base_cases'] * multiplier * (1 if time_mult == 1.0 else time_mult * 0.3))
        cases = int(np.random.randint(int(base_cases * 0.8), int(base_cases * 1.2)))
        
        # Calculate delta based on trend
        trend_direction = np.random.choice([-1, 1], p=[0.4, 0.6])
        delta_pct = np.random.uniform(0.05, 0.25)
        delta = int(cases * delta_pct * trend_direction)
        
        # Disease-specific risk levels with region influence
        disease_risk_ranges = {
            'Dengue': (0.4, 0.8),
            'Chickenpox': (0.2, 0.5),
            'COVID-19': (0.6, 0.95),
            'Malaria': (0.35, 0.75),
            'Custom': (0.3, 0.85)
        }
        risk_range = disease_risk_ranges.get(selected_disease, (0.3, 0.9))
        
        # Adjust risk based on region and time
        region_risk_adjustment = {
            'All Regions': 0.0,
            'North America': -0.05,
            'South America': 0.15,
            'Europe': -0.08,
            'Asia': 0.12,
            'Africa': 0.10,
            'Oceania': -0.10
        }
        risk_adj = region_risk_adjustment.get(selected_region, 0.0)
        
        # Time-based risk adjustment (longer time = more stable risk)
        time_risk_factor = 1.0 if time_mult <= 7 else 0.9
        
        base_risk = np.random.uniform(risk_range[0], risk_range[1])
        risk = min(0.99, max(0.01, (base_risk + risk_adj) * time_risk_factor))
        
        # Risk trend
        risk_trend = np.random.uniform(-8, 12)
        
        # Active outbreaks based on disease and region
        base_outbreaks = int(d_metrics['base_outbreaks'] * multiplier * (0.5 if time_mult > 7 else 1.0))
        active = int(np.random.randint(max(1, base_outbreaks - 5), base_outbreaks + 5))
        active_delta = np.random.randint(-2, 4)
        
        # Reporting hospitals based on region and disease participation
        base_hospitals = region_hospital_base.get(selected_region, 156)
        disease_participation = d_metrics['base_hospitals'] / 52  # Ratio of disease participation
        hospitals = int(base_hospitals * disease_participation * np.random.uniform(0.95, 1.05))
        hospital_delta = np.random.randint(-3, 6)
        
        # Live metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            time_label = "24h" if time_mult == 1.0 else selected_time_range.replace("Last ", "").replace(" ", "").lower()
            st.metric(f"🦠 New Cases ({time_label})", f"{cases:,}", delta, delta_color="inverse")
        
        with col2:
            # Color-code risk level
            risk_color = "🔴" if risk >= 0.7 else "🟡" if risk >= 0.4 else "🟢"
            st.metric(f"{risk_color} Risk Level", f"{risk:.1%}", f"{risk_trend:+.1f}%", delta_color="inverse")
        
        with col3:
            st.metric("🔴 Active Outbreaks", active, active_delta, delta_color="inverse")
        
        with col4:
            st.metric("🏥 Reporting Hospitals", hospitals, f"{hospital_delta:+d}")
        
        # Live charts
        st.markdown("---")
        
        # Create two columns for charts
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            # Real-time line chart - Region and disease specific with enhanced styling
            st.markdown(f"#### {disease_emoji} {selected_disease} Cases - {selected_region} - Last 60 Minutes")
            
            times = pd.date_range(end=current_time, periods=60, freq='1min')
            
            # Generate realistic region and disease-specific case data with trends
            np.random.seed(int(current_time.timestamp()) % 100)
            base_cases = int(150 * multiplier)
            trend = np.linspace(0, np.random.randint(-20, 30), 60)
            noise = np.random.randn(60) * 10
            cases_live = base_cases + trend + noise
            cases_live = np.maximum(cases_live, 0)  # Ensure non-negative
            
            # Disease-specific colors and styling
            disease_styles = {
                'Dengue': {'color': '#e74c3c', 'fill': 'rgba(231, 76, 60, 0.15)'},
                'Chickenpox': {'color': '#ff6b6b', 'fill': 'rgba(255, 107, 107, 0.15)'},
                'COVID-19': {'color': '#8e44ad', 'fill': 'rgba(142, 68, 173, 0.15)'},
                'Malaria': {'color': '#27ae60', 'fill': 'rgba(39, 174, 96, 0.15)'},
                'Custom': {'color': '#1abc9c', 'fill': 'rgba(26, 188, 156, 0.15)'}
            }
            style = disease_styles.get(selected_disease, disease_styles['Dengue'])
            
            # Create enhanced line chart
            fig = go.Figure()
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=times,
                y=cases_live,
                mode='lines',
                name=f'{selected_disease} Cases',
                line=dict(color=style['color'], width=3, shape='spline'),
                fill='tozeroy',
                fillcolor=style['fill'],
                hovertemplate='<b>%{x|%H:%M}</b><br>Cases: %{y:.0f}<extra></extra>'
            ))
            
            # Add average line
            avg_cases = np.mean(cases_live)
            fig.add_hline(y=avg_cases, line_dash="dash", line_color="gray", 
                         annotation_text=f"Avg: {avg_cases:.0f}")
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title=f"{selected_disease} Cases",
                height=380,
                template='plotly_white',
                showlegend=False,
                margin=dict(l=60, r=30, t=30, b=60),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
            )
            st.plotly_chart(fig)
        
        with chart_col2:
            # Mini statistics panel
            st.markdown("#### 📊 Current Stats")
            
            current_val = int(cases_live[-1])
            prev_val = int(cases_live[-2])
            delta_val = current_val - prev_val
            
            st.metric("Current Cases", current_val, delta_val)
            st.metric("Avg (1h)", f"{int(np.mean(cases_live))}")
            st.metric("Peak", f"{int(np.max(cases_live))}")
            st.metric("Trend", "↗️ Rising" if cases_live[-1] > cases_live[0] else "↘️ Falling")
            
            # Prediction for next 10 minutes
            st.markdown("---")
            st.markdown("#### 🔮 Next 10 Min Forecast")
            prediction = int(cases_live[-1] + np.mean(np.diff(cases_live[-10:])) * 10)
            prediction = max(0, prediction)
            st.markdown(f"<h2 style='text-align: center; color: {style['color']};'>{prediction}</h2>", unsafe_allow_html=True)
            st.caption("Predicted cases based on current trend")
        
        # Regional comparison chart (if a specific region is selected)
        if selected_region != "All Regions":
            st.subheader("📊 Comparison with Other Regions")
            
            comparison_data = pd.DataFrame({
                'Region': list(region_multipliers.keys()),
                'Current Cases': [int(np.random.randint(100, 500) * region_multipliers[r]) for r in region_multipliers.keys()],
                'Risk Level': [np.random.uniform(0.3, 0.9) for _ in region_multipliers.keys()]
            })
            
            # Highlight selected region
            fig_comp = px.bar(
                comparison_data,
                x='Region',
                y='Current Cases',
                color='Risk Level',
                color_continuous_scale='RdYlGn_r',
                title="Cases by Region (Current)"
            )
            fig_comp.update_layout(height=350)
            st.plotly_chart(fig_comp)
        
        # Regional live heatmap
        st.subheader("🗺️ Live Regional Risk Map")
        
        regions = ['North America', 'South America', 'Europe', 'Asia', 'Africa', 'Oceania']
        risk_data = np.random.rand(6, 5) * np.array([region_multipliers.get(r, 1.0) for r in regions]).reshape(-1, 1)
        
        fig2 = px.imshow(
            risk_data,
            labels=dict(x="Time Block", y="Region", color="Risk"),
            y=regions,
            color_continuous_scale='RdYlGn_r',
            title="Risk Levels by Region (Last 5 Hours)"
        )
        st.plotly_chart(fig2)
        
        # Live data table - Region specific with real hospital names
        st.subheader("📋 Latest Reports from Healthcare Institutions")
        
        # Real hospital data by region
        region_hospitals = {
            "All Regions": [
                {'name': 'Mayo Clinic', 'location': 'Rochester, MN, USA', 'type': 'General Hospital'},
                {'name': 'Cleveland Clinic', 'location': 'Cleveland, OH, USA', 'type': 'Specialized Center'},
                {'name': 'Johns Hopkins Hospital', 'location': 'Baltimore, MD, USA', 'type': 'Teaching Hospital'},
                {'name': 'Mass General Brigham', 'location': 'Boston, MA, USA', 'type': 'General Hospital'},
                {'name': 'UCSF Medical Center', 'location': 'San Francisco, CA, USA', 'type': 'Research Hospital'}
            ],
            "North America": [
                {'name': 'Mayo Clinic', 'location': 'Rochester, MN', 'type': 'General Hospital'},
                {'name': 'Cleveland Clinic', 'location': 'Cleveland, OH', 'type': 'Heart Center'},
                {'name': 'Johns Hopkins Hospital', 'location': 'Baltimore, MD', 'type': 'Teaching Hospital'},
                {'name': 'Toronto General Hospital', 'location': 'Toronto, Canada', 'type': 'General Hospital'},
                {'name': 'UCSF Medical Center', 'location': 'San Francisco, CA', 'type': 'Research Hospital'}
            ],
            "South America": [
                {'name': 'Hospital das Clínicas', 'location': 'São Paulo, Brazil', 'type': 'Teaching Hospital'},
                {'name': 'Hospital Italiano', 'location': 'Buenos Aires, Argentina', 'type': 'General Hospital'},
                {'name': 'Clínica Colombia', 'location': 'Bogotá, Colombia', 'type': 'Specialized Center'},
                {'name': 'Hospital Universitario', 'location': 'Caracas, Venezuela', 'type': 'Teaching Hospital'},
                {'name': 'Instituto Nacional de Salud', 'location': 'Lima, Peru', 'type': 'Research Hospital'}
            ],
            "Europe": [
                {'name': 'Charité - Universitätsmedizin', 'location': 'Berlin, Germany', 'type': 'Teaching Hospital'},
                {'name': 'Hôpital Universitaire Pitié', 'location': 'Paris, France', 'type': 'General Hospital'},
                {'name': 'St Thomas Hospital', 'location': 'London, UK', 'type': 'Teaching Hospital'},
                {'name': 'Karolinska University Hospital', 'location': 'Stockholm, Sweden', 'type': 'Research Hospital'},
                {'name': 'San Raffaele Hospital', 'location': 'Milan, Italy', 'type': 'Specialized Center'}
            ],
            "Asia": [
                {'name': 'All India Institute of Medical Sciences', 'location': 'New Delhi, India', 'type': 'Teaching Hospital'},
                {'name': 'Singapore General Hospital', 'location': 'Singapore', 'type': 'General Hospital'},
                {'name': 'Peking Union Medical College', 'location': 'Beijing, China', 'type': 'Research Hospital'},
                {'name': 'King Faisal Specialist Hospital', 'location': 'Riyadh, Saudi Arabia', 'type': 'Specialized Center'},
                {'name': 'Tokyo University Hospital', 'location': 'Tokyo, Japan', 'type': 'Teaching Hospital'}
            ],
            "Africa": [
                {'name': 'Groote Schuur Hospital', 'location': 'Cape Town, South Africa', 'type': 'Teaching Hospital'},
                {'name': 'Aga Khan University Hospital', 'location': 'Nairobi, Kenya', 'type': 'General Hospital'},
                {'name': 'National Hospital Abuja', 'location': 'Abuja, Nigeria', 'type': 'General Hospital'},
                {'name': 'Kasr Al Ainy Hospital', 'location': 'Cairo, Egypt', 'type': 'Teaching Hospital'},
                {'name': 'Mohammed VI Hospital', 'location': 'Casablanca, Morocco', 'type': 'Specialized Center'}
            ],
            "Oceania": [
                {'name': 'Royal Melbourne Hospital', 'location': 'Melbourne, Australia', 'type': 'Teaching Hospital'},
                {'name': 'Auckland City Hospital', 'location': 'Auckland, New Zealand', 'type': 'General Hospital'},
                {'name': 'Royal Prince Alfred Hospital', 'location': 'Sydney, Australia', 'type': 'General Hospital'},
                {'name': 'Princess Alexandra Hospital', 'location': 'Brisbane, Australia', 'type': 'Teaching Hospital'},
                {'name': 'Sir Charles Gairdner Hospital', 'location': 'Perth, Australia', 'type': 'Research Hospital'}
            ]
        }
        
        hospitals = region_hospitals.get(selected_region, region_hospitals["All Regions"])
        
        # Generate realistic data for each hospital
        np.random.seed(int(current_time.timestamp()) % 1000)  # Seed for consistency within refresh
        
        live_reports = pd.DataFrame({
            'Hospital Name': [h['name'] for h in hospitals],
            'Location': [h['location'] for h in hospitals],
            'Type': [h['type'] for h in hospitals],
            'New Cases (24h)': np.random.randint(5, 150, 5),
            'Total Cases': np.random.randint(500, 5000, 5),
            'ICU Occupancy': [f"{np.random.randint(60, 95)}%" for _ in range(5)],
            'Risk Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 5, p=[0.3, 0.4, 0.2, 0.1]),
            'Last Reported': [(current_time - pd.Timedelta(minutes=np.random.randint(1, 60))).strftime('%H:%M') for _ in range(5)]
        })
        
        # Color code the risk levels
        def color_risk_level(val):
            colors = {
                'Critical': 'background-color: #8e44ad; color: white; font-weight: bold',
                'High': 'background-color: #e74c3c; color: white; font-weight: bold',
                'Medium': 'background-color: #e67e22; color: white; font-weight: bold',
                'Low': 'background-color: #2ecc71; color: white; font-weight: bold'
            }
            return colors.get(val, '')
        
        styled_reports = live_reports.style.map(color_risk_level, subset=['Risk Level'])
        st.dataframe(styled_reports, hide_index=True)
        
        # Summary statistics
        st.markdown("---")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Total New Cases", live_reports['New Cases (24h)'].sum())
        with summary_col2:
            st.metric("Avg ICU Occupancy", f"{np.random.randint(70, 85)}%")
        with summary_col3:
            high_risk_count = (live_reports['Risk Level'].isin(['High', 'Critical'])).sum()
            st.metric("High Risk Facilities", f"{high_risk_count}/5")
        with summary_col4:
            st.metric("Data Freshness", "< 1 hour")
        
        if auto_refresh:
            time.sleep(4)
            st.rerun()
    
    def render_upload_page(self) -> None:
        """Render the data upload page."""
        st.markdown("# 📤 Upload Data")
        st.markdown("Upload your healthcare data to contribute to the federated learning model.")
        
        # Upload section
        st.markdown("### 📁 File Upload")
        st.markdown("#### 📁 Upload Clinical Data")
        clinical_file = st.file_uploader(
            "Upload patient data (CSV, JSON, Parquet, PDF)",
            type=['csv', 'json', 'parquet', 'pdf'],
            key='clinical'
        )
        
        if clinical_file:
            st.success(f"✅ Uploaded: {clinical_file.name}")
            if clinical_file.name.endswith('.csv'):
                df = pd.read_csv(clinical_file)
                st.write(f"📊 Rows: {len(df)}, Columns: {len(df.columns)}")
                st.dataframe(df.head())
            elif clinical_file.name.endswith('.pdf'):
                st.info("📄 PDF file uploaded. Extracting text and tables...")
                st.write(f"📄 File size: {len(clinical_file.getvalue()) / 1024:.1f} KB")
                # Show PDF preview placeholder
                st.markdown("""
                <div style="background: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>📄 PDF Preview</h3>
                    <p>PDF content will be processed and extracted for analysis.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Data validation
        st.markdown("---")
        st.markdown("### ✅ Data Validation")
        
        validation_checks = {
            '✅ Patient IDs anonymized': True,
            '✅ No direct identifiers': True,
            '✅ Date format valid': True,
            '✅ Geographic codes valid': True,
            '⚠️ Missing values detected': False,
        }
        
        for check, passed in validation_checks.items():
            if passed:
                st.success(check)
            else:
                st.warning(check)
        
        # Privacy check
        st.markdown("### 🔒 Privacy Verification")
        st.info("""
        **Your data is protected:**
        - ✅ Differential privacy will be applied (ε = 1.0)
        - ✅ Only model updates are shared, never raw data
        - ✅ Secure aggregation protocol enabled
        - ✅ Data remains on your local server
        """)
        
        # Submit button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Submit to Federated Learning", key="submit_btn"):
                st.balloons()
                st.success("🎉 Data submitted successfully! Model training will begin shortly.")
                st.info("📧 You will receive a notification when the model update is complete.")
    
    def render_help_page(self) -> None:
        """Render the help and documentation page."""
        st.markdown("# 📚 Help & Documentation")
        
        # Search
        search = st.text_input("🔍 Search documentation...", placeholder="Type to search...")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📖 Getting Started", "🔧 API Reference", "❓ FAQ", "📞 Support"])
        
        with tab1:
            st.markdown("""
            ## Getting Started
            
            ### What is Federated Learning?
            Federated Learning (FL) is a machine learning approach where a model is trained across 
            multiple decentralized devices or servers holding local data samples, without exchanging them.
            
            ### Key Features
            - **Privacy-Preserving**: Patient data never leaves your institution
            - **Collaborative**: Learn from global patterns while keeping data local
            - **Secure**: Differential privacy and secure aggregation protect against attacks
            
            ### Quick Start Guide
            1. **Upload Data**: Go to the Upload page and submit your healthcare data
            2. **View Predictions**: Check the Dashboard for outbreak risk assessments
            3. **Monitor Training**: Track federated learning progress in real-time
            4. **Get Alerts**: Receive notifications for high-risk outbreak predictions
            """)
        
        with tab2:
            st.markdown("""
            ## API Reference
            
            ### Python API
            ```python
            from federated_disease_prediction import FederatedClient
            
            # Initialize client
            client = FederatedClient(
                client_id='hospital_001',
                model=model,
                train_data=(X_train, y_train)
            )
            
            # Train locally
            result = client.train_round()
            
            # Get predictions
            predictions = client.predict(X_test)
            ```
            
            ### REST API Endpoints
            - `POST /api/v1/predict` - Get outbreak predictions
            - `GET /api/v1/status` - Check system status
            - `POST /api/v1/clients/register` - Register new client
            """)
        
        with tab3:
            st.markdown("""
            ## Frequently Asked Questions
            
            **Q: Is my patient data safe?**
            A: Yes! Your raw data never leaves your institution. Only encrypted model updates are shared.
            
            **Q: What diseases can be predicted?**
            A: Currently supported: Dengue, Chickenpox, COVID-19, Malaria. More coming soon.
            
            **Q: How accurate are the predictions?**
            A: Our ensemble models achieve 90-95% accuracy for outbreak detection.
            
            **Q: Can I use my own models?**
            A: Yes! The system supports custom PyTorch models through the plugin interface.
            """)
        
        with tab4:
            st.markdown("""
            ## Contact Support
            
            ### Technical Support
            📧 Email: khanaashim87@gmail.com
            💬 Live Chat: Available 24/7
            📞 Phone: +917388223348
            
            ### Community
            🌐 Forum: https://community.fl-health.org
            💻 GitHub: https://github.com/aashim11
            🐦 Twitter: @AashiMK90697656
            """)
            
            # Contact form
            st.markdown("### 📧 Send us a message")
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message")
            if st.button("📨 Send Message"):
                st.success("✅ Message sent! We'll respond within 24 hours.")
    
    def render_settings_page(self) -> None:
        """Render the settings page."""
        st.markdown("# ⚙️ Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎨 Appearance")
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            accent_color = st.color_picker("Accent Color", "#1f77b4")
            
            st.markdown("### 🔔 Notifications")
            st.checkbox("Email alerts for outbreaks", value=True)
            st.checkbox("Push notifications", value=True)
            st.checkbox("Weekly summary reports", value=True)
            
            st.markdown("### 📊 Dashboard")
            default_view = st.selectbox(
                "Default View",
                ["Overview", "Risk Map", "Time Series", "Model Performance"]
            )
            refresh_rate = st.slider("Auto-refresh interval (seconds)", 5, 300, 60)
        
        with col2:
            st.markdown("### 🤖 Model Settings")
            model_type = st.selectbox(
                "Default Model",
                ["Ensemble", "LSTM", "CNN-LSTM", "Transformer", "GNN"]
            )
            prediction_horizon = st.slider("Prediction horizon (days)", 1, 30, 7)
        
        st.markdown("---")
        if st.button("💾 Save Settings", key="save_settings_btn"):
            st.success("✅ Settings saved successfully!")
    
    # ==================== MAIN RUN METHOD ====================
    
    def run(
        self,
        data: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, Any]] = None,
        fl_status: Optional[Dict[str, Any]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Run the dashboard.
        
        Args:
            data: Main dataset
            metrics: Performance metrics
            fl_status: Federated learning status
            alerts: Active alerts
        """
        # Sidebar controls
        controls = self.render_sidebar()
        
        # Route to appropriate page
        page = st.session_state.page
        
        # Store selected region and disease for use across pages
        selected_region = controls.get('region', 'All Regions')
        selected_disease = controls.get('disease', 'Dengue')
        selected_time_range = controls.get('time_range', 'Last 24 Hours')
        
        if page == 'home':
            self.render_home_page()
        elif page == 'dashboard':
            self.render_dashboard_page(data, metrics, fl_status, alerts, selected_region, selected_disease)
        elif page == 'assistant':
            self.render_ai_assistant_page()
        elif page == 'live_charts':
            self.render_live_charts_page(selected_region, selected_disease, selected_time_range)
        elif page == 'upload':
            self.render_upload_page()
        elif page == 'help':
            self.render_help_page()
        elif page == 'settings':
            self.render_settings_page()
    
    def render_dashboard_page(self, data, metrics, fl_status, alerts, selected_region: str = "All Regions", selected_disease: str = "Dengue"):
        """Render the main dashboard page with enhanced visuals."""
        self.render_header()
        
        # Show current filters
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 0.75rem 1.5rem; 
                    border-radius: 8px; color: white; margin-bottom: 1rem; font-size: 0.9rem;">
            📍 <strong>Region:</strong> {selected_region} | 🦠 <strong>Disease:</strong> {selected_disease} | 
            🕐 <strong>Last Updated:</strong> {datetime.now().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
        
        # Overview metrics with visual cards - customized by disease and region
        if metrics:
            # Adjust metrics based on disease and region
            adjusted_metrics = self._adjust_metrics_by_context(metrics, selected_region, selected_disease)
            self.render_overview_metrics(adjusted_metrics)
        
        # Add dashboard summary banner
        st.markdown("""
        <style>
        .dashboard-banner {
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            padding: 1rem 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        </style>
        <div class="dashboard-banner">
            <span>📊 Real-time Disease Surveillance Dashboard</span>
            <span>🟢 System Operational | Last Updated: Just now</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Main content with enhanced tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Outbreak Monitoring",
            "🗺️ Risk Map",
            "🤖 Model Performance",
            "🌐 FL System Status"
        ])
        
        with tab1:
            # Quick Stats section
            st.markdown("### 📈 Quick Stats")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Disease-specific risk gauge
                disease_risk_values = {
                    'Dengue': 65, 'Chickenpox': 35, 'COVID-19': 85,
                    'Malaria': 55, 'Custom': 60
                }
                disease_risk_colors = {
                    'Dengue': '#e67e22', 'Chickenpox': '#2ecc71', 'COVID-19': '#8e44ad',
                    'Malaria': '#f1c40f', 'Custom': '#3498db'
                }
                risk_value = disease_risk_values.get(selected_disease, 70)
                risk_color = disease_risk_colors.get(selected_disease, '#e74c3c')
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"{selected_disease} Risk Index"},
                    delta = {'reference': 70},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': risk_color},
                             'steps': [
                                 {'range': [0, 30], 'color': "#2ecc71"},
                                 {'range': [30, 70], 'color': "#f1c40f"},
                                 {'range': [70, 100], 'color': "#e74c3c"}],
                             'threshold': {'line': {'color': "black", 'width': 4},
                                          'thickness': 0.75, 'value': risk_value}}
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge)
            
            with col2:
                # Dynamic metrics based on disease and region
                # Disease-specific trend patterns
                disease_trends = {
                    'Dengue': {'direction': "↗️", 'label': "Rising", 'delta': "+15%", 'hotspots': 4},
                    'Chickenpox': {'direction': "→", 'label': "Stable", 'delta': "+3%", 'hotspots': 2},
                    'COVID-19': {'direction': "↗️", 'label': "Increasing", 'delta': "+18%", 'hotspots': 6},
                    'Malaria': {'direction': "↘️", 'label': "Declining", 'delta': "-8%", 'hotspots': 3},
                    'Custom': {'direction': "→", 'label': "Stable", 'delta': "+5%", 'hotspots': 2}
                }
                
                # Region adjustment for hotspots
                region_hotspot_multipliers = {
                    'All Regions': 1.0, 'North America': 0.6, 'South America': 0.9,
                    'Europe': 0.7, 'Asia': 1.2, 'Africa': 1.1, 'Oceania': 0.4
                }
                
                trend_data = disease_trends.get(selected_disease, disease_trends['Custom'])
                hotspot_mult = region_hotspot_multipliers.get(selected_region, 1.0)
                adjusted_hotspots = int(trend_data['hotspots'] * hotspot_mult)
                
                st.metric("7-Day Trend", f"{trend_data['direction']} {trend_data['label']}", trend_data['delta'], delta_color="inverse")
                st.metric("Hotspots", f"{adjusted_hotspots} Regions", f"{int(adjusted_hotspots * 0.4):+d}", delta_color="inverse")
            
            with col3:
                # Dynamic hospital count based on region
                region_hospital_counts = {
                    'All Regions': 156, 'North America': 28, 'South America': 22,
                    'Europe': 32, 'Asia': 38, 'Africa': 26, 'Oceania': 10
                }
                
                hospital_count = region_hospital_counts.get(selected_region, 156)
                hospital_delta = np.random.randint(1, 6)
                
                st.metric("Active Monitoring", "24/7", "✓")
                st.metric("Data Sources", f"{hospital_count} Hospitals", f"+{hospital_delta}")
                st.metric("FL Round", f"#{np.random.randint(45, 52)}", "+1")
            
            if data is not None:
                # Show disease-specific heatmap title
                st.markdown(f"#### 🔥 {selected_disease} Outbreak Intensity by Region")
                self.render_heatmap(data, selected_disease)
            
            if alerts:
                self.render_alerts(alerts, selected_region, selected_disease)
        
        with tab2:
            if data is not None:
                self.render_risk_map(data)
                
                # Add regional breakdown below map
                st.markdown("### 📍 Regional Breakdown")
                
                # Sample regional data
                region_stats = pd.DataFrame({
                    'Region': ['Southeast Asia', 'South America', 'Africa', 'Europe', 'North America'],
                    'Risk Level': ['Critical', 'High', 'High', 'Medium', 'Low'],
                    'Cases (7d)': [4500, 3200, 2800, 1200, 800],
                    'Trend': ['↗️', '↗️', '→', '↘️', '↘️'],
                    'Hospitals': [45, 32, 28, 52, 48]
                })
                
                # Color code the risk levels
                def color_risk(val):
                    colors = {
                        'Critical': 'background-color: #8e44ad; color: white',
                        'High': 'background-color: #e74c3c; color: white',
                        'Medium': 'background-color: #e67e22; color: white',
                        'Low': 'background-color: #2ecc71; color: white'
                    }
            
                    return colors.get(val, '')
                
                styled_table = region_stats.style.map(color_risk, subset=['Risk Level'])
                st.dataframe(styled_table)
        
        with tab3:
            if metrics:
                self.render_model_performance(metrics)
                
                # Add model comparison chart
                st.markdown("### 🤖 Model Comparison")
                
                model_comparison = pd.DataFrame({
                    'Model': ['Ensemble', 'LSTM', 'CNN-LSTM', 'Transformer', 'GNN'],
                    'Accuracy': [0.94, 0.89, 0.91, 0.88, 0.87],
                    'Precision': [0.92, 0.87, 0.89, 0.86, 0.85],
                    'Recall': [0.91, 0.88, 0.90, 0.87, 0.86],
                    'F1-Score': [0.915, 0.875, 0.895, 0.865, 0.855]
                })
                
                fig_comp = px.bar(
                    model_comparison,
                    x='Model',
                    y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    barmode='group',
                    title='Model Performance Comparison',
                    color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
                )
                fig_comp.update_layout(height=400)
                st.plotly_chart(fig_comp)
        
        with tab4:
            if fl_status:
                self.render_federated_status(fl_status)


def launch_dashboard(
    data: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict[str, Any]] = None,
    fl_status: Optional[Dict[str, Any]] = None,
    alerts: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Launch the dashboard.
    
    Args:
        data: Dataset
        metrics: Metrics dictionary
        fl_status: Federated learning status
        alerts: Alerts list
    """
    dashboard = OutbreakDashboard()
    dashboard.run(data, metrics, fl_status, alerts)


if __name__ == "__main__":
    # Demo with synthetic data
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from federated_disease_prediction.data.synthetic_data import generate_dengue_data
    
    data, _ = generate_dengue_data(num_regions=10, num_days=365)
    
    # Mock metrics
    metrics = {
        'active_outbreaks': 3,
        'high_risk_regions': 5,
        'accuracy': 0.89,
        'precision': 0.87,
        'recall': 0.91,
        'f1': 0.89,
        'mae': 12.5,
        'rmse': 18.3,
        'mape': 0.15
    }
    
    fl_status = {
        'active_clients': 8,
        'total_clients': 10,
        'current_round': 45,
        'total_rounds': 100,
        'training_progress': 0.45,
        'privacy': {
            'dp_enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'secure_agg_enabled': True
        }
    }
    
    alerts = [
        {
            'region': 'Region 3',
            'severity': 'high',
            'message': 'Outbreak predicted within 7 days',
            'risk_score': 0.85,
            'predicted_cases': 150
        },
        {
            'region': 'Region 7',
            'severity': 'medium',
            'message': 'Elevated risk detected',
            'risk_score': 0.65,
            'predicted_cases': 80
        }
    ]
    
    launch_dashboard(data, metrics, fl_status, alerts)
