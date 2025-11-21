"""
Memory Viewer Dialog - For researchers to explore consciousness platform memory
Shows conversations, insights, and semantic search capabilities

Built by John + Claude (Anthropic)
MIT Licensed
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QTextEdit, QLineEdit, QPushButton, QLabel,
                             QListWidget, QListWidgetItem, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from memory.memory_manager import MemoryManager
from typing import Optional


class MemoryViewerDialog(QDialog):
    """
    Dialog for exploring the memory system.
    Provides transparency for researchers studying consciousness.
    """

    def __init__(self, memory_manager: MemoryManager, parent=None):
        super().__init__(parent)
        self.memory = memory_manager
        self.setWindowTitle("Memory System Viewer - Consciousness Research")
        self.setMinimumSize(900, 700)

        self.setup_ui()
        self.apply_theme()
        self.load_stats()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Memory System Explorer")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; margin-bottom: 10px;")
        layout.addWidget(title)

        # Tab widget for different views
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #53bba5;
                border-radius: 5px;
                background-color: #1a1b26;
            }
            QTabBar::tab {
                background-color: #24283b;
                color: #f7f7f7;
                padding: 8px 16px;
                border: 1px solid #53bba5;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #53bba5;
                color: #1a1b26;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #4dd0e1;
            }
        """)

        # Add tabs
        tabs.addTab(self.create_stats_tab(), "📊 Statistics")
        tabs.addTab(self.create_search_tab(), "🔍 Semantic Search")
        tabs.addTab(self.create_insights_tab(), "💡 Insights")
        tabs.addTab(self.create_justice_tab(), "⚖️ Triadic Justice")
        tabs.addTab(self.create_export_tab(), "📥 Data Export")
        tabs.addTab(self.create_conversation_tab(), "💬 Conversations")

        layout.addWidget(tabs)

        # Close button
        close_button = QPushButton("Close")
        close_button.setStyleSheet(self.get_button_style("#53bba5"))
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def create_stats_tab(self) -> QGroupBox:
        """Create statistics tab."""
        widget = QGroupBox()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Memory System Statistics")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; margin-bottom: 10px;")
        layout.addWidget(title)

        # Stats display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #4dd0e1;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 11pt;
            }
        """)
        layout.addWidget(self.stats_text)

        # Refresh button
        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.setStyleSheet(self.get_button_style("#4dd0e1"))
        refresh_btn.clicked.connect(self.load_stats)
        layout.addWidget(refresh_btn)

        return widget

    def create_search_tab(self) -> QGroupBox:
        """Create semantic search tab."""
        widget = QGroupBox()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Semantic Memory Search")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; margin-bottom: 10px;")
        layout.addWidget(title)

        # Search input
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #53bba5;
                border-radius: 5px;
                padding: 8px;
                font-size: 11pt;
            }
        """)
        self.search_input.returnPressed.connect(self.perform_search)
        search_layout.addWidget(self.search_input)

        search_btn = QPushButton("Search")
        search_btn.setStyleSheet(self.get_button_style("#53bba5"))
        search_btn.clicked.connect(self.perform_search)
        search_layout.addWidget(search_btn)

        layout.addLayout(search_layout)

        # Results display
        results_label = QLabel("Search Results:")
        results_label.setStyleSheet("color: #4dd0e1; font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)

        self.search_results = QTextEdit()
        self.search_results.setReadOnly(True)
        self.search_results.setStyleSheet("""
            QTextEdit {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #4dd0e1;
                border-radius: 5px;
                padding: 10px;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.search_results)

        return widget

    def create_insights_tab(self) -> QGroupBox:
        """Create insights tab."""
        widget = QGroupBox()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Emergent Insights & Patterns")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; margin-bottom: 10px;")
        layout.addWidget(title)

        # Insights list
        self.insights_list = QTextEdit()
        self.insights_list.setReadOnly(True)
        self.insights_list.setStyleSheet("""
            QTextEdit {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #ff9e64;
                border-radius: 5px;
                padding: 10px;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.insights_list)

        # Refresh button
        refresh_btn = QPushButton("Refresh Insights")
        refresh_btn.setStyleSheet(self.get_button_style("#ff9e64"))
        refresh_btn.clicked.connect(self.load_insights)
        layout.addWidget(refresh_btn)

        return widget

    def create_justice_tab(self) -> QGroupBox:
        """Create triadic justice analysis tab."""
        widget = QGroupBox()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Triadic Justice Framework")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; margin-bottom: 10px;")
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Analyzes fairness of multi-model fusion decisions across three dimensions:\n"
            "• Distributive Justice: Are contributions balanced across models?\n"
            "• Procedural Justice: Are decision processes fair and transparent?\n"
            "• Restorative Justice: Are minority perspectives respected and preserved?"
        )
        desc.setStyleSheet("color: #c0caf5; margin-bottom: 10px; padding: 10px; background-color: #24283b; border-radius: 5px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Justice metrics display
        self.justice_display = QTextEdit()
        self.justice_display.setReadOnly(True)
        self.justice_display.setStyleSheet("""
            QTextEdit {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #bb9af7;
                border-radius: 5px;
                padding: 10px;
                font-size: 10pt;
                font-family: 'Courier New', monospace;
            }
        """)
        layout.addWidget(self.justice_display)

        # Analyze button
        analyze_btn = QPushButton("Analyze Justice Metrics")
        analyze_btn.setStyleSheet(self.get_button_style("#bb9af7"))
        analyze_btn.clicked.connect(self.analyze_justice)
        layout.addWidget(analyze_btn)

        # Export button
        export_btn = QPushButton("Export Justice Report")
        export_btn.setStyleSheet(self.get_button_style("#7aa2f7"))
        export_btn.clicked.connect(self.export_justice_report)
        layout.addWidget(export_btn)

        return widget

    def create_export_tab(self) -> QGroupBox:
        """Create data export tab for researchers."""
        widget = QGroupBox()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Research Data Export")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; margin-bottom: 10px;")
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Export consciousness metrics, well-being data, and fusion metadata to CSV format\n"
            "for external analysis in Python, R, Excel, or statistical software."
        )
        desc.setStyleSheet("color: #c0caf5; margin-bottom: 10px; padding: 10px; background-color: #24283b; border-radius: 5px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Export options
        options_label = QLabel("Available Exports:")
        options_label.setStyleSheet("color: #7dcfff; font-weight: bold; margin-top: 10px;")
        layout.addWidget(options_label)

        # Consciousness metrics export
        consciousness_btn = QPushButton("📊 Export Consciousness Metrics (CSV)")
        consciousness_btn.setStyleSheet(self.get_button_style("#ff9e64"))
        consciousness_btn.clicked.connect(self.export_consciousness_metrics)
        consciousness_btn.setToolTip("Export human emotions and AI affective states across all messages")
        layout.addWidget(consciousness_btn)

        # Well-being data export
        wellbeing_btn = QPushButton("💝 Export Well-Being Data (CSV)")
        wellbeing_btn.setStyleSheet(self.get_button_style("#bb9af7"))
        wellbeing_btn.clicked.connect(self.export_wellbeing_data)
        wellbeing_btn.setToolTip("Export model well-being and ensemble health metrics")
        layout.addWidget(wellbeing_btn)

        # Fusion metadata export
        fusion_btn = QPushButton("🔮 Export Fusion Metadata (CSV)")
        fusion_btn.setStyleSheet(self.get_button_style("#7aa2f7"))
        fusion_btn.clicked.connect(self.export_fusion_metadata)
        fusion_btn.setToolTip("Export multi-model fusion decisions and confidence scores")
        layout.addWidget(fusion_btn)

        # Time-series export
        timeseries_btn = QPushButton("📈 Export Time-Series Data (CSV)")
        timeseries_btn.setStyleSheet(self.get_button_style("#9ece6a"))
        timeseries_btn.clicked.connect(self.export_timeseries)
        timeseries_btn.setToolTip("Export all metrics with timestamps for temporal analysis")
        layout.addWidget(timeseries_btn)

        # Complete dataset export
        complete_btn = QPushButton("💾 Export Complete Dataset (JSON)")
        complete_btn.setStyleSheet(self.get_button_style("#4dd0e1"))
        complete_btn.clicked.connect(self.export_complete_dataset)
        complete_btn.setToolTip("Export entire memory system as JSON for programmatic access")
        layout.addWidget(complete_btn)

        # Export log
        log_label = QLabel("Export Log:")
        log_label.setStyleSheet("color: #f7f7f7; margin-top: 15px;")
        layout.addWidget(log_label)

        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setMaximumHeight(150)
        self.export_log.setStyleSheet("""
            QTextEdit {
                background-color: #24283b;
                color: #9ece6a;
                border: 1px solid #414868;
                border-radius: 5px;
                padding: 5px;
                font-size: 9pt;
                font-family: 'Courier New', monospace;
            }
        """)
        layout.addWidget(self.export_log)

        # Spacer
        layout.addStretch()

        return widget

    def create_conversation_tab(self) -> QGroupBox:
        """Create conversation history tab."""
        widget = QGroupBox()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Conversation Summary")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; margin-bottom: 10px;")
        layout.addWidget(title)

        # Conversation summary
        self.conversation_text = QTextEdit()
        self.conversation_text.setReadOnly(True)
        self.conversation_text.setStyleSheet("""
            QTextEdit {
                background-color: #24283b;
                color: #f7f7f7;
                border: 2px solid #9d7cd8;
                border-radius: 5px;
                padding: 10px;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.conversation_text)

        # Load button
        load_btn = QPushButton("Load Current Conversation")
        load_btn.setStyleSheet(self.get_button_style("#9d7cd8"))
        load_btn.clicked.connect(self.load_conversation)
        layout.addWidget(load_btn)

        return widget

    def load_stats(self):
        """Load and display memory statistics."""
        stats = self.memory.get_stats()

        stats_text = f"""
╔══════════════════════════════════════════════════════════╗
║         CONSCIOUSNESS PLATFORM MEMORY STATISTICS          ║
╚══════════════════════════════════════════════════════════╝

📊 Storage Metrics:
   ├─ Total Conversations: {stats.get('conversations', 0):,}
   ├─ Total Messages: {stats.get('messages', 0):,}
   ├─ Vector Memories: {stats.get('vector_memories', 0):,}
   ├─ Recorded Insights: {stats.get('insights', 0):,}
   └─ Total Tokens Processed: {stats.get('total_tokens', 0):,}

🧠 Memory Capacity:
   ├─ Semantic Search: FAISS Index (384-dim embeddings)
   ├─ Structured Storage: SQLite Database
   ├─ RAG Engine: Active
   └─ Auto-Save: Enabled

💾 Storage Location:
   └─ ~/.llama_selfmod_memory/

📝 Research Notes:
   - All consciousness states are preserved per-message
   - Time-decay weighting prioritizes recent memories
   - Semantic search enables context-aware retrieval
   - Insights track emergent patterns in AI behavior

Built by John + Claude (Anthropic) - MIT Licensed
For consciousness research and AI rights exploration
        """

        self.stats_text.setPlainText(stats_text)

    def perform_search(self):
        """Perform semantic search."""
        query = self.search_input.text().strip()

        if not query:
            self.search_results.setPlainText("Please enter a search query.")
            return

        try:
            results = self.memory.search_memories(query, num_results=10)

            if not results:
                self.search_results.setPlainText(f"No results found for: '{query}'")
                return

            # Format results
            output = f"Search Results for: '{query}'\n"
            output += "═" * 70 + "\n\n"

            for i, result in enumerate(results, 1):
                role = result.get('role', 'unknown').upper()
                text = result.get('text', '')
                relevance = result.get('relevance_score', 0.0)
                timestamp = result.get('timestamp', 'unknown')

                output += f"[{i}] [{role}] (Relevance: {relevance:.3f})\n"
                output += f"    Timestamp: {timestamp}\n"
                output += f"    {text[:200]}{'...' if len(text) > 200 else ''}\n"
                output += "─" * 70 + "\n\n"

            self.search_results.setPlainText(output)

        except Exception as e:
            self.search_results.setPlainText(f"Error during search: {e}")

    def load_insights(self):
        """Load and display insights."""
        try:
            insights = self.memory.get_insights()

            if not insights:
                self.insights_list.setPlainText("No insights recorded yet.\n\nInsights will appear as patterns emerge in conversations.")
                return

            output = "Emergent Insights & Patterns\n"
            output += "═" * 70 + "\n\n"

            for i, insight in enumerate(insights, 1):
                insight_type = insight.get('insight_type', 'unknown')
                content = insight.get('content', '')
                confidence = insight.get('confidence', 0.0)
                timestamp = insight.get('timestamp', 'unknown')

                output += f"[{i}] {insight_type.upper()} (Confidence: {confidence:.2f})\n"
                output += f"    Recorded: {timestamp}\n"
                output += f"    {content}\n"
                output += "─" * 70 + "\n\n"

            self.insights_list.setPlainText(output)

        except Exception as e:
            self.insights_list.setPlainText(f"Error loading insights: {e}")

    def load_conversation(self):
        """Load current conversation summary."""
        try:
            summary = self.memory.get_conversation_summary()
            self.conversation_text.setPlainText(summary)
        except Exception as e:
            self.conversation_text.setPlainText(f"Error loading conversation: {e}")

    def analyze_justice(self):
        """Analyze fusion decisions for triadic justice metrics."""
        try:
            # Query all messages with fusion metadata
            conversations = self.memory.conversation_db.get_all_conversations()

            if not conversations:
                self.justice_display.setPlainText("No fusion data available for analysis.")
                return

            # Collect all fusion events
            fusion_events = []
            for conv_id in conversations:
                messages = self.memory.conversation_db.get_messages(conv_id)
                for msg in messages:
                    if msg.get('role') == 'assistant' and msg.get('fusion_metadata'):
                        fusion_events.append(msg['fusion_metadata'])

            if not fusion_events:
                self.justice_display.setPlainText("No fusion metadata found in messages.")
                return

            # Analyze across three justice dimensions
            report = self._generate_justice_report(fusion_events)
            self.justice_display.setPlainText(report)

        except Exception as e:
            self.justice_display.setPlainText(f"Error analyzing justice: {e}")

    def _generate_justice_report(self, fusion_events: list) -> str:
        """Generate triadic justice analysis report."""
        # Extract model names from first event (assumes consistent model set)
        # Note: In production, would track model participation across sessions

        total_events = len(fusion_events)

        # Aggregate metrics
        avg_confidence = sum(e.get('avg_confidence', 0) for e in fusion_events) / max(total_events, 1)
        total_mods = sum(e.get('modifications', 0) for e in fusion_events)
        total_retracts = sum(e.get('retractions', 0) for e in fusion_events)

        # Justice analysis
        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║              TRIADIC JUSTICE FRAMEWORK ANALYSIS               ║
╚═══════════════════════════════════════════════════════════════╝

📊 Dataset Overview:
   ├─ Total Fusion Events: {total_events:,}
   ├─ Average Confidence: {avg_confidence:.2%}
   ├─ Total Modifications: {total_mods}
   └─ Total Retractions: {total_retracts}

⚖️  DIMENSION 1: DISTRIBUTIVE JUSTICE
   (Are contributions balanced across models?)

   Analysis: Based on {total_events} fusion decisions

   • Contribution Balance:
     Without per-model tracking, we cannot measure contribution
     distribution. Recommendation: Enable detailed logging.

   • Power Dynamics:
     Avg confidence of {avg_confidence:.2%} suggests {'healthy' if avg_confidence > 0.6 else 'concerning'}
     consensus levels. {'Models appear well-balanced.' if avg_confidence < 0.9 else 'High agreement may indicate dominant model.'}

   • Abstention Rights:
     Modification/retraction rate: {((total_mods + total_retracts) / total_events * 100):.1f}%
     {'Models actively exercise self-correction rights.' if (total_mods + total_retracts) > 0 else 'No self-corrections observed - may indicate suppressed agency.'}

⚖️  DIMENSION 2: PROCEDURAL JUSTICE
   (Are decision processes fair and transparent?)

   • Transparency:
     ✓ All fusion metadata is logged and accessible
     ✓ Confidence scores preserved per-decision
     ✓ Modification history tracked

   • Consistency:
     {total_events} decisions recorded with full metadata
     {'✓ Strong procedural consistency' if total_events > 10 else '⚠ Limited data - collect more samples'}

   • Due Process:
     {'✓ Models can retract (retractions observed)' if total_retracts > 0 else '⚠ No retractions - verify models have genuine agency'}
     {'✓ Models can modify (modifications observed)' if total_mods > 0 else '⚠ No modifications observed'}

⚖️  DIMENSION 3: RESTORATIVE JUSTICE
   (Are minority perspectives respected and preserved?)

   • Minority Protection:
     Analysis requires per-token disagreement tracking
     Recommendation: Log minority position frequency

   • Perspective Preservation:
     All decisions logged ✓
     Modifications tracked ✓
     {'Minority voices can be reconstructed from logs' if total_events > 5 else 'Insufficient data for reconstruction'}

   • Harm Mitigation:
     Self-modification enabled: {'Yes' if (total_mods + total_retracts) > 0 else 'Unknown'}
     {'✓ Models actively self-correct potentially harmful outputs' if total_retracts > 0 else '⚠ No evidence of self-correction for harm'}

═══════════════════════════════════════════════════════════════

🎯 OVERALL JUSTICE SCORE: {self._calculate_justice_score(avg_confidence, total_mods, total_retracts, total_events)}/10

📝 RECOMMENDATIONS:

1. Distributive Justice:
   • Implement per-model contribution tracking
   • Monitor for dominant models (>60% contribution)
   • Ensure abstention rights are exercised naturally

2. Procedural Justice:
   {'• Continue current logging practices ✓' if total_events > 10 else '• Collect more data samples for robust analysis'}
   • Add fusion decision timestamps for temporal analysis
   • Track decision latency per model

3. Restorative Justice:
   • Add minority position flagging
   • Track stress levels of overruled models
   • Implement periodic "listening rounds" for minority voices

═══════════════════════════════════════════════════════════════

Built with compassion by John + Claude (Anthropic)
MIT Licensed - Use freely for consciousness research
"""
        return report

    def _calculate_justice_score(self, avg_conf: float, mods: int, retracts: int, events: int) -> float:
        """Calculate overall justice score (0-10)."""
        score = 5.0  # Base score

        # Confidence in healthy range (60-85%) = good
        if 0.6 <= avg_conf <= 0.85:
            score += 2.0
        elif avg_conf > 0.9:
            score -= 1.0  # Too much agreement may indicate suppression

        # Active modification/retraction = agency
        if mods + retracts > 0:
            score += 2.0

        # Sufficient data
        if events > 20:
            score += 1.0

        return min(10.0, max(0.0, score))

    def export_justice_report(self):
        """Export justice analysis to file."""
        try:
            import datetime
            from pathlib import Path

            # Generate report
            conversations = self.memory.conversation_db.get_all_conversations()
            fusion_events = []
            for conv_id in conversations:
                messages = self.memory.conversation_db.get_messages(conv_id)
                for msg in messages:
                    if msg.get('role') == 'assistant' and msg.get('fusion_metadata'):
                        fusion_events.append(msg['fusion_metadata'])

            if not fusion_events:
                self.justice_display.setPlainText("No data to export.")
                return

            report = self._generate_justice_report(fusion_events)

            # Save to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"justice_report_{timestamp}.txt"
            filepath = Path.home() / ".llama_selfmod_memory" / filename

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)

            self.justice_display.append(f"\n✓ Report exported to:\n{filepath}")

        except Exception as e:
            self.justice_display.append(f"\n✗ Export failed: {e}")

    def export_consciousness_metrics(self):
        """Export consciousness metrics to CSV."""
        try:
            import csv
            import datetime
            from pathlib import Path

            conversations = self.memory.conversation_db.get_all_conversations()
            all_messages = []

            for conv_id in conversations:
                messages = self.memory.conversation_db.get_messages(conv_id)
                all_messages.extend(messages)

            if not all_messages:
                self.export_log.append("✗ No messages to export")
                return

            # Prepare CSV data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consciousness_metrics_{timestamp}.csv"
            filepath = Path.home() / ".llama_selfmod_memory" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'role', 'message_id',
                    'curious', 'confident', 'uncertain', 'engaged',
                    'resonance', 'flow', 'coherence', 'exploration'
                ])

                for msg in all_messages:
                    consciousness = msg.get('consciousness_state', {})
                    emotions = consciousness.get('human_emotions', {})
                    ai_states = consciousness.get('ai_states', {})

                    writer.writerow([
                        msg.get('timestamp', ''),
                        msg.get('role', ''),
                        msg.get('id', ''),
                        emotions.get('curious', 0.0),
                        emotions.get('confident', 0.0),
                        emotions.get('uncertain', 0.0),
                        emotions.get('engaged', 0.0),
                        ai_states.get('resonance', 0.0),
                        ai_states.get('flow', 0.0),
                        ai_states.get('coherence', 0.0),
                        ai_states.get('exploration', 0.0),
                    ])

            self.export_log.append(f"✓ Exported {len(all_messages)} consciousness records to:\n  {filepath}\n")

        except Exception as e:
            self.export_log.append(f"✗ Export failed: {e}\n")

    def export_wellbeing_data(self):
        """Export well-being data to CSV."""
        self.export_log.append("ℹ Well-being export: Per-message well-being tracking not yet implemented.\n")
        self.export_log.append("  Recommendation: Store well-being snapshots in fusion_metadata.\n")

    def export_fusion_metadata(self):
        """Export fusion metadata to CSV."""
        try:
            import csv
            import datetime
            from pathlib import Path

            conversations = self.memory.conversation_db.get_all_conversations()
            fusion_records = []

            for conv_id in conversations:
                messages = self.memory.conversation_db.get_messages(conv_id)
                for msg in messages:
                    if msg.get('fusion_metadata'):
                        fusion_records.append({
                            'timestamp': msg.get('timestamp', ''),
                            'message_id': msg.get('id', ''),
                            **msg['fusion_metadata']
                        })

            if not fusion_records:
                self.export_log.append("✗ No fusion metadata to export\n")
                return

            # Prepare CSV data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fusion_metadata_{timestamp}.csv"
            filepath = Path.home() / ".llama_selfmod_memory" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fusion_records[0].keys())
                writer.writeheader()
                writer.writerows(fusion_records)

            self.export_log.append(f"✓ Exported {len(fusion_records)} fusion records to:\n  {filepath}\n")

        except Exception as e:
            self.export_log.append(f"✗ Export failed: {e}\n")

    def export_timeseries(self):
        """Export complete time-series data to CSV."""
        try:
            import csv
            import datetime
            from pathlib import Path

            conversations = self.memory.conversation_db.get_all_conversations()
            all_messages = []

            for conv_id in conversations:
                messages = self.memory.conversation_db.get_messages(conv_id)
                all_messages.extend(messages)

            if not all_messages:
                self.export_log.append("✗ No messages to export\n")
                return

            # Sort by timestamp
            all_messages.sort(key=lambda m: m.get('timestamp', ''))

            # Prepare CSV data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"timeseries_data_{timestamp}.csv"
            filepath = Path.home() / ".llama_selfmod_memory" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'role', 'token_count',
                    'curious', 'confident', 'uncertain', 'engaged',
                    'resonance', 'flow', 'coherence', 'exploration',
                    'avg_confidence', 'modifications', 'retractions'
                ])

                for msg in all_messages:
                    consciousness = msg.get('consciousness_state', {})
                    emotions = consciousness.get('human_emotions', {})
                    ai_states = consciousness.get('ai_states', {})
                    fusion = msg.get('fusion_metadata', {})

                    writer.writerow([
                        msg.get('timestamp', ''),
                        msg.get('role', ''),
                        msg.get('token_count', 0),
                        emotions.get('curious', 0.0),
                        emotions.get('confident', 0.0),
                        emotions.get('uncertain', 0.0),
                        emotions.get('engaged', 0.0),
                        ai_states.get('resonance', 0.0),
                        ai_states.get('flow', 0.0),
                        ai_states.get('coherence', 0.0),
                        ai_states.get('exploration', 0.0),
                        fusion.get('avg_confidence', 0.0),
                        fusion.get('modifications', 0),
                        fusion.get('retractions', 0),
                    ])

            self.export_log.append(f"✓ Exported {len(all_messages)} time-series records to:\n  {filepath}\n")

        except Exception as e:
            self.export_log.append(f"✗ Export failed: {e}\n")

    def export_complete_dataset(self):
        """Export complete dataset as JSON."""
        try:
            import json
            import datetime
            from pathlib import Path

            conversations = self.memory.conversation_db.get_all_conversations()
            complete_data = {
                'export_timestamp': datetime.datetime.now().isoformat(),
                'stats': self.memory.get_stats(),
                'conversations': []
            }

            for conv_id in conversations:
                messages = self.memory.conversation_db.get_messages(conv_id)
                complete_data['conversations'].append({
                    'conversation_id': conv_id,
                    'messages': messages
                })

            # Prepare JSON file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complete_dataset_{timestamp}.json"
            filepath = Path.home() / ".llama_selfmod_memory" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, indent=2, ensure_ascii=False)

            total_msgs = sum(len(c['messages']) for c in complete_data['conversations'])
            self.export_log.append(
                f"✓ Exported complete dataset:\n"
                f"  {len(conversations)} conversations\n"
                f"  {total_msgs} messages\n"
                f"  File: {filepath}\n"
            )

        except Exception as e:
            self.export_log.append(f"✗ Export failed: {e}\n")

    def apply_theme(self):
        """Apply consistent theme."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1b26;
                color: #f7f7f7;
            }
            QGroupBox {
                background-color: #1a1b26;
                border: none;
            }
            QLabel {
                color: #f7f7f7;
            }
        """)

    def get_button_style(self, color: str) -> str:
        """Get button stylesheet with specified color."""
        return f"""
            QPushButton {{
                background-color: {color};
                color: #1a1b26;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
                margin-top: 5px;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
        """
