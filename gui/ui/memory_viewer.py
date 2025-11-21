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
