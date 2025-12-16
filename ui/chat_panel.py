from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QTextEdit, QLineEdit, QPushButton
from PyQt6.QtCore import pyqtSignal

class ChatPanel(QDockWidget):
    messageSent = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Chat", parent)
        container = QWidget()
        layout = QVBoxLayout(container)

        self.chat_label = QLabel("Assistant Output:")
        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)

        self.chat_input = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._on_send)
        self.chat_input.returnPressed.connect(self._on_send)

        layout.addWidget(self.chat_label)
        layout.addWidget(self.chat_output)
        layout.addWidget(self.chat_input)
        layout.addWidget(self.send_button)

        self.setWidget(container)

    def _on_send(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.messageSent.emit(text)
        # assistant output temporarily disabled
        # self.append_response("(offline) I’m not connected to Gemini.")
        self.chat_input.clear()

    def append_response(self, text: str):
        self.chat_output.append(text)
