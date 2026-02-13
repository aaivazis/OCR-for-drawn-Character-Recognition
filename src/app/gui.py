import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath


class DrawingCanvas(QWidget):
    """A widget that allows users to draw lines following the mouse cursor."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: white; border: 4px solid #000000;")
        
        # Drawing state
        self.paths = []  # List to store all drawn paths
        self.current_path = None
        self.last_point = None
        self.drawing = False
        
        # Pen settings
        self.pen_color = QColor(0, 0, 0)  # Black
        self.pen_width = 2
        
    def mousePressEvent(self, event):
        """Handle mouse press events - start drawing."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            # Start a new path
            self.current_path = QPainterPath()
            self.current_path.moveTo(self.last_point)
            
    def mouseMoveEvent(self, event):
        """Handle mouse move events - draw line following cursor."""
        if self.drawing and self.last_point is not None:
            current_point = event.pos()
            # Add line to current path
            self.current_path.lineTo(current_point)
            self.last_point = current_point
            self.update()  # Trigger repaint
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events - stop drawing."""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_path:
                # Save the completed path
                self.paths.append(self.current_path)
                self.current_path = None
                self.last_point = None
                self.update()
                
    def paintEvent(self, event):
        """Paint all paths on the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set pen for drawing
        pen = QPen(self.pen_color, self.pen_width)
        painter.setPen(pen)
        
        # Draw all saved paths
        for path in self.paths:
            painter.drawPath(path)
        
        # Draw current path being drawn
        if self.current_path:
            painter.drawPath(self.current_path)
        
    def clear_canvas(self):
        """Clear all paths from the canvas."""
        self.paths = []
        self.current_path = None
        self.last_point = None
        self.update()


class DrawingApp(QMainWindow):
    """Main application window with drawing canvas."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Canvas")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main vertical layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(main_layout)
        
        # Top layout for button (aligned to left)
        top_layout = QHBoxLayout()
        top_layout.setAlignment(Qt.AlignLeft)
        
        # Clear canvas button
        clear_btn = QPushButton("Clear Canvas")
        clear_btn.clicked.connect(self.clear_canvas)
        clear_btn.setMinimumWidth(120)
        top_layout.addWidget(clear_btn)
        top_layout.addStretch()  # Push button to the left
        
        main_layout.addLayout(top_layout)
        
        # Drawing canvas
        self.canvas = DrawingCanvas()
        main_layout.addWidget(self.canvas)
        
    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.clear_canvas()


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

