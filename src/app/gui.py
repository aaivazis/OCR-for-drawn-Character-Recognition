import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath
from src.app.export import save_strokes_to_json


class DrawingCanvas(QWidget):
    """A widget that allows users to draw lines following the mouse cursor."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: white; border: 4px solid #000000;")
        
        # Drawing state
        self.strokes = []  # List of strokes, each stroke is a list of (x, y, t) tuples
        self.current_stroke = []  # Current stroke being drawn
        self.current_path = None  # For visual rendering
        self.last_point = None
        self.drawing = False
        self.stroke_start_time = None
        
        # Pen settings
        self.pen_color = QColor(0, 0, 0)  # Black
        self.pen_width = 2
        
    def mousePressEvent(self, event):
        """Handle mouse press events - start drawing."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            # Record start time for this stroke
            self.stroke_start_time = time.time()
            # Start a new stroke with first point (t=0.0)
            self.current_stroke = [(event.x(), event.y(), 0.0)]
            # Start a new path for visual rendering
            self.current_path = QPainterPath()
            self.current_path.moveTo(self.last_point)
            
    def mouseMoveEvent(self, event):
        """Handle mouse move events - draw line following cursor."""
        if self.drawing and self.stroke_start_time is not None:
            current_point = event.pos()
            # Calculate time elapsed since stroke started
            elapsed_time = time.time() - self.stroke_start_time
            # Add point with timestamp
            self.current_stroke.append((event.x(), event.y(), elapsed_time))
            # Update visual path
            self.current_path.lineTo(current_point)
            self.last_point = current_point
            self.update()  # Trigger repaint
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events - stop drawing."""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_stroke:
                # Save the completed stroke
                self.strokes.append(self.current_stroke.copy())
                self.current_stroke = []
                self.current_path = None
                self.stroke_start_time = None
                self.last_point = None
                self.update()
                
    def paintEvent(self, event):
        """Paint all strokes on the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set pen for drawing
        pen = QPen(self.pen_color, self.pen_width)
        painter.setPen(pen)
        
        # Draw all saved strokes
        for stroke in self.strokes:
            if len(stroke) > 1:
                path = QPainterPath()
                path.moveTo(stroke[0][0], stroke[0][1])
                for point in stroke[1:]:
                    path.lineTo(point[0], point[1])
                painter.drawPath(path)
        
        # Draw current stroke being drawn
        if self.current_path:
            painter.drawPath(self.current_path)
        
    def clear_canvas(self):
        """Clear all strokes from the canvas."""
        self.strokes = []
        self.current_stroke = []
        self.current_path = None
        self.stroke_start_time = None
        self.last_point = None
        self.update()
    
    def get_all_strokes(self):
        """Return all strokes as a list of lists of (x, y, t) tuples."""
        return self.strokes.copy()
    
    def has_strokes(self):
        """Check if there are any strokes drawn."""
        return len(self.strokes) > 0


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
        
        # Top layout for buttons (aligned to left)
        top_layout = QHBoxLayout()
        top_layout.setAlignment(Qt.AlignLeft)
        
        # Clear canvas button
        clear_btn = QPushButton("Clear Canvas")
        clear_btn.clicked.connect(self.clear_canvas)
        clear_btn.setMinimumWidth(120)
        top_layout.addWidget(clear_btn)
        
        # Send button
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_letter)
        send_btn.setMinimumWidth(120)
        send_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        top_layout.addWidget(send_btn)
        
        top_layout.addStretch()  # Push buttons to the left
        
        main_layout.addLayout(top_layout)
        
        # Drawing canvas
        self.canvas = DrawingCanvas()
        main_layout.addWidget(self.canvas)
        
    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.clear_canvas()
    
    def send_letter(self):
        """Save all strokes and clear canvas for next letter."""
        if not self.canvas.has_strokes():
            return  # Nothing to save
        
        # Get all strokes from canvas
        all_strokes = self.canvas.get_all_strokes()
        
        # Send strokes to export.py to save as JSON in data/raw
        try:
            filepath = save_strokes_to_json(len(all_strokes), all_strokes, output_dir="data/raw")
            print(f"Letter saved to {filepath}")
            print(f"Total strokes: {len(all_strokes)}")
            print(f"Total points: {sum(len(stroke) for stroke in all_strokes)}")
        except Exception as e:
            print(f"Error saving letter: {e}")
        
        # Clear canvas for next letter
        self.canvas.clear_canvas()


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()