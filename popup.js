document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const status = document.getElementById('status');
    const startButton = document.getElementById('start');
    const stopButton = document.getElementById('stop');
  
    let stream = null;
    let detectionActive = false;
  
    // Request access to the camera
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function(mediaStream) {
        video.srcObject = mediaStream;
        stream = mediaStream;
      })
      .catch(function(err) {
        console.error("Error accessing the camera: ", err);
        status.textContent = 'Status: Error accessing camera';
      });
  
    startButton.addEventListener('click', function() {
      if (stream && !detectionActive) {
        fetch('http://127.0.0.1:5000/start')  // Ensure this matches your Flask server
          .then(response => response.json())
          .then(data => {
            status.textContent = 'Status: Detection Started';
            detectionActive = true;
          })
          .catch(error => console.error('Error starting detection:', error));
      }
    });
  
    stopButton.addEventListener('click', function() {
      if (detectionActive) {
        fetch('http://127.0.0.1:5000/stop')
          .then(response => response.json())
          .then(data => {
            status.textContent = 'Status: Detection Stopped';
            detectionActive = false;
          })
          .catch(error => console.error('Error stopping detection:', error));
      }
    });
  });
  